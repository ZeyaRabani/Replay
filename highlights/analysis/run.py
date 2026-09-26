"""Analyse-match orchestrator for multi-angle projects.

Stages: teams (YOLO shirt-colour pass on the reference angle), stats,
summary — writing <project>/analysis/{teams,match_stats,summary}.json.

    python -m highlights.analysis.run --project-dir P [--force]

Time axes (shared with the rest of the multiangle pipeline):
  shared_t = file_t(angle i) + sync.offsets[i]
  t_out    = shared_t - lo_out   (UI player time; lo_out = clipped cut lo)
  t_file   = shared_t - offsets[ref_angle]  (reference-angle file seconds)
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

from highlights.io import write_json_atomic
from highlights.pipeline.errors import PipelineError
from highlights.pipeline.joblock import job_slot, workdir_for
from highlights.pipeline.run import _stdout_is
from highlights.pipeline.status import StatusWriter

from . import teams as teams_mod
from .stats import compute_match_stats
from .summary import build_summary

FPS = 0.5
VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".webm", ".avi"]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _angle_dirs(project_dir: Path) -> list[Path]:
    ad = project_dir / "angles"
    if not ad.is_dir():
        return []
    return sorted(d for d in ad.iterdir()
                  if d.is_dir() and d.name.startswith("a"))


def _angle_duration(angle_dir: Path) -> float:
    pr = _load_json(angle_dir / "pipeline" / "probe.json")
    if pr and pr.get("duration_s"):
        return float(pr["duration_s"])
    st = _load_json(angle_dir / "pipeline" / "status.json")
    dur = ((st or {}).get("video") or {}).get("duration_s")
    if dur:
        return float(dur)
    feat = _load_json(angle_dir / "track" / "features_1s.json")
    rows = (feat or {}).get("rows") or []
    if rows:
        cols = feat.get("columns") or []
        ti = cols.index("t") if "t" in cols else 0
        return float(rows[-1][ti]) + 1.0
    return 0.0


def _match_ext_video(angle_dir: Path) -> Path | None:
    for ext in VIDEO_EXTS:
        p = angle_dir / f"match{ext}"
        if p.exists():
            return p
    return None


def resolve_context(project_dir: Path) -> dict:
    """Everything the analysis stages need, on the shared-T axis."""
    project_dir = Path(project_dir)
    ma = project_dir / "multiangle"
    sync = _load_json(ma / "sync.json")
    if not sync or "offsets" not in sync:
        raise PipelineError("no multiangle/sync.json — run the pipeline first")
    offsets = [float(o) for o in sync["offsets"]]
    ulo, uhi = (sync.get("coverage") or {}).get("union") or (0.0, 0.0)
    lo, hi = float(ulo), float(uhi)
    cr = _load_json(ma / "cut_range.json")
    if cr and cr.get("hi") is not None and cr.get("lo") is not None:
        lo2, hi2 = max(lo, float(cr["lo"])), min(hi, float(cr["hi"]))
        if hi2 > lo2:
            lo, hi = lo2, hi2

    dirs = _angle_dirs(project_dir)
    durs = [_angle_duration(d) for d in dirs]
    ref_angle = int(max(range(len(durs)), key=lambda i: durs[i])) if dirs else 0

    ref_dir = project_dir / "angles" / f"a{ref_angle}"
    video = None
    proxy_offset = 0.0
    proxy = ref_dir / "analysis_480p.mp4"
    side = ref_dir / "analysis_480p.json"
    if proxy.exists() and side.exists():
        meta = _load_json(side) or {}
        video = proxy
        proxy_offset = float(meta.get("offset") or 0.0)
    else:
        video = _match_ext_video(ref_dir)
    if video is None:
        raise PipelineError(
            f"no video for angle {ref_angle} — sources purged; "
            f"re-run tracking to rebuild the 480p proxy")

    cands = _load_candidates(project_dir, lo)
    return {
        "project_dir": project_dir,
        "analysis_dir": project_dir / "analysis",
        "offsets": offsets,
        "union": [ulo, uhi],
        "window": (lo, hi),
        "lo_out": lo,
        "ref_angle": ref_angle,
        "n_angles": len(dirs) or len(offsets),
        "video": video,
        "proxy_offset": proxy_offset,
        "shared_offset": offsets[ref_angle] if ref_angle < len(offsets) else 0.0,
        "window_file": (lo - (offsets[ref_angle] if ref_angle < len(offsets) else 0.0),
                        hi - (offsets[ref_angle] if ref_angle < len(offsets) else 0.0)),
        "ref_features": _load_json(ref_dir / "track" / "features_1s.json"),
        "candidates": cands,
        "pitch_type": (_load_json(project_dir / "project.json") or {}).get("pitch_type"),
    }


def _load_candidates(project_dir: Path, lo_out: float) -> list[dict]:
    """Candidates -> [{t_shared, type, status, confidence}]."""
    out = []
    pj = _load_json(project_dir / "project.json") or {}
    for c in pj.get("candidates") or []:
        try:
            t_out = float(c.get("t", c.get("t_out", 0.0)))
        except (TypeError, ValueError):
            continue
        out.append({"t_shared": t_out + lo_out,
                    "type": str(c.get("type", "other")),
                    "status": str(c.get("status") or "pending"),
                    "confidence": float(c.get("confidence") or 0.0)})
    if out:
        return out
    fused = _load_json(project_dir / "multiangle" / "fused_candidates.json") or {}
    for e in fused.get("events") or []:
        try:
            t_out = float(e.get("t", 0.0))
        except (TypeError, ValueError):
            continue
        status = ("rejected" if e.get("cross_validation") == "rejected"
                  else str(e.get("status") or "pending"))
        out.append({"t_shared": t_out + lo_out,
                    "type": str(e.get("type", "other")),
                    "status": status,
                    "confidence": float(e.get("confidence") or 0.0)})
    return out


def estimate_minutes(window_s: float) -> int:
    """Rough runtime: ~4 fps CPU YOLO at imgsz 960, 0.5 fps sampling."""
    return max(1, math.ceil(window_s * FPS / 4 / 60))


def _teams_fresh(path: Path, ctx: dict) -> bool:
    """teams.json is reusable when newer than the video and on the same
    shared window."""
    if not path.exists():
        return False
    if path.stat().st_mtime < ctx["video"].stat().st_mtime:
        return False
    d = _load_json(path) or {}
    ws = d.get("window_shared")
    want = [round(ctx["window"][0], 3), round(ctx["window"][1], 3)]
    return [round(float(x), 3) for x in (ws or [])] == want


def run_analysis(project_dir: Path, log=print, force: bool = False,
                 status: StatusWriter | None = None, detector=None,
                 run_teams=None) -> dict:
    ctx = resolve_context(project_dir)
    adir = ctx["analysis_dir"]
    adir.mkdir(parents=True, exist_ok=True)
    teams_path = adir / "teams.json"
    stats_path = adir / "match_stats.json"
    summary_path = adir / "summary.json"

    def _upd(**kw):
        if status is not None:
            status.update(**kw)

    teams_path_fresh = _teams_fresh(teams_path, ctx)
    if teams_path_fresh and not force:
        log("teams: skip (teams.json up to date)")
        teams = _load_json(teams_path)
    else:
        _upd(stage="teams", progress=0.05,
             message=f"teams pass ~{estimate_minutes(ctx['window'][1] - ctx['window'][0])} min")
        fn = run_teams or teams_mod.run_teams_pass
        teams = fn(str(ctx["video"]), teams_path,
                   window_file=ctx["window_file"],
                   proxy_offset=ctx["proxy_offset"],
                   shared_offset=ctx["shared_offset"],
                   log=log, detector=detector)

    _upd(stage="stats", progress=0.85, message="computing stats")
    stats = compute_match_stats(
        teams, ctx["ref_features"], ctx["candidates"],
        match_window=ctx["window"], lo_out=ctx["lo_out"],
        offsets=ctx["offsets"], ref_angle=ctx["ref_angle"],
        pitch_type=ctx.get("pitch_type"))
    write_json_atomic(stats_path, stats, indent=1)
    log(f"stats: {len(stats['halves'])} halves, {len(stats['shots'])} shots")

    _upd(stage="summary", progress=0.95, message="writing summary")
    summary = build_summary(stats)
    write_json_atomic(summary_path, summary, indent=1)

    _upd(stage="done", progress=1.0, message="done")
    return {"teams": teams, "stats": stats, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.run")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    project_dir = args.project_dir
    adir = project_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(adir / "status.json")

    def on_sigterm(signum, frame):
        status.update(state="failed", error="cancelled",
                      finished_at=time.time(), force=True)
        sys.exit(1)
    signal.signal(signal.SIGTERM, on_sigterm)

    log_path = adir / "log.txt"
    with open(log_path, "a") as log_fh:
        def log(msg: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            if not _stdout_is(log_fh):
                log_fh.write(line + "\n")
                log_fh.flush()
        try:
            status.update(state="running", force=True)
            with job_slot(workdir_for(project_dir), status=status, log=log):
                run_analysis(project_dir, log=log, force=args.force,
                             status=status)
        except PipelineError as e:
            log(f"FAILED: {e}")
            status.update(state="failed", error=str(e),
                          finished_at=time.time(), force=True)
            return 1
        except Exception as e:
            log(f"FAILED ({type(e).__name__}): {e}")
            status.update(state="failed", error=str(e),
                          finished_at=time.time(), force=True)
            return 1
        status.update(state="done", stage="done", progress=1.0,
                      message="done", finished_at=time.time(), force=True)
        log("analysis done")
        return 0


if __name__ == "__main__":
    sys.exit(main())
