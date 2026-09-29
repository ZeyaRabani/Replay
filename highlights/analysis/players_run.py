"""Per-player analysis orchestrator for multi-angle projects.

Runs the players pass (ByteTrack tracklets + team votes + crops) over
the reference-angle window, writing <project>/analysis/players/
{status.json,log.txt,tracklets.json,crops/}.

    python -m highlights.analysis.players_run --project-dir P [--force]
        [--groups-only]   # rebuild groups.json from existing tracklets
"""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
import time
from pathlib import Path

from highlights.io import write_json_atomic
from highlights.pipeline.errors import PipelineError
from highlights.pipeline.joblock import job_slot, workdir_for
from highlights.pipeline.run import _stdout_is
from highlights.pipeline.status import StatusWriter

from .players import run_players_pass
from .run import _angle_dirs, _load_json, _match_ext_video, estimate_minutes, resolve_context


def run_players(project_dir: Path, log=print, force: bool = False,
                status: StatusWriter | None = None, tracker=None,
                groups_only: bool = False) -> dict:
    ctx = resolve_context(project_dir)
    adir = ctx["analysis_dir"] / "players"
    adir.mkdir(parents=True, exist_ok=True)
    teams_path = ctx["analysis_dir"] / "teams.json"
    teams = _load_json(teams_path)
    if not teams or not teams.get("teams"):
        raise PipelineError("run Team analysis first")

    def _upd(**kw):
        if status is not None:
            status.update(**kw)

    tracklets_path = adir / "tracklets.json"
    if groups_only:
        if not tracklets_path.exists():
            raise PipelineError("groups-only needs tracklets.json")
        doc = _load_json(tracklets_path)
    elif tracklets_path.exists() and not force:
        log("tracklets: skip (up to date)")
        doc = _load_json(tracklets_path)
    else:
        window_s = ctx["window"][1] - ctx["window"][0]
        _upd(stage="players", progress=0.05,
             message=f"players pass ~{estimate_minutes(window_s)} min")
        crops_dir = adir / "crops"
        if crops_dir.is_dir():
            shutil.rmtree(crops_dir)
        doc = run_players_pass(
            str(ctx["video"]), adir,
            window_file=ctx["window_file"],
            proxy_offset=ctx["proxy_offset"],
            shared_offset=ctx["shared_offset"],
            teams=teams, pitch_type=ctx.get("pitch_type"),
            log=log, status=status, tracker=tracker)
        # tracklet ids are renumbered on every re-run — clear the
        # roster's stale assignments but keep names/teams/scorers
        roster_path = adir / "roster.json"
        roster = _load_json(roster_path)
        if roster:
            clean = {"players": [dict(pl, tracklet_ids=[])
                                 for pl in roster.get("players") or []],
                     "scorers": roster.get("scorers") or {},
                     "hidden_tracklet_ids": []}
            write_json_atomic(roster_path, clean)
            log(f"roster: {len(clean['players'])} players kept, "
                "tracklet assignments cleared (ids renumbered)")

    _upd(stage="players", progress=0.95, message="grouping tracklets")
    try:
        from .groups import build_groups
        g = build_groups(adir)
        log(f"groups: {len(g['groups'])} groups "
            f"({g['n_grouped']}/{g['n_tracklets']} tracklets)")
    except Exception as e:
        log(f"groups: skipped ({type(e).__name__}: {e})")

    _upd(stage="done", progress=1.0, message="done")
    return doc


def run_players_v2(project_dir: Path, log=print, force: bool = False,
                   status: StatusWriter | None = None) -> dict:
    """Multi-view hi-res tracking: detect_hr per angle, then fuse into
    pitch-space tracks under analysis/players_v2/."""
    ctx = resolve_context(project_dir)
    adir = ctx["analysis_dir"] / "players_v2"
    adir.mkdir(parents=True, exist_ok=True)
    teams_path = ctx["analysis_dir"] / "teams.json"
    teams = _load_json(teams_path)
    if not teams or not teams.get("teams"):
        raise PipelineError("run Team analysis first")

    calib = _load_json(project_dir / "multiangle" / "calib.json") or {}
    n = int(ctx["n_angles"])
    missing = [a for a in range(n) if str(a) not in calib.get("angles", {})]
    if missing:
        raise PipelineError(
            f"pitch calibration missing for angle(s) {missing} — "
            "set landmarks first")

    def _upd(**kw):
        if status is not None:
            status.update(**kw)

    from .detect_hr import detect_angle
    from .fuse_tracks import run_fuse
    lo, hi = ctx["window"]
    dirs = _angle_dirs(project_dir)
    for a in range(n):
        v = _match_ext_video(dirs[a]) if a < len(dirs) else None
        if v is None:
            raise PipelineError(
                f"no source video for angle {a} — sources purged?")
        off = ctx["offsets"][a] if a < len(ctx["offsets"]) else 0.0
        _upd(stage=f"detect a{a}", progress=0.05 + 0.8 * a / max(1, n),
             stage_progress=0.0,
             message=f"hi-res detection angle {a + 1}/{n}")
        detect_angle(str(v), adir / f"det_a{a}.npz",
                     start_s=max(0.0, lo - off),
                     end_s=max(0.0, hi - off),
                     teams=teams, log=log)
    _upd(stage="fuse", progress=0.9, message="fusing tracks")
    videos = {a: _match_ext_video(dirs[a]) for a in range(n)
              if a < len(dirs)}
    crops_dir = adir / "crops"
    crops_dir.mkdir(exist_ok=True)

    def _crops_for(tr):
        """Up to 6 crops per track: best box from evenly spaced steps,
        cut from that angle's 1080p source via ffmpeg."""
        import subprocess as sp
        refs = tr.get("refs") or []
        if not refs:
            return []
        picks = [refs[int(i * (len(refs) - 1) / 5)] for i in
                 range(min(6, len(refs)))]
        jobs: list[tuple[str, Path, sp.Popen | None]] = []
        for j, (_k, cands) in enumerate(picks):
            best = max(cands, key=lambda r: (r["box"][2] - r["box"][0])
                       * (r["box"][3] - r["box"][1]))
            v = videos.get(best["angle"])
            if v is None:
                continue
            x1, y1, x2, y2 = [int(b) for b in best["box"]]
            name = f"v2_{tr['id']}_{j}.jpg"
            dst = crops_dir / name
            proc = None
            if not dst.exists():
                proc = sp.Popen(["ffmpeg", "-v", "error", "-ss",
                                 f"{best['t_file']:.2f}", "-i", str(v),
                                 "-frames:v", "1", "-vf",
                                 f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}",
                                 "-q:v", "3", "-y", str(dst)],
                                stdout=sp.DEVNULL, stderr=sp.DEVNULL)
            jobs.append((name, dst, proc))
        out = []
        for name, dst, proc in jobs:
            if proc is not None:
                proc.wait()
            if dst.exists():
                out.append(name)
        return out

    doc = run_fuse(project_dir, adir, log=log, crops_for=_crops_for)
    _upd(stage="done", progress=1.0, message="done")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.players_run")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--groups-only", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="multi-view hi-res pass: detect+ fusion")
    args = ap.parse_args(argv)

    project_dir = args.project_dir
    pdir = project_dir / "analysis" / "players"
    pdir.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(pdir / "status.json")

    def on_sigterm(signum, frame):
        status.update(state="failed", error="cancelled",
                      finished_at=time.time(), force=True)
        sys.exit(1)
    signal.signal(signal.SIGTERM, on_sigterm)

    log_path = pdir / "log.txt"
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
                if args.v2:
                    run_players_v2(project_dir, log=log,
                                   force=args.force, status=status)
                else:
                    run_players(project_dir, log=log, force=args.force,
                                status=status,
                                groups_only=args.groups_only)
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
        log("players analysis done")
        return 0


if __name__ == "__main__":
    sys.exit(main())
