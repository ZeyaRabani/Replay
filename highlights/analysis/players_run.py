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


def _relabel_det_one(args: tuple) -> dict:
    """ProcessPoolExecutor worker: kit-relabel one angle's det npz."""
    video, npz, teams, fps, lo, hi = args
    from .kit import relabel_detections
    return relabel_detections(video, npz, teams, fps=fps,
                              start_s=lo, end_s=hi)


def run_players_v2(project_dir: Path, log=print, force: bool = False,
                   relabel_dets: bool = False, stabilize_only: bool = False,
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
    lo, hi = ctx["window"]
    dirs = _angle_dirs(project_dir)
    if not stabilize_only:
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
        if relabel_dets:
            _upd(stage="relabel", progress=0.85,
                 message="kit-relabelling detections")
            from .detect_hr import FPS
            jobs = []
            for a in range(n):
                v = _match_ext_video(dirs[a]) if a < len(dirs) else None
                off = ctx["offsets"][a] if a < len(ctx["offsets"]) else 0.0
                npz = adir / f"det_a{a}.npz"
                if v is not None and npz.exists():
                    jobs.append((str(v), npz, teams, FPS,
                                 max(0.0, lo - off), max(0.0, hi - off)))
            if jobs:
                import concurrent.futures as cf
                with cf.ProcessPoolExecutor(max_workers=3) as ex:
                    for a, res in zip((j[0] for j in jobs),
                                      ex.map(_relabel_det_one, jobs)):
                        log(f"relabel {Path(a).parent.name}: "
                            f"{res['changed']}/{res['n_dets']} changed, "
                            f"{res['blank']} blank")
    _upd(stage="stabilize", progress=0.88,
         message="stabilising cameras")
    try:
        from .stabilize import run_stabilize
        run_stabilize(project_dir, adir, force=force, log=log)
    except Exception as e:
        log(f"stabilize: skipped ({type(e).__name__}: {e})")
    return _post_detect(project_dir, adir, dirs, ctx, lo, hi, teams,
                        log=log, _upd=_upd)


def _post_detect(project_dir: Path, adir: Path, dirs, ctx: dict,
                 lo: float, hi: float, teams: dict, *, log,
                 _upd) -> dict:
    """Tail of run_players_v2 after detection: crops -> fuse -> kit
    relabel -> groups -> identities."""
    n = int(ctx["n_angles"])
    from .fuse_tracks import run_fuse
    _upd(stage="fuse", progress=0.9, message="fusing tracks")
    videos = {a: _match_ext_video(dirs[a]) for a in range(n)
              if a < len(dirs)}
    crops_dir = adir / "crops"
    if crops_dir.is_dir():
        # track ids restart at 100001 every fusion, so leftover
        # v2_*_*.jpg crops would be reused for unrelated tracks
        shutil.rmtree(crops_dir)
        log("crops: cleared stale directory before re-fuse")
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
    _upd(stage="kit relabel", progress=0.96,
         message="re-voting track teams from crops")
    try:
        from .kit import relabel_tracks
        relabel_tracks(adir, teams, log=log)
    except Exception as e:
        log(f"kit relabel: skipped ({type(e).__name__}: {e})")
    _upd(stage="groups", progress=0.97, message="grouping tracks")
    try:
        from .groups_v2 import build_groups_v2
        g = build_groups_v2(adir)
        log(f"groups: {len(g['groups'])} groups "
            f"({g['n_grouped']}/{g['n_tracks']} tracks)")
    except Exception as e:
        log(f"groups: skipped ({type(e).__name__}: {e})")
    _upd(stage="identities", progress=0.98, message="linking identities")
    try:
        from .identity import build_identities
        build_identities(adir, log=log)
    except Exception as e:
        log(f"identities: skipped ({type(e).__name__}: {e})")
    _upd(stage="done", progress=1.0, message="done")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.players_run")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--groups-only", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="multi-view hi-res pass: detect+ fusion")
    ap.add_argument("--relabel-dets", action="store_true",
                    help="v2: kit-relabel saved det npz before fusion")
    ap.add_argument("--stabilize-only", action="store_true",
                    help="v2: skip detection; stabilize cameras then "
                    "re-fuse from the saved det npz")
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
                                   force=args.force,
                                   relabel_dets=args.relabel_dets,
                                   stabilize_only=args.stabilize_only,
                                   status=status)
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
