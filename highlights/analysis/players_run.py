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
from .run import _load_json, estimate_minutes, resolve_context


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.players_run")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--groups-only", action="store_true")
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
                run_players(project_dir, log=log, force=args.force,
                            status=status, groups_only=args.groups_only)
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
