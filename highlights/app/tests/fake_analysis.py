"""Fake analysis runner for tests.

Mimics `python -m highlights.analysis.run`: writes analysis/argv.json,
progresses analysis/status.json queued -> running -> done, and emits
analysis/{match_stats,summary}.json fixtures.

Env: FAKE_A_SLEEP seconds between state transitions (default 0.05).
"""

import argparse
import json
import os
import time
from pathlib import Path


def write_status(pdir: Path, status: dict) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    tmp = pdir / "status.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(pdir / "status.json")


STATS = {
    "match_window": [0.0, 600.0], "lo_out": 0.0, "ref_angle": 0,
    "offsets": [0.0, -748.2, -520.2], "pitch_type": None,
    "teams": {"A": {"name": "orange", "hex": "#f08c00"},
              "B": {"name": "white", "hex": "#ebebeb"}},
    "team_confidence": 0.72,
    "halves": [
        {"index": 1, "start": 0.0, "end": 300.0, "start_out": 0.0,
         "start_file": 0.0, "ball_visible_pct": 40.0, "contested_pct": 5.0,
         "teams": {"A": {"possession_pct": 58.0, "attacking_third_s": 90,
                         "shots": 2, "goals_confirmed": 1,
                         "goals_estimated": 1, "attacks_right": True},
                   "B": {"possession_pct": 42.0, "attacking_third_s": 30,
                         "shots": 1, "goals_confirmed": 0,
                         "goals_estimated": 1, "attacks_right": False}}},
        {"index": 2, "start": 300.0, "end": 600.0, "start_out": 300.0,
         "start_file": 300.0, "ball_visible_pct": 38.0, "contested_pct": 6.0,
         "teams": {"A": {"possession_pct": 51.0, "attacking_third_s": 60,
                         "shots": 1, "goals_confirmed": 0,
                         "goals_estimated": 0, "attacks_right": False},
                   "B": {"possession_pct": 49.0, "attacking_third_s": 55,
                         "shots": 2, "goals_confirmed": 0,
                         "goals_estimated": 0, "attacks_right": True}}},
    ],
    "totals": {
        "A": {"possession_pct": 55.0, "attacking_third_s": 150, "shots": 3,
              "goals_confirmed": 1, "goals_estimated": 1,
              "distance_norm": 30.0, "distance_m_est": 3000.0, "sprints": 10,
              "tracked_player_seconds": 4000},
        "B": {"possession_pct": 45.0, "attacking_third_s": 85, "shots": 3,
              "goals_confirmed": 0, "goals_estimated": 1,
              "distance_norm": 25.0, "distance_m_est": 2500.0, "sprints": 8,
              "tracked_player_seconds": 3600},
    },
    "shots": [{"t_shared": 100.0, "t_out": 100.0, "t_file": 100.0,
               "type": "goal", "confidence": 0.9, "status": "confirmed",
               "team": "A", "half": 1}],
    "caveats": ["Estimated from footage — not official statistics"],
}

SUMMARY = {
    "text": "Orange vs white — estimated from 3 angles' footage. "
            "Estimated from footage — not official statistics.",
    "bullets": [{"t_shared": 100.0, "t_out": 100.0, "t_file": 100.0,
                 "label": "goal (A)"}],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    adir = Path(args.project_dir) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "argv.json").write_text(json.dumps(os.sys.argv[1:]))

    sleep = float(os.environ.get("FAKE_A_SLEEP", "0.05"))
    now = time.time()
    status = {"state": "queued", "stage": "teams", "progress": 0.0,
              "stage_progress": 0.0, "message": "queued", "error": None,
              "started_at": now, "updated_at": now, "finished_at": None,
              "pid": os.getpid()}
    write_status(adir, status)
    time.sleep(sleep)

    status.update(state="running", progress=0.5, message="running",
                  updated_at=time.time())
    write_status(adir, status)
    time.sleep(sleep)

    (adir / "match_stats.json").write_text(json.dumps(STATS, indent=1))
    (adir / "summary.json").write_text(json.dumps(SUMMARY, indent=1))

    status.update(state="done", stage="done", progress=1.0,
                  message="done", updated_at=time.time(),
                  finished_at=time.time())
    write_status(adir, status)


if __name__ == "__main__":
    main()
