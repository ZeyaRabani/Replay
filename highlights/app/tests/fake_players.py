"""Fake players-analysis runner for tests.

Mimics `python -m highlights.analysis.players_run`: progresses
analysis/players/status.json queued -> running -> done and emits
analysis/players/{tracklets.json,argv.json} plus three crop jpgs.

Env:
  FAKE_P_SLEEP   seconds between state transitions (default 0.05)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def write_status(pdir: Path, status: dict) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    tmp = pdir / "status.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(pdir / "status.json")


def _tr(tid, team, t0, t1, dist):
    return {"id": tid, "team": team, "t_start": t0, "t_end": t1,
            "n_samples": int((t1 - t0) * 2), "distance_m": dist,
            "sprints": 1, "crops": [f"{tid}_{i}.jpg" for i in range(3)],
            "path": [[t0 + i * 0.5, 0.5, 0.9] for i in range(4)]}


TRACKLETS = {
    "fps": 2.0,
    "window_shared": [0.0, 20.0],
    "shared_offset": 0.0,
    "pitch_len_m": 100.0,
    "n_frames": 40,
    "generated_at": 0.0,
    "tracklets": [
        _tr(1, "A", 0.0, 10.0, 55.0),
        _tr(2, "A", 2.0, 15.0, 30.0),
        _tr(3, "B", 5.0, 20.0, 80.0),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pdir = Path(args.project_dir) / "analysis" / "players"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "argv.json").write_text(json.dumps(sys.argv[1:]))

    sleep = float(os.environ.get("FAKE_P_SLEEP", "0.05"))
    now = time.time()
    status = {"state": "queued", "stage": "players", "progress": 0.0,
              "stage_progress": 0.0, "message": "queued", "error": None,
              "started_at": now, "updated_at": now, "finished_at": None,
              "pid": os.getpid()}
    write_status(pdir, status)
    time.sleep(sleep)

    status.update(state="running", progress=0.5, message="running",
                  updated_at=time.time())
    write_status(pdir, status)
    time.sleep(sleep)

    (pdir / "tracklets.json").write_text(json.dumps(TRACKLETS, indent=1))
    crops = pdir / "crops"
    crops.mkdir(exist_ok=True)
    import cv2
    import numpy as np
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    for tid in (1, 2, 3):
        for i in range(3):
            img[:, :] = (tid * 40, 128, i * 80)
            cv2.imwrite(str(crops / f"{tid}_{i}.jpg"), img)

    status.update(state="done", stage="done", progress=1.0,
                  message="done", updated_at=time.time(),
                  finished_at=time.time())
    write_status(pdir, status)


if __name__ == "__main__":
    main()
