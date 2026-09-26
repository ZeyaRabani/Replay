"""Fake pipeline runner for tests.

Mimics highlights.pipeline.run: writes pipeline/argv.json, progresses
status.json queued -> running -> done, and emits pipeline/candidates.json.

Env:
  FAKE_PIPELINE_SLEEP  seconds between state transitions (default 0.05)
  FAKE_PIPELINE_HANG=1 stay in running state forever
  FAKE_PIPELINE_CRASH=1 write running then os._exit(1)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--youtube-url")
    ap.add_argument("--video")
    ap.add_argument("--stages")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cookies")
    args = ap.parse_args()

    root = Path(args.project_dir)
    pdir = root / "pipeline"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "argv.json").write_text(json.dumps(sys.argv[1:]))

    sleep = float(os.environ.get("FAKE_PIPELINE_SLEEP", "0.05"))
    now = time.time()
    status = {
        "state": "queued", "stage": "download" if args.youtube_url else "probe",
        "progress": 0.0, "stage_progress": 0.0, "message": "queued",
        "error": None, "started_at": now, "updated_at": now,
        "finished_at": None, "pid": os.getpid(),
        "video_path": None, "video": None, "download": None,
    }
    write_status(pdir, status)
    print("fake pipeline: queued", flush=True)
    time.sleep(sleep)

    status.update(state="running", stage="probe", progress=0.5,
                  stage_progress=0.5, message="running", updated_at=time.time())
    write_status(pdir, status)
    print("fake pipeline: running", flush=True)

    if os.environ.get("FAKE_PIPELINE_CRASH") == "1":
        os._exit(1)

    if os.environ.get("FAKE_PIPELINE_HANG") == "1":
        while True:
            time.sleep(0.2)
            status["updated_at"] = time.time()
            write_status(pdir, status)

    time.sleep(sleep)

    video_path = None
    video = None
    if args.video:
        video_path = str(Path(args.video).resolve())
        video = {"duration_s": 20.0, "width": 320, "height": 240, "fps": 10}
        try:
            from highlights.app.backend.ffmpeg import probe
            video = probe(video_path)
        except Exception:
            pass
    status.update(state="done", stage="done", progress=1.0,
                  stage_progress=1.0, message="done",
                  updated_at=time.time(), finished_at=time.time(),
                  video_path=video_path, video=video)
    write_status(pdir, status)

    cands = {
        "source": "fake_pipeline",
        "video_duration_s": 20.0,
        "events": [
            {"type": "shot", "t": 5.0, "t_start": 4.0, "t_end": 6.0,
             "confidence": 0.7, "signals": {}, "notes": "",
             "cross_validation": "pipeline_only"},
            {"type": "chance", "t": 12.0, "t_start": 11.0, "t_end": 13.0,
             "confidence": 0.4, "signals": {}, "notes": "",
             "cross_validation": "pipeline_only"},
        ],
    }
    (pdir / "candidates.json").write_text(json.dumps(cands, indent=2))
    print("fake pipeline: done", flush=True)


if __name__ == "__main__":
    main()
