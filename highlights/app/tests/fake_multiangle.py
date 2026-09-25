"""Fake multiangle runner for tests.

Mimics highlights.multiangle.run: writes multiangle/argv.json, progresses
multiangle/status.json queued -> running -> terminal, and emits
multiangle/{sync,director}.json plus pipeline/{candidates,stats}.json.

Env:
  FAKE_MA_SLEEP    seconds between state transitions (default 0.05)
  FAKE_MA_MODE     "done" (default) or "needs_input" (low sync confidence)
  FAKE_MA_VIDEO    path to a video copied to <project>/match.mp4
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

DUR = 20.0


def write_status(pdir: Path, status: dict) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    tmp = pdir / "status.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(pdir / "status.json")


def n_angles(root: Path) -> int:
    try:
        src = json.loads((root / "project.json").read_text()).get("source", {})
        return len(src.get("angles") or [])
    except Exception:
        adir = root / "angles"
        return len([d for d in adir.iterdir() if d.is_dir()]) if adir.is_dir() else 0


def write_angle_statuses(root: Path) -> None:
    adir = root / "angles"
    if not adir.is_dir():
        return
    for d in sorted(adir.iterdir()):
        if not d.is_dir():
            continue
        pdir = d / "pipeline"
        write_status(pdir, {
            "state": "done", "stage": "done", "progress": 1.0,
            "stage_progress": 1.0, "message": "done", "error": None,
            "started_at": time.time(), "updated_at": time.time(),
            "finished_at": time.time(), "pid": os.getpid(),
            "video_path": str((d / "match.mp4").resolve()),
            "video": {"duration_s": DUR, "width": 320, "height": 240, "fps": 10},
            "download": None,
        })


def sync_json(offsets: list[float] | None, n: int, needs_input: bool) -> dict:
    manual = offsets is not None
    if offsets is None:
        offsets = [0.0] + [12.34, -3.21, 7.0][: max(0, n - 1)]
    pairs = []
    for b in range(1, n):
        pairs.append({
            "a": 0, "b": b,
            "offset": offsets[b] if b < len(offsets) else 0.0,
            "pnr": 14.2, "r2": 0.31,
            "confident": not (needs_input and b == 1),
        })
    return {
        "reference": 0,
        "method": "manual" if manual else "xcorr",
        "offsets": offsets,
        "pairs": pairs,
        "triangle_residual_s": 0.08,
        "needs_manual": [1] if needs_input else [],
        "coverage": {"reference_s": DUR},
    }


DIRECTOR = {
    "segments": [
        {"t_start": 0.0, "t_end": 4.0, "angle": 0, "rule": "coverage",
         "score": 0.3, "runner_up": {"angle": 1, "score": 0.2}},
        {"t_start": 4.0, "t_end": 8.5, "angle": 1, "rule": "ball",
         "score": 0.8, "runner_up": {"angle": 0, "score": 0.5}},
        {"t_start": 8.5, "t_end": 12.0, "angle": 2, "rule": "cluster",
         "score": 0.7, "runner_up": {"angle": 1, "score": 0.4}},
        {"t_start": 12.0, "t_end": 15.5, "angle": 0, "rule": "hold",
         "score": 0.6, "runner_up": {"angle": 2, "score": 0.3}},
        {"t_start": 15.5, "t_end": 18.0, "angle": 2, "rule": "ball",
         "score": 0.9, "runner_up": {"angle": 0, "score": 0.5}},
        {"t_start": 18.0, "t_end": 20.0, "angle": 1, "rule": "cluster",
         "score": 0.6, "runner_up": {"angle": 2, "score": 0.4}},
    ],
    "per_second_rule": {"ball": 812, "cluster": 4310, "hold": 220},
    "ratios": {"ball": 0.152, "cluster": 0.807, "hold": 0.041},
    "n_cuts": 133,
    "mean_hold_s": 40.2,
    "angle_share": {"0": 0.35, "1": 0.31, "2": 0.34},
}


def candidates_json() -> dict:
    return {
        "source": "multiangle",
        "video_duration_s": DUR,
        "events": [
            {"type": "goal", "t": 5.0, "t_start": 4.0, "t_end": 6.0,
             "confidence": 0.9,
             "signals": {"team": "home", "angles": [0, 1],
                         "angle_conf": {"0": 0.9, "1": 0.8}},
             "notes": "cross-confirmed by angles 0,1",
             "cross_validation": "confirmed"},
            {"type": "goal", "t": 12.0, "t_start": 11.0, "t_end": 13.0,
             "confidence": 0.5,
             "signals": {"angles": [2], "angle_conf": {"2": 0.5}},
             "notes": "single-angle (angle 2: Far side)",
             "cross_validation": "pipeline_only"},
            {"type": "shot", "t": 16.0, "t_start": 15.0, "t_end": 17.0,
             "confidence": 0.6,
             "signals": {"angles": [0, 2], "angle_conf": {"0": 0.6, "2": 0.5},
                         "disputed": True, "types": ["shot", "chance"]},
             "notes": "",
             "cross_validation": "pipeline_only"},
        ],
    }


def stats_json(sync: dict, director: dict) -> dict:
    return {
        "duration_s": DUR,
        "match_window": [0.0, DUR],
        "halves": [],
        "bin_s": 60,
        "timeline": [],
        "events_by_type": {"goal": 2, "shot": 1},
        "events_per_10min": [],
        "top_moments": [],
        "whistles": [],
        "activity": {"mean_motion": 0.0, "peak_motion_t": 0.0,
                     "loudest_t": 0.0, "quietest_stretch": [0.0, 0.0]},
        "pipeline": {"model": "fake_multiangle", "auroc_reference": None,
                     "notes": "test fixture"},
        "multiangle": {
            "sync": sync,
            "director": {
                "ratios": director["ratios"],
                "n_cuts": director["n_cuts"],
                "mean_hold_s": director["mean_hold_s"],
                "angle_share": director["angle_share"],
            },
            "confirmation": {"cross": 1, "single": 1, "disputed": 1},
            "score": {
                "home": {"label": "Green end", "goals": 0},
                "away": {"label": "Blue end", "goals": 0},
                "basis": "confirmed goals with team set",
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--stages")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cookies")
    ap.add_argument("--offsets")
    args = ap.parse_args()

    root = Path(args.project_dir)
    pdir = root / "multiangle"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "argv.json").write_text(json.dumps(sys.argv[1:]))

    offsets = None
    if args.offsets:
        offsets = [float(x) for x in args.offsets.split(",")]

    sleep = float(os.environ.get("FAKE_MA_SLEEP", "0.05"))
    now = time.time()
    status = {
        "state": "queued", "stage": "download",
        "progress": 0.0, "stage_progress": 0.0, "message": "queued",
        "error": None, "started_at": now, "updated_at": now,
        "finished_at": None, "pid": os.getpid(),
        "video_path": None, "video": None, "download": None,
    }
    write_status(pdir, status)
    print("fake multiangle: queued", flush=True)
    time.sleep(sleep)

    status.update(state="running", stage="sync", progress=0.5,
                  stage_progress=0.5, message="running",
                  updated_at=time.time())
    write_status(pdir, status)
    print("fake multiangle: running", flush=True)
    time.sleep(sleep)

    n = n_angles(root)
    needs_input = os.environ.get("FAKE_MA_MODE") == "needs_input"
    sync = sync_json(offsets, n, needs_input)
    (pdir / "sync.json").write_text(json.dumps(sync, indent=2))
    write_angle_statuses(root)

    if needs_input:
        status.update(state="needs_input", stage="sync",
                      message="Sync confidence low for angle(s) 1 — "
                              "enter offsets",
                      error=None, updated_at=time.time(),
                      finished_at=time.time())
        write_status(pdir, status)
        print("fake multiangle: needs_input", flush=True)
        return

    director = dict(DIRECTOR)
    (pdir / "director.json").write_text(json.dumps(director, indent=2))

    video_path = None
    video = None
    src_video = os.environ.get("FAKE_MA_VIDEO")
    if src_video and Path(src_video).is_file():
        dst = root / "match.mp4"
        shutil.copyfile(src_video, dst)
        video_path = str(dst.resolve())
        video = {"duration_s": DUR, "width": 320, "height": 240, "fps": 10}
        try:
            from highlights.app.backend.ffmpeg import probe
            video = probe(video_path)
        except Exception:
            pass

    pl = root / "pipeline"
    pl.mkdir(parents=True, exist_ok=True)
    (pl / "candidates.json").write_text(json.dumps(candidates_json(), indent=2))
    (pl / "stats.json").write_text(json.dumps(stats_json(sync, director), indent=2))

    status.update(state="done", stage="done", progress=1.0,
                  stage_progress=1.0, message="done",
                  updated_at=time.time(), finished_at=time.time(),
                  video_path=video_path, video=video)
    write_status(pdir, status)
    print("fake multiangle: done", flush=True)


if __name__ == "__main__":
    main()
