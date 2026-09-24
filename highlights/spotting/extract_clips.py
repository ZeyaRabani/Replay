#!/usr/bin/env python3
"""Cut short review clips (and a contact-sheet of frames) for the top-N events in events.json."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("events_json")
    ap.add_argument("-o", "--out_dir", default=str(Path.home() / "match/review_clips"))
    ap.add_argument("-n", "--top", type=int, default=10)
    ap.add_argument("--clip_s", type=float, default=6.0)
    ap.add_argument("--match_only", action="store_true", help="skip events with signals.kind == warmup")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.events_json) as fp:
        events = json.load(fp)["events"]
    if args.match_only:
        events = [e for e in events if e["signals"].get("kind") != "warmup"]
    for rank, e in enumerate(events[: args.top], 1):
        start = max(0.0, e["t"] - args.clip_s / 2)
        tag = f"{rank:02d}_{e['signals']['class'].replace(' ', '_')}_{e['t']:.0f}s_p{e['confidence']:.2f}"
        subprocess.check_call([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", args.video,
            "-t", f"{args.clip_s:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-an", str(out / f"{tag}.mp4"),
        ])
        # 6-frame contact sheet, one frame per second, full source resolution
        subprocess.check_call([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", args.video,
            "-t", f"{args.clip_s:.2f}", "-vf", "fps=1,scale=640:-1,tile=3x2", "-frames:v", "1", str(out / f"{tag}.jpg"),
        ])
        print(tag)


if __name__ == "__main__":
    main()
