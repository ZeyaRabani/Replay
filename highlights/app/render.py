"""CLI: cut highlight clips + build a reel straight from a candidates.json.

    python -m highlights.app.render --video match.mp4 --candidates candidates.json --out reel/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .backend import ffmpeg as fx
from .backend.schemas import CandidatesFile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="highlights.app.render")
    p.add_argument("--video", required=True, help="input video file")
    p.add_argument("--candidates", required=True, help="candidates.json")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--only", choices=["confirmed", "all"], default=None,
                   help="default: events whose cross_validation != rejected")
    p.add_argument("--types", default=None, help="comma list of event types to keep")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--pad-goal", type=float, default=5.0)
    p.add_argument("--pad-default", type=float, default=3.0)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--reencode", action="store_true")
    p.add_argument("--max-clips", type=int, default=None, help="top N by confidence")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    video = Path(args.video)
    if not video.is_file():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1
    cf = CandidatesFile(**json.loads(Path(args.candidates).read_text()))
    info = fx.probe(video)
    duration = info["duration_s"] or cf.video_duration_s

    types = set(args.types.split(",")) if args.types else None
    events = []
    for i, ev in enumerate(cf.events):
        if args.only == "all":
            pass
        elif args.only == "confirmed":
            if ev.cross_validation != "confirmed":
                continue
        elif ev.cross_validation == "rejected":
            continue
        if types and ev.type not in types:
            continue
        if ev.confidence < args.min_confidence:
            continue
        events.append((i, ev))
    events.sort(key=lambda x: -x[1].confidence)
    if args.max_clips:
        events = events[: args.max_clips]

    items = []
    for rank, (i, ev) in enumerate(events, start=1):
        pad = args.pad_goal if ev.type == "goal" else args.pad_default
        start = max(0.0, ev.t - pad)
        end = min(duration, ev.t + pad) if duration > 0 else ev.t + pad
        items.append(
            {
                "id": f"c{i + 1:03d}",
                "t": ev.t,
                "type": ev.type,
                "clip_start": start,
                "clip_end": end,
                "confidence": ev.confidence,
                "name": f"{rank:02d}_{ev.type}_{ev.t:07.1f}.mp4",
            }
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def cb(frac: float, msg: str) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}", flush=True)

    res = fx.render_reel(video, items, out, overlay=not args.no_overlay, reencode=args.reencode, progress_cb=cb)

    manifest = [
        {
            "id": it["id"], "t": it["t"],
            "clip_start": it["clip_start"], "clip_end": it["clip_end"],
            "path": r["path"], "duration": r["duration"],
        }
        for it, r in zip(items, res["clips"], strict=True)
    ]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    counts: dict[str, int] = {}
    xv: dict[str, int] = {}
    for ev in cf.events:
        counts[ev.type] = counts.get(ev.type, 0) + 1
        xv[ev.cross_validation] = xv.get(ev.cross_validation, 0) + 1
    stats = {
        "source": cf.source,
        "video_duration_s": duration,
        "n_candidates": len(cf.events),
        "counts_by_type": counts,
        "counts_by_cross_validation": xv,
        "rendered": {
            "n_clips": len(res["clips"]),
            "total_clip_s": sum(r["duration"] for r in res["clips"]),
            "reel_s": res["reel_s"],
        },
        "timeline": [
            {
                "id": it["id"], "type": it["type"], "t": it["t"],
                "clip_start": it["clip_start"], "clip_end": it["clip_end"],
                "confidence": it["confidence"],
            }
            for it in sorted(items, key=lambda c: c["t"])
        ],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"done: {len(res['clips'])} clips, reel {res['reel']}, {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
