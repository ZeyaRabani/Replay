#!/usr/bin/env python3
"""Render selected fusion candidates to clips + a concatenated reel.

Clips: re-encoded libx264 CRF 28 preset veryfast, AAC 96k, max width 960
(aspect preserved), +faststart, exact clip_start/clip_end from
candidates.json. reel.mp4 = concat demuxer, stream copy. manifest.json
maps event ids to files. If total output exceeds 120 MB, re-render with
CRF 31 and max width 800.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAND = os.path.join(HERE, "outputs", "candidates.json")
DEFAULT_OUT = os.path.join(HERE, "outputs", "rendered")

LIMIT_MB = 120


def render(video, cands_path, out_dir, crf, width):
    with open(cands_path) as f:
        doc = json.load(f)
    events = [c for c in doc["candidates"] if c["selected"]]
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    manifest = []
    for c in events:
        name = f"{c['rank']:02d}_{c['type']}_{round(c['t'])}s.mp4"
        path = os.path.join(clips_dir, name)
        dur = c["clip_end"] - c["clip_start"]
        vf = f"scale='min({width},iw)':-2"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-ss", f"{c['clip_start']:.3f}", "-i", video,
               "-t", f"{dur:.3f}", "-vf", vf,
               "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
               "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
               "-y", path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {name}: {r.stderr[-300:]}", file=sys.stderr)
            continue
        manifest.append({"id": c["id"], "rank": c["rank"],
                         "type": c["type"], "t": c["t"],
                         "clip_start": c["clip_start"],
                         "clip_end": c["clip_end"],
                         "path": path, "duration_s": dur})
        print(f"  {name} ({dur:.1f}s)", flush=True)

    concat = os.path.join(clips_dir, "concat.txt")
    with open(concat, "w") as f:
        for m in sorted(manifest, key=lambda m: m["rank"]):
            f.write(f"file '{os.path.basename(m['path'])}'\n")
    reel = os.path.join(out_dir, "reel.mp4")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", concat,
                        "-c", "copy", "-y", reel],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"reel FAIL: {r.stderr[-300:]}", file=sys.stderr)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(os.path.getsize(m["path"]) for m in manifest)
    if os.path.exists(reel):
        total += os.path.getsize(reel)
    return manifest, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/home/ubuntu/match/match.mp4")
    ap.add_argument("--candidates", default=DEFAULT_CAND)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    for crf, w in [(28, 960), (31, 800)]:
        manifest, total = render(args.video, args.candidates, args.out,
                                 crf, w)
        mb = total / 1e6
        print(f"crf={crf} width={w}: {len(manifest)} clips, total {mb:.1f} MB",
              flush=True)
        if mb <= LIMIT_MB:
            break
        print(f"over {LIMIT_MB} MB -> rerender with crf=31 w=800", flush=True)
    for m in manifest:
        sz = os.path.getsize(m["path"]) / 1e6
        if sz > 100:
            print(f"WARNING {m['path']} = {sz:.1f} MB", file=sys.stderr)
    print("done", flush=True)


if __name__ == "__main__":
    main()
