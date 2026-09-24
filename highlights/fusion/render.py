#!/usr/bin/env python3
"""Render selected fusion candidates to clips + a concatenated reel.

Quality presets (--quality, default high):
  preview : CRF28 veryfast, max width 960, AAC 96k  (previous deliverable;
            also the only mode where a size-over-120MB fallback re-render
            applies, at CRF31/width 800)
  high    : source resolution (no scale), CRF14 preset slow, yuv420p,
            AAC 256k — visually transparent, MP4, app/browser-compatible
  max     : source resolution (no scale), CRF0 preset veryslow, yuv420p,
            FLAC — mathematically lossless video + lossless audio in MKV
            (Matroska; FLAC-in-MP4 is not consistently supported).
            Archival quality, NOT for web playback.

Exact clip_start/clip_end cuts at the original fps; no scaling, frame-rate
conversion, or filters in high/max. Manifest is a top-level object
{quality, codec_settings, clips: [...]}.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAND = os.path.join(HERE, "outputs", "candidates.json")

PREVIEW_LIMIT_MB = 120


def codec_args(quality):
    """(extension, video codec args, audio codec args, scale_filter|None)."""
    if quality == "preview":
        return ("mp4",
                ["-vf", "scale='min(960,iw)':-2", "-c:v", "libx264",
                 "-crf", "28", "-preset", "veryfast"],
                ["-c:a", "aac", "-b:a", "96k"])
    if quality == "high":
        return ("mp4",
                ["-c:v", "libx264", "-crf", "14", "-preset", "slow",
                 "-pix_fmt", "yuv420p"],
                ["-c:a", "aac", "-b:a", "256k"])
    if quality == "max":
        return ("mkv",
                ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
                 "-pix_fmt", "yuv420p"],
                ["-c:a", "flac"])
    raise ValueError(f"unknown quality {quality}")


def render(video, cands, out_dir, quality):
    ext, vargs, aargs = codec_args(quality)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    clips = []
    for c in cands:
        name = f"{c['rank']:02d}_{c['type']}_{round(c['t'])}s.{ext}"
        path = os.path.join(clips_dir, name)
        dur = c["clip_end"] - c["clip_start"]
        cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{c['clip_start']:.3f}", "-i", video,
                "-t", f"{dur:.3f}"]
               + vargs + aargs + ["-movflags", "+faststart", "-y", path])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {name}: {r.stderr[-300:]}", file=sys.stderr)
            continue
        clips.append({"id": c["id"], "rank": c["rank"], "type": c["type"],
                      "t": c["t"], "clip_start": c["clip_start"],
                      "clip_end": c["clip_end"], "path": path,
                      "duration_s": dur})
        print(f"  {name} ({dur:.1f}s)", flush=True)

    reel = os.path.join(out_dir, f"reel.{ext}")
    if clips:
        concat = os.path.join(clips_dir, "concat.txt")
        with open(concat, "w") as f:
            for m in sorted(clips, key=lambda m: m["rank"]):
                f.write(f"file '{os.path.basename(m['path'])}'\n")
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-f", "concat", "-safe", "0", "-i", concat,
                            "-c", "copy", "-y", reel],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"reel FAIL: {r.stderr[-300:]}", file=sys.stderr)

    manifest = {
        "quality": quality,
        "codec_settings": {"container": ext, "vcodec_args": vargs,
                           "acodec_args": aargs},
        "clips": clips,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(os.path.getsize(m["path"]) for m in clips)
    if os.path.exists(reel):
        total += os.path.getsize(reel)
    return clips, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/home/ubuntu/match/match.mp4")
    ap.add_argument("--candidates", default=DEFAULT_CAND)
    ap.add_argument("--quality", choices=["preview", "high", "max"],
                    default="high")
    ap.add_argument("--out", default=None,
                    help="default: outputs/rendered_<quality> "
                         "(preview keeps legacy outputs/rendered)")
    ap.add_argument("--only-t", type=float, default=None,
                    help="render only the event at this t")
    args = ap.parse_args()

    out = args.out or os.path.join(
        HERE, "outputs",
        "rendered" if args.quality == "preview"
        else f"rendered_{args.quality}")

    with open(args.candidates) as f:
        doc = json.load(f)
    cands = [c for c in doc["candidates"] if c["selected"]]
    if args.only_t is not None:
        cands = [c for c in cands if abs(c["t"] - args.only_t) < 1e-6]

    if args.quality == "preview":
        for degrade in (False, True):
            clips, total = render_preview(args, cands, out, degrade)
            mb = total / 1e6
            print(f"preview: {len(clips)} clips, total {mb:.1f} MB",
                  flush=True)
            if mb <= PREVIEW_LIMIT_MB or degrade:
                break
            print(f"over {PREVIEW_LIMIT_MB} MB -> CRF31/width800 rerender",
                  flush=True)
    else:
        clips, total = render(args.video, cands, out, args.quality)
        print(f"{args.quality}: {len(clips)} clips, "
              f"total {total / 1e6:.1f} MB", flush=True)

    for m in clips:
        sz = os.path.getsize(m["path"]) / 1e6
        if sz > 100:
            print(f"WARNING {m['path']} = {sz:.1f} MB", file=sys.stderr)
    print("done", flush=True)


def render_preview(args, cands, out, degrade):
    """Preview render with optional CRF31/width800 size fallback."""
    ext, vargs, aargs = codec_args("preview")
    if degrade:
        vargs = ["-vf", "scale='min(800,iw)':-2", "-c:v", "libx264",
                 "-crf", "31", "-preset", "veryfast"]
    clips_dir = os.path.join(out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    clips = []
    for c in cands:
        name = f"{c['rank']:02d}_{c['type']}_{round(c['t'])}s.{ext}"
        path = os.path.join(clips_dir, name)
        dur = c["clip_end"] - c["clip_start"]
        cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{c['clip_start']:.3f}", "-i", args.video,
                "-t", f"{dur:.3f}"]
               + vargs + aargs + ["-movflags", "+faststart", "-y", path])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {name}: {r.stderr[-300:]}", file=sys.stderr)
            continue
        clips.append({"id": c["id"], "rank": c["rank"], "type": c["type"],
                      "t": c["t"], "clip_start": c["clip_start"],
                      "clip_end": c["clip_end"], "path": path,
                      "duration_s": dur})
    reel = os.path.join(out, "reel.mp4")
    if clips:
        concat = os.path.join(clips_dir, "concat.txt")
        with open(concat, "w") as f:
            for m in sorted(clips, key=lambda m: m["rank"]):
                f.write(f"file '{os.path.basename(m['path'])}'\n")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", concat,
                        "-c", "copy", "-y", reel],
                       capture_output=True, text=True)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"quality": "preview" + ("-degraded" if degrade else ""),
                   "codec_settings": {"container": ext, "vcodec_args": vargs,
                                      "acodec_args": aargs},
                   "clips": clips}, f, indent=1)
    total = sum(os.path.getsize(m["path"]) for m in clips)
    if os.path.exists(reel):
        total += os.path.getsize(reel)
    return clips, total


if __name__ == "__main__":
    main()
