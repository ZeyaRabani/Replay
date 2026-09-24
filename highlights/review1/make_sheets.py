#!/usr/bin/env python3
"""Build timestamped contact sheets from a video for manual visual review.

Coarse pass: one frame every N seconds tiled into 5x4 sheets.
Fine pass:  frames at a given fps over a short window, tiled into sheets.

Usage:
  python make_sheets.py coarse VIDEO OUT_DIR --start 0 --end 1350 --step 2
  python make_sheets.py fine   VIDEO OUT_DIR --start 610 --end 650 --fps 2
"""
import argparse
import math
import os
import subprocess


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sheet(video, out_png, t0, t1, step, cols, rows, width, crop=None):
    """One sheet covering [t0, t1) sampling every `step` seconds."""
    n = cols * rows
    crop_f = f"crop={crop}," if crop else ""
    vf = (
        f"fps=1/{step}:round=up,{crop_f}scale={width}:-2,"
        f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"text='%{{eif\\:{t0}+t\\:d}}s':x=8:y=8:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6,"
        f"tile={cols}x{rows}:padding=2:margin=2"
    )
    run([
        "ffmpeg", "-y", "-ss", str(t0), "-t", str(t1 - t0), "-i", video,
        "-vf", vf, "-frames:v", "1", "-vsync", "vfr", out_png,
    ])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coarse", "fine", "far"],
                    help="far = fine pass on a full-res crop of the far-goal region")
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--step", type=float, default=2.0, help="coarse: seconds between frames")
    ap.add_argument("--fps", type=float, default=2.0, help="fine: frames per second")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--crop", default="560:200:0:0", help="far mode: w:h:x:y crop of source")
    a = ap.parse_args()
    crop = a.crop if a.mode == "far" else None
    os.makedirs(a.out_dir, exist_ok=True)

    step = a.step if a.mode == "coarse" else 1.0 / a.fps
    per_sheet = a.cols * a.rows * step
    n = math.ceil((a.end - a.start) / per_sheet)
    for i in range(n):
        t0 = a.start + i * per_sheet
        t1 = min(a.end, t0 + per_sheet)
        name = f"{a.mode}_{int(t0):05d}_{math.ceil(t1):05d}.png"
        sheet(a.video, os.path.join(a.out_dir, name), t0, t1, step, a.cols, a.rows, a.width, crop)
        print(name)


if __name__ == "__main__":
    main()
