#!/usr/bin/env python3
"""Build timestamped contact sheets from a video segment for manual review.

Coarse mode: 1 frame / 2 s, 5x4 tiles (40 s per sheet).
Fine mode:   N fps over a short window, 5x4 tiles.

Usage:
  python make_sheets.py coarse VIDEO OUT_DIR --start 4050 [--end END]
  python make_sheets.py fine   VIDEO OUT_DIR --start 4200 --end 4240 --fps 2
"""
import argparse
import math
import os
import subprocess

COLS, ROWS = 5, 4
TILE_W = 640


def duration(video: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video])
    return float(out.strip())


def make_sheet(video, start, step, out_png, n=COLS * ROWS, crop=None,
               tile_w=TILE_W):
    """One sheet: n frames starting at `start`, `step` seconds apart.

    crop: optional "w:h:x:y" region (source pixels) to zoom into.
    """
    # drawtext shows absolute video time: start + n_frame*step
    crop_f = f"crop={crop}," if crop else ""
    vf = (
        f"fps=1/{step}:round=up,{crop_f}scale={tile_w}:-2,"
        f"drawtext=text='%{{eif\\:{start}+n*{step}\\:d}}s':x=8:y=8:"
        f"fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6,"
        f"tile={COLS}x{ROWS}:padding=2:margin=2"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-i", video,
         "-frames:v", str(n), "-vf", vf, "-frames:v", "1", out_png],
        check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coarse", "fine"])
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fps", type=float, default=2.0, help="fine mode only")
    ap.add_argument("--crop", default=None, help="w:h:x:y source crop")
    ap.add_argument("--tile-w", type=int, default=TILE_W)
    ap.add_argument("--tag", default="", help="suffix for output names")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    end = a.end if a.end is not None else duration(a.video)
    step = 2.0 if a.mode == "coarse" else 1.0 / a.fps
    span = step * COLS * ROWS
    n_sheets = math.ceil((end - a.start) / span)
    for i in range(n_sheets):
        s = a.start + i * span
        name = (f"{a.mode}{a.tag}_{int(s):05d}-"
                f"{int(min(s + span, end)):05d}.png")
        make_sheet(a.video, s, step, os.path.join(a.out_dir, name),
                   crop=a.crop, tile_w=a.tile_w)
        print(name, flush=True)


if __name__ == "__main__":
    main()
