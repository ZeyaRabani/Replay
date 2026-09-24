#!/usr/bin/env python3
"""Build timestamped contact sheets from a video for manual visual review.

Coarse pass (1 frame / 2 s, 5x4 tiles = 40 s per sheet):
    python make_sheets.py VIDEO --start 2700 --end 4050 --step 2 --out sheets/coarse

Fine pass (4 fps over a window, 5x4 tiles = 5 s per sheet):
    python make_sheets.py VIDEO --start 3010 --end 3030 --fps 4 --out sheets/fine_3010

Each tile has the video timestamp (seconds from file start) burned in.
"""
import argparse
import math
import os
import subprocess
import sys

COLS, ROWS = 5, 4
TILE_W = 640


def sheet_cmd(video, t0, n_frames, interval, out_png, cols, rows, tile_w, crop):
    dur = n_frames * interval
    vf = [f"fps=1/{interval}" if interval >= 1 else f"fps={1/interval:g}"]
    if crop:
        vf.append(f"crop={crop}")
    vf.extend((
        f"scale={tile_w}:-2",
        # burn timestamp = t0 + n*interval (pts is relative to -ss with copyts off)
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"text='%{{eif\\:{t0}+n*{interval}\\:d}}s':x=8:y=8:fontsize=28:"
        "fontcolor=yellow:box=1:boxcolor=black@0.6",
        f"tile={cols}x{rows}:padding=2:color=black",
    ))
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{t0}", "-t", f"{dur}", "-i", video,
        "-vf", ",".join(vf), "-frames:v", "1", out_png,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--step", type=float, default=None, help="seconds between frames")
    ap.add_argument("--fps", type=float, default=None, help="frames per second (fine pass)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--tile-w", type=int, default=TILE_W)
    ap.add_argument("--crop", help="ffmpeg crop expression, e.g. iw*0.45:ih:0:0")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if (a.step is None) == (a.fps is None):
        sys.exit("give exactly one of --step or --fps")
    interval = a.step if a.step is not None else 1.0 / a.fps
    per_sheet = a.cols * a.rows
    os.makedirs(a.out, exist_ok=True)
    n_sheets = math.ceil((a.end - a.start) / (interval * per_sheet))
    procs = []
    for i in range(n_sheets):
        t0 = a.start + i * interval * per_sheet
        n = min(per_sheet, math.ceil((a.end - t0) / interval))
        out_png = os.path.join(a.out, f"sheet_{round(t0):05d}.png")
        procs.append(subprocess.Popen(sheet_cmd(a.video, t0, n, interval, out_png, a.cols, a.rows, a.tile_w, a.crop)))
        if len(procs) >= a.jobs:
            procs.pop(0).wait()
    for p in procs:
        p.wait()
    print(f"wrote {n_sheets} sheets to {a.out}")


if __name__ == "__main__":
    main()
