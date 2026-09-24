#!/usr/bin/env python3
"""Build timestamped contact sheets from a video segment for manual review.

Usage:
  python make_sheets.py VIDEO OUTDIR --start 1350 --end 2700 --step 2 --cols 5 --rows 4
  python make_sheets.py VIDEO OUTDIR --start 1500 --end 1540 --step 0.25 --cols 5 --rows 4 --prefix fine

Each tile has its video timestamp (seconds from file start) burned in.
One sheet covers cols*rows*step seconds.
"""
import argparse
import os
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("outdir")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--prefix", default="coarse")
    ap.add_argument("--crop", default=None, help="ffmpeg crop w:h:x:y before scaling (far goal: 640:200:0:40)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    per_sheet = a.cols * a.rows
    span = per_sheet * a.step
    t = a.start
    i = 0
    while t < a.end:
        dur = min(span, a.end - t)
        n = round(dur / a.step)
        out = os.path.join(a.outdir, f"{a.prefix}_{int(t):05d}.png")
        # burned timestamp = segment start + frame pts (pts restarts at 0 after -ss)
        crop = f"crop={a.crop}," if a.crop else ""
        vf = (
            f"fps=1/{a.step},{crop}scale={a.width}:-2,"
            f"drawtext=text='%{{eif\\:{t}+t\\:d}}.%{{eif\\:mod(t*10\\,10)\\:d}}s':"
            "fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6:x=8:y=8,"
            f"tile={a.cols}x{a.rows}"
        )
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{dur + a.step / 2:.3f}",
            "-i", a.video, "-vf", vf, "-frames:v", "1", out,
        ]
        subprocess.run(cmd, check=True)
        print(out, f"{t:.1f}-{t + dur:.1f}s", n, "frames")
        t += span
        i += 1


if __name__ == "__main__":
    main()
