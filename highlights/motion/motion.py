#!/usr/bin/env python3
"""Cheap per-second motion-feature extractor.

Decodes the video at 5 fps grayscale 320x144 via an ffmpeg rawvideo pipe,
computes frame-difference features per 1-second bin, and writes a JSON
timeseries plus a PNG plot of near-goal ROI energy and net disturbance.

Feature layout (scaled coords are orig/4):
  - near-goal ROI : orig x[890,1220] y[155,395]  (verified on f3000.png)
  - far region    : orig x<700, y<230
  - near third    : orig x>640 or y>300
"""
import argparse
import json
import subprocess
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 5
W, H = 320, 144          # decoded size (orig 1280x576 / 4)
SCALE = 4.0
DIFF_THRESH = 10         # suppress compression noise
ALPHA = 0.01             # background EMA rate

# scaled-coordinate regions
GOAL = (890 // 4, 155 // 4, 1220 // 4, 395 // 4)   # x1,y1,x2,y2 = (222,38,305,98)
FAR = (0, 0, 700 // 4, 230 // 4)                    # (0,0,175,57)
NEAR_X, NEAR_Y = 640 / 4, 300 / 4                   # 160, 75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-plot", default=None)
    args = ap.parse_args()

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", args.video,
           "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo",
           "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            bufsize=W * H * 64)

    gx1, gy1, gx2, gy2 = GOAL
    fx1, fy1, fx2, fy2 = FAR

    yy, xx = np.mgrid[0:H, 0:W]
    near_mask = (xx > NEAR_X) | (yy > NEAR_Y)

    bg = None                     # EMA background of goal ROI
    prev = None
    bins = {}                     # sec -> accumulators
    fi = 0
    frame_bytes = W * H
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.int16)

        roi_cur = cur[gy1:gy2, gx1:gx2].astype(np.float32)
        if bg is None:
            bg = roi_cur.copy()
        net = float(np.abs(roi_cur - bg).mean())
        bg = (1 - ALPHA) * bg + ALPHA * roi_cur

        if prev is not None:
            diff = np.abs(cur - prev).astype(np.float32)
            diff[diff < DIFF_THRESH] = 0
            sec = fi // FPS
            a = bins.setdefault(sec, dict(tot=0.0, roi=0.0, far=0.0,
                                          sx=0.0, sy=0.0, w=0.0,
                                          near=0.0, net=0.0, n=0))
            a["tot"] += float(diff.sum())
            a["roi"] += float(diff[gy1:gy2, gx1:gx2].sum())
            a["far"] += float(diff[fy1:fy2, fx1:fx2].sum())
            a["near"] += float(diff[near_mask].sum())
            wsum = float(diff.sum())
            a["w"] += wsum
            a["sx"] += float((diff * xx).sum())
            a["sy"] += float((diff * yy).sum())
            a["net"] += net
            a["n"] += 1
        prev = cur
        fi += 1
        if fi % 5000 == 0:
            print(f"{fi} frames (~{fi//FPS}s)", file=sys.stderr, flush=True)
    proc.stdout.close()
    proc.wait()
    print(f"decoded {fi} frames", file=sys.stderr)

    columns = ["t", "motion_total", "motion_goal_roi", "motion_far",
               "centroid_x", "centroid_y", "near_frac", "net_disturbance"]
    rows = []
    for sec in sorted(bins):
        a = bins[sec]
        cx = (a["sx"] / a["w"] * SCALE) if a["w"] > 0 else float("nan")
        cy = (a["sy"] / a["w"] * SCALE) if a["w"] > 0 else float("nan")
        rows.append([float(sec), a["tot"], a["roi"], a["far"],
                     cx, cy, a["near"] / a["tot"] if a["tot"] else 0.0,
                     a["net"] / a["n"]])

    with open(args.out_json, "w") as f:
        json.dump({"source": "motion", "step_s": 1.0,
                   "columns": columns, "rows": rows}, f)
    print(f"wrote {args.out_json}: {len(rows)} bins", file=sys.stderr)

    if not args.out_plot:
        return

    t = np.array([r[0] for r in rows])
    roi_e = np.array([r[2] for r in rows])
    net_v = np.array([r[7] for r in rows])
    sm = np.convolve(roi_e, np.ones(3) / 3, mode="same")
    work = sm.copy()
    peaks = []
    for _ in range(15):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]) or work[i] <= 0:
            break
        peaks.append((t[i], float(roi_e[i]), float(sm[i])))
        lo, hi = max(0, i - 20), min(len(work), i + 21)
        work[lo:hi] = -np.inf
    peaks.sort()

    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.plot(t, roi_e, lw=0.6, color="tab:blue", alpha=0.5,
             label="goal ROI energy")
    ax1.plot(t, sm, lw=1.2, color="tab:blue", label="goal ROI energy (3s)")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("motion energy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, net_v, lw=0.7, color="tab:red", alpha=0.8,
             label="net disturbance")
    ax2.set_ylabel("net disturbance", color="tab:red")
    for pt, pv, _ in peaks:
        ax1.annotate(f"{int(pt)}", xy=(pt, pv), xytext=(0, 14),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="black",
                     arrowprops=dict(arrowstyle="-", lw=0.5))
        ax1.axvline(pt, color="gray", lw=0.3, alpha=0.5)
    ax1.set_title("Near-goal ROI motion energy & net disturbance "
                  "(peaks = 3s-smoothed, min sep 20s)")
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=120)
    print(f"wrote {args.out_plot}", file=sys.stderr)

    print("\nTop 15 peaks (t, roi_energy, smoothed):")
    for pt, pv, ps in peaks:
        print(f"  t={int(pt):5d}s  roi={pv:12.0f}  sm={ps:12.0f}")


if __name__ == "__main__":
    main()
