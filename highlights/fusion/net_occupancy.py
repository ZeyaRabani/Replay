#!/usr/bin/env python3
"""Independent 'person inside the near net' occupancy detector.

After a near-end goal the keeper typically climbs into the net to fetch
the ball, so a person whose feet land inside the net interior for a
sustained run is a goal-retrieval cue independent of the reviewers.

Decode match.mp4 at 2 fps over [1050, 4990], crop x[880,1280] y[100,420]
(full-res crop around the near goal), run YOLO person detection (conf 0.3),
and test each bbox bottom-centre against the net-interior polygon:

  (950,160) (1060,148) (1215,330) (1060,350) (940,325)

  - "in net"     : bottom-centre inside polygon AND y_bottom > 250
                   (feet at/behind the goal-line depth)
  - "goal zone"  : bottom-centre inside polygon regardless of y

Writes outputs/net_occupancy_1s.json (t, n_in_net, n_goal_zone) and prints
merged occupancy runs (>=1.5 s consecutive, gaps <6 s merged) used for
contact sheets.
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "outputs", "net_occupancy_1s.json")

T_LO, T_HI = 1050.0, 4990.0
FPS = 2
CROP = (880, 100, 1280, 420)          # x1,y1,x2,y2 full-res crop
POLY = np.array([(950, 160), (1060, 148), (1215, 330),
                 (1060, 350), (940, 325)], np.int32)
CONF = 0.3
MIN_RUN_S = 1.5
MERGE_GAP_S = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/home/ubuntu/match/match.mp4")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--t-lo", type=float, default=T_LO)
    ap.add_argument("--t-hi", type=float, default=T_HI)
    args = ap.parse_args()

    x1, y1, x2, y2 = CROP
    cw, ch = x2 - x1, y2 - y1
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(args.t_lo), "-t", str(args.t_hi - args.t_lo),
           "-i", args.video,
           "-vf", f"fps={FPS},crop={cw}:{ch}:{x1}:{y1}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            bufsize=cw * ch * 3 * 8)
    model = YOLO(args.model)

    frame_bytes = cw * ch * 3
    fi = 0
    samples = []          # (t, n_in_net, n_goal_zone) per 2fps sample
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        t = args.t_lo + fi / FPS
        im = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
        res = model(im, classes=[0], conf=CONF, verbose=False)[0]
        n_net = n_zone = 0
        for b in res.boxes.xyxy.cpu().numpy():
            bx1, _by1, bx2, by2 = b + np.array([x1, y1, x1, y1])
            cx, by = (bx1 + bx2) / 2.0, by2
            inside = cv2.pointPolygonTest(POLY, (float(cx), float(by)),
                                          False) >= 0
            if inside:
                n_zone += 1
                if by > 250:
                    n_net += 1
        samples.append((t, n_net, n_zone))
        fi += 1
        if fi % 1000 == 0:
            print(f"{fi} frames (~{t:.0f}s)", flush=True)
    proc.stdout.close()
    proc.wait()
    print(f"decoded {fi} frames", flush=True)

    # per-second aggregation (max occupancy of the two samples)
    per_sec = {}
    for t, nn, nz in samples:
        s = int(t)
        a = per_sec.setdefault(s, [0, 0])
        a[0] = max(a[0], nn)
        a[1] = max(a[1], nz)
    rows = [[float(s)] + v for s, v in sorted(per_sec.items())]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"source": "net_occupancy", "step_s": 1.0,
                   "columns": ["t", "n_in_net", "n_goal_zone"],
                   "rows": rows}, f)
    print(f"wrote {args.out}: {len(rows)} bins", flush=True)

    # runs: consecutive flagged samples -> merge runs <MERGE_GAP_S apart
    # -> keep merged runs spanning >=MIN_RUN_S. (Merging before the length
    # filter keeps detections like the keeper-in-net event at ~2738-2742,
    # where YOLO misses him through the net on some frames.)
    flagged = [t for t, nn, _ in samples if nn >= 1]
    runs = []
    for t in flagged:
        if runs and t - runs[-1][1] <= 1.0 / FPS + 1e-6:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [[a, b, max(per_sec[int(s)][0]
                         for s in np.arange(a, b + 1e-6, 0.5))]
              for a, b in merged if b - a + 1.0 / FPS >= MIN_RUN_S]
    print(f"{len(merged)} occupancy runs:")
    for a, b, m in merged:
        print(f"  {a:7.1f} - {b:7.1f}   max n_in_net={m}")
    with open("/tmp/net_runs.json", "w") as f:
        json.dump([[a, b, m] for a, b, m in merged], f)


if __name__ == "__main__":
    main()
