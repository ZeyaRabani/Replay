"""Render a 30 s debug clip for one candidate event.

Left panel: video frame with tracked player boxes and ids.
Right panel: top-down radar (pitch 0..L m x +-16 m, near goal on the left)
with player dots coloured by coarse team colour, plus per-second feature
readout. Saves debug_<t>.mp4 and three PNG frames (start / event t / end).
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .calibrate import GroundModel, to_ref
from .features import BOX_DEPTH_M, BOX_HALF_W_M, CENTRE_RADIUS_M, team_of

RADAR_W, RADAR_H = 400, 260
MATCH_TOL = 0.26


def frame_reader(video, width, height, fps, t_start, duration):
    frame_bytes = width * height * 3
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-ss", str(t_start), "-i", video,
            "-t", str(duration), "-vf", f"fps={fps}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        stdout=subprocess.PIPE,
        bufsize=frame_bytes * 8,
    )

    def gen():
        while True:
            buf = b""
            while len(buf) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    return
                buf += chunk
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

    return proc, gen()


def probe_size(video):
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", video,
        ]
    ).decode().strip().split(",")
    return int(out[0]), int(out[1])


def nearest_idx(ts, t, tol=MATCH_TOL):
    i = int(np.argmin(np.abs(ts - t)))
    return i if abs(ts[i] - t) <= tol else None


def draw_radar(L, world_pts, team_labels):
    """world_pts: Nx2 (x along 0..L, y +-16). team_labels: 0 lime,1 orange,2 grey."""
    img = np.full((RADAR_H, RADAR_W, 3), (34, 80, 34), np.uint8)
    sx = (RADAR_W - 20) / L
    sy = (RADAR_H - 20) / 32.0
    s = min(sx, sy)

    def px(x, y):
        return int(10 + x * s), int(RADAR_H / 2 - y * s)

    p00 = px(0, -16)
    p11 = px(L, 16)
    cv2.rectangle(img, p00, p11, (230, 230, 230), 1)
    cv2.rectangle(img, px(0, -BOX_HALF_W_M), px(BOX_DEPTH_M, BOX_HALF_W_M),
                  (230, 230, 230), 1)
    cv2.rectangle(img, px(L - BOX_DEPTH_M, -BOX_HALF_W_M),
                  px(L, BOX_HALF_W_M), (230, 230, 230), 1)
    cv2.line(img, px(L / 2, -16), px(L / 2, 16), (230, 230, 230), 1)
    cv2.circle(img, px(L / 2, 0), int(CENTRE_RADIUS_M * s), (230, 230, 230), 1)
    colours = [(80, 255, 80), (60, 140, 255), (180, 180, 180)]
    for (x, y), tl in zip(world_pts, team_labels):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, px(x, y), 4, colours[tl], -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    with open(args.events) as fh:
        events = json.load(fh)["events"]
    if args.t_start is None:
        cand = [e for e in events
                if e["type"] == "chance"
                and e["signals"].get("goal_end") == "near"]
        best = max(cand, key=lambda e: e["confidence"])
        t_start = best["t_start"]
        t_event = best["t"]
    else:
        t_start = args.t_start
        t_event = t_start + args.duration / 2

    with open(args.calib) as fh:
        cal = json.load(fh)
    g = GroundModel.from_json(cal["model"])
    L = g.length

    d = np.load(args.tracks)
    det, frames = d["det"], d["frames"]
    t_det, t_fr = det[:, 1], frames[:, 1]
    Hs = frames[:, 4:13].reshape(-1, 3, 3).astype(np.float64)

    feat = None
    if args.features:
        with open(args.features) as fh:
            fj = json.load(fh)
        cols = {c: i for i, c in enumerate(fj["columns"])}
        feat = (fj["rows"], cols)

    width, height = probe_size(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc, gen = frame_reader(args.video, width, height, args.fps,
                             t_start, args.duration)
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    vw = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (width + RADAR_W + 20, height))
    png_at = {"start": t_start, "peak": t_event,
              "end": t_start + args.duration}
    saved = set()

    i = 0
    for frame in gen:
        t = t_start + i / args.fps
        i += 1
        fi = nearest_idx(t_fr, t)

        canvas = np.zeros((height, width + RADAR_W + 20, 3), np.uint8)
        vis = frame.copy()
        world_pts, team_labels = [], []
        if fi is not None:
            H = Hs[fi]
            m = np.where(np.abs(t_det - t) <= MATCH_TOL)[0]
            for r in det[m]:
                x1, y1, x2, y2 = r[3:7]
                tid = int(r[2])
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(vis, str(tid), (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                feet = np.array([[(x1 + x2) / 2, y2]])
                heads = np.array([[(x1 + x2) / 2, y1]])
                W = g.world(to_ref(feet, H), to_ref(heads, H))[0]
                lime, orange = team_of(
                    np.array([r[8]]), np.array([r[9]]), np.array([r[10]]))
                team_labels.append(0 if lime[0] else (1 if orange[0] else 2))
                world_pts.append(W)
        canvas[:, :width] = vis
        radar = draw_radar(L, np.asarray(world_pts).reshape(-1, 2),
                           team_labels)
        canvas[10:10 + RADAR_H, width + 10:width + 10 + RADAR_W] = radar

        y = 10 + RADAR_H + 30
        cv2.putText(canvas, f"t = {t:.1f}s", (width + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y += 28
        if feat is not None:
            rows, cols = feat
            fr = nearest_idx(np.array([r[0] for r in rows]), t, 0.6)
            if fr is not None:
                for name in ("n_players", "n_near_box", "rush_near_3s"):
                    v = rows[fr][cols[name]]
                    cv2.putText(canvas, f"{name}: {v}", (width + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (220, 220, 220), 1)
                    y += 24

        vw.write(canvas)
        for k, tk in png_at.items():
            if k not in saved and abs(t - tk) <= 0.5 / args.fps + 1e-6:
                cv2.imwrite(str(out_dir / f"debug_{int(t_start)}_{k}.png"),
                            canvas)
                saved.add(k)

    if "end" not in saved and i > 0:
        cv2.imwrite(str(out_dir / f"debug_{int(t_start)}_end.png"), canvas)
        saved.add("end")

    vw.release()
    proc.stdout.close()
    proc.wait()

    mp4 = out_dir / f"debug_{int(t_start)}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    tmp.unlink()
    print(f"wrote {mp4} + PNGs {sorted(saved)} -> {out_dir}")


if __name__ == "__main__":
    main()
