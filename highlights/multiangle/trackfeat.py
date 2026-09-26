"""Per-angle tracking features for the director stage.

YOLO (ultralytics yolov8n by default) at 1 fps over an ffmpeg pipe: person +
sports-ball detections reduced to one row per second:

    t, n_players, ball_conf, ball_size, ball_x, ball_y,
    players_cx, players_cy, players_spread, players_height_mean, cluster_score,
    players_xy   (JSON string of [[fx, fy], ...] feet points, "[]" if none)

Usage:
    python -m highlights.multiangle.trackfeat --video match.mp4 \
        --out track/features_1s.json [--model highlights/multiangle/models/yolov8n.pt]
        [--imgsz 960] [--max-seconds 120] [--fps 1]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CLASSES = [0, 32]          # person, sports ball
CONF = 0.25
BALL_OK = 0.35
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
FRAME_W = 960
COLS = ["t", "n_players", "ball_conf", "ball_size", "ball_x", "ball_y",
        "players_cx", "players_cy", "players_spread", "players_height_mean",
        "cluster_score", "players_xy"]


def _ensure_model(model: Path) -> Path:
    """Return a usable weights path; auto-downloads yolov8n.pt once."""
    if model.exists():
        return model
    model.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    YOLO("yolov8n.pt")  # downloads into cwd
    dl = Path("yolov8n.pt")
    if dl.exists() and dl.resolve() != model.resolve():
        shutil.move(str(dl), model)
    return model if model.exists() else Path("yolov8n.pt")


def _probe_dims(video: str, tries: int = 3, delay: float = 2.0):
    """ffprobe (w, h) of v:0; retry on non-zero rc or empty stdout —
    observed to fail transiently under load right after a render finishes."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", video]
    last = ""
    for attempt in range(tries):
        probe = subprocess.run(cmd, capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip():
            try:
                w, h = (int(x) for x in probe.stdout.strip().split(","))
                return w, h
            except ValueError:
                last = f"unparseable ffprobe output {probe.stdout!r}"
        else:
            last = (f"ffprobe rc={probe.returncode} "
                    f"stderr={probe.stderr.strip()!r}")
        if attempt < tries - 1:
            time.sleep(delay)
    raise RuntimeError(f"ffprobe failed for {video}: {last}")


def _frame_reader(video: str, fps: float, width: int,
                  start_s: float = 0.0, end_s: float | None = None):
    """Yield (t_seconds, bgr frame) via an ffmpeg rawvideo pipe at scale
    w=width. t counts FILE seconds starting at start_s. -ss goes before -i
    (fast seek), -t bounds the decode to [start_s, end_s)."""
    w_in, h_in = _probe_dims(video)
    w = width & ~1
    h = round(width * h_in / w_in) & ~1
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-i", video]
    if end_s is not None:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    t = float(start_s)
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        yield t, np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        t += 1.0 / fps
    proc.wait()


def compute_rows(video: str, model_path: Path, imgsz: int, fps: float,
                 max_seconds: float | None, log=print,
                 progress_file: Path | None = None,
                 start_s: float = 0.0, end_s: float | None = None
                 ) -> tuple[list[list], int, int]:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "0")) or
                          (os.cpu_count() or 1))
    model = YOLO(str(model_path))
    rows: list[list] = []
    frames = 0
    fh = None
    for t, frame in _frame_reader(video, fps, FRAME_W, start_s, end_s):
        if fh is None:
            fh = frame.shape[0]
        if max_seconds is not None and t - start_s >= max_seconds:
            break
        res = model.predict(frame, imgsz=imgsz, conf=CONF, classes=CLASSES,
                            verbose=False)[0]
        boxes = res.boxes
        persons, balls = [], []
        if boxes is not None and len(boxes):
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for k, c in enumerate(cls):
                (persons if c == 0 else balls).append((xyxy[k], conf[k]))
        n_p = len(persons)
        if balls:
            bidx = int(np.argmax([b[1] for b in balls]))
            bx, bconf = balls[bidx]
            bh = bx[3] - bx[1]
            ball_size = float(bh / frame.shape[0])
            ball_x = float((bx[0] + bx[2]) / 2 / frame.shape[1])
            ball_y = float((bx[1] + bx[3]) / 2 / frame.shape[0])
        else:
            bconf, ball_size, ball_x, ball_y = 0.0, 0.0, 0.0, 0.0
        if persons:
            pfeet = [[round(float(b[0][0] + b[0][2]) / 2 / frame.shape[1], 3),
                      round(float(b[0][3]) / frame.shape[0], 3)] for b in persons]
            cx = np.array([(b[0][0] + b[0][2]) / 2 / frame.shape[1] for b in persons])
            cy = np.array([(b[0][1] + b[0][3]) / 2 / frame.shape[0] for b in persons])
            hh = np.array([(b[0][3] - b[0][1]) / frame.shape[0] for b in persons])
            players_cx, players_cy = float(cx.mean()), float(cy.mean())
            players_spread = float(cx.std())
            players_height_mean = float(hh.mean())
        else:
            pfeet = []
            players_cx = players_cy = players_spread = players_height_mean = 0.0
        cluster = n_p * (1 - abs(players_cx - 0.5) * 2 * 0.5) * players_height_mean
        rows.append([round(t, 3), n_p, round(float(bconf), 3), round(ball_size, 4),
                     round(ball_x, 4), round(ball_y, 4), round(players_cx, 4),
                     round(players_cy, 4), round(players_spread, 4),
                     round(players_height_mean, 4), round(cluster, 4),
                     json.dumps(pfeet)])
        frames += 1
        if frames % 30 == 0:
            log(f"trackfeat @{t:.0f}s")
            if progress_file is not None:
                progress_file.write_text(
                    json.dumps({"t": round(t, 3), "frames": frames}))
    ball_rate = (sum(1 for r in rows if r[2] >= BALL_OK) / len(rows)) if rows else 0.0
    return rows, frames, ball_rate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.multiangle.trackfeat")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="start decoding at this file second")
    ap.add_argument("--end-s", type=float, default=None,
                    help="stop decoding at this file second")
    ap.add_argument("--progress-file", type=Path, default=None,
                    help="write {t, frames} here every 30 frames")
    args = ap.parse_args(argv)

    model = _ensure_model(args.model)
    rows, frames, ball_rate = compute_rows(
        args.video, model, args.imgsz, args.fps, args.max_seconds,
        progress_file=args.progress_file,
        start_s=args.start_s, end_s=args.end_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from highlights.io import write_json_atomic
    write_json_atomic(args.out, {
        "fps": args.fps, "model": str(model), "imgsz": args.imgsz,
        "columns": COLS, "rows": rows,
        "meta": {"ball_rate": round(ball_rate, 4), "n_frames": frames,
                 "start_s": args.start_s, "end_s": args.end_s},
    }, indent=0)
    print(f"trackfeat: {frames} frames -> {args.out} (ball_rate {ball_rate:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
