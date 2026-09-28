"""Hi-res person detection per angle for players v2.

Decodes the ORIGINAL source video at native size (not the 480p proxy),
runs YOLO on a 2x2 tile grid with 10% overlap, merges tile boxes with
NMS (IoU 0.5) in full-frame coords and writes analysis/players_v2/
det_a{i}.npz {t, x1, y1, x2, y2, conf, team}.

Resume-safe: every ~5 min of wall time the partial npz is flushed; on
restart, seconds already in the file are skipped.

    python -m highlights.analysis.detect_hr --video V --out det_a0.npz \
        --start-s S --end-s E [--fps 2] [--model yolov8n.pt]
        [--backend pt|onnx]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from highlights.io import write_json_atomic
from highlights.multiangle.trackfeat import _ensure_model, _probe_dims
from highlights.pipeline.errors import PipelineError

FPS = 2.0
CONF = 0.15
CLS = [0]                     # person only
IMGSZ = 1088
TILE_OVERLAP = 0.10
NMS_IOU = 0.5
FLUSH_S = 300.0               # wall-clock checkpoint interval
ETA_LIMIT_H = 8.0
ETA_PROBE = 20                # frames timed for the projection


def tiles(w: int, h: int, n: int = 2, overlap: float = TILE_OVERLAP
          ) -> list[tuple[int, int, int, int]]:
    """n x n tile boxes (x1,y1,x2,y2) covering wxh with `overlap`."""
    tw = int(w / n * (1 + overlap))
    th = int(h / n * (1 + overlap))
    out = []
    for r in range(n):
        for c in range(n):
            x1 = 0 if c == 0 else w - tw if c == n - 1 else \
                int(c * (w - tw) / (n - 1))
            y1 = 0 if r == 0 else h - th if r == n - 1 else \
                int(r * (h - th) / (n - 1))
            out.append((x1, y1, x1 + tw, y1 + th))
    return out


def nms(boxes: np.ndarray, conf: np.ndarray,
        iou: float = NMS_IOU) -> np.ndarray:
    """Greedy NMS -> indices kept, highest-conf first."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(-conf)
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                - inter)
        suppressed |= (inter / np.maximum(area, 1e-9)) > iou
        suppressed[i] = False
    return np.array(keep, dtype=int)


def _load_model(model_path: Path, backend: str, log):
    """YOLO handle for `predict(frame, imgsz, conf, classes)`; backend
    'onnx' exports the .pt once then serves through ultralytics' ONNX
    support (onnxruntime on aarch64)."""
    from ultralytics import YOLO
    if backend == "onnx":
        onnx = model_path.with_suffix(".onnx")
        if not onnx.exists():
            log(f"detect_hr: exporting {model_path.name} -> onnx "
                f"(imgsz={IMGSZ})")
            YOLO(str(model_path)).export(format="onnx", imgsz=IMGSZ,
                                         dynamic=False)
        return YOLO(str(onnx))
    return YOLO(str(model_path))


def _predict(model, frame, imgsz: int):
    """[(xyxy np[4], conf)] person detections on one (tile) frame."""
    res = model.predict(frame, imgsz=imgsz, conf=CONF, classes=CLS,
                        verbose=False)[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            if cls[i] == 0:
                out.append((xyxy[i], float(conf[i])))
    return out


def _detect_frame(model, frame, predict):
    """Tiled detection -> merged full-frame (boxes, conf)."""
    h, w = frame.shape[:2]
    boxes, confs = [], []
    for tx1, ty1, tx2, ty2 in tiles(w, h):
        tile = frame[ty1:ty2, tx1:tx2]
        for xyxy, c in predict(model, tile, IMGSZ):
            b = xyxy.copy()
            b[0] += tx1
            b[2] += tx1
            b[1] += ty1
            b[3] += ty1
            boxes.append(b)
            confs.append(c)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    boxes = np.array(boxes)
    confs = np.array(confs)
    keep = nms(boxes, confs)
    return boxes[keep], confs[keep]


def detect_angle(video: str | Path, out: Path, *,
                 model_path: Path | None = None, backend: str = "pt",
                 fps: float = FPS, start_s: float = 0.0,
                 end_s: float | None = None, teams: dict | None = None,
                 model=None, predict=_predict, frames=None,
                 log=print, status=None) -> dict:
    """Detect persons at `fps` over [start_s, end_s) file seconds.

    model/predict/frames injectable for tests. teams -> assign_team per
    detection (stored 'A'/'B'/''). Returns the meta dict."""
    from highlights.multiangle.trackfeat import _frame_reader

    from .players import assign_team
    from .teams import torso_descriptor

    w, h = _probe_dims(str(video)) if frames is None else (0, 0)
    if model is None:
        model_path = _ensure_model(model_path or Path("yolov8n.pt"))
        model = _load_model(model_path, backend, log)
    if frames is None:
        frames = _frame_reader(str(video), fps, w, start_s, end_s)

    rows_t: list[float] = []
    rows_b: list[list[float]] = []
    rows_c: list[float] = []
    rows_tm: list[str] = []
    done_s = -1.0
    if out.exists():
        z = np.load(out, allow_pickle=False)
        if "t" in z and len(z["t"]):
            rows_t = z["t"].tolist()
            rows_b = np.stack([z["x1"], z["y1"], z["x2"], z["y2"]],
                              axis=1).tolist()
            rows_c = z["conf"].tolist()
            rows_tm = z["team"].tolist()
            done_s = float(max(rows_t))
            log(f"detect_hr: resume — {len(rows_t)} dets, "
                f"through t={done_s:.0f}s")

    n_frames = 0
    timed = 0.0
    n_pred = 0
    flush_at = time.time() + FLUSH_S
    total_s = (end_s - start_s) if end_s is not None else None
    for t, frame in frames:
        if total_s is not None and t - start_s >= total_s + 1e-6:
            break
        if t <= done_s:
            continue
        t0 = time.time()
        boxes, conf = _detect_frame(model, frame, predict)
        timed += time.time() - t0
        n_pred += 1
        if n_pred == ETA_PROBE:
            per = timed / n_pred
            if total_s is not None:
                eta_h = per * total_s * fps / 3600.0
                log(f"detect_hr: {per:.2f} s/frame -> ETA {eta_h:.1f} h")
                if eta_h > ETA_LIMIT_H:
                    raise PipelineError(
                        f"detection projected {eta_h:.1f} h > "
                        f"{ETA_LIMIT_H:.0f} h — aborting")
        if w == 0:
            h, w = frame.shape[:2]
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            tm = ""
            if teams:
                desc = torso_descriptor(frame, (x1, y1, x2, y2))
                if desc is not None:
                    tm = assign_team(desc, teams) or ""
            rows_t.append(round(t, 3))
            rows_b.append([x1, y1, x2, y2])
            rows_c.append(conf[i])
            rows_tm.append(tm)
        n_frames += 1
        if n_frames % 30 == 0:
            log(f"detect_hr @{t:.0f}s")
        if status is not None and total_s:
            status.update(stage_progress=min(
                0.99, (t - start_s) / max(1.0, total_s)))
        if time.time() >= flush_at:
            _flush(out, rows_t, rows_b, rows_c, rows_tm)
            flush_at = time.time() + FLUSH_S
    _flush(out, rows_t, rows_b, rows_c, rows_tm)
    meta = {"w": w, "h": h, "fps": fps, "imgsz": IMGSZ,
            "model": str(model_path or ""), "backend": backend,
            "n_frames": n_frames, "n_dets": len(rows_t),
            "start_s": start_s, "end_s": end_s}
    write_json_atomic(out.with_suffix(".meta.json"), meta, indent=1)
    log(f"detect_hr: {n_frames} frames, {len(rows_t)} dets -> {out.name}")
    return meta


def _flush(out: Path, t, boxes, conf, team) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    b = np.asarray(boxes) if boxes else np.zeros((0, 4))
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, t=np.asarray(t), x1=b[:, 0] if len(b) else [],
             y1=b[:, 1] if len(b) else [],
             x2=b[:, 2] if len(b) else [],
             y2=b[:, 3] if len(b) else [],
             conf=np.asarray(conf), team=np.asarray(team))
    tmp.replace(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.detect_hr")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--backend", choices=["pt", "onnx"], default="pt")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=None)
    args = ap.parse_args(argv)
    detect_angle(args.video, args.out, model_path=args.model,
                 backend=args.backend, fps=args.fps,
                 start_s=args.start_s, end_s=args.end_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
