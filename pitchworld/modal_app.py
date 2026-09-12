"""Modal GPU worker: YOLO (Ultralytics) + ByteTrack over a whole clip.

Deployed lazily by the CLI (``app.run()``); can also be deployed with
``modal deploy pitchworld/modal_app.py`` and looked up by name.
"""

from __future__ import annotations

import modal

APP_NAME = "pitchworld-track"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.3,<9", "opencv-python-headless", "lap>=0.5", "numpy")
)
weights = modal.Volume.from_name("pitchworld-weights", create_if_missing=True)
app = modal.App(APP_NAME)


def _track_impl(video_bytes: bytes, model_name: str, conf: float, imgsz: int, classes: list[int],
                weights_dir: str | None, ball_imgsz: int = 0) -> dict:
    import os
    import tempfile

    import cv2
    from ultralytics import YOLO

    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        os.chdir(weights_dir)  # ultralytics downloads weights into cwd
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(model_name)
    frames = []
    results = model.track(source=path, stream=True, persist=True, tracker="bytetrack.yaml", conf=conf,
                          iou=0.6, imgsz=imgsz, classes=classes, verbose=False, half=True)
    for i, r in enumerate(results):
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(xyxy)
            for b, c, k, t in zip(xyxy, cf, cls, ids):
                dets.append({"id": int(t), "cls": int(k), "conf": round(float(c), 3),
                             "box": [round(float(v), 1) for v in b]})
        frames.append(dets)
    if 32 in classes and ball_imgsz > 0:
        ball_results = model.predict(source=path, stream=True, conf=min(conf, 0.15), imgsz=ball_imgsz,
                                     classes=[32], verbose=False, half=True)
        for i, r in enumerate(ball_results):
            if i >= len(frames) or r.boxes is None or not len(r.boxes):
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, cf):
                frames[i].append({"id": -1, "cls": 32, "conf": round(float(c), 3),
                                  "box": [round(float(v), 1) for v in b]})
    return {"fps": fps, "width": w, "height": h, "model": model_name, "frames": frames}


@app.function(image=image, gpu="A10G", timeout=60 * 30, volumes={"/weights": weights})
def track_video(video_bytes: bytes, model_name: str = "yolov8m.pt", conf: float = 0.25, imgsz: int = 1280,
                classes: list[int] = (0,), ball_imgsz: int = 0) -> dict:
    out = _track_impl(video_bytes, model_name, conf, imgsz, list(classes), "/weights", ball_imgsz)
    weights.commit()
    return out


def track_local(video_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 1280,
                classes: list[int] = (0,), ball_imgsz: int = 0) -> dict:
    """CPU fallback (slow) for debugging without Modal."""
    from pathlib import Path

    return _track_impl(Path(video_path).read_bytes(), model_name, conf, imgsz, list(classes), None, ball_imgsz)
