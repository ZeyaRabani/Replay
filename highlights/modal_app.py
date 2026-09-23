"""Modal GPU worker: player ByteTrack + tiled ball detection per chunk.

Deployed lazily by ``highlights run``; can also be deployed with
``modal deploy highlights/modal_app.py``.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "highlights-detect"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.3,<9", "opencv-python-headless", "lap>=0.5", "numpy",
                 "supervision>=0.25", "huggingface_hub")
)
weights = modal.Volume.from_name("highlights-weights", create_if_missing=True)
footage = modal.Volume.from_name("football-footage", create_if_missing=False)
FOOTAGE_DIR = "/footage"
app = modal.App(APP_NAME)


def _ffprobe(path: str) -> dict:
    import json
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,r_frame_rate,width,height",
         "-show_entries", "format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    fps = eval(video["r_frame_rate"]) if video else 0.0
    return {"duration": float(info["format"]["duration"]), "fps": fps,
            "width": int(video["width"]) if video else 0, "height": int(video["height"]) if video else 0,
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
            "streams": [s["codec_type"] for s in info["streams"]]}


def _run(cmd: list[str]) -> None:
    import subprocess

    subprocess.run(cmd, check=True)


@app.function(image=image, volumes={FOOTAGE_DIR: footage}, timeout=600)
def probe_remote(path: str) -> dict:
    return _ffprobe(f"{FOOTAGE_DIR}/{path}")


@app.function(image=image, volumes={FOOTAGE_DIR: footage}, timeout=600)
def read_frame_remote(path: str, t: float = 1.0) -> bytes:
    """JPEG bytes of the frame at t seconds."""
    import subprocess

    return subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", f"{FOOTAGE_DIR}/{path}",
                           "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                          check=True, capture_output=True).stdout


@app.function(image=image, volumes={FOOTAGE_DIR: footage}, timeout=1800)
def audio_envelope_remote(path: str, hop_s: float = 0.1) -> dict:
    """RMS dB envelope computed in-container; returns compact arrays."""
    import subprocess

    import librosa
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{FOOTAGE_DIR}/{path}", "-vn", "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    frame = max(1, int(hop_s * 16000))
    rms = librosa.feature.rms(y=x, frame_length=2 * frame, hop_length=frame)[0]
    db = 20.0 * np.log10(rms + 1e-8)
    return {"hop_s": hop_s, "t": (np.arange(len(db)) * hop_s).tolist(), "rms_db": db.tolist()}


@app.function(image=image, volumes={FOOTAGE_DIR: footage}, timeout=1800)
def split_remote(path: str, chunk_s: float, chunk_times: list[float] | None = None) -> list[dict]:
    """Stream-copy split into /footage/work/<stem>/chunks/; returns chunk metadata."""
    from pathlib import Path

    import numpy as np

    src = Path(f"{FOOTAGE_DIR}/{path}")
    out_dir = src.parent / "work" / src.stem / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = _ffprobe(str(src))["duration"]
    times = chunk_times if chunk_times else list(np.arange(0.0, total - 0.05, chunk_s))
    chunks = []
    for idx, t in enumerate(times):
        dst = out_dir / f"chunk_{idx:03d}.mp4"
        _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-t", f"{chunk_s:.3f}",
              "-c", "copy", str(dst)])
        if not dst.exists() or dst.stat().st_size == 0:
            break
        chunks.append({"idx": idx, "path": str(dst.relative_to(FOOTAGE_DIR)), "t0": t,
                       "duration": _ffprobe(str(dst))["duration"]})
    footage.commit()
    return chunks


@app.function(image=image, volumes={FOOTAGE_DIR: footage}, timeout=1800)
def extract_clips_remote(path: str, clips: list[dict]) -> dict:
    """clips: [{"id","start","end"}] -> {id: mp4 bytes}, 720p re-encode."""
    import subprocess

    out = {}
    for c in clips:
        out[c["id"]] = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, c['start']):.3f}", "-i", f"{FOOTAGE_DIR}/{path}",
             "-t", f"{c['end'] - c['start']:.3f}", "-vf", "scale=-2:720", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof", "pipe:1"],
            check=True, capture_output=True).stdout
    return out


@app.function(image=image, gpu=os.environ.get("HIGHLIGHTS_GPU", "A10G") or None, timeout=60 * 30, volumes={"/weights": weights, FOOTAGE_DIR: footage})
def detect_chunk_path(chunk_path: str, cfg_dict: dict) -> dict:
    """Detect on a chunk already stored on the footage volume."""
    with open(f"{FOOTAGE_DIR}/{chunk_path}", "rb") as f:
        data = f.read()
    out = _detect_impl(data, cfg_dict, "/weights")
    weights.commit()
    return out


def _load_ball_model(spec: str, weights_dir: str | None):
    """spec: "hf:<repo>:<file>" | "coco" | local path | ultralytics model name."""
    import os

    from ultralytics import YOLO

    if spec == "coco":
        return YOLO("yolov8x.pt"), 32  # COCO class 32 = sports ball
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        _, repo, fname = spec.split(":", 2)
        kwargs = {"local_dir": weights_dir} if weights_dir else {}
        return YOLO(hf_hub_download(repo_id=repo, filename=fname, **kwargs)), 0
    if weights_dir and not spec.startswith("/"):
        cached = os.path.join(weights_dir, spec)
        if os.path.exists(cached):
            return YOLO(cached), 0
    return YOLO(spec), 0


def _detect_impl(video_bytes: bytes, cfg: dict, weights_dir: str | None) -> dict:
    import os
    import tempfile

    import cv2
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        os.chdir(weights_dir)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    player_model = YOLO(cfg["player_model"])
    ball_model, ball_cls = _load_ball_model(cfg["ball_model"], weights_dir)

    def ball_cb(tile: np.ndarray) -> sv.Detections:
        r = ball_model.predict(tile, conf=cfg["ball_conf"], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(r)
        if ball_cls > 0:
            dets = dets[dets.class_id == ball_cls]
        return dets

    tile_w, tile_h = cfg["ball_tile"]
    ov = (int(cfg["ball_overlap"] * tile_w), int(cfg["ball_overlap"] * tile_h))
    slicer = sv.InferenceSlicer(callback=ball_cb, slice_wh=(int(tile_w), int(tile_h)),
                                overlap_wh=ov, iou_threshold=0.5)
    stride = int(cfg["stride"])
    players, ball = [], []
    n_proc = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            r = player_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=cfg["player_conf"],
                                   iou=0.6, imgsz=cfg["player_imgsz"], classes=[0], verbose=False, half=True)[0]
            pd = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
                       else np.full(len(xyxy), -1))
                for b, c, t in zip(xyxy, cf, ids):
                    pd.append({"id": int(t), "conf": round(float(c), 3),
                               "box": [round(float(v), 1) for v in b]})
            bd = []
            dets = slicer(frame)
            if dets.confidence is not None and len(dets):
                order = np.argsort(-dets.confidence)[: int(cfg["ball_keep"])]
                for j in order:
                    bd.append({"conf": round(float(dets.confidence[j]), 3),
                               "box": [round(float(v), 1) for v in dets.xyxy[j]]})
            players.append(pd)
            ball.append(bd)
            n_proc += 1
        i += 1
    cap.release()
    return {"fps": fps, "width": w, "height": h, "n_frames": n_proc, "stride": stride,
            "players": players, "ball": ball}


@app.function(image=image, gpu=os.environ.get("HIGHLIGHTS_GPU", "A10G") or None, timeout=60 * 30, volumes={"/weights": weights})
def detect_chunk(video_bytes: bytes, cfg_dict: dict) -> dict:
    out = _detect_impl(video_bytes, cfg_dict, "/weights")
    weights.commit()
    return out


def detect_local(video_path: str, cfg: dict) -> dict:
    """CPU fallback for debugging without Modal."""
    from pathlib import Path

    return _detect_impl(Path(video_path).read_bytes(), cfg, None)


def run_detection(chunks, cfg, out_dir, local: bool = False) -> list:
    """Detect all chunks; returns list of cached JSON paths."""
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"chunk_{c.idx:03d}.json" for c in chunks]
    todo = [(c, p) for c, p in zip(chunks, paths) if not p.exists()]
    if todo:
        if local:
            results = [detect_local(str(c.path), cfg.to_dict()) for c, _ in todo]
        else:
            results = list(detect_chunk.map([c.path.read_bytes() for c, _ in todo],
                                            [cfg.to_dict()] * len(todo)))
        for (_, p), r in zip(todo, results):
            p.write_text(json.dumps(r))
    return paths
