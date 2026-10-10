"""Visual calibration check overlays.

Camera mode (default): the frame from a camera video at shared time t
with (a) pitch lines projected through inverse stab + inverse H in
green, (b) that camera's own detections as blue boxes, (c) tracks.json
positions reprojected into the frame as red circles.

    python -m highlights.analysis.overlay_check P --angle 0 --t 2500 \
        --video /path/match.mp4 --out frame.png

Radar mode: top-down pitch view with each camera's projected detections
in a different colour:

    python -m highlights.analysis.overlay_check P --t 2500 \
        --radar --out radar.png
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from .calib import apply_h, effective_h
from .fuse_tracks import load_dets
from .stabilize import load_stab, warp_at

DET_TOL_S = 0.35
TRACK_TOL_S = 0.75
CAM_COLORS = [(255, 80, 80), (80, 255, 80), (80, 140, 255),
              (255, 220, 80), (255, 80, 255), (80, 255, 255)]


def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _stab_at(stab: dict | None, file_t: float) -> np.ndarray:
    """Current->reference-frame H nearest file_t (identity if none)."""
    if not stab or stab.get("t") is None or not len(stab["t"]):
        return np.eye(3)
    i = int(np.abs(stab["t"] - file_t).argmin())
    return np.asarray(stab["H"][i], dtype=float)


def _pitch_polylines(pitch: dict) -> list[np.ndarray]:
    """Pitch line segments in metres (small template + full fallback)."""
    L = float(pitch.get("len_m") or 100.0)
    W = float(pitch.get("wid_m") or 64.0)
    lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),
             np.array([[L / 2, 0], [L / 2, W]], float)]
    if pitch.get("template") == "small":
        r = float(pitch.get("d_radius_m") or 9.0)
        cy = W / 2
        th = np.linspace(-math.pi / 2, math.pi / 2, 25)
        lines.append(np.c_[r * np.cos(th), cy + r * np.sin(th)])
        lines.append(np.c_[L - r * np.cos(th), cy + r * np.sin(th)])
        gw = float(pitch.get("goal_w_m") or 3.66)
        lines.append(np.array([[0, cy - gw / 2], [0, cy + gw / 2]]))
        lines.append(np.array([[L, cy - gw / 2], [L, cy + gw / 2]]))
    else:
        th = np.linspace(0, 2 * math.pi, 48)
        lines.append(np.c_[L / 2 + 9.15 * np.cos(th),
                           W / 2 + 9.15 * np.sin(th)])
    return lines


def _to_frame(H: np.ndarray, stabH: np.ndarray, x: float, y: float
              ) -> tuple[float, float]:
    """Pitch metres -> current-frame normalized fx,fy (inverse H then
    inverse stab warp)."""
    Hi = np.linalg.inv(H)
    Si = np.linalg.inv(stabH)
    u, v = apply_h(Hi, x, y)
    if not (math.isfinite(u) and math.isfinite(v)):
        return (math.inf, math.inf)
    return apply_h(Si, u, v)


def _grab_frame(video: Path, file_t: float) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, file_t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        # fallback via ffmpeg seek
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{file_t:.3f}", "-i",
             str(video), "-frames:v", "1", "-f", "image2pipe",
             "-vcodec", "png", "-"], capture_output=True)
        buf = np.frombuffer(out.stdout, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"no frame at {file_t:.1f}s in {video}")
    return frame


def _track_xy_at(tr: dict, t: float, t0: float, step: float):
    k = round((t - t0) / step)
    i = k - round((tr["start"] - t0) / step)
    xy = tr.get("xy") or []
    if 0 <= i < len(xy) and xy[i][0] is not None:
        return float(xy[i][0]), float(xy[i][1])
    return None


def camera_overlay(project_dir: Path, angle: int, t: float,
                   video: Path, out: Path,
                   players_v2: Path | None = None) -> Path:
    import cv2
    project_dir = Path(project_dir)
    pv2 = players_v2 or project_dir / "analysis" / "players_v2"
    calib = _load_json(project_dir / "multiangle" / "calib.json") or {}
    sync = _load_json(project_dir / "multiangle" / "sync.json") or {}
    offsets = [float(o) for o in sync.get("offsets") or []]
    entry = (calib.get("angles") or {}).get(str(angle)) or {}
    H = np.asarray(effective_h(entry), dtype=float)
    pitch = calib.get("pitch") or {}
    file_t = t - (offsets[angle] if angle < len(offsets) else 0.0)
    stab = load_stab(pv2 / f"stab_a{angle}.npz")
    stabH = _stab_at(stab, file_t)

    frame = _grab_frame(video, file_t)
    h, w = frame.shape[:2]

    def px(fx, fy):
        return round(fx * w), round(fy * h)

    # (a) pitch lines in green — densify long segments so the warp
    # bends them correctly
    for line in _pitch_polylines(pitch):
        pts = []
        for i in range(len(line)):
            if i:
                seg = line[i] - line[i - 1]
                nseg = max(1, int(np.hypot(*seg) / 1.0))
                for f in np.linspace(0, 1, nseg, endpoint=False):
                    x, y = line[i - 1] + seg * f
                    fx, fy = _to_frame(H, stabH, float(x), float(y))
                    pts.append((fx, fy))
            x, y = line[i]
            fx, fy = _to_frame(H, stabH, float(x), float(y))
            pts.append((fx, fy))
        pix = np.array([px(fx, fy) for fx, fy in pts
                        if math.isfinite(fx)], np.int32)
        if len(pix) >= 2:
            cv2.polylines(frame, [pix], False, (0, 255, 0), 2)

    # (b) own detections as blue boxes
    det = load_dets(pv2 / f"det_a{angle}.npz")
    sw, sh = w / float(det["w"]), h / float(det["h"])
    for i in np.flatnonzero(np.abs(det["t"] - file_t) <= DET_TOL_S):
        x1, y1, x2, y2 = det["box"][i]
        cv2.rectangle(frame, (int(x1 * sw), int(y1 * sh)),
                      (int(x2 * sw), int(y2 * sh)), (255, 0, 0), 2)

    # (c) tracks.json positions reprojected as red circles
    tracks = _load_json(pv2 / "tracks.json") or {}
    t0 = float(tracks.get("t0") or 0.0)
    step = float(tracks.get("step") or 0.5)
    for tr in tracks.get("tracks") or []:
        p = _track_xy_at(tr, t, t0, step)
        if p is None:
            continue
        fx, fy = _to_frame(H, stabH, p[0], p[1])
        if math.isfinite(fx):
            cx, cy = px(fx, fy)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)
            cv2.putText(frame, str(tr.get("id", "")), (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out


def radar_overlay(project_dir: Path, t: float, out: Path,
                  players_v2: Path | None = None, scale: int = 16
                  ) -> Path:
    """Top-down pitch with every camera's projected dets at t."""
    import cv2
    project_dir = Path(project_dir)
    pv2 = players_v2 or project_dir / "analysis" / "players_v2"
    calib = _load_json(project_dir / "multiangle" / "calib.json") or {}
    sync = _load_json(project_dir / "multiangle" / "sync.json") or {}
    offsets = [float(o) for o in sync.get("offsets") or []]
    pitch = calib.get("pitch") or {}
    L = float(pitch.get("len_m") or 100.0)
    W = float(pitch.get("wid_m") or 64.0)
    img = np.full((int(W * scale) + 20, int(L * scale) + 20, 3), 24,
                  np.uint8)

    def px(x, y):
        return round(x * scale) + 10, round(y * scale) + 10

    for line in _pitch_polylines(pitch):
        pix = np.array([px(float(x), float(y)) for x, y in line],
                       np.int32)
        cv2.polylines(img, [pix], False, (200, 200, 200), 1)
    for a, ent in sorted((calib.get("angles") or {}).items(),
                         key=lambda kv: int(kv[0])):
        a = int(a)
        H = effective_h(ent)
        if not H:
            continue
        H = np.asarray(H, dtype=float)
        f = pv2 / f"det_a{a}.npz"
        if not f.is_file():
            continue
        det = load_dets(f)
        off = offsets[a] if a < len(offsets) else 0.0
        stab = load_stab(pv2 / f"stab_a{a}.npz")
        col = CAM_COLORS[a % len(CAM_COLORS)]
        file_t = t - off
        for i in np.flatnonzero(np.abs(det["t"] - file_t) <= DET_TOL_S):
            if det["conf"][i] < 0.4:
                continue
            fx, fy = float(det["foot"][i][0]), float(det["foot"][i][1])
            if stab is not None:
                fx, fy = warp_at(stab, float(det["t"][i]), fx, fy)
            x, y = apply_h(H, fx, fy)
            if math.isfinite(x) and math.isfinite(y):
                cv2.circle(img, px(x, y), 4, col, -1)
        cv2.putText(img, f"a{a}", (10 + 30 * a, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="highlights.analysis.overlay_check")
    ap.add_argument("project_dir", type=Path)
    ap.add_argument("--angle", type=int)
    ap.add_argument("--t", type=float, required=True,
                    help="shared project seconds")
    ap.add_argument("--video", type=Path,
                    help="camera video path (camera mode)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radar", action="store_true")
    ap.add_argument("--players-v2", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.radar:
        p = radar_overlay(args.project_dir, args.t, args.out,
                          players_v2=args.players_v2)
    else:
        if args.angle is None or args.video is None:
            ap.error("camera mode needs --angle and --video")
        p = camera_overlay(args.project_dir, args.angle, args.t,
                           args.video, args.out,
                           players_v2=args.players_v2)
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
