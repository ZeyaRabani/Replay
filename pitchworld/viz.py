"""Debug renders: top-down pitch map video, per-camera overlay video, calibration check images."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .calibrate import CameraCalibration
from .pitch import PitchModel

PALETTE = [(255, 99, 71), (30, 144, 255), (50, 205, 50), (255, 215, 0), (186, 85, 211), (0, 206, 209),
           (255, 140, 0), (220, 20, 60), (127, 255, 0), (255, 105, 180), (70, 130, 180), (244, 164, 96)]


def _color(i: int):
    r, g, b = PALETTE[i % len(PALETTE)]
    return (b, g, r)


class PitchCanvas:
    def __init__(self, pitch: PitchModel, scale: float = 12.0, pad: int = 40):
        self.pitch, self.scale, self.pad = pitch, scale, pad
        self.w = int(pitch.length * scale) + 2 * pad
        self.h = int(pitch.width * scale) + 2 * pad

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        # y up on the pitch -> image rows go down, so flip
        return int(self.pad + x * self.scale), int(self.h - self.pad - y * self.scale)

    def base(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), (40, 110, 40), np.uint8)
        for a, b in self.pitch.segments():
            cv2.line(img, self.to_px(*a), self.to_px(*b), (235, 235, 235), 2)
        for (cx, cy), r, s, e in self.pitch.arcs():
            pts = [self.to_px(cx + r * math.cos(math.radians(t)), cy + r * math.sin(math.radians(t)))
                   for t in np.linspace(s, e, 48)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 200, 60), 2)
        g = self.pitch.goal_width / 2
        for x in (0, self.pitch.length):
            cv2.line(img, self.to_px(x, self.pitch.width / 2 - g), self.to_px(x, self.pitch.width / 2 + g), (255, 255, 255), 6)
        return img


def render_pitch_map(timeline: list[dict], pitch: PitchModel, fps: float, out: Path, trails: int = 15) -> None:
    canvas = PitchCanvas(pitch)
    base = canvas.base()
    tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.w, canvas.h))
    history: dict[int, list[tuple[int, int]]] = {}
    for fr in timeline:
        img = base.copy()
        for p in fr["players"]:
            pt = canvas.to_px(p["x"], p["y"])
            history.setdefault(p["id"], []).append(pt)
            history[p["id"]] = history[p["id"]][-trails:]
            col = _color(p["id"])
            if len(history[p["id"]]) > 1:
                cv2.polylines(img, [np.array(history[p["id"]], np.int32)], False, col, 1)
            cv2.circle(img, pt, 7, col, -1)
            cv2.circle(img, pt, 7, (0, 0, 0), 1)
            cv2.putText(img, f"{p['id']}", (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(img, "".join(str(c) for c in p["cameras"]), (pt[0] + 8, pt[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"t={fr['t']:.2f}s  players={len(fr['players'])}", (10, canvas.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(img)
    vw.release()
    _h264(tmp, out)


def render_overlay(video: Path, timeline: list[dict], cam: int, out: Path, pitch: PitchModel,
                   cal: CameraCalibration, max_frames: int | None = None) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    grid = _pitch_grid_pixels(pitch, cal, w, h)
    for fr in timeline:
        ok, img = cap.read()
        if not ok or (max_frames and fr["frame"] >= max_frames):
            break
        for seg in grid:
            cv2.polylines(img, [seg], False, (0, 255, 255), 1, cv2.LINE_AA)
        for p in fr["players"]:
            for d in p["detections"]:
                if d["camera"] != cam:
                    continue
                x1, y1, x2, y2 = map(int, d["box"])
                col = _color(p["id"])
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                label = f"P{p['id']} ({p['x']:.0f},{p['y']:.0f}) cams={''.join(map(str, p['cameras']))}"
                cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(img, f"cam{cam} t={fr['t']:.2f}s", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    cap.release()
    _h264(tmp, out)


def _pitch_grid_pixels(pitch: PitchModel, cal: CameraCalibration, w: int, h: int) -> list[np.ndarray]:
    """Pitch line segments (and arcs) re-projected into the image, clipped to the frame."""
    Hinv = np.linalg.inv(cal.matrix)
    segs: list[np.ndarray] = []
    polylines = [np.linspace(np.array(a), np.array(b), 40) for a, b in pitch.segments()]
    for (cx, cy), r, s, e in pitch.arcs():
        ts = np.radians(np.linspace(s, e, 60))
        polylines.append(np.stack([cx + r * np.cos(ts), cy + r * np.sin(ts)], 1))
    for pl in polylines:
        # drop points behind the camera (w <= 0 in homogeneous coords)
        hom = np.c_[pl, np.ones(len(pl))] @ Hinv.T
        keep = hom[:, 2] > 1e-6
        px = hom[keep, :2] / hom[keep, 2:3]
        inside = (px[:, 0] > -w) & (px[:, 0] < 2 * w) & (px[:, 1] > -h) & (px[:, 1] < 2 * h)
        if inside.sum() >= 2:
            segs.append(px[inside].astype(np.int32).reshape(-1, 1, 2))
    return segs


def calibration_check_image(frame: np.ndarray, pitch: PitchModel, cal: CameraCalibration, out: Path) -> None:
    img = frame.copy()
    h, w = img.shape[:2]
    for seg in _pitch_grid_pixels(pitch, cal, w, h):
        cv2.polylines(img, [seg], False, (0, 255, 255), 2, cv2.LINE_AA)
    for p in cal.points:
        u, v = map(int, p["pixel"])
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, str(p["world"]), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for tr in cal.lines + cal.arcs + cal.parallels:
        for u, v in tr["pixels"]:
            cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), 2)
        u, v = tr["pixels"][0]
        cv2.putText(img, str(tr["world"]), (int(u) + 8, int(v) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    pose = f"  cam@({cal.pose['x']:.1f},{cal.pose['y']:.1f}) h={cal.pose['height']:.1f}m" if cal.pose else ""
    cv2.putText(img, f"{cal.method}  err={cal.reproj_error_m:.2f}m/{cal.reproj_error_px:.1f}px  conf={cal.confidence:.2f}{pose}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out), img)


def sync_contact_sheet(clips: list[Path], times: list[float], out: Path, tile_w: int = 640) -> None:
    rows = []
    for t in times:
        tiles = []
        for c in clips:
            cap = cv2.VideoCapture(str(c))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            cap.release()
            if not ok:
                f = np.zeros((360, 640, 3), np.uint8)
            f = cv2.resize(f, (tile_w, int(f.shape[0] * tile_w / f.shape[1])))
            cv2.putText(f, f"{c.name} t={t:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            tiles.append(f)
        hmin = min(t.shape[0] for t in tiles)
        rows.append(np.hstack([t[:hmin] for t in tiles]))
    cv2.imwrite(str(out), np.vstack(rows))


def _h264(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", str(dst)], check=True)
    src.unlink(missing_ok=True)
