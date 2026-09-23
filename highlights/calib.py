"""Calibration wrapper + goal zones for single-camera footage.

Returns a ``CameraCalibration`` when a homography can be trusted, otherwise
``None`` and downstream code works in pixel space.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pitchworld.calibrate import CameraCalibration, auto_calibrate, manual_calibrate, read_frame
from pitchworld.pitch import PitchModel


class Space(enum.Enum):
    PITCH = "pitch"
    PIXEL = "pixel"


@dataclass
class GoalZone:
    goal: str                # "A" (x=0 end) or "B" (x=length end)
    attack_dir: float        # +1 = x increasing (toward B), -1 = toward A
    goal_line_x: float       # pitch metres, or polygon for pixel space
    poly: list[list[float]]  # rectangle [x,y]x4 in pitch m, or [u,v]x4 pixels


@dataclass
class CalibResult:
    cal: CameraCalibration | None
    zones: dict[str, GoalZone]
    space: Space
    warnings: list[str] = field(default_factory=list)


def default_landmarks(pitch: PitchModel) -> list[str]:
    """8 penalty-box corners for standard pitches; 4 posts + 4 corners otherwise."""
    if pitch.penalty_depth > 0:
        return ["A_box_S_line", "A_box_S", "A_box_N", "A_box_N_line",
                "B_box_S_line", "B_box_S", "B_box_N", "B_box_N_line"]
    return ["A_post_S", "A_post_N", "B_post_S", "B_post_N",
            "corner_A_S", "corner_A_N", "corner_B_S", "corner_B_N"]


def goal_zones_pitch(pitch: PitchModel) -> dict[str, GoalZone]:
    L, W, cy = pitch.length, pitch.width, pitch.width / 2
    depth = pitch.penalty_depth or 0.3 * L
    half = (pitch.penalty_width / 2) or 0.4 * W
    return {
        "A": GoalZone("A", -1.0, 0.0, [[0, cy - half], [depth, cy - half], [depth, cy + half], [0, cy + half]]),
        "B": GoalZone("B", +1.0, float(L), [[L - depth, cy - half], [L, cy - half], [L, cy + half], [L - depth, cy + half]]),
    }


def goal_zones_pixel(path: Path) -> dict[str, GoalZone]:
    data = json.loads(Path(path).read_text())
    zones = {}
    for g in ("A", "B"):
        poly = [[float(u), float(v)] for u, v in data[g]]
        zones[g] = GoalZone(g, 0.0, float("nan"), poly)
    # attack direction = mean u of the zone relative to frame centre, set later when width known
    return zones


def _manual_entry(manual_calib_path: Path) -> dict | None:
    data = json.loads(Path(manual_calib_path).read_text())
    entry = (data.get("cameras", {}) or {}).get("0")
    if not entry:
        return None
    return {k: entry.get(k) for k in ("points", "lines", "arcs", "parallels")}


def calibrate_frame(frame: np.ndarray, pitch: PitchModel, manual_entry: dict | None = None,
                    goal_zones_px: Path | None = None, auto: bool = False) -> CalibResult:
    """Calibrate from a decoded BGR frame (used for remote sources)."""
    return _fit(frame, pitch, manual_entry, goal_zones_px, auto)


def calibrate(video: Path, pitch: PitchModel, manual_calib_path: Path | None = None,
              frame_time: float = 1.0, goal_zones_px: Path | None = None, auto: bool = False) -> CalibResult:
    frame = read_frame(video, frame_time)
    return _fit(frame, pitch, _manual_entry(manual_calib_path) if manual_calib_path else None,
                goal_zones_px, auto)


def _fit(frame: np.ndarray, pitch: PitchModel, manual_entry: dict | None,
         goal_zones_px: Path | None = None, auto: bool = False) -> CalibResult:
    warnings: list[str] = []
    h, w = frame.shape[:2]
    cal: CameraCalibration | None = None

    auto_cal = None
    if auto:
        auto_cal, reason, _ = auto_calibrate(frame, pitch)
        if auto_cal is not None and auto_cal.confidence >= 0.6:
            cal = auto_cal
        else:
            warnings.append(reason if auto_cal is None
                            else f"auto confidence {auto_cal.confidence:.2f} < 0.6")
    if cal is None and manual_entry is not None:
        cal = manual_calibrate(manual_entry, pitch, frame_size=(w, h))
        warnings += cal.notes

    if cal is None:
        warnings.append("no calibration -> falling back to PIXEL space; speed thresholds are heuristic")
        zones = goal_zones_pixel(goal_zones_px) if goal_zones_px else {}
        if zones:
            cx = w / 2
            for z in zones.values():
                mu = float(np.mean([p[0] for p in z.poly]))
                z.attack_dir = 1.0 if mu > cx else -1.0
        return CalibResult(None, zones, Space.PIXEL, warnings)
    return CalibResult(cal, goal_zones_pitch(pitch), Space.PITCH, warnings)


def in_zone(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(4,2) polygon, (N,2) points -> bool[N]. Ray casting, space-agnostic."""
    inside = np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= hit
        j = i
    return inside


def render_check(video: Path, cal: CameraCalibration | None, pitch: PitchModel,
                 zones: dict[str, GoalZone], out: Path, frame_time: float = 1.0) -> Path:
    return render_check_frame(read_frame(video, frame_time), cal, pitch, zones, out)


def render_check_frame(frame: np.ndarray, cal: CameraCalibration | None, pitch: PitchModel,
                       zones: dict[str, GoalZone], out: Path) -> Path:
    """Draw the unprojected pitch lines and goal zones on a frame."""
    import cv2
    if cal is not None:
        for a, b in pitch.segments():
            pa, pb = cal.unproject(np.array([a])), cal.unproject(np.array([b]))
            cv2.line(frame, tuple(pa[0].astype(int)), tuple(pb[0].astype(int)), (60, 200, 255), 2)
        for (cx, cy), r, _, _ in pitch.arcs():
            pts = [cal.unproject(np.array([[cx + r * np.cos(t), cy + r * np.sin(t)]]))[0]
                   for t in np.linspace(0, 2 * np.pi, 48)]
            cv2.polylines(frame, [np.array(pts, np.int32)], True, (60, 200, 255), 2)
    for g, z in zones.items():
        pts = cal.unproject(np.array(z.poly)).astype(np.int32) if cal is not None else np.array(z.poly, np.int32)
        cv2.polylines(frame, [pts], True, (0, 80, 255), 3)
        c = pts.mean(0).astype(int)
        cv2.putText(frame, f"goal {g}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3)
    cv2.imwrite(str(out), frame)
    return out
