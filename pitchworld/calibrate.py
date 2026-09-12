"""Per-camera ground-plane homography (pixel -> pitch metres).

Two paths:
  * ``auto_calibrate`` — detects white pitch lines and their intersections and
    tries to fit the pitch template. It returns None (with a reason) when the
    footage does not expose enough line structure, which is the normal case for
    pitch-level amateur cameras. It never silently returns a bad homography.
  * ``manual_calibrate`` — user-supplied constraints: landmark<->pixel pairs
    and/or pixels traced along known pitch lines / arcs (see ``calib_tool.py``
    for the click UI, or write the JSON by hand). Fitted as a physical camera
    pose (``posefit``), so a goal line + a "D" + one more line is enough.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .pitch import PitchModel
from .posefit import Constraints, Pose, fit_pose, homography_pixel_to_world, rotation


@dataclass
class CameraCalibration:
    H: list[list[float]]  # 3x3 pixel -> world
    method: str  # "auto" | "manual"
    reproj_error_m: float
    reproj_error_px: float
    n_points: int
    confidence: float  # 0..1
    notes: list[str] = field(default_factory=list)
    points: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    arcs: list[dict] = field(default_factory=list)
    parallels: list[dict] = field(default_factory=list)
    pose: dict | None = None  # camera position/orientation/focal when fitted via posefit

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.H, dtype=np.float64)

    def project(self, px: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,2) metres."""
        px = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(px, self.matrix).reshape(-1, 2)

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(xy, np.linalg.inv(self.matrix)).reshape(-1, 2)

    def to_dict(self) -> dict:
        return asdict(self)


def read_frame(video: Path, t: float = 1.0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame at {t}s from {video}")
    return frame


# --------------------------------------------------------------------------
# Manual
# --------------------------------------------------------------------------
def _fit(world: np.ndarray, pixel: np.ndarray) -> tuple[np.ndarray, float, float]:
    if len(world) == 4:
        H = cv2.getPerspectiveTransform(pixel.astype(np.float32), world.astype(np.float32)).astype(np.float64)
    else:
        H, _ = cv2.findHomography(pixel, world, method=0)  # least squares on all user points
        if H is None:
            raise ValueError("homography fit failed (degenerate points?)")
    # sign convention: pixels in front of the camera get a positive homogeneous scale
    if (H @ np.array([*pixel.mean(0), 1.0]))[2] < 0:
        H = -H
    proj = cv2.perspectiveTransform(pixel.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_m = float(np.sqrt(((proj - world) ** 2).sum(1).mean()))
    back = cv2.perspectiveTransform(world.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    err_px = float(np.sqrt(((back - pixel) ** 2).sum(1).mean()))
    return H, err_m, err_px


def _constraint_dof(points: list[dict], lines: list[dict], arcs: list[dict], parallels: list[dict]) -> int:
    dof = 2 * len(points)
    dof += sum(2 for l in lines if len(l["pixels"]) >= 2)
    dof += sum(min(5, len(a["pixels"])) for a in arcs if len(a["pixels"]) >= 3)
    dof += sum(1 for p in parallels if len(p["pixels"]) >= 2)
    return dof


def manual_calibrate(entry: dict | list[dict], pitch: PitchModel, frame_size: tuple[int, int] | None = None
                     ) -> CameraCalibration:
    """Fit one camera from user constraints.

    ``entry`` is either the legacy list of points or a dict with any of
    ``points`` [{"world": name|[x,y], "pixel": [u,v]}], ``lines`` [{"world": name, "pixels": [[u,v],..]}],
    ``arcs`` [{"world": name, "pixels": [[u,v],..]}], ``parallels`` [{"world": line name, "pixels": ..}]
    (pixels on an unidentified line known to be parallel to a named one). With ``frame_size`` (w, h) the fit is a physical
    camera pose (recommended); without it only >= 4 points are accepted and a plain DLT is used.
    """
    if isinstance(entry, list):
        entry = {"points": entry}
    points = entry.get("points") or []
    lines = entry.get("lines") or []
    arcs = entry.get("arcs") or []
    parallels = entry.get("parallels") or []
    notes: list[str] = []

    if frame_size is None or (not lines and not arcs and not parallels and len(points) >= 6):
        if len(points) < 4:
            raise ValueError("need at least 4 landmark points per camera (or traced lines/arcs with a frame size)")
        world = np.array([pitch.resolve(p["world"]) for p in points], dtype=np.float64)
        pixel = np.array([p["pixel"] for p in points], dtype=np.float64)
        H, err_m, err_px = _fit(world, pixel)
        if not np.all(np.isfinite(H)):
            raise ValueError("non-finite homography")
        conf = float(np.clip(1.0 - err_px / 15.0, 0.0, 1.0))
        if len(points) == 4:
            conf = min(conf, 0.6)
            notes.append("only 4 points: fit is exact, so reprojection error cannot validate it; add more landmarks")
        if err_px > 8:
            notes.append(f"high reprojection error {err_px:.1f}px / {err_m:.2f}m: landmark clicks or pitch dimensions are off")
        return CameraCalibration(H=H.tolist(), method="manual_dlt", reproj_error_m=err_m, reproj_error_px=err_px,
                                 n_points=len(points), confidence=conf, notes=notes, points=points)

    dof = _constraint_dof(points, lines, arcs, parallels)
    if dof < 7:
        raise ValueError(f"constraints give only {dof} degrees of freedom; a camera pose needs >= 7 "
                         "(e.g. 4 points, or 1 line + 1 arc + 1 more line/point)")
    w, h = frame_size
    cons = Constraints.build(pitch, points, lines, arcs, parallels)
    pose, res = fit_pose(cons, pitch, w, h)
    H = homography_pixel_to_world(pose, w, h)
    err_m = float(np.sqrt(np.mean(np.square(res))))
    err_px = float(np.sqrt(np.mean(np.square(cons.residuals_px(H, np.linalg.inv(H))))))
    conf = float(np.clip(1.0 - err_m / 1.0, 0.0, 1.0))
    if dof < 9:
        conf = min(conf, 0.6)
        notes.append(f"constraints are thin ({dof} dof for a 7-dof pose): add another line/landmark to validate")
    if err_m > 0.5:
        notes.append(f"high residual {err_m:.2f}m rms: traced pixels or pitch dimensions (length/width/d_radius) are off")
    if not 0.8 <= pose.z <= 20:
        conf *= 0.5
        notes.append(f"implausible camera height {pose.z:.1f}m: check pitch dimensions / constraints")
    n = len(points) + sum(len(t["pixels"]) for t in lines + arcs + parallels)
    return CameraCalibration(H=H.tolist(), method="manual_pose", reproj_error_m=err_m, reproj_error_px=err_px,
                             n_points=n, confidence=conf, notes=notes, points=points, lines=lines, arcs=arcs,
                             parallels=parallels, pose=pose.to_dict())


def snap_polyline_to_mask(mask: np.ndarray, polyline: list[list[float]], band: float = 10.0,
                          step: float = 25.0) -> list[list[float]]:
    """Refine a roughly-drawn polyline onto a binary line mask.

    Walks along the polyline every ``step`` px, and at each station takes the centroid of mask pixels within
    ``band`` px of the station (perpendicular window). Stations without mask support are dropped, so the
    result is a clean set of on-line pixels suitable as a line/arc constraint.
    """
    pl = np.asarray(polyline, dtype=np.float64)
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(np.float64)
    out: list[list[float]] = []
    for a, b in itertools.pairwise(pl):
        seg = b - a
        n = max(int(np.linalg.norm(seg) // step), 1)
        for k in range(n + 1):
            s = a + seg * (k / n)
            near = pts[np.linalg.norm(pts - s, axis=1) < band]
            if len(near) >= 3:
                out.append([float(v) for v in near.mean(0)])
    return out


def colour_mask(frame: np.ndarray, colour: str) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if colour == "white":
        return cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    if colour == "blue":
        return cv2.inRange(hsv, (95, 60, 40), (130, 255, 255))
    if colour == "yellow":
        return cv2.inRange(hsv, (20, 80, 100), (40, 255, 255))
    if colour == "red":
        return cv2.inRange(hsv, (0, 90, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 80), (180, 255, 255))
    raise KeyError(f"unknown line colour {colour!r}")


def metres_per_unit_from_people(cal: CameraCalibration, boxes: np.ndarray, frame_size: tuple[int, int],
                                person_height_m: float = 1.75) -> tuple[float, int]:
    """Absolute scale from detected people, for calibrations fitted in a pitch model of unknown size.

    For each (x1,y1,x2,y2) person box, back-project the feet to the ground and the head onto the vertical
    through that point; the median implied person height (in model units) vs ``person_height_m`` gives
    metres-per-unit. Needs a pose-fitted calibration. Returns (scale, n_used).
    """
    if not cal.pose:
        raise ValueError("scale from people needs a pose-fitted calibration")
    p = cal.pose
    pose = Pose(p["x"], p["y"], p["height"], math.radians(p["yaw_deg"]), math.radians(p["pitch_deg"]),
                math.radians(p["roll_deg"]), p["focal_px"])
    w, h = frame_size
    Hpw = homography_pixel_to_world(pose, w, h)
    feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    heads = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1]]
    g = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hpw).reshape(-1, 2)
    # ray through head pixel: C + s * dir ; find s where it passes above g -> height
    R = rotation(pose.yaw, pose.pitch, pose.roll)
    K_inv = np.linalg.inv(np.array([[pose.f, 0, w / 2], [0, pose.f, h / 2], [0, 0, 1]]))
    dirs = (R.T @ (K_inv @ np.c_[heads, np.ones(len(heads))].T)).T  # world directions
    C = np.array([pose.x, pose.y, pose.z])
    heights = []
    for gi, d in zip(g, dirs):
        # closest approach of the ray to the vertical line through gi (in xy)
        dxy = d[:2]
        if np.linalg.norm(dxy) < 1e-9:
            continue
        s = (gi - C[:2]) @ dxy / (dxy @ dxy)
        if s <= 0:
            continue
        z = C[2] + s * d[2]
        if 0.2 < z < 10 * pose.z:
            heights.append(z)
    if len(heights) < 5:
        return float("nan"), len(heights)
    return person_height_m / float(np.median(heights)), len(heights)


# --------------------------------------------------------------------------
# Automatic (line based) — best effort, refuses when structure is insufficient
# --------------------------------------------------------------------------
def white_line_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # grass mask: hue ~ 35..90, decent saturation
    grass = cv2.inRange(hsv, (30, 40, 40), (95, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    grass = cv2.dilate(grass, np.ones((15, 15), np.uint8))
    white = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
    white = cv2.bitwise_and(white, grass)  # only white *on the pitch*
    # top-hat to keep thin bright structures (lines), drop big blobs (kits, goals)
    th = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    lines = cv2.subtract(white, th)
    return lines


def detect_lines(frame: np.ndarray, min_len_frac: float = 0.12) -> np.ndarray:
    """Return (K,4) line segments [x1,y1,x2,y2] of long white pitch lines."""
    mask = white_line_mask(frame)
    _h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80, minLineLength=int(min_len_frac * w), maxLineGap=40)
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(float)
    # merge near-duplicate segments (same angle, close offset)
    merged: list[np.ndarray] = []
    for s in segs:
        ang = math.atan2(s[3] - s[1], s[2] - s[0]) % math.pi
        dup = False
        for i, m in enumerate(merged):
            mang = math.atan2(m[3] - m[1], m[2] - m[0]) % math.pi
            if min(abs(ang - mang), math.pi - abs(ang - mang)) < math.radians(3):
                # distance from s midpoint to line m
                mid = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
                p, q = m[:2], m[2:]
                d = abs(np.cross(q - p, mid - p)) / (np.linalg.norm(q - p) + 1e-9)
                if d < 12:
                    # keep the longer extent
                    pts = np.array([s[:2], s[2:], p, q])
                    axis = (q - p) / (np.linalg.norm(q - p) + 1e-9)
                    t = pts @ axis
                    merged[i] = np.concatenate([pts[np.argmin(t)], pts[np.argmax(t)]])
                    dup = True
                    break
        if not dup:
            merged.append(s.copy())
    return np.array(merged) if merged else np.zeros((0, 4))


def _intersections(lines: np.ndarray, w: int, h: int) -> list[tuple[float, float]]:
    pts = []
    for a, b in itertools.combinations(lines, 2):
        x1, y1, x2, y2 = a
        x3, y3, x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            continue
        ang_a = math.atan2(y2 - y1, x2 - x1)
        ang_b = math.atan2(y4 - y3, x4 - x3)
        d = abs(ang_a - ang_b) % math.pi
        if min(d, math.pi - d) < math.radians(10):
            continue  # near-parallel
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        if -0.2 * w <= px <= 1.2 * w and -0.2 * h <= py <= 1.2 * h:
            pts.append((px, py))
    return pts


def auto_calibrate(frame: np.ndarray, pitch: PitchModel, min_intersections: int = 4) -> tuple[CameraCalibration | None, str, dict]:
    """Try to fit the pitch from detected line intersections.

    Returns (calibration | None, reason, debug). We only trust the result when
    at least ``min_intersections`` distinct intersections of long white lines
    are visible *and* the best template assignment reprojects within tolerance.
    """
    h, w = frame.shape[:2]
    lines = detect_lines(frame)
    inter = _intersections(lines, w, h)
    debug = {"lines": lines.tolist(), "intersections": inter}
    if len(lines) < 3 or len(inter) < min_intersections:
        return None, (f"auto: found {len(lines)} long white line(s), {len(inter)} intersection(s); need >= 3 lines and "
                      f">= {min_intersections} intersections to fit the pitch template"), debug

    # Candidate template points: line intersections of the model (corners, box corners, half-way ends)
    lm = pitch.landmarks()
    cand_names = [n for n in lm if n.startswith(("corner_", "half_", "A_box", "B_box", "A_6yd", "B_6yd"))]
    cand = np.array([lm[n] for n in cand_names])
    img_pts = np.array(inter[:8])  # keep it tractable
    best = None
    for img_combo in itertools.combinations(range(len(img_pts)), 4):
        P = img_pts[list(img_combo)]
        for world_combo in itertools.permutations(range(len(cand)), 4):
            Wd = cand[list(world_combo)]
            try:
                H = cv2.getPerspectiveTransform(P.astype(np.float32), Wd.astype(np.float32))
            except cv2.error:
                continue
            # Score: how many *other* intersections land near *other* template points
            proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2).astype(np.float64), H.astype(np.float64)).reshape(-1, 2)
            d = np.sqrt(((proj[:, None, :] - cand[None, :, :]) ** 2).sum(-1)).min(1)
            inliers = int((d < 1.0).sum())
            if best is None or inliers > best[0]:
                best = (inliers, H, d)
    if best is None or best[0] < 5:
        return None, f"auto: template fit unsupported (best {best[0] if best else 0} consistent intersections, need >= 5)", debug
    inliers, H, d = best
    err_m = float(np.sqrt((d[d < 1.0] ** 2).mean()))
    conf = float(np.clip(inliers / 8.0, 0, 1) * np.clip(1 - err_m, 0, 1))
    return CameraCalibration(H=H.tolist(), method="auto", reproj_error_m=err_m, reproj_error_px=float("nan"),
                             n_points=inliers, confidence=conf, notes=[f"auto fit from {inliers} intersections"]), "ok", debug


# --------------------------------------------------------------------------
def load_calibration_file(path: Path) -> dict:
    return json.loads(path.read_text())


def calibrate_camera(video: Path, pitch: PitchModel, manual: dict | list[dict] | None,
                     frame_time: float = 1.0) -> tuple[CameraCalibration, list[str]]:
    """Auto first; fall back to manual constraints if given, else raise with guidance."""
    warnings: list[str] = []
    frame = read_frame(video, frame_time)
    h, w = frame.shape[:2]
    auto, reason, _ = auto_calibrate(frame, pitch)
    if auto is not None and auto.confidence >= 0.6 and not manual:
        return auto, warnings
    if auto is None:
        warnings.append(reason)
    else:
        warnings.append(f"auto calibration low confidence ({auto.confidence:.2f}); {'using manual constraints' if manual else 'no manual constraints'}")
    if manual:
        cal = manual_calibrate(manual, pitch, frame_size=(w, h))
        warnings += cal.notes
        return cal, warnings
    raise RuntimeError(
        f"{video.name}: automatic calibration failed ({reason}). Run\n"
        f"  pitchworld calibrate {video} --camera <i> --pitch <pitch.json> --out calib.json\n"
        f"and click landmarks / trace lines, then re-run with --calib calib.json")
