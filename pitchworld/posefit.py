"""Camera pose fit from ground-plane constraints (points, traced lines, traced arcs).

A pinhole camera at height above a flat pitch is parameterised by
(x, y, z, yaw, pitch, roll, f). Its ground-plane homography is derived from
the pose, so the fit stays physically plausible even when only a few
distinct landmarks are visible (the usual case for pitch-level phone
footage: one goal line, a "D", maybe a touchline).

Constraint types (all pixel coordinates of the *calibration frame*):
  points: [{"world": "B_D_S" | [x, y], "pixel": [u, v]}]
  lines:  [{"world": "goal_line_B" | {"point": [x, y], "dir": [dx, dy]}, "pixels": [[u, v], ...]}]
  arcs:   [{"world": "B_D" | {"centre": [x, y], "radius": r},          "pixels": [[u, v], ...]}]

Residuals are measured in metres on the ground plane; the solver is a
multi-start Levenberg-Marquardt over the pose.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .pitch import PitchModel

BEHIND_PENALTY_M = 50.0  # metres of residual for pixels that back-project behind the camera
BEHIND_PENALTY_PX = 2000.0


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float  # rad, direction of view in the ground plane (0 = +x)
    pitch: float  # rad, positive = looking down
    roll: float  # rad
    f: float  # focal length in pixels

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw, self.pitch, self.roll, self.f], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> Pose:
        return cls(*map(float, v))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "height": self.z, "yaw_deg": math.degrees(self.yaw),
                "pitch_deg": math.degrees(self.pitch), "roll_deg": math.degrees(self.roll), "focal_px": self.f}


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """World (x east, y north, z up) -> camera (x right, y down, z forward)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    # camera looking along +x with z up: cam_x = -world_y, cam_y = -world_z, cam_z = world_x
    base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    r_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])  # rotate world so that view dir -> +x
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])  # tilt down about camera x
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])  # about camera z
    return r_roll @ r_pitch @ base @ r_yaw


def homography_world_to_pixel(pose: Pose, w: int, h: int) -> np.ndarray:
    R = rotation(pose.yaw, pose.pitch, pose.roll)
    C = np.array([pose.x, pose.y, pose.z])
    t = -R @ C
    K = np.array([[pose.f, 0, w / 2], [0, pose.f, h / 2], [0, 0, 1]])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def homography_pixel_to_world(pose: Pose, w: int, h: int) -> np.ndarray:
    # keep the sign: (H @ [u, v, 1])[2] > 0 exactly for pixels that see the ground in front of the camera
    H = np.linalg.inv(homography_world_to_pixel(pose, w, h))
    return H / abs(H[2, 2])


def _ground_points(Hpw: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.c_[px, np.ones(len(px))] @ Hpw.T
    depth_ok = hom[:, 2] > 1e-9
    xy = np.where(depth_ok[:, None], hom[:, :2] / np.where(depth_ok, hom[:, 2], 1.0)[:, None], 0.0)
    return xy, depth_ok


# --------------------------------------------------------------------------
def resolve_line(pitch: PitchModel, spec) -> tuple[np.ndarray, np.ndarray]:
    """-> (point, unit direction) of a named or explicit pitch line."""
    L, W = pitch.length, pitch.width
    named = {
        "goal_line_A": ((0, 0), (0, 1)), "goal_line_B": ((L, 0), (0, 1)),
        "touch_S": ((0, 0), (1, 0)), "touch_N": ((0, W), (1, 0)), "half": ((L / 2, 0), (0, 1)),
    }
    if isinstance(spec, str):
        if spec not in named:
            raise KeyError(f"unknown line {spec!r}; known: {sorted(named)} or {{'point': [x,y], 'dir': [dx,dy]}}")
        p, d = named[spec]
    else:
        p, d = spec["point"], spec["dir"]
    d = np.asarray(d, dtype=np.float64)
    return np.asarray(p, dtype=np.float64), d / np.linalg.norm(d)


def resolve_arc(pitch: PitchModel, spec) -> tuple[np.ndarray, float]:
    L, cy = pitch.length, pitch.width / 2
    named = {"A_D": ((0, cy), pitch.d_radius), "B_D": ((L, cy), pitch.d_radius),
             "centre_circle": ((L / 2, cy), pitch.centre_circle_radius)}
    if isinstance(spec, str):
        if spec not in named:
            raise KeyError(f"unknown arc {spec!r}; known: {sorted(named)} or {{'centre': [x,y], 'radius': r}}")
        c, r = named[spec]
    else:
        c, r = spec["centre"], spec["radius"]
    if r <= 0:
        raise ValueError(f"arc {spec!r} has radius 0 in this pitch model (set d_radius / centre_circle_radius)")
    return np.asarray(c, dtype=np.float64), float(r)


@dataclass
class Constraints:
    pt_px: np.ndarray  # (P,2)
    pt_world: np.ndarray  # (P,2)
    line_px: list[np.ndarray]
    line_geom: list[tuple[np.ndarray, np.ndarray]]
    arc_px: list[np.ndarray]
    arc_geom: list[tuple[np.ndarray, float]]
    par_px: list[np.ndarray] = field(default_factory=list)  # pixels on *some* line parallel to a known one
    par_dir: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def build(cls, pitch: PitchModel, points: list[dict] | None, lines: list[dict] | None,
              arcs: list[dict] | None, parallels: list[dict] | None = None) -> Constraints:
        points, lines, arcs, parallels = points or [], lines or [], arcs or [], parallels or []
        pt_px = np.array([p["pixel"] for p in points], dtype=np.float64).reshape(-1, 2)
        pt_world = np.array([pitch.resolve(p["world"]) for p in points], dtype=np.float64).reshape(-1, 2)
        return cls(pt_px, pt_world,
                   [np.asarray(l["pixels"], dtype=np.float64).reshape(-1, 2) for l in lines],
                   [resolve_line(pitch, l["world"]) for l in lines],
                   [np.asarray(a["pixels"], dtype=np.float64).reshape(-1, 2) for a in arcs],
                   [resolve_arc(pitch, a["world"]) for a in arcs],
                   [np.asarray(p["pixels"], dtype=np.float64).reshape(-1, 2) for p in parallels],
                   [resolve_line(pitch, p["world"])[1] for p in parallels])

    @property
    def n_residuals(self) -> int:
        return (2 * len(self.pt_px) + sum(len(p) for p in self.line_px) + sum(len(p) for p in self.arc_px)
                + sum(len(p) for p in self.par_px))

    def all_pixels(self) -> np.ndarray:
        parts = [self.pt_px] + self.line_px + self.arc_px + self.par_px
        return np.concatenate(parts) if self.n_residuals else np.zeros((0, 2))

    def _parallel_px(self, Hwp: np.ndarray) -> np.ndarray:
        """Distance of each 'parallel' pixel from the image line through its group's centroid and the
        vanishing point of the reference direction."""
        out = []
        for px, d in zip(self.par_px, self.par_dir):
            vp = Hwp @ np.array([d[0], d[1], 0.0])
            c = px.mean(0)
            if abs(vp[2]) < 1e-9:  # vanishing point at infinity -> direction only
                u = vp[:2] / np.linalg.norm(vp[:2])
            else:
                u = vp[:2] / vp[2] - c
                u /= max(np.linalg.norm(u), 1e-9)
            n = np.array([-u[1], u[0]])
            out.append(np.abs((px - c) @ n))
        return np.concatenate(out) if out else np.zeros(0)

    def _parallel_m(self, Hpw: np.ndarray) -> np.ndarray:
        out = []
        for px, d in zip(self.par_px, self.par_dir):
            xy, ok = _ground_points(Hpw, px)
            n = np.array([-d[1], d[0]])
            r = np.abs((xy - xy.mean(0)) @ n)
            r[~ok] = BEHIND_PENALTY_M
            out.append(r)
        return np.concatenate(out) if out else np.zeros(0)

    def closest_world(self, Hpw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For every constraint pixel: (ground point, closest world point satisfying the constraint, in-front mask)."""
        ground, target, ok_all = [], [], []
        if len(self.pt_px):
            xy, ok = _ground_points(Hpw, self.pt_px)
            ground.append(xy); target.append(self.pt_world); ok_all.append(ok)
        for px, (p0, d) in zip(self.line_px, self.line_geom):
            xy, ok = _ground_points(Hpw, px)
            t = (xy - p0) @ d
            ground.append(xy); target.append(p0 + t[:, None] * d); ok_all.append(ok)
        for px, (c, rad) in zip(self.arc_px, self.arc_geom):
            xy, ok = _ground_points(Hpw, px)
            v = xy - c
            nrm = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            ground.append(xy); target.append(c + v / nrm * rad); ok_all.append(ok)
        if not ground:
            z = np.zeros((0, 2))
            return z, z, np.zeros(0, bool)
        return np.concatenate(ground), np.concatenate(target), np.concatenate(ok_all)

    def residuals_m(self, Hpw: np.ndarray) -> np.ndarray:
        """Ground-plane distance (m) of each constraint pixel from its constraint."""
        g, t, ok = self.closest_world(Hpw)
        r = np.linalg.norm(g - t, axis=1)
        r[~ok] = BEHIND_PENALTY_M
        return np.concatenate([r, self._parallel_m(Hpw)])

    def residuals_px(self, Hpw: np.ndarray, Hwp: np.ndarray) -> np.ndarray:
        """Pixel distance between each constraint pixel and the re-projection of its closest valid world point."""
        _, t, ok = self.closest_world(Hpw)
        hom = np.c_[t, np.ones(len(t))] @ Hwp.T
        ok &= hom[:, 2] > 1e-9
        proj = hom[:, :2] / np.where(ok, hom[:, 2], 1.0)[:, None]
        n_geom = len(t)
        r = np.linalg.norm(proj - self.all_pixels()[:n_geom], axis=1)
        r[~ok] = BEHIND_PENALTY_PX
        return np.concatenate([r, self._parallel_px(Hwp)])


# --------------------------------------------------------------------------
def _residuals(v: np.ndarray, cons: Constraints, w: int, h: int) -> np.ndarray:
    pose = Pose.from_vector(v)
    Hwp = homography_world_to_pixel(pose, w, h)
    try:
        Hpw = np.linalg.inv(Hwp)
    except np.linalg.LinAlgError:
        return np.full(cons.n_residuals, BEHIND_PENALTY_PX)
    return cons.residuals_px(Hpw, Hwp)


def _starts(pitch: PitchModel, w: int) -> list[np.ndarray]:
    L, W = pitch.length, pitch.width
    xs = np.linspace(-0.4 * L, 1.4 * L, 8)
    ys = np.linspace(-0.5 * W, 1.5 * W, 8)
    yaws = np.linspace(-math.pi, math.pi, 16, endpoint=False)
    out = []
    for x, y, yaw in itertools.product(xs, ys, yaws):
        for z, pt in ((1.5, math.radians(6)), (2.5, math.radians(12)), (5.0, math.radians(22))):
            for f in (0.6 * w, 1.1 * w):
                out.append(np.array([x, y, z, yaw, pt, 0.0, f]))
    return out


def fit_pose(cons: Constraints, pitch: PitchModel, w: int, h: int, n_coarse: int = 60, n_fine: int = 6,
             init: Pose | None = None) -> tuple[Pose, np.ndarray]:
    """Multi-start LM on pixel residuals. Returns (pose, per-constraint-pixel residual in metres)."""
    lo = np.array([-3 * pitch.length, -3 * pitch.width, 0.5, -np.inf, math.radians(-10), math.radians(-45), 0.2 * w])
    hi = np.array([4 * pitch.length, 4 * pitch.width, 60.0, np.inf, math.radians(80), math.radians(45), 6.0 * w])

    def solve(s: np.ndarray, loss: str, nfev: int):
        s = np.clip(s, lo + 1e-6, hi - 1e-6)
        return least_squares(_residuals, s, args=(cons, w, h), bounds=(lo, hi), loss=loss, f_scale=3.0, max_nfev=nfev,
                             x_scale="jac")

    if init is not None:
        starts = [init.as_vector()]
    else:
        cand = _starts(pitch, w)
        costs = np.array([float(np.square(_residuals(v, cons, w, h)).sum()) for v in cand])
        starts = [cand[i] for i in np.argsort(costs)[:n_coarse]]
    coarse = []
    for s in starts:
        try:
            sol = solve(s, "soft_l1", 150)
        except ValueError:
            continue
        coarse.append((float(np.square(sol.fun).sum()), sol.x))
    if not coarse:
        raise RuntimeError("pose fit failed from every start")
    coarse.sort(key=lambda c: c[0])
    best: tuple[float, np.ndarray] | None = None
    for _, x0 in coarse[:n_fine]:
        sol = solve(x0, "linear", 2000)
        cost = float(np.square(sol.fun).sum())
        if best is None or cost < best[0]:
            best = (cost, sol.x)
    pose = Pose.from_vector(best[1])
    res_m = cons.residuals_m(homography_pixel_to_world(pose, w, h))
    return pose, res_m
