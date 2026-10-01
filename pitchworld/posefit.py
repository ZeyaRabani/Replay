"""Camera pose fit from ground-plane constraints (points, traced lines, traced arcs).

A pinhole camera at height above a flat pitch is parameterised by
(x, y, z, yaw, pitch, roll, f). Its ground-plane homography is derived from
the pose, so the fit stays physically plausible even when only a few
distinct landmarks are visible (the usual case for pitch-level phone
footage: one goal line, a "D", maybe a touchline).

Constraint types (all pixel coordinates of the *calibration frame*):
  points: [{"world": "B_D_S" | [x, y], "pixel": [u, v]}]
  lines:  [{"world": "goal_line_B" | {"point": [x, y], "dir": [dx, dy]}, "pixels": [[u, v], ...]}]
  arcs:   [{"world": "B_D" | {"centre": [x, y], "radius": r, "start_deg": a0, "end_deg": a1}, "pixels": [...]}]
  parallels: [{"world": "goal_line_A", "pixels": [[u, v], ...]}]  (pixels on *some* line parallel to it)

Arcs are matched to their angular range (a "D" is a semicircle, not a full
circle): matching to the full circle admits a second, equally exact pose
rotated 180 deg about the arc centre (camera behind the goal instead of in
front of it), which silently flips player positions.

Residuals are pixel distances; the solver is a multi-start trust-region
least-squares over the pose with bounds. ``fit_pose_report`` additionally
reports the constraint dof count, remaining 180-degree twin poses and the
pose covariance propagated to ground-point error over the visible pitch, so
thin or ambiguous constraint sets are flagged instead of silently fitted.

Degrees-of-freedom rule (``Constraints.dof``): a point fixes 2, a traced
line 2 (an image line), a traced arc min(5, n_pixels) (an image conic), a
"parallel" trace 1 (a vanishing point on a known line). The pose has 7
unknowns, so dof >= 7 is required, dof >= 9 leaves room to validate the fit,
and any 180-degree rotation about a pitch point that maps every constraint
onto itself doubles the solution set regardless of the count.
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
    """Inverse ground homography, scaled so that (H @ [u, v, 1])[2] > 0 exactly for pixels whose ray
    hits the ground in front of the camera (the inverse of K[r1 r2 t] already has that sign because
    H_wp @ [x, y, 1] = depth * [u, v, 1]; a positive scale factor keeps it)."""
    H = np.linalg.inv(homography_world_to_pixel(pose, w, h))
    scale = abs(H[2, 2])
    if scale < 1e-9 * np.linalg.norm(H):  # horizon passes through the principal point
        scale = np.linalg.norm(H)
    return H / scale


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


def resolve_arc(pitch: PitchModel, spec) -> tuple[np.ndarray, float, float, float]:
    """-> (centre, radius, start angle rad, end angle rad) of a named or explicit arc (counter-clockwise)."""
    L, cy = pitch.length, pitch.width / 2
    named = {"A_D": ((0, cy), pitch.d_radius, -90.0, 90.0), "B_D": ((L, cy), pitch.d_radius, 90.0, 270.0),
             "centre_circle": ((L / 2, cy), pitch.centre_circle_radius, 0.0, 360.0)}
    if isinstance(spec, str):
        if spec not in named:
            raise KeyError(f"unknown arc {spec!r}; known: {sorted(named)} or {{'centre': [x,y], 'radius': r}}")
        c, r, a0, a1 = named[spec]
    else:
        c, r = spec["centre"], spec["radius"]
        a0, a1 = spec.get("start_deg", 0.0), spec.get("end_deg", 360.0)
    if r <= 0:
        raise ValueError(f"arc {spec!r} has radius 0 in this pitch model (set d_radius / centre_circle_radius)")
    return np.asarray(c, dtype=np.float64), float(r), math.radians(a0), math.radians(a1)


def _clamp_angle(theta: np.ndarray, a0: float, a1: float) -> np.ndarray:
    """Clamp angles onto the counter-clockwise arc [a0, a1] (nearest endpoint when outside)."""
    span = a1 - a0
    if span >= 2 * math.pi - 1e-9:
        return theta
    rel = np.mod(theta - a0, 2 * math.pi)
    inside = rel <= span
    # outside: distance to a1 going forward vs to a0 going backward
    to_end = rel - span
    to_start = 2 * math.pi - rel
    return np.where(inside, theta, np.where(to_end < to_start, a1, a0))


@dataclass
class Constraints:
    pt_px: np.ndarray  # (P,2)
    pt_world: np.ndarray  # (P,2)
    line_px: list[np.ndarray]
    line_geom: list[tuple[np.ndarray, np.ndarray]]
    arc_px: list[np.ndarray]
    arc_geom: list[tuple[np.ndarray, float, float, float]]  # centre, radius, start rad, end rad (ccw)
    par_px: list[np.ndarray] = field(default_factory=list)  # pixels on *some* line parallel to a known one
    par_dir: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        # accept legacy (centre, radius) arcs as full circles
        self.arc_geom = [(np.asarray(g[0], dtype=np.float64), float(g[1]), 0.0, 2 * math.pi) if len(g) == 2 else g
                         for g in self.arc_geom]
        self.line_geom = [(np.asarray(p0, dtype=np.float64), np.asarray(d, dtype=np.float64) / np.linalg.norm(d))
                          for p0, d in self.line_geom]
        self.par_dir = [np.asarray(d, dtype=np.float64) / np.linalg.norm(d) for d in self.par_dir]

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

    def dof(self) -> int:
        """Constraint count under the documented rule (point 2, line 2, arc min(5, n), parallel 1)."""
        return (2 * len(self.pt_px) + sum(2 for p in self.line_px if len(p) >= 2)
                + sum(min(5, len(p)) for p in self.arc_px if len(p) >= 3) + sum(1 for p in self.par_px if len(p) >= 2))

    def twin_centres(self, pitch: PitchModel) -> list[tuple[float, float]]:
        """Pitch points about which a 180-degree rotation maps every constraint onto itself.

        For such a centre the rotated pose (2cx - x, 2cy - y, yaw + pi) reproduces every constraint pixel
        exactly, so the fit cannot tell the two apart. Reflections are not physical for a pinhole
        camera, so only rotations are checked.
        """
        L, W = pitch.length, pitch.width
        out = []
        for c in ((L / 2, W / 2), (0.0, W / 2), (L, W / 2)):
            if self._invariant_under_rotation(np.asarray(c)):
                out.append((float(c[0]), float(c[1])))
        return out

    def _invariant_under_rotation(self, c: np.ndarray, tol: float = 1e-6) -> bool:
        if self.n_residuals == 0:
            return False
        if len(self.pt_world) and np.any(np.linalg.norm(self.pt_world - c, axis=1) > tol):
            return False  # a labelled point is only invariant if it *is* the centre
        for p0, d in self.line_geom:
            q = 2 * c - p0  # rotated point must lie on the same line
            off = (q - p0) - ((q - p0) @ d) * d
            if np.linalg.norm(off) > tol:
                return False
        for cen, _r, a0, a1 in self.arc_geom:
            if np.linalg.norm(cen - c) > tol or (a1 - a0) < 2 * math.pi - 1e-9:
                return False  # only a full circle centred on c is invariant
        return True  # parallels (directions) are always invariant under a half turn

    @property
    def n_residuals(self) -> int:
        return (len(self.pt_px) + sum(len(p) for p in self.line_px) + sum(len(p) for p in self.arc_px)
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
        for px, (c, rad, a0, a1) in zip(self.arc_px, self.arc_geom):
            xy, ok = _ground_points(Hpw, px)
            v = xy - c
            theta = _clamp_angle(np.arctan2(v[:, 1], v[:, 0]), a0, a1)
            ground.append(xy); target.append(c + rad * np.c_[np.cos(theta), np.sin(theta)]); ok_all.append(ok)
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


def rotate_pose(pose: Pose, centre: tuple[float, float]) -> Pose:
    """The pose rotated 180 degrees about a ground point (its 'twin' for constraint sets symmetric under it)."""
    yaw = math.remainder(pose.yaw + math.pi, 2 * math.pi)
    return Pose(2 * centre[0] - pose.x, 2 * centre[1] - pose.y, pose.z, yaw, pose.pitch, pose.roll, pose.f)


def pose_bounds(pitch: PitchModel, w: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([-3 * pitch.length, -3 * pitch.width, 0.5, -np.inf, math.radians(-10), math.radians(-45), 0.2 * w])
    hi = np.array([4 * pitch.length, 4 * pitch.width, 60.0, np.inf, math.radians(80), math.radians(45), 6.0 * w])
    return lo, hi


@dataclass
class FitReport:
    pose: Pose
    res_m: np.ndarray  # per-constraint-pixel residual on the ground (m)
    res_px: np.ndarray  # per-constraint-pixel reprojection residual (px)
    cost_px2: float
    dof: int
    twins: list[Pose]  # equally exact poses (180-degree rotations the constraints cannot distinguish)
    alternatives: list[Pose]  # other local minima found within 1% of the best cost, > 1 m away
    predicted_error_m: float  # median 1-sigma ground error over the visible pitch, from the pose covariance
    predicted_error_p90_m: float
    notes: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return not self.notes


def fit_pose(cons: Constraints, pitch: PitchModel, w: int, h: int, n_coarse: int = 60, n_fine: int = 6,
             init: Pose | None = None, camera_hint: tuple[float, float] | None = None) -> tuple[Pose, np.ndarray]:
    """Multi-start least squares on pixel residuals. Returns (pose, per-constraint-pixel residual in metres).

    ``camera_hint`` is an approximate camera (x, y) on the pitch plane used only to choose between poses the
    constraints cannot distinguish (see ``FitReport.twins``)."""
    rep = fit_pose_report(cons, pitch, w, h, n_coarse=n_coarse, n_fine=n_fine, init=init, camera_hint=camera_hint)
    return rep.pose, rep.res_m


def _robust_cost(r: np.ndarray, f_scale: float = 3.0) -> float:
    z = np.square(r / f_scale)
    return float(np.sum(2 * (np.sqrt(1 + z) - 1)))  # soft_l1


def fit_pose_report(cons: Constraints, pitch: PitchModel, w: int, h: int, n_coarse: int = 60, n_fine: int = 6,
                    init: Pose | None = None, camera_hint: tuple[float, float] | None = None,
                    noise_px: float = 0.5, n_hop: int = 8) -> FitReport:
    lo, hi = pose_bounds(pitch, w)

    def solve(s: np.ndarray, loss: str, nfev: int):
        s = np.clip(s, lo + 1e-6, hi - 1e-6)
        return least_squares(_residuals, s, args=(cons, w, h), bounds=(lo, hi), loss=loss, f_scale=3.0, max_nfev=nfev,
                             x_scale="jac")

    if init is not None:
        starts = [init.as_vector()]
    else:
        cand = _starts(pitch, w)
        # rank starts by a robust cost so a few behind-camera pixels do not hide an otherwise good basin
        costs = np.array([_robust_cost(_residuals(v, cons, w, h)) for v in cand])
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
    fine = []
    for _, x0 in coarse[:n_fine]:
        sol = solve(x0, "linear", 2000)
        fine.append((float(np.square(sol.fun).sum()), sol))
    fine.sort(key=lambda c: c[0])
    # basin hopping around the best solution: escape shallow local minima (e.g. a wrong roll/f trade-off)
    rng = np.random.default_rng(0)
    jit = np.array([1.0, 1.0, 0.3, math.radians(3), math.radians(2), math.radians(2), 0.05 * w])
    for _ in range(n_hop):
        try:
            sol = solve(fine[0][1].x + rng.normal(size=7) * jit, "linear", 500)
        except ValueError:
            continue
        fine.append((float(np.square(sol.fun).sum()), sol))
    fine.sort(key=lambda c: c[0])
    best_cost, best = fine[0]
    poses = [Pose.from_vector(best.x)]
    twin_centres = cons.twin_centres(pitch)
    twins = [rotate_pose(poses[0], c) for c in twin_centres]
    if camera_hint is not None and twins:
        hx, hy = camera_hint
        poses = sorted(poses + twins, key=lambda p: math.hypot(p.x - hx, p.y - hy))
        twins = poses[1:]
    pose = poses[0]
    alternatives = []
    for cost, sol in fine[1:]:
        p = Pose.from_vector(sol.x)
        far = all(math.hypot(p.x - q.x, p.y - q.y, p.z - q.z) > 1.0 for q in [pose] + twins + alternatives)
        if cost <= best_cost * 1.01 + 1e-6 and far:
            alternatives.append(p)

    Hpw = homography_pixel_to_world(pose, w, h)
    res_m = cons.residuals_m(Hpw)
    res_px = cons.residuals_px(Hpw, np.linalg.inv(Hpw))
    med, p90 = predicted_ground_error(pose, cons, pitch, w, h, noise_px=noise_px)

    dof = cons.dof()
    notes = []
    if dof < 7:
        notes.append(f"under-determined: {dof} constraint dof for a 7-dof pose")
    elif dof < 9:
        notes.append(f"thin constraints: {dof} dof for a 7-dof pose, the fit cannot be validated")
    if twin_centres and camera_hint is None:
        cs = ", ".join(f"({c[0]:.1f}, {c[1]:.1f})" for c in twin_centres)
        notes.append(f"ambiguous: constraints are symmetric under a 180-degree rotation about {cs}; "
                     "add an asymmetric landmark or a camera position hint")
    if alternatives:
        notes.append(f"{len(alternatives)} other pose(s) fit the constraints equally well")
    if not math.isfinite(med) or med > 0.5:
        notes.append(f"ill-conditioned: predicted ground error {med:.2f} m (median) for {noise_px:.1f} px trace noise")
    return FitReport(pose, res_m, res_px, best_cost, dof, twins, alternatives, med, p90, notes)


def predicted_ground_error(pose: Pose, cons: Constraints, pitch: PitchModel, w: int, h: int,
                           noise_px: float = 0.5) -> tuple[float, float]:
    """Propagate trace-pixel noise through the fit to ground-point error on the visible part of the pitch.

    Returns (median, 90th percentile) 1-sigma error in metres over a grid of pitch points visible in the frame,
    or (inf, inf) when the constraint Jacobian is rank deficient (a pose direction is unconstrained).
    """
    v0 = pose.as_vector()
    n = cons.n_residuals
    if n < 7:
        return math.inf, math.inf
    steps = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-2])
    J = np.empty((n, 7))
    for i in range(7):
        d = np.zeros(7); d[i] = steps[i]
        J[:, i] = (_residuals(v0 + d, cons, w, h) - _residuals(v0 - d, cons, w, h)) / (2 * steps[i])
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] <= 1e-9 * sv[0]:
        return math.inf, math.inf
    cov = noise_px**2 * np.linalg.pinv(J.T @ J)

    L, W = pitch.length, pitch.width
    gx, gy = np.meshgrid(np.linspace(0, L, 11), np.linspace(0, W, 7))
    grid = np.c_[gx.ravel(), gy.ravel()]
    Hwp = homography_world_to_pixel(pose, w, h)
    hom = np.c_[grid, np.ones(len(grid))] @ Hwp.T
    front = hom[:, 2] > 1e-9
    px = hom[:, :2] / np.where(front, hom[:, 2], 1.0)[:, None]
    vis = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    if not vis.any():
        return math.inf, math.inf
    px = px[vis]
    G = np.empty((len(px), 2, 7))
    for i in range(7):
        d = np.zeros(7); d[i] = steps[i]
        gp = _ground_points(homography_pixel_to_world(Pose.from_vector(v0 + d), w, h), px)[0]
        gm = _ground_points(homography_pixel_to_world(Pose.from_vector(v0 - d), w, h), px)[0]
        G[:, :, i] = (gp - gm) / (2 * steps[i])
    var = np.einsum("nij,jk,nlk->nil", G, cov, G)
    err = np.sqrt(np.maximum(var[:, 0, 0] + var[:, 1, 1], 0.0))
    return float(np.median(err)), float(np.percentile(err, 90))
