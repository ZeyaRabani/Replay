"""Joint multi-camera refinement using the players themselves as shared ground points.

A single pitch-level camera with a couple of traced lines/arcs has an almost exactly determined pose
(7 unknowns, ~8 constraints), so small tracing errors produce metre-scale disagreement between cameras.
The players fix that: at any synced instant the feet of a player seen from two cameras are the *same*
ground point. We alternate

  1. match detections across camera pairs (Hungarian, gated) using the current poses,
  2. re-solve all poses (+ optionally the unknown D / circle radius of the pitch model) by least squares on
       - each camera's own line/arc/parallel pixel residuals,
       - metre disagreement between matched feet,
       - a weak person-height prior (absolute scale when the pitch model's size is not known).

Everything stays physically parameterised (position, height, yaw/pitch/roll, focal), so the result is
also what Stage 2/3 need for camera anchors.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .calibrate import CameraCalibration
from .pitch import PitchModel
from .posefit import (
    Constraints,
    Pose,
    _ground_points,
    homography_pixel_to_world,
    homography_world_to_pixel,
    rotation,
)

PERSON_HEIGHT_M = 1.75


@dataclass
class JointResult:
    poses: list[Pose]
    pitch: PitchModel
    pair_median_m: dict[str, float]  # "0-1" -> median matched-feet disagreement after refinement
    n_matches: int
    notes: list[str] = field(default_factory=list)


@dataclass
class _CamDets:
    frame: np.ndarray  # (N,) frame index
    feet: np.ndarray  # (N,2) px
    height_px: np.ndarray  # (N,)
    tracks: dict[int, dict[int, int]]  # track id -> {frame: global idx}


def _collect(frames: list[list[dict]], step: int, min_conf: float, min_track_frames: int = 5) -> _CamDets:
    fr, ft, hp = [], [], []
    tracks: dict[int, dict[int, int]] = {}
    for fi in range(0, len(frames), step):
        for d in frames[fi]:
            if d["conf"] < min_conf or d.get("id") is None:
                continue
            x0, y0, x1, y1 = d["box"]
            tracks.setdefault(int(d["id"]), {})[fi] = len(fr)
            fr.append(fi)
            ft.append(((x0 + x1) / 2, y1))
            hp.append(y1 - y0)
    tracks = {k: v for k, v in tracks.items() if len(v) >= min_track_frames}
    return _CamDets(np.array(fr, dtype=int), np.array(ft, dtype=np.float64).reshape(-1, 2),
                    np.array(hp, dtype=np.float64), tracks)


def _pose_from_dict(p: dict) -> Pose:
    return Pose(p["x"], p["y"], p["height"], math.radians(p["yaw_deg"]), math.radians(p["pitch_deg"]),
                math.radians(p["roll_deg"]), p["focal_px"])


def _ground(pose: Pose, w: int, h: int, feet: np.ndarray) -> np.ndarray:
    g, ok = _ground_points(homography_pixel_to_world(pose, w, h), feet)
    g[~ok] = np.nan
    return g


def _implied_heights(pose: Pose, w: int, h: int, feet: np.ndarray, heights_px: np.ndarray) -> np.ndarray:
    """Person height (m) implied by each box under ``pose`` (nan when behind camera)."""
    g = _ground(pose, w, h, feet)
    R = rotation(pose.yaw, pose.pitch, pose.roll)
    K_inv = np.linalg.inv(np.array([[pose.f, 0, w / 2], [0, pose.f, h / 2], [0, 0, 1]]))
    heads = np.c_[feet[:, 0], feet[:, 1] - heights_px, np.ones(len(feet))]
    dirs = (R.T @ (K_inv @ heads.T)).T
    C = np.array([pose.x, pose.y, pose.z])
    dxy = dirs[:, :2]
    den = np.einsum("ij,ij->i", dxy, dxy)
    s = np.einsum("ij,ij->i", np.nan_to_num(g) - C[:2], dxy) / np.where(den > 1e-12, den, 1.0)
    z = C[2] + s * dirs[:, 2]
    bad = np.isnan(g[:, 0]) | (den <= 1e-12) | (s <= 0)
    return np.where(bad, np.nan, z)


def _match(G: list[np.ndarray], dets: list[_CamDets], gate: float,
           min_common: int = 5) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Track-level Hungarian matching between camera pairs.

    Two tracks are candidates when they overlap for >= ``min_common`` sampled frames; their cost is the mean
    ground distance over the overlap. Returns [(cam a, cam b, global idx in a, global idx in b)] over the
    common frames of every matched track pair whose mean distance is < ``gate`` metres.
    """
    n = len(G)
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            ta, tb = list(dets[a].tracks), list(dets[b].tracks)
            cost = np.full((len(ta), len(tb)), 1e6)
            pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
            for i, ida in enumerate(ta):
                fa = dets[a].tracks[ida]
                for j, idb in enumerate(tb):
                    fb = dets[b].tracks[idb]
                    common = fa.keys() & fb.keys()
                    if len(common) < min_common:
                        continue
                    ia = np.array([fa[f] for f in common])
                    ib = np.array([fb[f] for f in common])
                    d = np.linalg.norm(G[a][ia] - G[b][ib], axis=1)
                    if np.isnan(d).mean() > 0.5:
                        continue
                    cost[i, j] = np.nanmean(d)
                    pairs[(i, j)] = (ia, ib)
            if not pairs:
                continue
            r, c = linear_sum_assignment(cost)
            keep = [(i, j) for i, j in zip(r, c) if cost[i, j] < gate]
            if keep:
                out.append((a, b, np.concatenate([pairs[k][0] for k in keep]), np.concatenate([pairs[k][1] for k in keep])))
    return out


def refine_joint(cals: list[CameraCalibration], pitch: PitchModel, raw_tracks: list[list[list[dict]]],
                 frame_sizes: list[tuple[int, int]], free_radius: bool = True, frame_step: int = 3,
                 min_conf: float = 0.4, gates_m: tuple[float, ...] = (5.0, 3.0, 2.0, 1.2),
                 w_match: float = 40.0, w_height: float = 15.0, person_height_m: float = PERSON_HEIGHT_M,
                 verbose: bool = True) -> JointResult:
    """Jointly refine pose-fitted calibrations. Returns refined poses (+ pitch with fitted d_radius).

    ``w_match`` / ``w_height`` are px-per-metre weights for one residual *if* the group had as many residuals as
    the line constraints; they are rescaled so the matched-feet group carries ~9x and the height prior ~1x the
    total weight of the traced lines regardless of detection count.
    """
    n = len(cals)
    if n < 2 or any(c.pose is None for c in cals):
        raise ValueError("joint refinement needs >= 2 pose-fitted cameras")
    poses = [_pose_from_dict(c.pose) for c in cals]
    dets = [_collect(t, frame_step, min_conf) for t in raw_tracks]
    notes: list[str] = []

    def build_cons(p: PitchModel) -> list[Constraints]:
        return [Constraints.build(p, c.points, c.lines, c.arcs, c.parallels) for c in cals]

    def unpack(v: np.ndarray) -> tuple[list[Pose], PitchModel]:
        ps = [Pose.from_vector(v[7 * i:7 * i + 7]) for i in range(n)]
        p = dataclasses.replace(pitch, d_radius=float(v[7 * n])) if free_radius else pitch
        return ps, p

    def ground_all(ps: list[Pose]) -> list[np.ndarray]:
        return [_ground(ps[c], *frame_sizes[c], dets[c].feet) for c in range(n)]

    cons_cache: dict[float, list[Constraints]] = {}
    n_line = sum(len(c.residuals_px(np.eye(3), np.eye(3))) for c in build_cons(pitch))
    n_height = sum(len(d.feet) for d in dets)
    w_h = 3.0 * w_height * math.sqrt(n_line / max(n_height, 1))

    def soft(r: np.ndarray, c: float) -> np.ndarray:
        """Per-residual soft-L1: linear near 0, ~sqrt beyond ``c`` (robust to wrong matches)."""
        return np.sign(r) * c * np.sqrt(2.0 * (np.sqrt(1.0 + (r / c) ** 2) - 1.0))

    def residuals(v: np.ndarray, matches) -> np.ndarray:
        n_match = sum(len(m[2]) for m in matches)
        w_m = 3.0 * w_match * math.sqrt(n_line / max(n_match, 1))
        ps, p = unpack(v)
        if p.d_radius not in cons_cache:
            cons_cache.clear()
            cons_cache[p.d_radius] = build_cons(p)
        cons = cons_cache[p.d_radius]
        parts = []
        for c in range(n):
            w, h = frame_sizes[c]
            Hwp = homography_world_to_pixel(ps[c], w, h)
            Hpw = np.linalg.inv(Hwp)
            parts.append(cons[c].residuals_px(Hpw, Hwp))
        G = ground_all(ps)
        for a, b, ia, ib in matches:
            d = G[a][ia] - G[b][ib]
            parts.append(soft(w_m * np.nan_to_num(d, nan=50.0).ravel(), w_m * 1.0))
        for c in range(n):
            z = _implied_heights(ps[c], *frame_sizes[c], dets[c].feet, dets[c].height_px)
            parts.append(soft(w_h * (np.nan_to_num(z, nan=5 * person_height_m) - person_height_m), w_h * 0.5))
        return np.concatenate(parts)

    v = np.concatenate([p.as_vector() for p in poses] + ([np.array([pitch.d_radius])] if free_radius else []))
    lo = np.concatenate([np.array([-1e3, -1e3, 1.0, -np.inf, math.radians(-5), math.radians(-45), 0.4 * w])
                         for (w, _) in frame_sizes] + ([np.array([0.25 * pitch.d_radius])] if free_radius else []))
    hi = np.concatenate([np.array([1e3, 1e3, 30.0, np.inf, math.radians(60), math.radians(45), 3.0 * w])
                         for (w, _) in frame_sizes] + ([np.array([4.0 * pitch.d_radius])] if free_radius else []))
    v = np.clip(v, lo + 1e-6, hi - 1e-6)
    matches: list = []
    for gate in gates_m:
        ps, p = unpack(v)
        matches = _match(ground_all(ps), dets, gate)
        n_m = sum(len(m[2]) for m in matches)
        if n_m < 20:
            notes.append(f"joint refinement: only {n_m} cross-camera matches within {gate} m; stopping")
            break
        sol = least_squares(residuals, v, args=(matches,), bounds=(lo, hi), x_scale="jac", max_nfev=200)
        v = sol.x
        ps, p = unpack(v)
        if verbose:
            print(f"    gate {gate:.1f} m: {n_m} matches, cost {sol.cost:.0f}, nfev {sol.nfev}, "
                  f"D radius {p.d_radius:.2f}, heights {[round(q.z, 2) for q in ps]}", flush=True)

    ps, p = unpack(v)
    G = ground_all(ps)
    pair_med: dict[str, float] = {}
    for a, b, ia, ib in _match(G, dets, 3.0):
        d = np.linalg.norm(G[a][ia] - G[b][ib], axis=1)
        pair_med[f"{a}-{b}"] = float(np.nanmedian(d)) if len(d) else float("nan")
    n_m = sum(len(m[2]) for m in matches)
    return JointResult(ps, p, pair_med, n_m, notes)


def cross_camera_consistency(cals: list[CameraCalibration], raw_tracks: list[list[list[dict]]],
                             frame_step: int = 3, min_conf: float = 0.4, gate_m: float = 3.0) -> dict[str, float]:
    """Median ground disagreement (m) of track-matched players for every camera pair, under the given
    calibrations. Works for any calibration (homography-only included); nan when a pair shares no tracks.
    A well-calibrated rig gives ~0.3-0.6 m (feet/box noise); metres mean the cameras do not share a frame."""
    dets = [_collect(t, frame_step, min_conf) for t in raw_tracks]
    G = []
    for cal, d in zip(cals, dets):
        H = np.asarray(cal.H, dtype=np.float64)
        if len(d.feet) and np.median(np.c_[d.feet, np.ones(len(d.feet))] @ H[2]) < 0:
            H = -H  # 4-point homographies have arbitrary sign; players' feet are in front of the camera
        g, ok = _ground_points(H, d.feet)
        g[~ok] = np.nan
        G.append(g)
    out: dict[str, float] = {}
    for a in range(len(cals)):
        for b in range(a + 1, len(cals)):
            out[f"{a}-{b}"] = float("nan")
    for a, b, ia, ib in _match(G, dets, gate_m):
        d = np.linalg.norm(G[a][ia] - G[b][ib], axis=1)
        out[f"{a}-{b}"] = float(np.nanmedian(d)) if len(d) else float("nan")
    return out


def apply_pose(cal: CameraCalibration, pose: Pose, frame_size: tuple[int, int], pitch: PitchModel,
               pair_median_m: float) -> CameraCalibration:
    """Return a copy of ``cal`` with the refined pose / homography and a consistency-based confidence."""
    w, h = frame_size
    H = homography_pixel_to_world(pose, w, h)
    cons = Constraints.build(pitch, cal.points, cal.lines, cal.arcs, cal.parallels)
    res_px = cons.residuals_px(H, np.linalg.inv(H))
    res_m = cons.residuals_m(H)
    conf = float(np.clip(1.0 - pair_median_m / 2.0, 0.0, 1.0)) * (0.9 if pose.z < 1.0 or pose.z > 25 else 1.0)
    notes = [n for n in cal.notes if not n.startswith("constraints are thin")]
    notes.append(f"joint multi-camera refinement: median cross-camera player disagreement {pair_median_m:.2f} m")
    if pair_median_m > 1.0:
        notes.append("cameras still disagree by > 1 m on player positions: calibration is LOW confidence")
    return dataclasses.replace(cal, H=H.tolist(), method=cal.method + "+joint",
                               reproj_error_m=float(np.sqrt(np.mean(np.square(res_m)))),
                               reproj_error_px=float(np.sqrt(np.mean(np.square(res_px)))),
                               confidence=conf, notes=notes, pose=pose.to_dict())
