"""Joint multi-camera refinement using the players themselves as shared ground points.

A single pitch-level camera with a couple of traced lines/arcs has an almost exactly determined pose
(7 unknowns, ~8 constraints), so small tracing errors produce metre-scale disagreement between cameras.
The players fix that: at any synced instant the feet of a player seen from two cameras are the *same*
ground point. We alternate

  1. match tracks across camera pairs (Hungarian, gated) using the current poses, reject gross outliers,
  2. re-solve all poses (+ optionally the unknown D / circle radius and a global pitch scale) by least squares on
       - each camera's own line/arc/parallel pixel residuals,
       - metre disagreement between matched feet (robust loss),
       - a robust per-track person-height prior (absolute scale when the pitch model's size is not known),
       - weak priors that keep every pose near its single-camera fit (no 27 m cameras, no 3x focals).

Everything stays physically parameterised (position, height, yaw/pitch/roll, focal), so the result is
also what Stage 2/3 need for camera anchors. ``JointResult.identifiability`` reports which parameters the data
actually constrain (from the Jacobian at the solution) so a degenerate rig is flagged instead of trusted.
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
PARAM_NAMES = ("x", "y", "height", "yaw", "pitch", "roll", "focal")

# Prior sigmas around the single-camera pose fit (weak; only there to forbid physically absurd solutions).
PRIOR_SIGMA = {"x": 4.0, "y": 4.0, "height": 1.5, "yaw": math.radians(8), "pitch": math.radians(6),
               "roll": math.radians(4), "focal_rel": 0.25}
# A parameter counts as "unconstrained" when its 1-sigma uncertainty from the data-only Jacobian exceeds this.
IDENT_TOL = {"x": 2.0, "y": 2.0, "height": 1.0, "yaw": math.radians(3), "pitch": math.radians(3),
             "roll": math.radians(3), "focal": 0.15, "d_radius": 0.5, "scale": 0.05}


@dataclass
class JointResult:
    poses: list[Pose]
    pitch: PitchModel
    pair_median_m: dict[str, float]  # "0-1" -> median matched-feet disagreement after refinement
    n_matches: int
    notes: list[str] = field(default_factory=list)
    pair_frac_within_1m: dict[str, float] = field(default_factory=dict)
    scale: float = 1.0  # global pitch scale factor (1.0 unless free_scale)
    identifiability: dict[str, float] = field(default_factory=dict)  # param name -> 1-sigma from data only
    unconstrained: list[str] = field(default_factory=list)  # param names whose sigma exceeds IDENT_TOL
    n_rejected_tracks: int = 0
    n_rejected_dets: int = 0
    person_height_m: float = PERSON_HEIGHT_M  # robust median implied height after refinement
    accepted: bool = True  # False -> the fit was physically implausible and the *input* poses/pitch are returned


class Consistency(dict):
    """``dict`` of pair -> median disagreement (m), plus ``frac_within_1m`` (pair -> fraction of matched detections
    that agree to < 1 m) and ``overall_frac_within_1m``. Behaves as a plain dict for existing callers."""

    frac_within_1m: dict[str, float]
    overall_frac_within_1m: float

    def __init__(self, med: dict[str, float], frac: dict[str, float], overall: float):
        super().__init__(med)
        self.frac_within_1m = frac
        self.overall_frac_within_1m = overall


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


def _track_heights(pose: Pose, w: int, h: int, d: _CamDets) -> np.ndarray:
    """Median implied person height per track (nan when the track is mostly behind the camera)."""
    z = _implied_heights(pose, w, h, d.feet, d.height_px)
    out = []
    for idx in d.tracks.values():
        zz = z[np.fromiter(idx.values(), dtype=int)]
        out.append(np.nanmedian(zz) if np.isfinite(zz).sum() >= max(3, len(zz) // 2) else np.nan)
    return np.array(out, dtype=np.float64)


@dataclass
class _TrackMatch:
    a: int
    b: int
    ida: int
    idb: int
    ia: np.ndarray
    ib: np.ndarray


def _match_tracks(G: list[np.ndarray], dets: list[_CamDets], gate: float, min_common: int = 5) -> list[_TrackMatch]:
    """Track-level Hungarian matching between camera pairs.

    Two tracks are candidates when they overlap for >= ``min_common`` sampled frames; their cost is the mean
    ground distance over the overlap. Returns one ``_TrackMatch`` (with the global indices over the common frames)
    for every assigned track pair whose mean distance is < ``gate`` metres.
    """
    out: list[_TrackMatch] = []
    for a in range(len(G)):
        for b in range(a + 1, len(G)):
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
            for i, j in zip(r, c):
                if cost[i, j] < gate:
                    out.append(_TrackMatch(a, b, ta[i], tb[j], *pairs[(i, j)]))
    return out


def _match(G: list[np.ndarray], dets: list[_CamDets], gate: float,
           min_common: int = 5) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Per-camera-pair view of :func:`_match_tracks`: [(cam a, cam b, global idx in a, global idx in b)]."""
    return _group(_match_tracks(G, dets, gate, min_common))


def _group(tms: list[_TrackMatch]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[int, int], list[_TrackMatch]] = {}
    for m in tms:
        grouped.setdefault((m.a, m.b), []).append(m)
    return [(a, b, np.concatenate([m.ia for m in ms]), np.concatenate([m.ib for m in ms]))
            for (a, b), ms in grouped.items()]


def _reject_outliers(G: list[np.ndarray], tms: list[_TrackMatch], det_factor: float = 4.0,
                     track_factor: float = 3.0) -> tuple[list[_TrackMatch], int, int]:
    """Drop track pairs whose median disagreement is a gross outlier w.r.t. all pairs (wrong identity), and
    individual detections far off their own track pair's median (id switches, box glitches).

    Thresholds are MAD based (median + k * 1.4826 * MAD) with a floor so a perfectly consistent rig is not trimmed.
    Returns (kept matches, n rejected tracks, n rejected detections)."""
    if not tms:
        return tms, 0, 0
    per = []
    for m in tms:
        d = np.linalg.norm(G[m.a][m.ia] - G[m.b][m.ib], axis=1)
        per.append((d, float(np.nanmedian(d)) if np.isfinite(d).any() else np.inf))
    meds = np.array([p[1] for p in per])
    finite = meds[np.isfinite(meds)]
    if not len(finite):
        return [], len(tms), 0
    centre = float(np.median(finite))
    mad = 1.4826 * float(np.median(np.abs(finite - centre)))
    thr_track = centre + track_factor * max(mad, 0.25)
    kept: list[_TrackMatch] = []
    n_tr = n_det = 0
    for m, (d, med) in zip(tms, per):
        if not np.isfinite(med) or med > thr_track:
            n_tr += 1
            continue
        dev = np.abs(np.nan_to_num(d, nan=np.inf) - med)
        mad_d = 1.4826 * float(np.nanmedian(dev[np.isfinite(dev)])) if np.isfinite(dev).any() else 0.0
        ok = dev <= det_factor * max(mad_d, 0.2)
        n_det += int((~ok).sum())
        if ok.sum() >= 3:
            kept.append(_TrackMatch(m.a, m.b, m.ida, m.idb, m.ia[ok], m.ib[ok]))
        else:
            n_tr += 1
    return kept, n_tr, n_det


def _scaled(pitch: PitchModel, s: float) -> PitchModel:
    """All linear pitch dimensions multiplied by ``s`` (the 'shared-detection scale')."""
    if s == 1.0:
        return pitch
    return dataclasses.replace(pitch, **{k: getattr(pitch, k) * s for k in (
        "length", "width", "goal_width", "d_radius", "penalty_depth", "penalty_width", "goal_area_depth",
        "goal_area_width", "centre_circle_radius")})


def _param_names(n: int, free_radius: bool, free_scale: bool) -> list[str]:
    names = [f"cam{c}.{p}" for c in range(n) for p in PARAM_NAMES]
    if free_radius:
        names.append("pitch.d_radius")
    if free_scale:
        names.append("pitch.scale")
    return names


def identifiability(jac: np.ndarray, v: np.ndarray, names: list[str], resid: np.ndarray | None = None,
                    rel_tol: float = 1e-8) -> tuple[dict[str, float], list[str]]:
    """1-sigma uncertainty of every parameter from a (data-only) Jacobian, and the names of the parameters the data
    does not constrain (null-space directions or sigma beyond ``IDENT_TOL``).

    Covariance is s2 * (J^T J)^+ with s2 the residual variance (unit when ``resid`` is not given). Focal and scale
    are reported relative to their values."""
    s2 = 1.0
    if resid is not None and len(resid) > len(v):
        s2 = float(resid @ resid) / (len(resid) - len(v))
    col = np.linalg.norm(jac, axis=0)
    zero = col < 1e-12
    D = np.where(zero, 1.0, col)
    JtJ = (jac / D).T @ (jac / D) / s2  # unit-norm columns: null test independent of parameter units
    s, U = np.linalg.eigh(JtJ)
    smax = float(s.max()) if len(s) else 1.0
    null = (s < rel_tol * smax) | (s <= 0)
    sig: dict[str, float] = {}
    inv = np.where(null, 0.0, 1.0 / np.where(null, 1.0, s))
    cov_diag = (U ** 2 * inv).sum(1) / D ** 2
    cov_diag[zero] = np.inf
    for i, nm in enumerate(names):
        sd = math.sqrt(max(cov_diag[i], 0.0))
        if nm.endswith(".focal"):
            sd /= max(abs(v[i]), 1.0)
        if nm.endswith(".scale"):
            sd /= max(abs(v[i]), 1e-6)
        sig[nm] = float(sd)
    bad: list[str] = []
    for k in np.flatnonzero(null):
        vec = np.abs(U[:, k])
        bad += [names[i] for i in np.flatnonzero(vec > 0.3 * vec.max())]
    for nm, sd in sig.items():
        key = nm.split(".")[-1]
        if sd > IDENT_TOL.get(key, np.inf):
            bad.append(nm)
    return sig, sorted(set(bad), key=names.index)


def refine_joint(cals: list[CameraCalibration], pitch: PitchModel, raw_tracks: list[list[list[dict]]],
                 frame_sizes: list[tuple[int, int]], free_radius: bool = True, frame_step: int = 3,
                 min_conf: float = 0.4, gates_m: tuple[float, ...] = (5.0, 3.0, 2.0, 1.2),
                 w_match: float = 40.0, w_height: float = 15.0, person_height_m: float = PERSON_HEIGHT_M,
                 free_scale: bool = False, w_prior: float = 1.0, match_soft_px: float = 15.0,
                 height_soft_m: float = 0.25, verbose: bool = True) -> JointResult:
    """Jointly refine pose-fitted calibrations. Returns refined poses (+ pitch with fitted d_radius / scale).

    Matched feet are scored as *pixel* re-projection of one shared ground point into both cameras (soft-L1 beyond
    ``match_soft_px``), so the group is scale-free; it is weighted to carry ~9x the total weight of the traced
    lines regardless of detection count (``w_match`` is kept for API compatibility). The person-height prior is
    one residual per *track* (median implied height, soft-L1 beyond ``height_soft_m``, so kids / crouching players
    have bounded pull) weighted ~``w_height`` px/m and ~1x the lines in total. ``w_prior`` scales the pose priors
    around the single-camera fit (sigmas in ``PRIOR_SIGMA``; total weight ~1x the traced lines).

    ``free_scale`` additionally estimates one global scale factor for *all* pitch dimensions from the person-height
    prior ("shared-detection scale"); only meaningful when the pitch size is guessed.
    """
    n = len(cals)
    if n < 2 or any(c.pose is None for c in cals):
        raise ValueError("joint refinement needs >= 2 pose-fitted cameras")
    poses = [_pose_from_dict(c.pose) for c in cals]
    dets = [_collect(t, frame_step, min_conf) for t in raw_tracks]
    notes: list[str] = []
    names = _param_names(n, free_radius, free_scale)

    def build_cons(p: PitchModel) -> list[Constraints]:
        return [Constraints.build(p, c.points, c.lines, c.arcs, c.parallels) for c in cals]

    def unpack(v: np.ndarray) -> tuple[list[Pose], PitchModel]:
        ps = [Pose.from_vector(v[7 * i:7 * i + 7]) for i in range(n)]
        p = pitch
        k = 7 * n
        if free_radius:
            p = dataclasses.replace(p, d_radius=float(v[k]))
            k += 1
        if free_scale:
            p = _scaled(p, float(v[k]))
        return ps, p

    def ground_all(ps: list[Pose]) -> list[np.ndarray]:
        return [_ground(ps[c], *frame_sizes[c], dets[c].feet) for c in range(n)]

    cons_cache: dict[tuple[float, float], list[Constraints]] = {}
    n_line = sum(len(c.residuals_px(np.eye(3), np.eye(3))) for c in build_cons(pitch))
    n_height = sum(len(d.tracks) for d in dets)
    w_h = 3.0 * w_height * math.sqrt(n_line / max(n_height, 1))
    w_p = w_prior * math.sqrt(n_line / (7 * n))
    v0 = np.concatenate([p.as_vector() for p in poses])
    prior_sig = np.concatenate([np.array([PRIOR_SIGMA["x"], PRIOR_SIGMA["y"], PRIOR_SIGMA["height"], PRIOR_SIGMA["yaw"],
                                          PRIOR_SIGMA["pitch"], PRIOR_SIGMA["roll"], PRIOR_SIGMA["focal_rel"] * p.f])
                                for p in poses])

    def soft(r: np.ndarray, c: float) -> np.ndarray:
        """Per-residual soft-L1: linear near 0, ~sqrt beyond ``c`` (robust to wrong matches)."""
        return np.sign(r) * c * np.sqrt(2.0 * (np.sqrt(1.0 + (r / c) ** 2) - 1.0))

    def residuals(v: np.ndarray, matches: list[_TrackMatch], with_prior: bool = True) -> np.ndarray:
        n_match = sum(len(m.ia) for m in matches)
        w_m = 3.0 * math.sqrt(n_line / max(n_match, 1))
        ps, p = unpack(v)
        key = (p.d_radius, p.length)
        if key not in cons_cache:
            cons_cache.clear()
            cons_cache[key] = build_cons(p)
        cons = cons_cache[key]
        parts = []
        for c in range(n):
            w, h = frame_sizes[c]
            Hwp = homography_world_to_pixel(ps[c], w, h)
            Hpw = np.linalg.inv(Hwp)
            parts.append(cons[c].residuals_px(Hpw, Hwp))
        G = ground_all(ps)
        Hwps = [homography_world_to_pixel(ps[c], *frame_sizes[c]) for c in range(n)]
        for m in matches:
            # bundle-adjustment style: both feet observations must re-project from one shared ground point.
            # Measured in pixels so that shrinking the scene cannot artificially shrink the disagreement.
            mid = 0.5 * (np.nan_to_num(G[m.a][m.ia], nan=1e3) + np.nan_to_num(G[m.b][m.ib], nan=1e3))
            hom = np.c_[mid, np.ones(len(mid))]
            for cam, idx in ((m.a, m.ia), (m.b, m.ib)):
                q = hom @ Hwps[cam].T
                ok = q[:, 2] > 1e-9
                uv = q[:, :2] / np.where(ok, q[:, 2], 1.0)[:, None]
                r = np.where(ok[:, None], uv - dets[cam].feet[idx], 500.0)
                parts.append(soft(w_m * r.ravel(), w_m * match_soft_px))
        for c in range(n):
            z = _track_heights(ps[c], *frame_sizes[c], dets[c])
            r = w_h * (np.nan_to_num(z, nan=5 * person_height_m) - person_height_m)
            # asymmetric: kids / crouching / sitting only ever make people *shorter*, so short tracks are trusted less
            parts.append(np.where(r < 0, soft(r, w_h * height_soft_m), soft(r, 2.5 * w_h * height_soft_m)))
        if with_prior:
            parts.append(w_p * (v[:7 * n] - v0) / prior_sig)
        return np.concatenate(parts)

    v = v0.copy()
    L, Wd = pitch.length, pitch.width
    lo_c = [np.array([-L, -Wd, 1.0, -np.inf, math.radians(-5), math.radians(-30), 0.5 * w]) for (w, _) in frame_sizes]
    hi_c = [np.array([2 * L, 2 * Wd, 15.0, np.inf, math.radians(60), math.radians(30), 2.5 * w]) for (w, _) in frame_sizes]
    lo, hi = np.concatenate(lo_c), np.concatenate(hi_c)
    if free_radius:
        v = np.append(v, pitch.d_radius)
        lo, hi = np.append(lo, 0.65 * pitch.d_radius), np.append(hi, 1.5 * pitch.d_radius)
    if free_scale:
        v = np.append(v, 1.0)
        lo, hi = np.append(lo, 0.7), np.append(hi, 1.4)
    v = np.clip(v, lo + 1e-6, hi - 1e-6)
    matches: list[_TrackMatch] = []
    n_rej_tr = n_rej_det = 0
    sol = None
    for gate in gates_m:
        ps, p = unpack(v)
        G = ground_all(ps)
        matches, r_tr, r_det = _reject_outliers(G, _match_tracks(G, dets, gate))
        n_rej_tr, n_rej_det = n_rej_tr + r_tr, n_rej_det + r_det
        n_m = sum(len(m.ia) for m in matches)
        if n_m < 20:
            notes.append(f"joint refinement: only {n_m} cross-camera matches within {gate} m; stopping")
            break
        sol = least_squares(residuals, v, args=(matches,), bounds=(lo, hi), x_scale="jac", max_nfev=200)
        v = sol.x
        ps, p = unpack(v)
        if verbose:
            print(f"    gate {gate:.1f} m: {n_m} matches ({r_tr} tracks / {r_det} dets rejected), cost {sol.cost:.0f}, "
                  f"nfev {sol.nfev}, D radius {p.d_radius:.2f}, scale {v[-1] if free_scale else 1.0:.3f}, "
                  f"heights {[round(q.z, 2) for q in ps]}", flush=True)

    ps, p = unpack(v)
    G = ground_all(ps)
    pair_med: dict[str, float] = {}
    pair_frac: dict[str, float] = {}
    final, _, _ = _reject_outliers(G, _match_tracks(G, dets, 3.0))
    for a, b, ia, ib in _group(final):
        d = np.linalg.norm(G[a][ia] - G[b][ib], axis=1)
        pair_med[f"{a}-{b}"] = float(np.nanmedian(d)) if len(d) else float("nan")
        pair_frac[f"{a}-{b}"] = float(np.nanmean(d < 1.0)) if len(d) else float("nan")
    n_m = sum(len(m.ia) for m in matches)

    sig: dict[str, float] = {}
    bad: list[str] = []
    if sol is not None:
        eps = np.maximum(1e-6 * np.abs(v), 1e-7)
        f0 = residuals(v, matches, with_prior=False)
        J = np.empty((len(f0), len(v)))
        for i in range(len(v)):
            dv = np.zeros_like(v)
            dv[i] = eps[i]
            J[:, i] = (residuals(v + dv, matches, with_prior=False) - f0) / eps[i]
        sig, bad = identifiability(J, v, names, f0)
        if bad:
            notes.append("joint refinement: parameters not constrained by lines+players (held by priors): "
                         + ", ".join(bad))
    zs = np.concatenate([_track_heights(ps[c], *frame_sizes[c], dets[c]) for c in range(n)])
    z_med = float(np.nanmedian(zs)) if np.isfinite(zs).any() else float("nan")
    reject: list[str] = []
    for c, q in enumerate(ps):
        if q.z < 1.2 or q.z > 12.0 or not (0.45 * frame_sizes[c][0] <= q.f <= 2.2 * frame_sizes[c][0]):
            reject.append(f"camera {c} pose implausible (height {q.z:.1f} m, focal {q.f:.0f} px)")
        moved = math.hypot(q.x - poses[c].x, q.y - poses[c].y)
        if moved > 3 * PRIOR_SIGMA["x"] and {f"cam{c}.x", f"cam{c}.y"} & set(bad):
            reject.append(f"camera {c} moved {moved:.1f} m from its line fit while its position is unconstrained")
    if free_radius and not (0.7 * pitch.d_radius <= p.d_radius <= 1.4 * pitch.d_radius):
        reject.append(f"fitted D radius {p.d_radius:.2f} m far from prior {pitch.d_radius:.2f} m")
    if math.isfinite(z_med) and not (0.8 * person_height_m <= z_med <= 1.2 * person_height_m):
        reject.append(f"implied player height {z_med:.2f} m (prior {person_height_m:.2f} m): scale not recovered")
    if n_rej_tr:
        notes.append(f"joint refinement: rejected {n_rej_tr} track pairs / {n_rej_det} detections as gross outliers")
    accepted = not reject
    if not accepted:
        # a degenerate joint solution is worse than none: keep the single-camera fits and say why
        notes.append("joint refinement REJECTED (input poses kept): " + "; ".join(reject))
        ps, p = poses, pitch
        G = ground_all(ps)
        pair_med, pair_frac = {}, {}
        for a, b, ia, ib in _group(_reject_outliers(G, _match_tracks(G, dets, 3.0))[0]):
            d = np.linalg.norm(G[a][ia] - G[b][ib], axis=1)
            pair_med[f"{a}-{b}"] = float(np.nanmedian(d)) if len(d) else float("nan")
            pair_frac[f"{a}-{b}"] = float(np.nanmean(d < 1.0)) if len(d) else float("nan")
    return JointResult(ps, p, pair_med, n_m, notes, pair_frac, float(v[-1]) if free_scale and accepted else 1.0,
                       sig, bad, n_rej_tr, n_rej_det, z_med, accepted)


def cross_camera_consistency(cals: list[CameraCalibration], raw_tracks: list[list[list[dict]]],
                             frame_step: int = 3, min_conf: float = 0.4, gate_m: float = 3.0) -> Consistency:
    """Median ground disagreement (m) of track-matched players for every camera pair, under the given
    calibrations. Works for any calibration (homography-only included); nan when a pair shares no tracks.
    A well-calibrated rig gives ~0.3-0.6 m (feet/box noise); metres mean the cameras do not share a frame.

    The returned dict also carries ``frac_within_1m`` (per pair, fraction of matched detections closer than 1 m)
    and ``overall_frac_within_1m``."""
    dets = [_collect(t, frame_step, min_conf) for t in raw_tracks]
    G = []
    for cal, d in zip(cals, dets):
        H = np.asarray(cal.H, dtype=np.float64)
        if len(d.feet) and np.median(np.c_[d.feet, np.ones(len(d.feet))] @ H[2]) < 0:
            H = -H  # 4-point homographies have arbitrary sign; players' feet are in front of the camera
        g, ok = _ground_points(H, d.feet)
        g[~ok] = np.nan
        G.append(g)
    med: dict[str, float] = {}
    frac: dict[str, float] = {}
    for a in range(len(cals)):
        for b in range(a + 1, len(cals)):
            med[f"{a}-{b}"] = frac[f"{a}-{b}"] = float("nan")
    all_d = []
    for a, b, ia, ib in _match(G, dets, gate_m):
        d = np.linalg.norm(G[a][ia] - G[b][ib], axis=1)
        d = d[np.isfinite(d)]
        med[f"{a}-{b}"] = float(np.median(d)) if len(d) else float("nan")
        frac[f"{a}-{b}"] = float(np.mean(d < 1.0)) if len(d) else float("nan")
        all_d.append(d)
    overall = float(np.mean(np.concatenate(all_d) < 1.0)) if all_d and sum(map(len, all_d)) else float("nan")
    return Consistency(med, frac, overall)


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
