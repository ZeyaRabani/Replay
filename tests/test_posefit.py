"""Synthetic tests for pitchworld.posefit: homography round trips, sign convention, pose recovery and
generalisation to held-out ground points, and reporting of degenerate / ambiguous constraint sets.

Run with ``pytest tests/`` (conftest puts the repo root on sys.path)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _synth import H, W, pitch, project, random_pose, trace_arc, trace_line, visible_ground_points

from pitchworld.posefit import (
    Constraints,
    Pose,
    _ground_points,
    _residuals,
    fit_pose,
    fit_pose_report,
    homography_pixel_to_world,
    homography_world_to_pixel,
    rotate_pose,
    rotation,
)

NOISE_PX = 0.5


# ------------------------------------------------------------------ helpers
def _synthetic_constraints(pose: Pose, rng: np.random.Generator, noise: float, *, touchline: bool = True,
                           points: bool = True, arc_deg=(-90, 90), n_line: int = 80, n_arc: int = 30):
    """Trace goal line A, the A 'D' and (optionally) touchline S + two halfway points as seen by ``pose``."""
    p = pitch()
    L, Wd, cy = p.length, p.width, p.width / 2
    lines, arcs, pts = [], [], []
    gl = trace_line(pose, (0, 0), (0, 1), (1, Wd - 1), n_line, rng, noise)
    lines.append({"world": "goal_line_A", "pixels": gl.tolist()})
    arc = trace_arc(pose, (0, cy), p.d_radius, arc_deg, n_arc, rng, noise)
    arcs.append({"world": "A_D", "pixels": arc.tolist()})
    if touchline:
        ts = trace_line(pose, (0, 0), (1, 0), (0, L), n_line, rng, noise)
        lines.append({"world": "touch_S", "pixels": ts.tolist()})
    if points:
        pw = np.array([[L / 2, 0], [L / 2, Wd]])
        px, vis = project(pose, pw)
        px = px + rng.normal(0, noise, px.shape)
        pts = [{"world": w.tolist(), "pixel": u.tolist()} for w, u, v in zip(pw, px, vis) if v]
    return Constraints.build(p, pts, lines, arcs), len(gl), len(arc)


def _well_seen_pose(rng: np.random.Generator, noise: float = NOISE_PX, **kw) -> tuple[Pose, Constraints]:
    """A random pose from which the traced markings are actually visible (>= 10 px on line and arc)."""
    for _ in range(500):
        pose = random_pose(rng)
        cons, n_gl, n_arc = _synthetic_constraints(pose, rng, noise, **kw)
        if n_gl >= 10 and n_arc >= 10 and len(cons.pt_px) == 2 * bool(kw.get("points", True)):
            return pose, cons
    raise RuntimeError("could not sample a pose that sees the markings")


def _seen_traces(rng: np.random.Generator, noise: float, arc_deg=(-90, 90), *, with_arc: bool = True):
    """Random pose + traced goal line A / A 'D' / a line parallel to the goal line at x=20, all well seen."""
    p = pitch()
    cy = p.width / 2
    for _ in range(500):
        true = random_pose(rng)
        gl = trace_line(true, (0, 0), (0, 1), (1, p.width - 1), 30, rng, noise)
        arc = trace_arc(true, (0, cy), p.d_radius, arc_deg, 20, rng, noise)
        par = trace_line(true, (20, 0), (0, 1), (2, p.width - 2), 20, rng, noise)
        if len(gl) >= 10 and (len(arc) >= 12 or not with_arc) and len(par) >= 8:
            return true, gl, arc, par
    raise RuntimeError("could not sample a pose that sees the markings")


def _angle_diff(a: float, b: float) -> float:
    return abs(math.remainder(a - b, 2 * math.pi))


def _held_out_error(true: Pose, est: Pose, rng: np.random.Generator, n: int = 300,
                    max_range_m: float = math.inf) -> np.ndarray:
    """Ground error (m) of held-out visible pitch points within ``max_range_m`` of the camera, back-projected
    through the estimated pose."""
    held = visible_ground_points(true, rng, n)
    held = held[np.hypot(held[:, 0] - true.x, held[:, 1] - true.y) <= max_range_m]
    assert len(held) >= 20
    px, _ = project(true, held)
    g, ok = _ground_points(homography_pixel_to_world(est, W, H), px)
    assert ok.all(), "held-out visible ground points back-projected behind the camera"
    return np.linalg.norm(g - held, axis=1)


# ------------------------------------------------------------------ (a) round trip
@pytest.mark.parametrize("seed", range(20))
def test_homography_round_trip(seed):
    rng = np.random.default_rng(seed)
    pose = random_pose(rng, near_goal_a=False)
    xy = visible_ground_points(pose, rng, 50)
    Hwp = homography_world_to_pixel(pose, W, H)
    Hpw = homography_pixel_to_world(pose, W, H)
    hom = np.c_[xy, np.ones(len(xy))] @ Hwp.T
    px = hom[:, :2] / hom[:, 2:]
    back, ok = _ground_points(Hpw, px)
    assert ok.all()
    assert np.abs(back - xy).max() < 1e-6
    # and the other way round: pixel -> ground -> pixel
    hom2 = np.c_[back, np.ones(len(back))] @ Hwp.T
    assert np.abs(hom2[:, :2] / hom2[:, 2:] - px).max() < 1e-6


def test_round_trip_through_horizon_at_principal_point():
    # pitch angle 0 puts the horizon through the principal point, i.e. H_pw[2, 2] == 0
    pose = Pose(-5.0, 15.0, 2.0, 0.0, 0.0, 0.0, 1500.0)
    Hpw = homography_pixel_to_world(pose, W, H)
    assert np.isfinite(Hpw).all()
    px = np.array([[W / 2, H / 2 + 200.0]])  # below the horizon -> ground in front
    g, ok = _ground_points(Hpw, px)
    assert ok.all()
    hom = np.c_[g, np.ones(1)] @ homography_world_to_pixel(pose, W, H).T
    assert np.abs(hom[:, :2] / hom[:, 2:] - px).max() < 1e-6


# ------------------------------------------------------------------ (b) sign convention
@pytest.mark.parametrize("seed", range(20))
def test_pixel_to_world_sign_is_depth_sign(seed):
    rng = np.random.default_rng(seed)
    pose = random_pose(rng, near_goal_a=False)
    Hpw = homography_pixel_to_world(pose, W, H)
    Hwp = homography_world_to_pixel(pose, W, H)
    # ground points all over the plane (in front and behind the camera), projected through the full camera
    xy = rng.uniform(-200, 200, size=(2000, 2))
    hom = np.c_[xy, np.ones(len(xy))] @ Hwp.T
    depth = hom[:, 2]
    keep = np.abs(depth) > 1e-3
    px = hom[keep, :2] / depth[keep, None]
    third = (np.c_[px, np.ones(len(px))] @ Hpw.T)[:, 2]
    assert np.all((third > 0) == (depth[keep] > 0))
    # and every pixel that sees ground in front of the camera is strictly positive
    vis = visible_ground_points(pose, rng, 100)
    vpx, _ = project(pose, vis)
    assert np.all((np.c_[vpx, np.ones(len(vpx))] @ Hpw.T)[:, 2] > 0)
    # a ray far above the horizon hits the ground plane behind the camera -> negative
    assert (np.array([W / 2, H / 2 - 5 * pose.f, 1.0]) @ Hpw.T)[2] < 0


# ------------------------------------------------------------------ (c) recovery + generalisation
@pytest.mark.parametrize("seed", range(6))
def test_fit_recovers_pose_and_generalises(seed):
    rng = np.random.default_rng(100 + seed)
    true, cons = _well_seen_pose(rng)
    rep = fit_pose_report(cons, pitch(), W, H)
    est = rep.pose
    # the optimiser must not stop short of the true pose (global minimum reached)
    cost_true = float(np.square(_residuals(true.as_vector(), cons, W, H)).sum())
    assert rep.cost_px2 <= cost_true * 1.05 + 1e-9
    d = np.array([est.x - true.x, est.y - true.y, est.z - true.z])
    dpos = float(np.linalg.norm(d))
    dang = max(_angle_diff(est.yaw, true.yaw), abs(est.pitch - true.pitch), abs(est.roll - true.roll))
    # Moving the camera along its optical axis while zooming (f) leaves the ground map almost unchanged
    # (dolly-zoom), so with 0.5 px noise on ~30 m distant markings that direction is noise-limited
    # (a few cm per 0.1 % of f).  Position transverse to the axis is what the markings pin down.
    axis = rotation(est.yaw, est.pitch, est.roll)[2]
    transverse = float(np.linalg.norm(d - axis * (d @ axis)))
    assert transverse < 0.05, (true, est, dpos, rep.predicted_error_m)
    assert dpos < 0.25 and abs(est.f - true.f) / true.f < 0.01, (true, est, dpos)
    assert math.degrees(dang) < 0.2, (true, est)
    # held-out ground points within 25 m of the camera (where players are projected from this camera)
    err = _held_out_error(true, est, rng, max_range_m=25.0)
    assert err.mean() < 0.1, (err.mean(), err.max(), rep.predicted_error_m)
    assert err.max() < 0.5
    # further out the ground error grows as range^2 / height (a 3 m camera at 35 m: 0.1 deg -> 0.6 m);
    # the report's covariance-based prediction must stay consistent with the realised error there
    err_all = _held_out_error(true, est, rng)
    assert err_all.mean() < 3 * rep.predicted_error_p90_m + 0.02, (err_all.mean(), rep.predicted_error_p90_m)
    assert not rep.twins and not rep.alternatives
    assert rep.dof >= 9


@pytest.mark.parametrize("seed", range(3))
def test_noise_free_fit_is_exact(seed):
    rng = np.random.default_rng(200 + seed)
    true, cons = _well_seen_pose(rng, noise=0.0)
    est, res_m = fit_pose(cons, pitch(), W, H)
    assert np.sqrt(np.mean(res_m**2)) < 1e-3
    assert math.hypot(est.x - true.x, est.y - true.y, est.z - true.z) < 1e-2
    assert math.degrees(_angle_diff(est.yaw, true.yaw)) < 0.01
    assert _held_out_error(true, est, rng).max() < 1e-2


def test_fit_from_init_refines_locally():
    rng = np.random.default_rng(7)
    true, cons = _well_seen_pose(rng)
    v = true.as_vector() + np.array([0.5, -0.5, 0.2, 0.02, -0.01, 0.005, 30.0])
    est, _ = fit_pose(cons, pitch(), W, H, init=Pose.from_vector(v))
    assert math.hypot(est.x - true.x, est.y - true.y, est.z - true.z) < 0.3


# ------------------------------------------------------------------ (d) degenerate / ambiguous sets
def test_dof_rule():
    p = pitch()
    px = [[10.0 * i, 500.0 + i] for i in range(6)]
    cons = Constraints.build(p, [{"world": "centre", "pixel": [1, 2]}], [{"world": "goal_line_A", "pixels": px}],
                             [{"world": "A_D", "pixels": px[:3]}], [{"world": "touch_S", "pixels": px[:2]}])
    assert cons.dof() == 2 + 2 + 3 + 1
    # too few pixels do not count
    thin = Constraints.build(p, [], [{"world": "goal_line_A", "pixels": px[:1]}], [{"world": "A_D", "pixels": px[:2]}],
                             [{"world": "touch_S", "pixels": px[:1]}])
    assert thin.dof() == 0


def test_full_circle_arc_plus_goal_line_is_reported_ambiguous():
    """The legacy behaviour: matching the D as a full circle admits a 180-degree twin about its centre."""
    rng = np.random.default_rng(11)
    p = pitch()
    cy = p.width / 2
    true, gl, arc, par = _seen_traces(rng, 0.0)
    cons = Constraints.build(p, [], [{"world": "goal_line_A", "pixels": gl.tolist()}],
                             [{"world": {"centre": [0, cy], "radius": p.d_radius}, "pixels": arc.tolist()}],
                             [{"world": "goal_line_A", "pixels": par.tolist()}])
    assert cons.twin_centres(p) == [(0.0, cy)]
    twin = rotate_pose(true, (0.0, cy))
    c_true = float(np.square(_residuals(true.as_vector(), cons, W, H)).sum())
    c_twin = float(np.square(_residuals(twin.as_vector(), cons, W, H)).sum())
    assert c_true < 1e-6 and c_twin < 1e-6  # both poses reproduce every traced pixel exactly
    rep = fit_pose_report(cons, p, W, H)
    assert any("ambiguous" in n for n in rep.notes)
    assert len(rep.twins) == 1
    # with a rough camera position the right twin is chosen
    rep2 = fit_pose_report(cons, p, W, H, camera_hint=(true.x + 3, true.y - 3))
    assert math.hypot(rep2.pose.x - true.x, rep2.pose.y - true.y) < 0.05
    assert not any("ambiguous" in n for n in rep2.notes)


def test_semicircle_arc_breaks_the_twin():
    """Matching the D to its actual angular range removes the 180-degree ambiguity (the real-footage case:
    goal line + D + a parallel to the goal line)."""
    rng = np.random.default_rng(11)
    p = pitch()
    cy = p.width / 2
    true, gl, arc, par = _seen_traces(rng, NOISE_PX, arc_deg=(-80, 80))
    cons = Constraints.build(p, [], [{"world": "goal_line_A", "pixels": gl.tolist()}],
                             [{"world": "A_D", "pixels": arc.tolist()}], [{"world": "goal_line_A", "pixels": par.tolist()}])
    assert cons.twin_centres(p) == []
    twin = rotate_pose(true, (0.0, cy))
    assert float(np.square(_residuals(twin.as_vector(), cons, W, H)).sum()) > 1e3
    rep = fit_pose_report(cons, p, W, H)
    assert not rep.twins
    assert rep.dof == 8  # still thin: flagged, not silently trusted
    assert any("thin" in n for n in rep.notes)
    assert math.hypot(rep.pose.x - true.x, rep.pose.y - true.y) < 1.0 + 3 * rep.predicted_error_m


def test_under_determined_set_is_flagged_not_trusted():
    rng = np.random.default_rng(5)
    p = pitch()
    _true, gl, _arc, par = _seen_traces(rng, 0.0, with_arc=False)
    cons = Constraints.build(p, [], [{"world": "goal_line_A", "pixels": gl.tolist()}], [],
                             [{"world": "goal_line_A", "pixels": par.tolist()}])
    assert cons.dof() == 3
    rep = fit_pose_report(cons, p, W, H, n_coarse=10, n_fine=3)
    assert any("under-determined" in n for n in rep.notes)
    assert not rep.reliable
    assert not math.isfinite(rep.predicted_error_m) or rep.predicted_error_m > 0.5


def test_whole_pitch_symmetric_set_reports_centre_twin():
    p = pitch()
    px = [[100.0 + 10 * i, 600.0 + i] for i in range(8)]
    # centre spot + halfway line + a line parallel to the touchlines: every element maps onto itself
    cons = Constraints.build(p, [{"world": "centre", "pixel": [900, 500]}], [{"world": "half", "pixels": px}],
                             [], [{"world": "touch_N", "pixels": px}])
    assert cons.twin_centres(p) == [(p.length / 2, p.width / 2)]
    # a labelled touchline breaks it (touch_S maps onto touch_N, a different constraint)
    cons2 = Constraints.build(p, [{"world": "centre", "pixel": [900, 500]}],
                              [{"world": "touch_S", "pixels": px}, {"world": "half", "pixels": px}], [], [])
    assert cons2.twin_centres(p) == []
