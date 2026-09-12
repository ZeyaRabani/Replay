"""Synthetic multi-camera validation of pitchworld.jointfit.

Three phone-like cameras look at the goal-A end of a 50x30 small-sided pitch; ~15 players random-walk on the ground
with YOLO-like box noise, missing frames, false tracks and ByteTrack-style id switches. The single-camera starts are
deliberately wrong (yaw +-5 deg, position +-2 m, D radius x1.3, then re-fitted to the traced lines under the wrong
radius, exactly like a real over-fitted 7-dof pose) and refine_joint has to recover the truth.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pytest

from pitchworld.calibrate import CameraCalibration
from pitchworld.jointfit import (
    IDENT_TOL,
    Consistency,
    _collect,
    _match_tracks,
    _pose_from_dict,
    _reject_outliers,
    _TrackMatch,
    cross_camera_consistency,
    identifiability,
    refine_joint,
)
from pitchworld.pitch import PitchModel
from pitchworld.posefit import Constraints, Pose, fit_pose, homography_pixel_to_world, rotation

W, H = 1920, 1080
TRUE_PITCH = PitchModel.small_sided(length=50, width=30, goal_width=3.66, d_radius=6.0)


def true_poses() -> list[Pose]:
    def look(x, y, z, tx, ty, f=0.95 * W):
        yaw = math.atan2(ty - y, tx - x)
        pitch = math.atan2(z, math.hypot(tx - x, ty - y)) * 0.9
        return Pose(x, y, z, yaw, pitch, math.radians(0.5), f)
    return [look(-8.0, 15.0, 3.5, 8.0, 15.0), look(12.0, -4.0, 3.2, 4.0, 14.0), look(11.0, 34.0, 3.4, 5.0, 16.0)]


def project(pose: Pose, X: np.ndarray) -> np.ndarray:
    """(N,3) world -> (N,2) pixels (nan behind the camera)."""
    R = rotation(pose.yaw, pose.pitch, pose.roll)
    C = np.array([pose.x, pose.y, pose.z])
    K = np.array([[pose.f, 0, W / 2], [0, pose.f, H / 2], [0, 0, 1]])
    p = (K @ (R @ (X - C).T)).T
    ok = p[:, 2] > 0.1
    uv = p[:, :2] / np.where(ok, p[:, 2], 1.0)[:, None]
    uv[~ok] = np.nan
    return uv


@dataclass
class Scenario:
    pitch: PitchModel
    poses: list[Pose]
    cals: list[CameraCalibration]  # corrupted single-camera fits (the input to refine_joint)
    raw_tracks: list[list[list[dict]]]
    sizes: list[tuple[int, int]]
    player_xy: np.ndarray  # (F, P, 2) ground truth


def simulate_players(rng: np.random.Generator, n_players: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.empty((n_frames, n_players, 2))
    xy[0] = np.c_[rng.uniform(1, 24, n_players), rng.uniform(3, 27, n_players)]
    vel = rng.normal(0, 0.08, (n_players, 2))
    for f in range(1, n_frames):
        vel = 0.9 * vel + rng.normal(0, 0.03, (n_players, 2))
        xy[f] = np.clip(xy[f - 1] + vel, [0.5, 1.0], [26.0, 29.0])
    heights = rng.normal(1.75, 0.07, n_players)
    heights[:2] = 1.35  # two kids
    return xy, heights


def detections(rng: np.random.Generator, pose: Pose, xy: np.ndarray, heights: np.ndarray, px_noise: float = 5.0,
               p_miss: float = 0.1, n_false: int = 2, id_switch_frame: int | None = None,
               crouch_player: int | None = None) -> list[list[dict]]:
    F, P, _ = xy.shape
    frames: list[list[dict]] = [[] for _ in range(F)]
    false_pos = rng.uniform([100, 300], [W - 100, H - 50], (n_false, 2))
    for f in range(F):
        h_eff = heights.copy()
        if crouch_player is not None and (f // 40) % 2 == 1:
            h_eff[crouch_player] *= 0.6
        feet = project(pose, np.c_[xy[f], np.zeros(P)])
        head = project(pose, np.c_[xy[f], h_eff])
        for p in range(P):
            if np.isnan(feet[p, 0]) or rng.random() < p_miss:
                continue
            u, v = feet[p]
            hp = v - head[p, 1]
            if not (0 <= u < W and 0 <= v < H and hp > 15):
                continue
            wbox = 0.4 * hp
            x0, x1 = u - wbox / 2 + rng.normal(0, px_noise), u + wbox / 2 + rng.normal(0, px_noise)
            y1 = v + rng.normal(0, px_noise)
            y0 = head[p, 1] + rng.normal(0, px_noise)
            pid = p + 1
            if id_switch_frame is not None and f >= id_switch_frame and p % 2 == 0:
                pid += 100
            frames[f].append({"id": pid, "cls": 0, "conf": float(rng.uniform(0.5, 0.95)), "box": [x0, y0, x1, y1]})
        false_pos += rng.normal(0, 3, false_pos.shape)
        for k in range(n_false):
            u, v = false_pos[k]
            frames[f].append({"id": 900 + k, "cls": 0, "conf": 0.6, "box": [u - 20, v - 90, u + 20, v]})
    return frames


def traces(rng: np.random.Generator, pose: Pose, pitch: PitchModel, px_noise: float = 1.0) -> dict:
    L, Wd, cy, r = pitch.length, pitch.width, pitch.width / 2, pitch.d_radius

    def vis(uv):
        ok = np.isfinite(uv[:, 0]) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
        return (uv[ok] + rng.normal(0, px_noise, (ok.sum(), 2))).round(1).tolist()

    gl = np.c_[np.zeros(40), np.linspace(0, Wd, 40), np.zeros(40)]
    th = np.linspace(-math.pi / 2, math.pi / 2, 25)
    arc = np.c_[r * np.cos(th), cy + r * np.sin(th), np.zeros(25)]
    par = np.c_[np.full(20, 0.35 * L), np.linspace(2, Wd - 2, 20), np.zeros(20)]  # some unknown parallel line
    out = {"points": [], "lines": [{"world": "goal_line_A", "pixels": vis(project(pose, gl))}],
           "arcs": [{"world": "A_D", "pixels": vis(project(pose, arc))}],
           "parallels": [{"world": "goal_line_A", "pixels": vis(project(pose, par))}]}
    return {k: [g for g in v if len(g["pixels"]) >= 3] if k != "points" else v for k, v in out.items()}


def make_scenario(seed: int = 0, n_players: int = 15, n_frames: int = 300, corrupt: bool = True,
                  radius_factor: float = 1.3) -> Scenario:
    rng = np.random.default_rng(seed)
    poses = true_poses()
    xy, heights = simulate_players(rng, n_players, n_frames)
    raw = [detections(rng, p, xy, heights, id_switch_frame=150 if c == 1 else None, crouch_player=3)
           for c, p in enumerate(poses)]
    wrong = PitchModel.small_sided(length=50, width=30, goal_width=3.66, d_radius=TRUE_PITCH.d_radius * radius_factor)
    cals = []
    for c, p in enumerate(poses):
        tr = traces(rng, p, TRUE_PITCH)
        start = p
        if corrupt:
            start = Pose(p.x + rng.choice([-2, 2]), p.y + rng.choice([-2, 2]), p.z, p.yaw + math.radians(rng.choice([-5, 5])),
                         p.pitch, p.roll, p.f)
        cons = Constraints.build(wrong, tr["points"], tr["lines"], tr["arcs"], tr["parallels"])
        fitted, res_m = fit_pose(cons, wrong, W, H, init=start)
        Hpw = homography_pixel_to_world(fitted, W, H)
        cals.append(CameraCalibration(H=Hpw.tolist(), method="pose", reproj_error_m=float(np.sqrt(np.mean(res_m ** 2))),
                                      reproj_error_px=0.0, n_points=0, confidence=0.5, pose=fitted.to_dict(), **tr))
    return Scenario(wrong, poses, cals, raw, [(W, H)] * 3, xy)


def pose_errors(est: Pose, truth: Pose) -> dict[str, float]:
    dyaw = (est.yaw - truth.yaw + math.pi) % (2 * math.pi) - math.pi
    return {"pos_m": math.hypot(est.x - truth.x, est.y - truth.y), "height_m": abs(est.z - truth.z),
            "yaw_deg": abs(math.degrees(dyaw)), "focal_rel": abs(est.f / truth.f - 1)}


@pytest.fixture(scope="module", params=[0, 1])
def scenario(request) -> Scenario:
    return make_scenario(seed=request.param)


@pytest.fixture(scope="module")
def refined(scenario: Scenario):
    return refine_joint(scenario.cals, scenario.pitch, scenario.raw_tracks, scenario.sizes, free_radius=True,
                        verbose=False)


def test_corrupted_start_disagrees(scenario: Scenario):
    cons = cross_camera_consistency(scenario.cals, scenario.raw_tracks)
    assert isinstance(cons, Consistency)
    assert set(cons) == {"0-1", "0-2", "1-2"}
    assert max(cons.values()) > 0.8, cons
    assert cons.overall_frac_within_1m < 0.8


def test_refine_recovers_poses_and_radius(scenario: Scenario, refined):
    assert all(v < 0.5 for v in refined.pair_median_m.values()), refined.pair_median_m
    assert all(v > 0.9 for v in refined.pair_frac_within_1m.values()), refined.pair_frac_within_1m
    assert abs(refined.pitch.d_radius / TRUE_PITCH.d_radius - 1) < 0.05, refined.pitch.d_radius
    for est, truth in zip(refined.poses, scenario.poses):
        e = pose_errors(est, truth)
        assert e["pos_m"] < 1.5 and e["yaw_deg"] < 1.5 and e["height_m"] < 0.6 and e["focal_rel"] < 0.1, e
    assert abs(refined.person_height_m - 1.75) < 0.15
    assert not any("implausible" in n for n in refined.notes), refined.notes


def test_refined_is_at_noise_floor(scenario: Scenario, refined):
    """The true poses cannot do better than the 5 px feet noise allows; refinement must reach that floor."""
    from pitchworld.jointfit import apply_pose
    truth = [apply_pose(c, p, s, TRUE_PITCH, 0.0) for c, p, s in zip(scenario.cals, scenario.poses, scenario.sizes)]
    floor = cross_camera_consistency(truth, scenario.raw_tracks)
    for k, v in refined.pair_median_m.items():
        assert v < floor[k] + 0.1, (k, v, floor[k])


def test_refined_consistency_fraction(scenario: Scenario, refined):
    from pitchworld.jointfit import apply_pose
    cals = [apply_pose(c, p, s, refined.pitch, m) for c, p, s, m in
            zip(scenario.cals, refined.poses, scenario.sizes, refined.pair_median_m.values())]
    cons = cross_camera_consistency(cals, scenario.raw_tracks)
    assert all(v < 0.5 for v in cons.values()), cons
    assert cons.overall_frac_within_1m > 0.9


def test_outlier_rejection_drops_wrong_tracks(scenario: Scenario, refined):
    # false tracks (ids 900+) are static image boxes: they never match a real player across cameras
    assert refined.n_rejected_tracks >= 1 or refined.n_rejected_dets >= 1
    rng = np.random.default_rng(1)
    Ga = rng.uniform(0, 30, (60, 2))
    Gb = Ga + rng.normal(0, 0.2, Ga.shape)
    Gb[50:60] += 4.0  # one wrongly matched track pair, 4 m off
    Gb[5] += np.array([3.0, 0.0])  # one id-switch glitch inside a good track
    tms = [_TrackMatch(0, 1, k, k, np.arange(10 * k, 10 * k + 10), np.arange(10 * k, 10 * k + 10)) for k in range(6)]
    kept, n_tr, n_det = _reject_outliers([Ga, Gb], tms)
    assert n_tr == 1 and n_det == 1
    assert {m.ida for m in kept} == {0, 1, 2, 3, 4}
    assert len(kept[0].ia) == 9 and 5 not in kept[0].ia
    assert _match_tracks([Ga, Gb], [_collect([], 3, 0.4)] * 2, gate=5.0) == []


def test_identifiability_flags_null_direction():
    names = ["cam0.x", "cam0.y", "cam0.height"]
    J = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])  # height unobserved
    sig, bad = identifiability(J, np.ones(3), names)
    assert bad == ["cam0.height"]
    assert sig["cam0.x"] < IDENT_TOL["x"]


def test_identifiability_on_scenario(refined):
    assert set(refined.identifiability) >= {"cam0.x", "cam2.focal", "pitch.d_radius"}
    assert refined.identifiability["pitch.d_radius"] < IDENT_TOL["d_radius"]
    assert "pitch.d_radius" not in refined.unconstrained


def test_free_scale_recovers_pitch_size():
    """Pitch guessed 15 % too large in every dimension: the person-height prior should pull the scale back."""
    sc = make_scenario(seed=3, corrupt=False, radius_factor=1.0)
    big = PitchModel.small_sided(length=50 * 1.15, width=30 * 1.15, goal_width=3.66 * 1.15, d_radius=6.0 * 1.15)
    cals = []
    for c in sc.cals:
        cons = Constraints.build(big, c.points, c.lines, c.arcs, c.parallels)
        fitted, _ = fit_pose(cons, big, W, H, init=_pose_from_dict(c.pose))
        cals.append(dataclasses.replace(c, pose=fitted.to_dict(), H=homography_pixel_to_world(fitted, W, H).tolist()))
    jr = refine_joint(cals, big, sc.raw_tracks, sc.sizes, free_radius=False, free_scale=True, verbose=False)
    assert abs(jr.scale * 1.15 - 1.0) < 0.06, jr.scale
    assert abs(jr.pitch.length - 50) < 3.0
