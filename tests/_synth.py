"""Synthetic camera helpers shared by the posefit tests."""

from __future__ import annotations

import math

import numpy as np

from pitchworld.pitch import PitchModel
from pitchworld.posefit import Pose, homography_world_to_pixel

W, H = 1920, 1080


def pitch() -> PitchModel:
    return PitchModel.small_sided(length=50, width=30, goal_width=3.66, d_radius=6.0)


def random_pose(rng: np.random.Generator, p: PitchModel | None = None, near_goal_a: bool = True) -> Pose:
    """A plausible phone camera: on a ring around the pitch, 1.5-6 m up, looking roughly at the centre.

    With ``near_goal_a`` the camera stands in the half-ring behind/beside goal A (x=0), i.e. within
    ~10-30 m of the markings the tests trace (goal line A, the A "D", touchline S).
    """
    p = p or pitch()
    cx, cy = p.length / 2, p.width / 2
    ang = math.pi + rng.uniform(-0.55 * math.pi, 0.55 * math.pi) if near_goal_a else rng.uniform(0, 2 * math.pi)
    rad = rng.uniform(0.7, 1.1) * math.hypot(cx, cy)
    x, y = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
    yaw = math.atan2(cy - y, cx - x) + rng.uniform(-0.25, 0.25)
    z = rng.uniform(1.5, 6.0)
    pitch_ang = math.atan2(z, rad * 0.8) + rng.uniform(-0.05, 0.1)
    roll = rng.uniform(-0.03, 0.03)
    f = rng.uniform(0.7, 1.6) * W
    return Pose(x, y, z, yaw, pitch_ang, roll, f)


def project(pose: Pose, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World ground points -> (pixels, visible mask). Visible = in front and inside the frame."""
    Hwp = homography_world_to_pixel(pose, W, H)
    hom = np.c_[xy, np.ones(len(xy))] @ Hwp.T
    front = hom[:, 2] > 1e-9
    px = hom[:, :2] / np.where(front, hom[:, 2], 1.0)[:, None]
    inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return px, front & inside


def visible_ground_points(pose: Pose, rng: np.random.Generator, n: int, p: PitchModel | None = None) -> np.ndarray:
    p = p or pitch()
    out = []
    while len(out) < n:
        cand = rng.uniform([-2, -2], [p.length + 2, p.width + 2], size=(4 * n, 2))
        _, vis = project(pose, cand)
        out.extend(cand[vis].tolist())
    return np.array(out[:n])


def trace_line(pose: Pose, p0, d, t_range, n, rng, noise_px) -> np.ndarray:
    t = np.linspace(*t_range, n)
    xy = np.asarray(p0, float) + t[:, None] * np.asarray(d, float)
    px, vis = project(pose, xy)
    return px[vis] + rng.normal(0, noise_px, size=(int(vis.sum()), 2))


def trace_arc(pose: Pose, c, r, ang_range_deg, n, rng, noise_px) -> np.ndarray:
    a = np.radians(np.linspace(*ang_range_deg, n))
    xy = np.asarray(c, float) + r * np.c_[np.cos(a), np.sin(a)]
    px, vis = project(pose, xy)
    return px[vis] + rng.normal(0, noise_px, size=(int(vis.sum()), 2))
