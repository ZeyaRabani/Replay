"""Synthetic 3-camera joint calibration test: two cameras' landmark
clicks are noisy, detections are generated from the true Hs, and the
joint refine must pull cross-camera residuals below 0.5 m."""
from __future__ import annotations

import numpy as np
import pytest

from highlights.analysis import joint_calib
from highlights.analysis.calib import apply_h, landmark_xy, landmarks_for

rng = np.random.default_rng(7)

PITCH = {"len_m": 55.0, "wid_m": 23.0, "template": "small",
         "goal_w_m": 4.0, "d_radius_m": 4.0}
L, W = 55.0, 23.0


def _true_H(seed: int) -> np.ndarray:
    """A plausible frame->pitch homography (affine + mild perspective)."""
    r = np.random.default_rng(seed)
    H = np.array([[40 + 20 * r.random(), -8 * r.random(), 5 + 5 * r.random()],
                  [3 * r.random(), 25 + 15 * r.random(), 8 + 4 * r.random()],
                  [0.01 * r.random() - 0.005, -0.4 * r.random(), 1.0]])
    return H


TRUE_H = {a: _true_H(a) for a in range(3)}
LM_XY = landmark_xy(L, W, PITCH)


def _clicks(a: int, noise: float) -> list[dict]:
    """Landmark clicks for camera a: true pitch lm -> frame coords via
    inverse true H, plus gaussian noise."""
    Hi = np.linalg.inv(TRUE_H[a])
    pts = []
    for lm in landmarks_for(PITCH):
        fx, fy = apply_h(Hi, lm["x"], lm["y"])
        pts.append({"name": lm["name"],
                    "fx": fx + float(rng.normal(0, noise)),
                    "fy": fy + float(rng.normal(0, noise))})
    return pts


def _dets(n_steps: int = 60, n_players: int = 10) -> dict[int, dict]:
    """The same random players each 1 s step seen by every camera;
    foot points projected through the inverse true H."""
    r = np.random.default_rng(100)
    steps_pos = [np.c_[r.uniform(2, L - 2, n_players),
                       r.uniform(1, W - 1, n_players)]
                 for _ in range(n_steps)]
    out = {}
    for a in range(3):
        Hi = np.linalg.inv(TRUE_H[a])
        ts, feet, confs, boxes = [], [], [], []
        for k in range(n_steps):
            pos = steps_pos[k]
            for x, y in pos:
                fx, fy = apply_h(Hi, float(x), float(y))
                ts.append(float(k))
                feet.append([fx, fy])
                confs.append(0.9)
                boxes.append([0.0, 0.0, 40.0, 100.0])
        out[a] = {"t": np.array(ts), "foot": np.array(feet),
                  "conf": np.array(confs), "box": np.array(boxes),
                  "team": np.array([""] * len(ts), dtype=object),
                  "w": 1.0, "h": 1.0}
    return out


def _pair_med(Hs: dict[int, np.ndarray], dets: dict[int, dict],
              a: int, b: int, k: int) -> float:
    """Median cross-camera distance at step k for matched players."""
    fa = [apply_h(Hs[a], *f) for i, f in enumerate(dets[a]["foot"])
          if dets[a]["t"][i] == float(k)]
    fb = [apply_h(Hs[b], *f) for i, f in enumerate(dets[b]["foot"])
          if dets[b]["t"][i] == float(k)]
    return float(np.median([np.hypot(p[0] - q[0], p[1] - q[1])
                            for p, q in zip(fa, fb)]))


@pytest.fixture(scope="module")
def refined():
    calib = {"pitch": dict(PITCH),
             "angles": {"0": {"pts": _clicks(0, 0.0)},
                        "1": {"pts": _clicks(1, 0.015)},
                        "2": {"pts": _clicks(2, 0.03)}}}
    dets = _dets()
    res = joint_calib.joint_refine(calib, dets, {}, [0.0, 0.0, 0.0],
                                   window=(0.0, 60.0), log=lambda m: None)
    return res, dets


def test_joint_refine_accepted(refined):
    res, _ = refined
    assert res["accepted"], res.get("reason")
    for a, rms in res["lm_rms"].items():
        assert rms <= joint_calib.MAX_LM_RMS_M, (a, rms)


def test_pair_residuals_below_half_metre(refined):
    res, _dets = refined
    for pair, med in res["pairs"].items():
        assert med["after"] < 0.5, (pair, med)
        assert med["after"] < med["before"], (pair, med)
    # sanity: landmark-only agreement really was worse than 0.5 m on
    # at least one pair, so the test isn't vacuous
    assert any(m["before"] > 0.5 for m in res["pairs"].values())
