"""Calibration tests: landmark table + homography solving."""

import numpy as np
import pytest

from highlights.analysis.calib import apply_h, landmark_xy, landmarks, solve_homography

LM = landmark_xy(100.0, 64.0)
LML = {l["name"]: l for l in landmarks(100.0, 64.0)}


def _inv_h(H, x, y):
    v = np.linalg.inv(np.asarray(H)) @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def test_landmarks_table():
    assert LM["corner_near_left"] == [0.0, 64.0]
    assert LM["corner_far_right"] == [100.0, 0.0]
    assert LM["centre_spot"] == [50.0, 32.0]
    assert LM["pen_spot_l"] == [11.0, 32.0]
    assert LM["pen_spot_r"] == [89.0, 32.0]
    # goal posts 7.32 m centred on a 64 m pitch
    assert LM["goalpost_l_near"][1] == pytest.approx(35.66, abs=0.01)
    # 4 corners + 2 halfway + centre + 8 pen + 8 six + 2 spots + 4 posts
    assert len(LM) == 29
    assert LML["corner_near_left"]["label"] == "Corner - near left"
    assert LML["centre_spot"]["label"] == "Centre spot"


def test_solve_recovers_known_h():
    # known H pitch->frame; frame pts = H·pitch, solve frame->pitch
    Hk = np.array([[400.0, 30.0, 100.0],
                   [20.0, 500.0, 200.0],
                   [0.0005, -0.0002, 1.0]])
    names = ["corner_near_left", "corner_far_right", "halfway_far",
             "centre_spot", "pen_spot_l", "six_r_edge_near"]
    pts = [{"name": n,
            "fx": _inv_h(Hk, *LM[n])[0],
            "fy": _inv_h(Hk, *LM[n])[1]}
           for n in names]
    H, rms = solve_homography(pts, 100.0, 64.0)
    assert rms < 1e-6
    for n in names:
        x, y = apply_h(H, _inv_h(Hk, *LM[n])[0],
                       _inv_h(Hk, *LM[n])[1])
        assert x == pytest.approx(LM[n][0], abs=1e-4)
        assert y == pytest.approx(LM[n][1], abs=1e-4)


def test_solve_rejects_bad_input():
    pts = [{"name": n, "fx": 0.1 * i, "fy": 0.1 * i}
           for i, n in enumerate(["corner_near_left",
                                  "corner_far_right", "centre_spot"])]
    with pytest.raises(ValueError, match="at least 4"):
        solve_homography(pts, 100.0, 64.0)
    pts4 = pts + [{"name": "nope", "fx": 0.5, "fy": 0.5}]
    with pytest.raises(ValueError, match="unknown landmark"):
        solve_homography(pts4, 100.0, 64.0)
    pts5 = pts + [{"name": "halfway_far", "fx": 5.0, "fy": 0.5}]
    with pytest.raises(ValueError, match=r"\[-1, 2\]"):
        solve_homography(pts5, 100.0, 64.0)
    # collinear -> degenerate
    col = [{"name": n, "fx": 0.1, "fy": 0.1 + 0.1 * i}
           for i, n in enumerate(["corner_near_left", "corner_far_right",
                                  "halfway_far", "centre_spot"])]
    with pytest.raises(ValueError):
        solve_homography(col, 100.0, 64.0)
