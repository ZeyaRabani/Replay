"""Calibration tests: landmark table + homography solving."""

import numpy as np
import pytest

from highlights.analysis.calib import apply_h, landmark_xy, landmarks, landmarks_for, solve_homography

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


def test_landmarks_small_template():
    pitch = {"len_m": 70.0, "wid_m": 45.0, "template": "small",
             "goal_w_m": 3.66, "d_radius_m": 9.0}
    lm = {l["name"]: [l["x"], l["y"]] for l in landmarks_for(pitch)}
    assert lm["corner_near_left"] == [0.0, 45.0]
    assert lm["corner_far_right"] == [70.0, 0.0]
    assert lm["halfway_near"] == [35.0, 45.0]
    assert lm["centre_spot"] == [35.0, 22.5]
    # goalposts centred, 3.66 m apart
    assert lm["goalpost_l_near"] == [0.0, pytest.approx(24.33)]
    assert lm["goalpost_l_far"] == [0.0, pytest.approx(20.67)]
    assert lm["goalpost_r_near"] == [70.0, pytest.approx(24.33)]
    # D arc meets goal line at y = cy +/- r, apex at x = r / L - r
    assert lm["d_l_near"] == [0.0, 31.5]
    assert lm["d_l_far"] == [0.0, 13.5]
    assert lm["d_r_near"] == [70.0, 31.5]
    assert lm["d_r_far"] == [70.0, 13.5]
    assert lm["d_l_apex"] == [9.0, 22.5]
    assert lm["d_r_apex"] == [61.0, 22.5]
    # 4 corners + 2 halfway + centre + 4 posts + 4 D-ends + 2 apices
    assert len(lm) == 17
    lab = {l["name"]: l["label"] for l in landmarks_for(pitch)}
    assert lab["d_l_apex"] == "D left - apex"
    # unknown template falls back to the full table
    assert len(landmarks_for({"len_m": 100.0, "wid_m": 64.0})) == 29


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


def test_solve_small_template_names():
    pitch = {"len_m": 70.0, "wid_m": 45.0, "template": "small",
             "goal_w_m": 3.66, "d_radius_m": 9.0}
    lm = {l["name"]: [l["x"], l["y"]] for l in landmarks_for(pitch)}
    Hk = np.array([[400.0, 30.0, 100.0],
                   [20.0, 500.0, 200.0],
                   [0.0005, -0.0002, 1.0]])
    names = ["corner_near_left", "corner_far_right", "halfway_far",
             "centre_spot", "d_l_apex", "d_r_near"]
    pts = [{"name": n,
            "fx": _inv_h(Hk, *lm[n])[0],
            "fy": _inv_h(Hk, *lm[n])[1]}
           for n in names]
    _H, rms = solve_homography(pts, 70.0, 45.0, pitch)
    assert rms < 1e-6
    # a full-template name is unknown on a small pitch
    with pytest.raises(ValueError, match="unknown landmark"):
        solve_homography(pts + [{"name": "pen_spot_l", "fx": 0.5,
                                 "fy": 0.5}], 70.0, 45.0, pitch)
