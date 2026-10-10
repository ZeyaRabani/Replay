"""stabilize: SIFT homography recovery, warp_at, gap fill, fuse hook."""

import numpy as np
import pytest

from highlights.analysis.stabilize import _fill_invalid, estimate_h, warp_at

cv2 = pytest.importorskip("cv2")


def _synthetic(rng, w=480, h=270, n=300):
    img = np.zeros((h, w), np.uint8)
    for _ in range(n):
        x, y = int(rng.integers(8, w - 8)), int(rng.integers(8, h - 8))
        r = int(rng.integers(2, 6))
        cv2.circle(img, (x, y), r, int(rng.integers(120, 255)), -1)
    noise = rng.normal(0, 8, (h, w)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_estimate_h_recovers_known_warp():
    rng = np.random.default_rng(3)
    ref = _synthetic(rng)
    h, w = ref.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.03)
    A[0, 2] += 12.0
    A[1, 2] -= 7.0
    Hk = np.vstack([A, [0, 0, 1.0]])
    cur = cv2.warpPerspective(ref, np.linalg.inv(Hk), (w, h))
    H, n_in = estimate_h(ref, cur, None, None)
    assert H is not None and n_in >= 25
    pts = rng.random((20, 2)) * np.array([w, h])
    got = cv2.perspectiveTransform(
        pts[None].astype(np.float64), H)[0]
    want = cv2.perspectiveTransform(
        pts[None].astype(np.float64), Hk)[0]
    assert np.abs(got - want).max() < 1.0


def test_estimate_h_textureless():
    assert estimate_h(np.zeros((270, 480), np.uint8),
                      np.zeros((270, 480), np.uint8), None,
                      None)[0] is None


def test_warp_at():
    H = np.stack([np.eye(3), np.array([[1, 0, 0.1], [0, 1, 0],
                                       [0, 0, 1.0]]), np.eye(3)])
    stab = {"t": np.array([0.0, 1.0, 2.0]), "H": H, "step_s": 1.0}
    x, y = warp_at(stab, 1.0, 0.5, 0.5)
    assert x == pytest.approx(0.6) and y == pytest.approx(0.5)
    assert warp_at(stab, 0.2, 0.5, 0.5) == (0.5, 0.5)
    assert warp_at(stab, 10.0, 0.5, 0.5) == (0.5, 0.5)


def test_fill_invalid_interpolates():
    H = np.stack([np.eye(3), np.eye(3), np.eye(3),
                  np.diag([2.0, 2.0, 1.0])])
    valid = np.array([True, False, False, True])
    out = _fill_invalid(H, valid)
    assert out[1][0, 0] == pytest.approx(4 / 3)
    assert out[2][0, 0] == pytest.approx(5 / 3)
    none = _fill_invalid(np.ones((2, 3, 3)), np.zeros(2, bool))
    assert (none == np.eye(3)).all()


def test_fuse_stab_shifts_position():
    from highlights.analysis.fuse_tracks import fuse
    L, W = 100.0, 64.0
    H_ID = [[L, 0, 0], [0, W, 0], [0, 0, 1.0]]
    ts, feet, confs, teams, boxes = [], [], [], [], []
    for k in range(24):
        ts.append(k * 0.5)
        feet.append([10.0 / L, 10.0 / W])
        confs.append(0.9)
        teams.append("A")
        boxes.append([0, 0, 20, 40])
    det = {"t": np.array(ts), "foot": np.array(feet),
           "conf": np.array(confs), "team": np.array(teams),
           "box": np.array(boxes), "w": 1.0, "h": 1.0}
    base = fuse({0: det}, {0: H_ID}, window=(0.0, 12.0),
                offsets=[0.0], pitch=(L, W), log=lambda m: None)
    Hs = np.tile(np.eye(3), (30, 1, 1))
    Hs[:, 0, 2] = 0.1
    stab = {"t": np.arange(30.0), "H": Hs, "step_s": 1.0}
    doc = fuse({0: det}, {0: H_ID}, window=(0.0, 12.0),
               offsets=[0.0], pitch=(L, W), stabs={0: stab},
               log=lambda m: None)
    x0 = np.nanmean([p[0] for p in base["tracks"][0]["xy"]
                     if p and p[0] is not None])
    x1 = np.nanmean([p[0] for p in doc["tracks"][0]["xy"]
                     if p and p[0] is not None])
    assert x1 - x0 == pytest.approx(0.1 * L, abs=0.2)
