"""fuse_tracks: two noisy angles + dropouts -> 6 stable tracks."""

import numpy as np

from highlights.analysis.fuse_tracks import STEP, fuse

L, W = 100.0, 64.0
# frame coords = pitch metres / pitch dims -> H scales back up
H_ID = [[L, 0, 0], [0, W, 0], [0, 0, 1.0]]

WINDOW = (0.0, 60.0)
N_STEPS = round((WINDOW[1] - WINDOW[0]) / STEP) + 1

# 6 players: fixed (x0,y0), drifting vx m/s along x, bouncing at edges
PLAYERS = [(10 + i * 15, 10 + i * 8, 0.6 + 0.2 * (i % 3), "A" if i < 3 else "B")
           for i in range(6)]


def _true_pos(p, k):
    x0, y0, vx, _team = p
    x = x0 + vx * k * STEP
    lo, hi = 10.0, L - 10
    span = hi - lo
    x = lo + abs((x - lo) % (2 * span) - span)  # reflect into [lo,hi]
    return x, y0


def _dets_for_angle(rng, missing_mod=None):
    """detections dict for one angle: every player every step,
    foot = (x/L, y/W) + 0.5 m noise; skip every `missing_mod` step."""
    ts, feet, confs, teams, boxes = [], [], [], [], []
    for k in range(N_STEPS):
        if missing_mod and k % missing_mod == 0:
            continue
        for p in PLAYERS:
            x, y = _true_pos(p, k)
            nx = x + rng.normal(0, 0.5)
            ny = y + rng.normal(0, 0.5)
            ts.append(k * STEP)
            feet.append([nx / L, ny / W])
            confs.append(0.8)
            teams.append(p[3])
            boxes.append([0, 0, 20, 40])
    return {"t": np.array(ts), "foot": np.array(feet),
            "conf": np.array(confs), "team": np.array(teams),
            "box": np.array(boxes), "w": 1.0, "h": 1.0}


def test_fuse_two_angles_six_players():
    rng = np.random.default_rng(7)
    dets = {0: _dets_for_angle(rng), 1: _dets_for_angle(rng, missing_mod=3)}
    doc = fuse(dets, {0: H_ID, 1: H_ID}, window=WINDOW,
               offsets=[0.0, 0.0], pitch=(L, W), log=lambda m: None)
    tracks = doc["tracks"]
    assert len(tracks) == 6
    by_team = {"A": 0, "B": 0}
    for tr in tracks:
        by_team[tr["team"]] += 1
        # each track covers >=95% of steps
        assert len(tr["xy"]) >= 0.95 * N_STEPS
        # positions within 1 m of the true path
        errs = []
        for k, (x, y) in enumerate(tr["xy"]):
            if x is None:
                continue
            xk, yk = _true_pos(min(
                PLAYERS, key=lambda p: np.hypot(
                    _true_pos(p, 0)[0] - tr["xy"][0][0],
                    _true_pos(p, 0)[1] - tr["xy"][0][1])), k)
            errs.append(np.hypot(x - xk, y - yk))
        assert np.mean(errs) < 1.0
    assert by_team == {"A": 3, "B": 3}
    assert doc["summary"]["n_tracks"] == 6
    assert 4 <= doc["summary"]["median_visible"] <= 6


# ownership: angle 0's camera is at the y=W end, angle 1's at y=0
H_CAM0 = [[L, 0, 0], [0, W, 0], [0, 0, 1.0]]
H_CAM1 = [[L, 0, 0], [0, -W, W], [0, 0, 1.0]]


def _det(angle: int, px: float, py: float, t: float = 0.0):
    """One-angle dets dict with a single detection whose foot projects
    to pitch (px, py) through that angle's H above."""
    foot = [px / L, py / W] if angle == 0 else [px / L, (W - py) / W]
    return {"t": np.array([t]), "foot": np.array([foot]),
            "conf": np.array([0.9]), "team": np.array([""]),
            "box": np.array([[0, 0, 20, 40]]), "w": 1.0, "h": 1.0}


def test_ownership_keeps_nearest_camera():
    """Same player at (30, 60) seen by both cameras: it is nearer to
    camera 0 (50,64) than camera 1 (50,0), so only angle 0's detection
    survives — one visible observation that step."""
    dets = {0: _det(0, 30.0, 60.0), 1: _det(1, 30.0, 60.0)}
    doc = fuse(dets, {0: H_CAM0, 1: H_CAM1}, window=(0.0, 0.0),
               offsets=[0.0, 0.0], pitch=(L, W), log=lambda m: None)
    assert doc["summary"]["visible_hist"] == [1]
    assert doc["summary"]["ownership"] is True
    assert doc["summary"]["cam_xy"] == {"0": [50.0, 64.0], "1": [50.0, 0.0]}


def test_ownership_off_keeps_both():
    """With ownership disabled, detections 8 m apart exceed MERGE_M and
    stay two observations."""
    dets = {0: _det(0, 30.0, 60.0), 1: _det(1, 38.0, 60.0)}
    doc = fuse(dets, {0: H_CAM0, 1: H_CAM1}, window=(0.0, 0.0),
               offsets=[0.0, 0.0], pitch=(L, W), ownership=False,
               log=lambda m: None)
    assert doc["summary"]["visible_hist"] == [2]
    assert doc["summary"]["ownership"] is False
    assert doc["summary"]["cam_xy"] == {}
