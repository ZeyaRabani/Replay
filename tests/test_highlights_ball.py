import numpy as np

from highlights.ball import ball_signals, link_ball
from highlights.calib import GoalZone, Space

W, H = 1920, 1080


def _mkdets(frames_truth: list[tuple[float, float] | None], jitter=(400, 200)):
    """Each frame: truth (or none) plus a false candidate at a fixed spot."""
    dets = []
    rng = np.random.default_rng(1)
    for p in frames_truth:
        fr = []
        if p is not None:
            fr.append({"conf": 0.8, "box": [p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5]})
        j = (jitter[0] + rng.normal(0, 20), jitter[1] + rng.normal(0, 20))
        fr.append({"conf": 0.5, "box": [j[0] - 5, j[1] - 5, j[0] + 5, j[1] + 5]})
        dets.append(fr)
    return dets


def test_link_follows_true_path():
    fps = 10.0
    n = 50
    truth = [(100 + 10 * i, 500) for i in range(n)]
    dets = _mkdets(truth)
    tr = link_ball(dets, fps, W, max_jump_frac=0.08)
    err = np.abs(tr.u - np.array([100 + 10 * i for i in range(n)]))
    assert np.nanmax(err) < 5
    assert tr.valid.sum() > 40


def test_attack_fires_only_toward_goal():
    fps = 10.0
    zone = {"B": GoalZone("B", +1.0, float("nan"), [[1500, 200], [1900, 200], [1900, 900], [1500, 900]])}
    bin_s = 0.5
    # ball moves toward B (x+) inside the zone for 20 frames, then away (x-) for 20
    truth = [(1600 + 20 * i, 500) for i in range(20)] + [(1980 - 20 * i, 500) for i in range(20)]
    truth += [None] * 20
    dets = _mkdets(truth)
    tr = link_ball(dets, fps, W)
    sig = ball_signals(tr, zone, Space.PIXEL, bin_s, v_shot=0.1)
    bins_toward = {int(tr.t[i] / bin_s) for i in range(10) if np.isfinite(tr.u[i])}
    assert any(sig["ball_attack_B"][b] > 0 for b in bins_toward)
    # bins fully inside the moving-away window (frames ~25-34) must not fire
    assert sig["ball_attack_B"][5:7].max() == 0.0
