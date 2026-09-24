import numpy as np

from highlights.calib import GoalZone, Space
from highlights.players import PlayerSigCfg, player_signals

W, H = 1920, 1080
BOX_H = 80  # -> mpp = 1.75/80 ~ 0.022 m/px


class _Pitch:
    length = 105.0
    width = 68.0


def _cfg():
    return PlayerSigCfg(v_run=3.0, cluster_radius=5.0, cluster_min_players=5,
                        cluster_slow=1.0, cluster_min_dur_s=2.0,
                        restart_half_width=12.0, restart_centre_r=4.0)


def _zones():
    return {"B": GoalZone("B", +1.0, float("nan"), [[1500, 200], [1900, 200], [1900, 900], [1500, 900]])}


def test_sprint_then_cluster():
    fps = 10.0
    frames = []
    for fi in range(120):
        t = fi / fps
        fr = []
        for pid in range(6):
            x = (400 + 400 * t + pid * 15) if t < 3.0 else (1550 + (pid % 3) * 8)
            y = 500 + (pid // 3) * 10
            fr.append({"id": pid, "conf": 0.9, "box": [x - 10, y - BOX_H, x + 10, y]})
        frames.append(fr)
    sig = player_signals(frames, fps, None, Space.PIXEL, _zones(), W, H, _Pitch(), _cfg(), bin_s=0.5)
    assert sig["attack_B"][: int(3.0 / 0.5)].max() > 0.5
    assert sig["cluster"].max() == 1.0
    assert sig["cluster_B"].max() == 1.0
    assert sig["cluster_size"].max() >= 1.0


def test_scattered_slow_players_no_signals():
    fps = 10.0
    rng = np.random.default_rng(2)
    spots = rng.uniform([100, 300], [1800, 900], size=(8, 2))
    frames = []
    for fi in range(60):
        fr = []
        for pid, (x, y) in enumerate(spots):
            x2 = x + rng.normal(0, 2)  # ~2px/frame jitter ~ 0.9 m/s << 3 m/s
            fr.append({"id": pid, "conf": 0.9, "box": [x2 - 10, y - BOX_H, x2 + 10, y]})
        frames.append(fr)
    sig = player_signals(frames, fps, None, Space.PIXEL, _zones(), W, H, _Pitch(), _cfg(), bin_s=0.5)
    assert sig["attack_B"].max() == 0.0
    assert sig["cluster"].max() == 0.0
