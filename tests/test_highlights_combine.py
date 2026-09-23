import numpy as np

from highlights.combine import combine
from highlights.config import Config


def _sig(n=400):
    z = np.zeros(n)
    return {"ball_attack_A": z.copy(), "ball_attack_B": z.copy(), "attack_A": z.copy(), "attack_B": z.copy(),
            "ball_lost_A": z.copy(), "ball_lost_B": z.copy(), "audio": z.copy(), "cluster": z.copy(),
            "cluster_A": z.copy(), "cluster_B": z.copy(), "restart": z.copy(), "n_players": np.full(n, 10.0)}


def test_goal_and_chance_and_merge():
    bin_s = 0.5
    cfg = Config()
    sig = _sig()
    # goal on B at t=40 (bins 80-81): attack sustained + audio + cluster + lost
    sig["ball_attack_B"][80] = 1.0
    sig["ball_attack_B"][81] = 0.7
    sig["audio"][84:90] = 0.9
    sig["cluster_B"][82:88] = 1.0
    sig["ball_lost_B"][79] = 1.0
    # second anchor 5s later on B (t=45, bins 90-91) — should merge
    sig["ball_attack_B"][90] = 0.8
    sig["ball_attack_B"][91] = 0.6
    # chance on A at t=120 (bins 240-241): attack + weak audio only
    sig["ball_attack_A"][240] = 0.6
    sig["ball_attack_A"][241] = 0.5
    sig["audio"][241] = 0.2
    # isolated bin with no players — must not anchor
    sig["n_players"][300] = 1.0
    sig["ball_attack_B"][300] = 0.9
    cands, windows = combine(sig, bin_s, cfg, duration_s=200)
    assert windows == [[0.0, 200.0]]
    assert len(cands) == 2
    top = cands[0]
    assert top.type == "goal" and top.confidence > 0.7 and top.goal == "B"
    other = cands[1]
    assert other.type == "chance" and other.confidence < 0.5
    assert [c.id for c in cands] == ["c01", "c02"]
    assert top.start <= top.t_event - cfg.roll_goal_s
    assert top.end >= top.t_event + cfg.roll_goal_s  # merged anchors union the windows
    assert other.end - other.start == 2 * cfg.roll_chance_s


def test_play_gate_drops_warmup_anchors():
    bin_s = 0.5
    cfg = Config()
    sig = _sig(600)  # 300 s
    sig["n_players"][:240] = 2.0   # first 120 s = warm-up, median < 10
    sig["ball_attack_B"][80] = 1.0  # strong anchor inside the warm-up window
    sig["ball_attack_B"][81] = 0.9
    sig["audio"][84:90] = 1.0
    sig["cluster_B"][82:88] = 1.0
    sig["ball_attack_B"][500] = 1.0  # same anchor during real play
    sig["ball_attack_B"][501] = 0.9
    sig["audio"][504:510] = 1.0
    sig["cluster_B"][502:508] = 1.0
    cands, windows = combine(sig, bin_s, cfg, duration_s=300)
    assert all(c.t_event > 120.0 for c in cands)
    assert len(cands) == 1
    assert windows == [[120.0, 300.0]]  # warm-up half gated off, real-play half active


def test_audio_anchors():
    bin_s = 0.5
    cfg = Config()
    sig = _sig(800)
    # 6 s audio episode at t=100 with a cluster near B -> goal via audio anchor
    sig["audio"][200:212] = 0.9
    sig["cluster_B"][204:210] = 1.0
    # 1.5 s spike at t=200, no support -> chance
    sig["audio"][400:403] = 0.9
    # episode during gated-off warm-up is ignored
    sig["audio"][20:30] = 0.9
    sig["n_players"][:200] = 2.0
    sig["n_players"][199] = 10.0  # spike bins still >0.5 at 100s only if gated; keep 100s inside gate
    sig["n_players"][200:] = 10.0
    cands, _ = combine(sig, bin_s, cfg, duration_s=400)
    assert len(cands) == 2
    goal = cands[0]
    assert goal.type == "goal" and goal.anchor == "audio" and goal.goal is None  # no attack -> "?"
    assert goal.signals["audio_dur_s"] == 6.0
    assert goal.confidence > 0.4  # 0.25*1.0 (audio) + 0.20*1.0 (cluster)
    ch = cands[1]
    assert ch.type == "chance" and ch.anchor == "audio"
    assert abs(ch.confidence - 0.25 * 0.6875) < 0.01  # 0.25 * min(1, 0.5+1.5/8)
