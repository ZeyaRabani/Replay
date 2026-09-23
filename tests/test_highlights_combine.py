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
    cands = combine(sig, bin_s, cfg, duration_s=200)
    assert len(cands) == 2
    top = cands[0]
    assert top.type == "goal" and top.confidence > 0.7 and top.goal == "B"
    other = cands[1]
    assert other.type == "chance" and other.confidence < 0.5
    assert [c.id for c in cands] == ["c01", "c02"]
    assert top.start == top.t_event - cfg.pre_roll_s
    assert top.end > top.t_event
