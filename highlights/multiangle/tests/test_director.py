"""Director rules over synthetic per-second tracks."""

import numpy as np

from highlights.multiangle import director as D


def _track(T: int, ball_conf=0.0, ball_size=0.0, cluster=0.0):
    return {"ball_conf": np.full(T, ball_conf),
            "ball_size": np.full(T, ball_size),
            "cluster": np.full(T, cluster)}


def _avail(n: int, T: int):
    return np.ones((n, T), dtype=bool)


def test_ball_rule_picks_larger_ball():
    T = 60
    tr = [_track(T, cluster=5.0), _track(T, ball_conf=0.5, ball_size=0.04)]
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)])
    assert all(s["angle"] == 1 for s in out["segments"])
    assert out["per_second_rule"]["ball"] > 0


def test_cluster_fallback_no_ball():
    T = 60
    tr = [_track(T, cluster=2.0), _track(T, cluster=9.0)]
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)])
    assert all(s["angle"] == 1 for s in out["segments"])
    assert out["per_second_rule"]["cluster"] > 0


def test_no_flicker_blip_below_margin():
    """A 1 s cluster blip must not cause a cut."""
    T = 90
    tr = [_track(T, cluster=8.0), _track(T, cluster=3.0)]
    tr[1]["cluster"][40] = 20.0  # one-second blip
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)])
    assert out["n_cuts"] == 0
    assert len(out["segments"]) == 1


def test_min_hold_respected():
    """Sustained better angle is only chosen after hold + confirm."""
    T = 60
    tr = [_track(T, cluster=8.0), _track(T, cluster=3.0)]
    tr[1]["cluster"][30:] = 20.0  # better from t=30 on
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)])
    assert out["n_cuts"] == 1
    cut = out["segments"][0]["t_end"]
    assert 28 <= cut <= 35  # confirm+hold delay, not instant


def test_hard_cut_on_unavailable():
    T = 60
    avail = _avail(2, T)
    avail[0, 30:] = False  # angle 0 dies at t=30
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    out = D.cut_director(tr, avail, [np.zeros(T), np.zeros(T)])
    assert out["n_cuts"] == 1
    assert out["segments"][1]["angle"] == 1
    assert out["segments"][0]["rule"] == "coverage"


def test_cluster_margin_blocks_and_allows():
    """Under cluster rule (margin 40%): challenger 1.2 vs 1.0 -> no cut;
    challenger 1.5 -> cut after CONFIRM + MIN_HOLD."""
    T = 90
    tr = [_track(T, cluster=1.0), _track(T, cluster=0.0)]
    tr[1]["cluster"][20:] = 1.2  # beats but within 40% margin
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)])
    assert out["n_cuts"] == 0

    tr[1]["cluster"][20:] = 1.5  # exceeds margin -> cuts
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)])
    assert out["n_cuts"] == 1
    cut = out["segments"][0]["t_end"]
    # CONFIRM satisfied at t=22; cut placed inside the [t-2, t+2] window
    assert 20 <= cut <= 24


def test_ratios_sum_to_one():
    T = 120
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    tr[1]["cluster"][60:] = 12.0
    out = D.cut_director(tr, _avail(2, T), [np.ones(T), np.ones(T)])
    assert abs(sum(out["ratios"].values()) - 1.0) < 1e-6
