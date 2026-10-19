"""Director rules over synthetic per-second tracks (v2 semantics)."""

import numpy as np
import pytest

from highlights.multiangle import director as D


@pytest.fixture(autouse=True)
def _players_cluster(monkeypatch):
    """These tests encode the players-crowd cluster behaviour."""
    monkeypatch.setattr(D, "CLUSTER_SOURCE", "players")


def _track(T: int, ball_conf=0.0, ball_size=0.0, cluster=0.0):
    return {"ball_conf": np.full(T, ball_conf),
            "ball_size": np.full(T, ball_size),
            "cluster": np.full(T, cluster)}


def _avail(n: int, T: int):
    return np.ones((n, T), dtype=bool)


def _mot(T: int):
    return np.zeros(T)


def test_ball_rule_picks_larger_ball():
    T = 60
    tr = [_track(T, cluster=5.0), _track(T, ball_conf=0.5, ball_size=0.04)]
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert all(s["angle"] == 1 for s in out["segments"])
    assert out["per_second_rule"]["ball"] > 0
    assert out["segments"][0]["rule"] == "start"


def test_ball_needs_two_sightings():
    """A single ball_conf hit in a 5 s window is NOT a sighting."""
    T = 60
    tr = [_track(T, cluster=8.0), _track(T, cluster=3.0)]
    tr[1]["ball_conf"][30] = 0.9  # one isolated hit
    tr[1]["ball_size"][30] = 0.5
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["per_second_rule"]["ball"] == 0


def test_cluster_fallback_no_ball():
    """No ball anywhere -> cluster rule picks the angle above its baseline."""
    T = 400
    tr = [_track(T, cluster=2.0), _track(T, cluster=1.0)]
    tr[1]["cluster"][370:] = 9.0  # 9x its own p90 baseline -> normalised win
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["per_second_rule"]["cluster"] > 0
    assert out["per_second_rule"]["ball"] == 0
    assert out["segments"][-1]["angle"] == 1


def test_no_flicker_blip_below_margin():
    """A 1 s cluster blip must not cause a cut."""
    T = 90
    tr = [_track(T, cluster=8.0), _track(T, cluster=3.0)]
    tr[1]["cluster"][40] = 40.0  # one-second blip
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["n_cuts"] == 0
    assert len(out["segments"]) == 1


def test_segment_fields_recorded_at_start():
    """rule/score/runner_up describe the selection at segment start, and
    runner_up is the displaced angle (never the new one)."""
    T = 400
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    tr[1]["cluster"][370:] = 20.0  # <10% rise keeps a1's p90 at 2.0
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["n_cuts"] == 1
    s0, s1 = out["segments"]
    assert s0["rule"] == "start" and s0["runner_up"] is None
    assert s1["rule"] == "cluster"
    assert s1["angle"] == 1
    assert s1["runner_up"]["angle"] == 0
    assert s1["runner_up"]["angle"] != s1["angle"]
    assert s1["score"] > 0
    assert s0["t_end"] == s1["t_start"]


def test_min_hold_respected():
    """Challenger better from t=5: CONFIRM at 11 but cut waits for hold>=20."""
    T = 400
    tr = [_track(T, cluster=8.0), _track(T, cluster=3.0)]
    tr[1]["cluster"][5:35] = 20.0  # 30 s rise, <10% of span -> p90 stays 3.0
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["n_cuts"] == 1
    cut = out["segments"][1]["t_start"]
    # cut lands inside the [t-2, t+2] window after hold reaches MIN_HOLD
    assert 18 <= cut <= 24


def test_cluster_margin_blocks_and_allows():
    """Margin 50% under cluster rule: 1.4x no cut; 1.6x cut."""
    T = 400
    tr = [_track(T, cluster=1.0), _track(T, cluster=1.0)]
    tr[1]["cluster"][370:] = 1.4  # 1.4x a1's own p90 = within 50% margin
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["n_cuts"] == 0

    tr[1]["cluster"][370:] = 2.0  # 2x -> exceeds margin
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["n_cuts"] == 1


def test_cluster_baseline_normalisation():
    """Different camera baselines must not bias the winner."""
    T = 400
    # angle 0: wide shot, high raw baseline; angle 1: tight, low baseline
    tr = [_track(T, cluster=10.0), _track(T, cluster=1.0)]
    tr[1]["cluster"][370:] = 5.0  # short <10% rise: p90 stays at the 1.0 base
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["cluster_baseline"] == [10.0, 1.0]
    assert out["n_cuts"] == 1
    assert out["segments"][1]["angle"] == 1


def test_dead_feed_recovery():
    """Incumbent at 0 for >= 6 s while challenger has signal -> cut."""
    T = 90
    tr = [_track(T, cluster=8.0), _track(T, cluster=1.0)]
    tr[0]["cluster"][20:] = 0.0   # a0's signal dies
    out = D.cut_director(tr, _avail(2, T), [_mot(T), _mot(T)])
    assert out["n_cuts"] == 1
    assert out["segments"][1]["angle"] == 1


def test_hard_cut_on_unavailable():
    T = 60
    avail = _avail(2, T)
    avail[0, 30:] = False  # angle 0 dies at t=30
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    out = D.cut_director(tr, avail, [_mot(T), _mot(T)])
    assert out["n_cuts"] == 1
    assert out["segments"][1]["angle"] == 1
    assert out["segments"][1]["rule"] == "coverage"


def test_cluster_source_motion_wins_over_players(monkeypatch):
    """a0: high crowd score but near-zero motion; a1: the reverse.
    Motion source must pick a1 for the bulk; players source picks a0."""
    T = 120
    tr = [_track(T, cluster=10.0), _track(T, cluster=1.0)]
    mot = [np.full(T, 1.0), np.full(T, 3.0)]
    mot[0][:20] = 10.0  # >10% early burst drags a0's motion p90 up to 10

    def bulk_angle():
        secs = {}
        for s in out["segments"]:
            secs[s["angle"]] = secs.get(s["angle"], 0) + (
                s["t_end"] - s["t_start"])
        return secs

    monkeypatch.setattr(D, "CLUSTER_SOURCE", "motion")
    out = D.cut_director(tr, _avail(2, T), mot)
    assert max(bulk_angle(), key=bulk_angle().get) == 1

    monkeypatch.setattr(D, "CLUSTER_SOURCE", "players")
    out = D.cut_director(tr, _avail(2, T), mot)
    assert max(bulk_angle(), key=bulk_angle().get) == 0


def test_cluster_source_motion_does_not_mutate_track(monkeypatch):
    monkeypatch.setattr(D, "CLUSTER_SOURCE", "motion")
    T = 60
    tr = [_track(T, cluster=5.0), _track(T, cluster=2.0)]
    snapshot = [{k: v.copy() for k, v in t.items()} for t in tr]
    D.cut_director(tr, _avail(2, T), [np.full(T, 1.0), np.full(T, 2.0)])
    for t, snap in zip(tr, snapshot):
        assert t.keys() == snap.keys()
        for k in t:
            np.testing.assert_array_equal(t[k], snap[k])


def test_ratios_sum_to_one():
    T = 120
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    tr[1]["cluster"][60:] = 30.0
    out = D.cut_director(tr, _avail(2, T), [np.ones(T), np.ones(T)])
    assert abs(sum(out["ratios"].values()) - 1.0) < 1e-6
    assert "cuts_per_10min" in out and "median_hold_s" in out
