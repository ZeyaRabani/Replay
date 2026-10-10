"""Director learned prefs: event_rule, angle_remap, angle_weight."""

import numpy as np

from highlights.multiangle import director as D
from highlights.multiangle.manual_direct import learn_prefs


def _track(T: int, ball_conf=0.0, ball_size=0.0, cluster=0.0, event=0.0):
    return {"ball_conf": np.full(T, ball_conf),
            "ball_size": np.full(T, ball_size),
            "cluster": np.full(T, cluster),
            "event": np.full(T, event)}


def _avail(n: int, T: int, off: int | None = None):
    a = np.ones((n, T), dtype=bool)
    if off is not None:
        a[off] = False
    return a


def test_event_rule_disabled():
    """prefs.event_rule=False: no second may resolve as rule 'event'."""
    T = 30
    tr = [_track(T, cluster=5.0), _track(T, event=0.9)]
    _a, _s, r, _S, _b, _zs, _zb = D.per_second(tr, _avail(2, T))
    assert (r == 3).any()                     # default: event fires
    _a, _s, r, _S, _b, _zs, _zb = D.per_second(
        tr, _avail(2, T), prefs={"event_rule": False})
    assert not (r == 3).any()
    out = D.cut_director(tr, _avail(2, T), [np.zeros(T), np.zeros(T)],
                         prefs={"event_rule": False})
    assert out["per_second_rule"]["event"] == 0
    assert out["prefs"]["event_rule"] is False


def test_angle_remap_applies_when_available():
    """remap cluster 1->0: angle 0 wins whenever it is available."""
    T = 30
    # angle1 cluster-leads every second; remap sends the leader to 0
    tr = [_track(T, cluster=2.0), _track(T, cluster=8.0)]
    p = {"angle_remap": {"cluster": {"1": 0}}}
    a, _s, _r, _S, _b, _zs, _zb = D.per_second(tr, _avail(2, T), prefs=p)
    assert (a == 0).all()
    # when 0 is off-camera the remap is skipped and 1 stays
    a, _s, _r, _S, _b, _zs, _zb = D.per_second(
        tr, _avail(2, T, off=0), prefs=p)
    assert (a == 1).all()


def test_learn_prefs_learns_zone_remap():
    """User always picks 1 where the zone rule says 2 -> remap zone 2->1
    and agreement can only improve."""
    T, lo = 30, 0.0
    sessions = [{"t_start": 0.0, "t_end": 10.0,
                 "choices": [{"t": 0, "angle": 1}]}]

    def cand(overrides, prefs):
        a = np.full(T, 2, dtype=int)
        r = np.full(T, 4, dtype=int)          # zone picks angle 2
        if prefs and prefs.get("angle_remap", {}).get("zone", {}).get("2"):
            a[:] = int(prefs["angle_remap"]["zone"]["2"])
        return a, r

    def replay(overrides, prefs=None):
        ang = 2
        if prefs:
            ang = int(prefs.get("angle_remap", {}).get("zone", {})
                      .get("2", 2))
        return [{"t_start": 0.0, "t_end": float(T), "angle": ang,
                 "rule": "zone"}]

    res = learn_prefs(sessions, replay, cand, range_lo=lo, n_angles=3)
    assert res["prefs"]["angle_remap"]["zone"]["2"] == 1
    assert res["votes"]["zone"]["2"]["1"] == 10
    after = res["agreement_pct_after"] or 0
    before = res["agreement_pct_before"] or 0
    assert after >= before
    assert res["n_sessions"] == 1
