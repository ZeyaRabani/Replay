"""Manual ball zones in the director."""

import numpy as np

from highlights.multiangle.director import _in_poly, cut_director


def _track(T, cluster=1.0, ball_conf=0.0, ball_xy=(0.0, 0.0)):
    return {
        "ball_conf": np.full(T, ball_conf),
        "ball_size": np.full(T, ball_conf * 0.1),
        "ball_x": np.full(T, ball_xy[0]),
        "ball_y": np.full(T, ball_xy[1]),
        "cluster": np.full(T, cluster),
        "event": np.zeros(T),
    }


def _run(zones, T=300, zone_ok=None):
    avail = np.ones((2, T), dtype=bool)
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    # angle 1 sees the ball at (.2,.5) during t=100..200
    tr[1]["ball_conf"][100:201] = 0.8
    tr[1]["ball_size"][100:201] = 0.05
    tr[1]["ball_x"][100:201] = 0.2
    tr[1]["ball_y"][100:201] = 0.5
    motion = [np.ones(T), np.ones(T)]
    return cut_director(tr, avail, motion, zones=zones, zone_ok=zone_ok)


def test_zone_cut_overrides_cluster():
    zones = [[], [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]]]
    d = _run(zones)
    zone_segs = [s for s in d["segments"] if s["rule"] == "zone"]
    assert zone_segs, "expected zone segments"
    for s in zone_segs:
        assert s["angle"] == 1
    t0 = min(s["t_start"] for s in zone_segs)
    t1 = max(s["t_end"] for s in zone_segs)
    assert t0 <= 103            # cuts to the zoned angle quickly
    assert t1 >= 200            # plus linger (~203)
    assert d["zones_used"] is True
    assert d["ratios"].get("zone", 0) > 0
    # while a zone angle is eligible the director never uses the cluster rule
    for s in d["segments"]:
        if s["rule"] == "cluster":
            assert s["t_end"] <= t0 or s["t_start"] >= t1
    # outside the zone window the strong cluster camera holds
    assert d["segments"][0]["angle"] == 0


def test_zones_none_identical():
    a = _run(None)
    b = _run(zones=None)
    assert a["segments"] == b["segments"]
    assert a["zones_used"] is False


def test_zone_ok_suspends():
    zones = [[], [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]]]
    ok = np.ones((2, 300), dtype=bool)
    ok[1, 120:] = False          # camera moved at t=120
    d = _run(zones, zone_ok=ok)
    zone_segs = [s for s in d["segments"] if s["rule"] == "zone"]
    assert zone_segs, "expected zone segments before the camera moved"
    # no new zone cuts once the view check fails at t=120
    assert all(s["t_start"] <= 123 for s in zone_segs)


def test_in_poly():
    poly = [[0, 0], [1, 0], [1, 1], [0, 1]]
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]])
    assert _in_poly(pts, poly).tolist() == [True, False, False]
