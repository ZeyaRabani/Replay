"""Event rule: an available angle whose event channel is > 0 wins outright;
the cut lands at the event second, raw confidence score, no margin/confirm/
hold, and normal rules resume afterwards."""

import numpy as np

from highlights.multiangle.director import cut_director, per_second


def _track(T, cluster, event=None):
    t = {"ball_conf": np.zeros(T), "ball_size": np.zeros(T),
         "cluster": np.asarray(cluster, dtype=float)}
    if event is not None:
        t["event"] = event
    return t


def _three_angles(T=200):
    """3 angles, all available; normalised cluster favours angle 2.

    Constant series all normalise to 1.0 under the p90 baseline, so a0/a1
    get a >10% early high period that drags their p90 up — their typical
    normalised score then sits ~0.5 while a2's sits at 1.0."""
    c0 = np.ones(T); c0[:25] = 3.0
    c1 = np.ones(T) * 2; c1[:25] = 5.0
    ev0 = np.zeros(T)
    ev0[100:110] = 0.9            # a0 sees a shot over [100, 109]
    tracks = [_track(T, c0, ev0), _track(T, c1), _track(T, np.full(T, 9.0))]
    avail = np.ones((3, T), dtype=bool)
    motion = [np.zeros(T) for _ in range(3)]
    return tracks, avail, motion


def test_event_rule_overrides_cluster():
    tracks, avail, motion = _three_angles()
    best_a, _, best_r, S, _, _, _ = per_second(tracks, avail)
    assert best_r[105] == 3 and best_a[105] == 0
    assert S[0, 105] == 0.9

    out = cut_director(tracks, avail, motion)
    segs = out["segments"]
    # cut lands exactly at t=100 on angle 0 with rule "event"
    seg = [s for s in segs if s["t_start"] == 100.0 and s["angle"] == 0]
    assert seg, [s for s in segs if s["angle"] == 0]
    assert seg[0]["rule"] == "event"
    assert seg[0]["runner_up"]["angle"] == 2
    # event seconds counted under their own key
    assert out["per_second_rule"]["event"] >= 1
    # back on angle 2 by ~140 (event ends ~109, MIN_HOLD 20 + cluster confirm)
    assert segs[-1]["angle"] == 2
    assert segs[-1]["t_start"] <= 140.0


def test_no_event_channel_same_behaviour():
    tracks, avail, motion = _three_angles()
    for t in tracks:
        t.pop("event", None)
    out = cut_director(tracks, avail, motion)
    # cluster rule eventually settles on angle 2; no event seconds
    assert out["segments"][-1]["angle"] == 2
    assert all(s["rule"] != "event" for s in out["segments"])
    assert out["per_second_rule"]["event"] == 0
