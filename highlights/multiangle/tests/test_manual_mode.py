"""manual mode: apply_manual overlays + manual_summary stats."""

import itertools

import numpy as np

from highlights.multiangle.manual_direct import apply_manual, manual_summary

T = 30
LO = 100.0
SEGS = [
    {"t_start": 0.0, "t_end": 10.0, "angle": 0, "rule": "start",
     "score": 0.5, "runner_up": None},
    {"t_start": 10.0, "t_end": 20.0, "angle": 1, "rule": "ball",
     "score": 0.8, "runner_up": {"angle": 0, "score": 0.3}},
    {"t_start": 20.0, "t_end": 30.0, "angle": 0, "rule": "cluster",
     "score": 0.6, "runner_up": {"angle": 1, "score": 0.2}},
]


def _assert_contiguous(out):
    assert out[0]["t_start"] == 0.0
    assert out[-1]["t_end"] == float(T)
    for a, b in itertools.pairwise(out):
        assert a["t_end"] == b["t_start"]


def test_no_choices_is_identity():
    out = apply_manual(SEGS, {}, LO, T)
    assert out == SEGS


def test_override_splits_segments_and_keeps_ai_fields():
    out = apply_manual(SEGS, {105: 2, 106: 2, 107: 2}, LO, T)
    _assert_contiguous(out)
    manual = [s for s in out if s["rule"] == "manual"]
    assert len(manual) == 1
    assert (manual[0]["t_start"], manual[0]["t_end"],
            manual[0]["angle"]) == (5.0, 8.0, 2)
    assert manual[0]["score"] is None and manual[0]["runner_up"] is None
    # the AI segment around the override is split, rule/score kept
    around = [s for s in out if s["rule"] == "start"]
    assert [(s["t_start"], s["t_end"]) for s in around] == [(0.0, 5.0),
                                                          (8.0, 10.0)]
    assert all(s["angle"] == 0 and s["score"] == 0.5 for s in around)


def test_unavailable_angle_choice_ignored():
    avail = np.ones((3, T), dtype=bool)
    avail[2, :] = False
    out = apply_manual(SEGS, {105: 2, 106: 2}, LO, T, avail)
    assert out == SEGS
    # but it applies when the angle IS available
    avail[2, 4:8] = True
    out = apply_manual(SEGS, {105: 2}, LO, T, avail)
    assert [s["angle"] for s in out if s["rule"] == "manual"] == [2]


def test_out_of_range_choice_ignored():
    out = apply_manual(SEGS, {90: 2, 500: 1}, LO, T)
    assert out == SEGS


def test_second_before_first_keeps_ai():
    # shared second 100 -> i=0; choice at i=0 overrides the AI start
    out = apply_manual(SEGS, {100: 1}, LO, T)
    assert out[0]["rule"] == "manual" and out[0]["angle"] == 1
    assert (out[0]["t_start"], out[0]["t_end"]) == (0.0, 1.0)


def test_manual_summary_stats():
    out = apply_manual(SEGS, {105: 2, 106: 2, 107: 2}, LO, T)
    s = manual_summary(out, T)
    assert s["n_cuts"] == len(out) - 1
    assert abs(sum(s["ratios"].values()) - 1.0) < 1e-3
    assert s["ratios"]["manual"] == 0.1  # 3 of 30 seconds
    assert s["per_second_rule"]["manual"] == 3
    assert abs(sum(s["angle_share"].values()) - 1.0) < 1e-3
    assert s["mean_hold_s"] > 0 and s["median_hold_s"] > 0
    assert s["cuts_per_10min"] > 0


def test_fractional_range_lo_keys():
    # choices keyed by shared float seconds: 1151.8 -> i=0, 1152.8 -> i=1
    out = apply_manual(SEGS, {1151.8: 2, 1152.8: 2}, 1151.8, T)
    assert out[0]["rule"] == "manual" and out[0]["angle"] == 2
    assert (out[0]["t_start"], out[0]["t_end"]) == (0.0, 2.0)
    assert out[1]["rule"] == "start"
