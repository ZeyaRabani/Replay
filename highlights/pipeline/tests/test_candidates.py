import itertools

import numpy as np
import pandas as pd

from highlights.pipeline.candidates import make_candidates


def _frame(n=1200):
    t = np.arange(n, dtype=float)
    learned = np.zeros(n)
    for pt in range(100, n, 50):          # peaks every 50 s -> 22 peaks < 60
        learned[pt] = 0.8
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "t": t,
        "motion_goal_roi": rng.normal(size=n),
        "net_disturbance": rng.normal(size=n),
        "z300_rms": rng.normal(size=n),
        "z60_rms": rng.normal(size=n),
        "motion_total": 1.0 + rng.random(n),
    })
    in_match = (t >= 60) & (t <= 1140)
    return df, learned, np.zeros(n), in_match


def test_candidates_basic():
    df, learned, rule, in_match = _frame()
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    assert out["source"] == "pipeline"
    assert out["events"] is out["candidates"] or out["events"] == out["candidates"]
    evs = out["events"]
    assert 0 < len(evs) <= 60
    ts = sorted(e["t"] for e in evs)
    assert all(b - a >= 12 for a, b in itertools.pairwise(ts))  # NMS spacing
    assert all(e["confidence"] <= 0.9 for e in evs)
    assert all(e["status"] == "pending" for e in evs)
    assert all(e["cross_validation"] == "pipeline_only" for e in evs)
    assert all(e["id"] == f"event_{e['rank']:03d}" for e in evs)
    assert all(e["t_start"] <= e["t"] <= e["t_end"] for e in evs)
    assert out["match_window"] == [60.0, 1140.0]


def test_shot_rule_on_goal_roi_spike():
    df, learned, rule, in_match = _frame()
    # strong goal-ROI spike under one of the peaks at t=200
    df.loc[df["t"] == 200, "motion_goal_roi"] = 1e6
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    shot = [e for e in out["events"] if abs(e["t"] - 200) <= 1]
    assert shot and shot[0]["type"] == "shot"
    assert shot[0]["signals"]["motion_goal_roi_z"] > 2
    assert any(e["type"] == "attack" for e in out["events"])


def _ev(out, t):
    return [e for e in out["events"] if abs(e["t"] - t) <= 1]


def test_goal_rule_needs_net_crowd_and_lull():
    df, learned, rule, in_match = _frame()
    # goal-ROI + net + crowd spikes at the t=200 peak, motion lull after it
    df.loc[df["t"] == 200, ["motion_goal_roi", "net_disturbance", "z60_rms"]] = 1e6
    df.loc[(df["t"] >= 215) & (df["t"] < 245), "motion_total"] = 0.01
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    e = _ev(out, 200)[0]
    assert e["type"] == "goal"
    assert e["signals"]["restart_lull"] is True
    assert e["signals"]["net_disturbance_z"] >= 2
    assert "goal" in e["notes"]
    # without the lull the same spikes classify as a shot
    df.loc[(df["t"] >= 215) & (df["t"] < 245), "motion_total"] = 5.0
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    assert _ev(out, 200)[0]["type"] == "shot"


def test_goalmouth_rule():
    df, learned, rule, in_match = _frame()
    # roi z between 1 and 2: modest positive spike at the t=200 peak
    v = df["motion_goal_roi"]
    med = v[in_match].median()
    mad = (v[in_match] - med).abs().median()
    scale = mad * 1.4826
    df.loc[df["t"] == 200, "motion_goal_roi"] = med + 1.5 * scale
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    e = _ev(out, 200)[0]
    assert e["type"] == "goalmouth"
    assert 1.0 <= e["signals"]["motion_goal_roi_z"] <= 2.0


def test_crowd_rule():
    df, learned, rule, in_match = _frame()
    # big audio spike, no goal-ROI motion
    df.loc[df["t"] == 200, "z60_rms"] = 1e6
    df.loc[df["t"] == 200, "motion_goal_roi"] = df["motion_goal_roi"].median()
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    e = _ev(out, 200)[0]
    assert e["type"] == "crowd"
    assert e["signals"]["z60_rms_z"] >= 2.0


def test_attack_is_default():
    df, learned, rule, in_match = _frame()
    # keep every signal at baseline so nothing else fires
    df.loc[df["t"] == 200, ["motion_goal_roi", "z60_rms"]] = \
        df[["motion_goal_roi", "z60_rms"]].median()
    out = make_candidates(df, learned, rule, in_match, 1200.0)
    e = _ev(out, 200)[0]
    assert e["type"] == "attack"


def _frame_long(n, peaks, in_match, prob=0.8):
    t = np.arange(n, dtype=float)
    learned = np.zeros(n)
    for pt in peaks:
        learned[pt] = prob
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "t": t,
        "motion_goal_roi": rng.normal(size=n),
        "net_disturbance": rng.normal(size=n),
        "z300_rms": rng.normal(size=n),
        "z60_rms": rng.normal(size=n),
        "motion_total": 1.0 + rng.random(n),
    })
    return df, learned, np.zeros(n), in_match


def test_cap_scales_with_90min_match():
    n = 5600
    in_match = np.zeros(n, dtype=bool)
    in_match[60:5460] = True                      # 90 min window -> cap 63
    peaks = np.arange(100, 5400, 75)              # ~71 peaks > 63
    df, learned, rule, im = _frame_long(n, peaks, in_match)
    out = make_candidates(df, learned, rule, im, float(n))
    assert len(out["events"]) == 63


def test_cap_scales_with_40min_match():
    n = 2500
    in_match = np.zeros(n, dtype=bool)
    in_match[60:2460] = True                      # 40 min -> cap 28
    peaks = np.arange(100, 2400, 60)              # ~38 peaks > 28
    df, learned, rule, im = _frame_long(n, peaks, in_match)
    out = make_candidates(df, learned, rule, im, float(n))
    assert len(out["events"]) == 28


def test_weak_peaks_trimmed_to_min_n():
    n = 5600
    in_match = np.zeros(n, dtype=bool)
    in_match[60:5460] = True                      # 90 min -> cap 63
    peaks = list(range(200, 200 + 5 * 30, 30)) + \
        list(range(700, 5400, 75))                # 5 strong + many weak
    t = np.arange(n, dtype=float)
    learned = np.full(n, 0.01)
    for pt in peaks[:5]:
        learned[pt] = 0.9
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "t": t,
        "motion_goal_roi": rng.normal(size=n),
        "net_disturbance": rng.normal(size=n),
        "z300_rms": rng.normal(size=n),
        "z60_rms": rng.normal(size=n),
        "motion_total": 1.0 + rng.random(n),
    })
    out = make_candidates(df, learned, np.zeros(n), in_match, float(n))
    assert len(out["events"]) == 15   # MIN_N floor, not 63
