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
    assert any(e["type"] == "chance" for e in out["events"])
