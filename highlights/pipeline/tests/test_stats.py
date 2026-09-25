import numpy as np
import pandas as pd

from highlights.pipeline.stats import compute_stats


def _frame(n=600):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "t": np.arange(n, dtype=float),
        "motion_total": rng.random(n),
        "rms_db": -30 + 10 * rng.random(n),
        "z300_speech": rng.normal(size=n),
        "motion_goal_roi": rng.random(n),
    })


def test_compute_stats_shape():
    events = [
        {"type": "goal", "t": 100.0, "confidence": 0.99, "notes": "net"},
        {"type": "shot", "t": 250.0, "confidence": 0.7},
        {"type": "chance", "t": 400.0, "confidence": 0.4, "cross_validation": "rejected"},
    ]
    s = compute_stats(_frame(), events, 600.0, [60, 540], [{"start": 60, "end": 300}], [12.5, 301.0])
    assert s["duration_s"] == 600.0
    assert s["match_window"] == [60.0, 540.0]
    assert s["bin_s"] == 30
    assert len(s["timeline"]) == 20
    row = s["timeline"][3]
    assert set(row) == {"t", "motion", "audio", "excitement", "events"}
    assert row["events"] == 1  # goal at 100 s falls in bin [90, 120)
    for r in s["timeline"]:
        for k in ("motion", "audio", "excitement"):
            assert 0.0 <= r[k] <= 1.0
    assert s["events_by_type"] == {"goal": 1, "shot": 1}
    assert len(s["events_per_10min"]) == 1
    assert s["events_per_10min"][0]["goal"] == 1
    assert s["top_moments"][0]["type"] == "goal"
    assert s["top_moments"][0]["reason"] == "net"
    assert s["whistles"] == [12.5, 301.0]
    assert s["halves"] == [{"start": 60.0, "end": 300.0}]
    assert set(s["activity"]) == {"mean_motion", "peak_motion_t", "loudest_t", "quietest_stretch"}
    assert s["pipeline"]["model"] == "audio_motion_lr v1"


def test_compute_stats_handles_missing_columns_and_window():
    df = pd.DataFrame({"t": np.arange(100, dtype=float)})
    s = compute_stats(df, [], 100.0, None, [], None)
    assert s["match_window"] == [0.0, 100.0]
    assert len(s["timeline"]) == 4
    assert s["events_by_type"] == {}
    assert s["top_moments"] == []
    assert s["match_stats"]["territory"] == {"near_goal_pct": 50.0,
                                           "far_goal_pct": 50.0}
    assert s["match_stats"]["halves"] == []


def test_match_stats_block():
    n = 600
    t = np.arange(n, dtype=float)
    motion = np.full(n, 5.0)
    motion[100:160] = 0.0     # 60 s quiet stretch -> low motion_n
    df = pd.DataFrame({
        "t": t,
        "motion_total": motion,
        "near_frac": np.full(n, 0.3),
    })
    events = [
        {"type": "goal", "t": 120.0, "confidence": 0.9},
        {"type": "shot", "t": 200.0, "confidence": 0.8},
        {"type": "attack", "t": 320.0, "confidence": 0.5},
        {"type": "attack", "t": 325.0, "confidence": 0.5},
        {"type": "crowd", "t": 500.0, "confidence": 0.6},
        {"type": "shot", "t": 590.0, "confidence": 0.4,
         "cross_validation": "rejected"},
    ]
    halves = [{"start": 60, "end": 290}, {"start": 310, "end": 540}]
    s = compute_stats(df, events, 600.0, [60, 540], halves, [65.0, 300.0])
    ms = s["match_stats"]
    assert ms["goals"] == 1
    assert ms["shots_on_goal"] == 2  # goal + shot; the rejected shot excluded
    assert ms["attacks"] == 2
    assert ms["crowd_reactions"] == 1
    assert ms["big_moments"] == 2
    assert ms["territory"]["near_goal_pct"] == 30.0
    assert ms["territory"]["far_goal_pct"] == 70.0
    assert ms["stoppages"]["whistles"] == 2
    assert ms["stoppages"]["quiet_stretches"] >= 1
    assert len(ms["halves"]) == 2
    assert ms["halves"][0]["goals"] == 1
    assert ms["halves"][1]["attacks"] == 2
    assert ms["peak_minute"]["events"] == 2  # t=300,305 in the same minute bin
    assert "single camera" in ms["basis"]
