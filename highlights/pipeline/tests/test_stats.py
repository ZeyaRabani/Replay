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
