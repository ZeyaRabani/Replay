"""Contract 5 stats computation unit tests."""

import json
import math

import pandas as pd

from highlights.app.backend.stats import compute_stats

KEYS = {
    "duration_s", "match_window", "halves", "bin_s", "timeline",
    "events_by_type", "events_per_10min", "top_moments", "whistles",
    "activity", "pipeline", "match_stats",
}


def _df():
    t = list(range(600))
    return pd.DataFrame({
        "t": t,
        "motion_total": [math.sin(x / 30.0) for x in t],
        "rms_db": [x / 600.0 for x in t],
    })


def _events():
    return [
        {"type": "goal", "t": 100.0, "confidence": 0.9, "cross_validation": "confirmed", "notes": "bang"},
        {"type": "shot", "t": 105.0, "confidence": 0.5, "cross_validation": "pipeline_only", "notes": ""},
        {"type": "shot", "t": 300.0, "confidence": 0.6, "cross_validation": "visual_only", "notes": ""},
        {"type": "chance", "t": 400.0, "confidence": 0.3, "cross_validation": "rejected", "notes": "no"},
    ]


def test_compute_stats_shape():
    s = compute_stats(_df(), _events(), 600.0, [60.0, 540.0],
                      [{"start": 0, "end": 270}], [12.345, 99.0],
                      model="m", auroc_reference=0.5, notes="n")
    assert set(s.keys()) == KEYS
    assert s["match_stats"]["goals"] == 1
    assert len(s["timeline"]) == 20
    for row in s["timeline"]:
        assert 0 <= row["motion"] <= 1
        assert 0 <= row["audio"] <= 1
        assert 0 <= row["excitement"] <= 1
        assert isinstance(row["t"], int)
    json.dumps(s)  # JSON round-trips, no numpy types


def test_events_exclude_rejected():
    s = compute_stats(_df(), _events(), 600.0, None, None, None)
    # rejected chance at t=400 not counted anywhere
    assert sum(r["events"] for r in s["timeline"]) == 3
    assert s["events_by_type"] == {"goal": 1, "shot": 2}
    # events_per_10min: 600s -> 1 bin covering all
    assert len(s["events_per_10min"]) == 1
    row = s["events_per_10min"][0]
    assert row["goal"] == 1 and row["shot"] == 2 and row["chance"] == 0


def test_top_moments_sorted_no_rejected():
    s = compute_stats(_df(), _events(), 600.0, None, None, None)
    tm = s["top_moments"]
    assert len(tm) == 3
    confs = [m["confidence"] for m in tm]
    assert confs == sorted(confs, reverse=True)
    assert all(m["type"] != "chance" for m in tm)
    assert tm[0]["reason"] == "bang"
    assert tm[1]["reason"]  # fallback reason is non-empty


def test_quietest_and_activity():
    s = compute_stats(_df(), _events(), 600.0, [60.0, 540.0], None, [1.111],
                      model="m", auroc_reference=0.5)
    act = s["activity"]
    assert 0 <= act["mean_motion"] <= 1
    qs = act["quietest_stretch"]
    assert qs[1] > qs[0]
    assert qs[0] >= 0 and qs[1] <= 600
    assert s["whistles"] == [1.11]
    assert s["pipeline"]["model"] == "m"
    assert s["pipeline"]["auroc_reference"] == 0.5
