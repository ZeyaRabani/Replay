"""build_summary determinism + shape."""

from __future__ import annotations

import re

from highlights.analysis.summary import build_summary

STATS = {
    "teams": {"A": {"name": "orange", "hex": "#ff6400"},
              "B": {"name": "white", "hex": "#ffffff"}},
    "team_confidence": 0.85,
    "offsets": [0.0, -100.0],
    "ref_angle": 0,
    "lo_out": 0.0,
    "halves": [
        {"index": 1, "start": 0.0, "end": 300.0, "start_out": 0.0,
         "start_file": 0.0,
         "teams": {"A": {"possession_pct": 62.0, "attacking_third_s": 80,
                         "shots": 3, "goals_confirmed": 1,
                         "goals_estimated": 2, "attacks_right": True},
                   "B": {"possession_pct": 38.0, "attacking_third_s": 20,
                         "shots": 1, "goals_confirmed": 0,
                         "goals_estimated": 0, "attacks_right": False}}},
        {"index": 2, "start": 300.0, "end": 600.0, "start_out": 300.0,
         "start_file": 300.0,
         "teams": {"A": {"possession_pct": 51.0, "attacking_third_s": 40,
                         "shots": 1, "goals_confirmed": 0,
                         "goals_estimated": 0, "attacks_right": False},
                   "B": {"possession_pct": 49.0, "attacking_third_s": 50,
                         "shots": 2, "goals_confirmed": 0,
                         "goals_estimated": 1, "attacks_right": True}}},
    ],
    "totals": {
        "A": {"possession_pct": 56.0, "attacking_third_s": 120, "shots": 4,
              "goals_confirmed": 1, "goals_estimated": 2,
              "distance_m_est": 3200.0, "sprints": 12},
        "B": {"possession_pct": 44.0, "attacking_third_s": 70, "shots": 3,
              "goals_confirmed": 0, "goals_estimated": 1,
              "distance_m_est": 2800.0, "sprints": 9},
    },
    "shots": [
        {"t_shared": 100.0, "t_out": 100.0, "t_file": 100.0, "type": "goal",
         "confidence": 0.95, "status": "confirmed", "team": "A", "half": 1},
        {"t_shared": 200.0, "t_out": 200.0, "t_file": 200.0, "type": "shot",
         "confidence": 0.8, "status": "pending", "team": "A", "half": 1},
        {"t_shared": 450.0, "t_out": 450.0, "t_file": 450.0, "type": "shot",
         "confidence": 0.7, "status": "pending", "team": "B", "half": 2},
    ],
    "caveats": ["Estimated from footage — not official statistics",
                "halves split at the midpoint of the match window"],
}


def test_summary_deterministic_and_shape():
    s1 = build_summary(STATS)
    s2 = build_summary(STATS)
    assert s1["text"] == s2["text"]
    n_sent = len([x for x in re.split(r"(?<=[.!?])\s+", s1["text"]) if x])
    assert 6 <= n_sent <= 12
    assert "orange" in s1["text"].lower() and "white" in s1["text"].lower()
    assert re.search(r"\d+:\d{2}", s1["text"])
    assert s1["bullets"]
    assert any("goal" in b["label"] for b in s1["bullets"])
    assert any("half-time" in b["label"] for b in s1["bullets"])
