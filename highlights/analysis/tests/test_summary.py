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
                         "shots": 3, "goals": 1,
                         "territory": {"def": 0.3, "mid": 0.4, "att": 0.3},
                         "attacks_right": True},
                   "B": {"possession_pct": 38.0, "attacking_third_s": 20,
                         "shots": 1, "goals": 0,
                         "territory": {"def": 0.5, "mid": 0.4, "att": 0.1},
                         "attacks_right": False}}},
        {"index": 2, "start": 300.0, "end": 600.0, "start_out": 300.0,
         "start_file": 300.0,
         "teams": {"A": {"possession_pct": 51.0, "attacking_third_s": 40,
                         "shots": 1, "goals": 0,
                         "territory": {"def": 0.4, "mid": 0.4, "att": 0.2},
                         "attacks_right": False},
                   "B": {"possession_pct": 49.0, "attacking_third_s": 50,
                         "shots": 2, "goals": 0,
                         "territory": {"def": 0.2, "mid": 0.4, "att": 0.4},
                         "attacks_right": True}}},
    ],
    "totals": {
        "A": {"possession_pct": 56.0, "attacking_third_s": 120, "shots": 4,
              "goals": 1,
              "distance_m_est": 3200.0, "sprints": 12},
        "B": {"possession_pct": 44.0, "attacking_third_s": 70, "shots": 3,
              "goals": 0,
              "distance_m_est": 2800.0, "sprints": 9},
    },
    "events": [
        {"t_shared": 100.0, "t_out": 100.0, "t_file": 100.0, "mmss": "1:40",
         "kind": "goal", "type": "goal",
         "confidence": 0.95, "status": "confirmed", "team": "A",
         "attribution": "high", "candidate_id": "c1", "half": 1},
        {"t_shared": 200.0, "t_out": 200.0, "t_file": 200.0, "mmss": "3:20",
         "kind": "shot", "type": "shot",
         "confidence": 0.8, "status": "confirmed", "team": "A",
         "attribution": "high", "candidate_id": "c2", "half": 1},
    ],
    "unreviewed": [],
    "n_unreviewed": 3,
    "momentum": [
        {"t_start_shared": 0.0, "t_start_out": 0.0, "t_start_file": 0.0,
         "value": 0.4, "n": 250},
        {"t_start_shared": 300.0, "t_start_out": 300.0,
         "t_start_file": 300.0, "value": -0.6, "n": 240},
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
    # score line + momentum swing come from the confirmed data
    assert "1 - 0" in s1["text"]
    assert "spell of pressure" in s1["text"]


def test_summary_zero_confirmed_goals():
    st = {**STATS, "events": [],
          "totals": {"A": {**STATS["totals"]["A"], "goals": 0},
                     "B": {**STATS["totals"]["B"], "goals": 0}}}
    s = build_summary(st)
    assert "no goals confirmed yet" in s["text"].lower()
    assert " - " not in s["text"].split("confirmed")[0]  # no fake scoreline
    assert not any("goal" in b["label"] for b in s["bullets"])
