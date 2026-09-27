"""manual_direct: pick_candidate / suggest_stretch / director_rows / compare."""

from highlights.multiangle.manual_direct import compare, director_rows, pick_candidate, suggest_stretch

SEGS = [
    {"t_start": 0.0, "t_end": 5.0, "angle": 0, "rule": "start"},
    {"t_start": 5.0, "t_end": 9.0, "angle": 1, "rule": "zone"},
    {"t_start": 9.0, "t_end": 14.0, "angle": 2, "rule": "cluster"},
]


def test_pick_candidate_prefers_confirmed_goal():
    fused = {"candidates": [
        {"t": 50.0, "type": "goal", "status": "pending",
         "confidence": 0.99},
        {"t": 20.0, "type": "goal", "status": "confirmed",
         "confidence": 0.5},
        {"t": 10.0, "type": "shot", "status": "confirmed",
         "confidence": 0.9},
    ]}
    c = pick_candidate(fused, (0.0, 100.0))
    assert c["t"] == 20.0          # confirmed goal beats pending high-conf
    # events key works too
    assert pick_candidate({"events": fused["candidates"]},
                          (0.0, 100.0))["t"] == 20.0
    assert pick_candidate(fused, (30.0, 60.0))["t"] == 50.0
    assert pick_candidate(fused, (90.0, 95.0)) is None


def test_suggest_stretch_clamps():
    assert suggest_stretch(100.0, (0.0, 500.0)) == (55.0, 130.0)
    assert suggest_stretch(10.0, (0.0, 500.0)) == (0.0, 40.0)
    assert suggest_stretch(490.0, (0.0, 500.0)) == (445.0, 500.0)


def test_director_rows_and_compare():
    # range_lo=10: shared seconds 10..14 map onto the segments' 0..4
    rows = director_rows(SEGS, range_lo=10.0, t_start=10.0, t_end=19.0)
    assert [r["t"] for r in rows] == [10, 11, 12, 13, 14, 15, 16, 17, 18]
    assert [r["angle"] for r in rows] == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert rows[6]["rule"] == "zone"

    # user picked angle 0 the whole stretch (choice at t=10 fills all)
    cmp = compare({10: 0}, rows)
    assert cmp["n_seconds"] == 9
    assert cmp["agreement_pct"] == round(5 / 9 * 100, 1)
    assert cmp["user_switches"] == 0
    assert cmp["director_switches"] == 1   # 0 -> 1 at t=15
    assert cmp["disagree_by_rule"] == {"start": {"n": 0, "of": 5},
                                       "zone": {"n": 4, "of": 4}}
    assert cmp["disagree_by_pair"] == {"you:0->dir:1": 4}
    assert cmp["rows"][5]["agree"] is False
    assert cmp["rows"][0]["agree"] is True

    # a mid-stretch change of mind is a user switch and counts rows
    cmp2 = compare({10: 0, 15: 1}, rows)
    assert cmp2["agreement_pct"] == 100.0
    assert cmp2["user_switches"] == 1

    # no choices before the first -> those seconds don't count
    cmp3 = compare({16: 1}, rows)
    assert cmp3["n_seconds"] == 3
    assert cmp3["rows"][0]["user"] is None
