"""compute_match_stats over a synthetic teams dict."""

from __future__ import annotations

from highlights.analysis.stats import compute_match_stats, density_peak_x

COLS = ["t_shared", "ball_x", "ball_y", "ball_conf",
        "teamA_xs", "teamB_xs", "teamA_n", "teamB_n"]


def _teams_dict(quiet_s=None, quiet_e=None, n=600):
    """600 s window. Half 1: A xs ~0.3, B ~0.7, ball rides with A.
    Half 2: swapped. Optional dead stretch [quiet_s, quiet_e] with
    zero players."""
    rows = []
    for s in range(n):
        if quiet_s is not None and quiet_s <= s <= quiet_e:
            a, b = [], []
            bx, bc = 0.0, 0.0
        elif s < 300:
            a, b = [0.28, 0.30, 0.32], [0.68, 0.70, 0.72]
            bx, bc = 0.30 + (s % 5) * 0.001, 0.9
        else:
            a, b = [0.68, 0.70, 0.72], [0.28, 0.30, 0.32]
            bx, bc = 0.28 + (s % 5) * 0.001, 0.9
        # shot candidate at t=280: ball travels +0.2 in x over [276, 280]
        if 276 <= s <= 280:
            bx = 0.30 + (s - 276) * 0.05
        rows.append([s, bx, 0.5, bc, a, b, len(a), len(b)])
    return {"columns": COLS, "rows": rows,
            "teams": {"A": {"name": "orange", "hex": "#ff6400"},
                      "B": {"name": "white", "hex": "#ffffff"}},
            "team_confidence": 0.9}


def _stats(**kw):
    cands = kw.pop("candidates", [])
    teams = kw.pop("teams", None) or _teams_dict(**kw)
    return compute_match_stats(
        teams, None, cands,
        match_window=(0.0, 600.0), lo_out=0.0,
        offsets=[0.0, -100.0, 50.0], ref_angle=0)


def test_possession_and_ends():
    st = _stats()
    h1 = st["halves"][0]["teams"]
    assert h1["A"]["possession_pct"] >= 90.0
    assert h1["A"]["attacks_right"] is True
    assert h1["B"]["attacks_right"] is False
    h2 = st["halves"][1]["teams"]
    assert h2["A"]["attacks_right"] is False
    assert st["ends_swapped"] is True
    # midpoint split when there is no quiet stretch
    assert st["halves"][1]["start"] == 300.0
    assert any("midpoint" in c for c in st["caveats"])


def test_quiet_stretch_split():
    st = _stats(quiet_s=260, quiet_e=360)
    split = st["halves"][1]["start"]
    assert 290.0 <= split <= 330.0
    assert any("half-time break" in c for c in st["caveats"])


def test_confirmed_events_only_in_headline():
    """Headline shots/goals count confirmed candidates only; pending and
    rejected land in the unreviewed list (top 10 by confidence)."""
    cands = [
        {"t_shared": 280.0, "type": "shot", "status": "pending",
         "confidence": 0.7, "id": "c-pend"},
        {"t_shared": 100.0, "type": "goal", "status": "confirmed",
         "confidence": 0.95, "id": "c-goal"},
        {"t_shared": 400.0, "type": "goal", "status": "rejected",
         "confidence": 0.9, "id": "c-rej"},
        {"t_shared": 450.0, "type": "goal", "status": "pending",
         "confidence": 0.85, "id": "c-pgoal"},
        {"t_shared": 200.0, "type": "shot", "status": "confirmed",
         "confidence": 0.6, "id": "c-shot"},
    ]
    st = _stats(candidates=cands)
    assert len(st["events"]) == 2      # the two confirmed only
    assert {e["candidate_id"] for e in st["events"]} == {"c-goal", "c-shot"}
    assert all(e["status"] == "confirmed" for e in st["events"])
    kinds = [e["kind"] for e in st["events"]]
    assert kinds.count("goal") == 1 and kinds.count("shot") == 1
    # goal at t=100 (half 1): ball rides with A at x~0.3 -> left third,
    # which is A's own-goal side -> attributed to B
    goal = next(e for e in st["events"] if e["kind"] == "goal")
    assert goal["team"] == "B" and goal["attribution"] == "high"
    # headline numbers come from events only
    assert st["totals"]["A"]["goals"] == 0
    assert st["totals"]["B"]["goals"] == 1
    assert st["totals"]["B"]["shots"] == 1   # shot at 200: x~0.3 -> B
    # unreviewed: pending only (rejected excluded), sorted by confidence
    assert st["n_unreviewed"] == 2
    assert [u["candidate_id"] for u in st["unreviewed"]] == [
        "c-pgoal", "c-pend"]
    assert all(u["status"] == "pending" for u in st["unreviewed"])


def test_territory_and_momentum():
    """Half 1: ball sits with A at x~0.3 (A's defensive third = B's
    attacking third)."""
    st = _stats()
    h1, h2 = st["halves"]
    tA, tB = h1["teams"]["A"]["territory"], h1["teams"]["B"]["territory"]
    assert tA is not None and tB is not None
    assert tA["def"] > 0.9            # A defends the left in half 1
    assert tB["att"] > 0.9            # B attacks the left in half 1
    tA2 = h2["teams"]["A"]["territory"]
    assert tA2["att"] > 0.9           # swapped: ball stays at ~0.3 -> A att
    # momentum: half 1 ball in B's attacking half -> negative
    m = st["momentum"]
    assert len(m) == 2                # 600 s window -> two 5-min bins
    assert m[0]["value"] is not None and m[0]["value"] <= -0.9
    assert m[1]["value"] is not None and m[1]["value"] >= 0.9
    assert m[0]["t_start_shared"] == 0.0
    assert m[0]["n"] > 0


def test_density_peak_x():
    # cluster at ~0.15 with two stragglers -> peak at the cluster
    xs = [0.14, 0.15, 0.15, 0.16, 0.15, 0.14, 0.80, 0.82]
    px = density_peak_x(xs)
    assert px is not None and abs(px - 0.15) < 0.05
    # too few players -> no estimate
    assert density_peak_x([0.2, 0.5, 0.7]) is None
    assert density_peak_x([]) is None


def test_play_x_fallback_when_ball_invisible():
    """ball_conf 0 everywhere: territory/momentum still work off player
    positions; possession is gated to null with a caveat."""
    teams = _teams_dict()
    for r in teams["rows"]:
        r[3] = 0.0          # ball never confidently tracked
    st = _stats(teams=teams)
    h1 = st["halves"][0]["teams"]
    assert h1["A"]["possession_pct"] is None
    assert h1["B"]["possession_pct"] is None
    assert st["totals"]["A"]["possession_pct"] is None
    # territory still populated from player positions — the symmetric
    # player centroid sits mid-pitch in this fixture
    tA = h1["A"]["territory"]
    assert tA is not None and tA["mid"] > 0.9
    tB = h1["B"]["territory"]
    assert tB is not None and tB["mid"] > 0.9
    # momentum bins still carry values: the two equal density peaks
    # (0.325/0.725) tie-break to 0.525 -> A's attacking half -> +1
    m = st["momentum"]
    assert m[0]["value"] is not None and abs(m[0]["value"]) == 1.0
    assert m[0]["n"] > 0
    assert any("ball rarely visible" in c for c in st["caveats"])
    assert any("territory/momentum estimated" in c for c in st["caveats"])


def test_low_confidence_caveat():
    teams = _teams_dict()
    teams["team_confidence"] = 0.3
    st = _stats(teams=teams)
    assert any("confidence is low" in c for c in st["caveats"])
    assert any("not official" in c for c in st["caveats"])
