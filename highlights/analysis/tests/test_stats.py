"""compute_match_stats over a synthetic teams dict."""

from __future__ import annotations

from highlights.analysis.stats import compute_match_stats

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


def test_shot_attribution_and_goals():
    cands = [
        {"t_shared": 280.0, "type": "shot", "status": "pending",
         "confidence": 0.7},
        {"t_shared": 100.0, "type": "goal", "status": "confirmed",
         "confidence": 0.95},
        {"t_shared": 400.0, "type": "goal", "status": "rejected",
         "confidence": 0.9},
        {"t_shared": 450.0, "type": "goal", "status": "pending",
         "confidence": 0.85},
    ]
    st = _stats(candidates=cands)
    shot = next(s for s in st["shots"] if s["type"] == "shot")
    assert shot["team"] == "A"          # ball travelled right in half 1
    assert shot["half"] == 1
    assert shot["t_file"] == 280.0      # ref offset 0
    h1 = st["halves"][0]["teams"]
    confirmed = h1["A"]["goals_confirmed"] + h1["B"]["goals_confirmed"]
    assert confirmed == 1
    est = (st["totals"]["A"]["goals_estimated"] +
           st["totals"]["B"]["goals_estimated"])
    est_none = sum(1 for s in st["shots"] if s["type"] == "goal"
                   and s["status"] != "rejected" and s["confidence"] >= 0.8)
    assert est == est_none == 2         # confirmed + pending 0.85
    assert st["totals"]["A"]["shots"] >= 1


def test_low_confidence_caveat():
    teams = _teams_dict()
    teams["team_confidence"] = 0.3
    st = _stats(teams=teams)
    assert any("confidence is low" in c for c in st["caveats"])
    assert any("not official" in c for c in st["caveats"])
