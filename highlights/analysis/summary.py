"""Deterministic prose summary from a match_stats dict.

`build_summary` maps stats -> {text, bullets, generated_at}; the same
stats always produce the same text (generated_at aside). Everything in
the text comes from the stats dict — score line and events list only
ever mention candidates confirmed in Review.
"""

from __future__ import annotations

import time

from .stats import mmss


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def build_summary(stats: dict) -> dict:
    teams = stats.get("teams") or {}
    nameA = (teams.get("A") or {}).get("name") or "Team A"
    nameB = (teams.get("B") or {}).get("name") or "Team B"
    halves = stats.get("halves") or []
    totals = stats.get("totals") or {}
    events = stats.get("events") or []
    momentum = stats.get("momentum") or []
    caveats = stats.get("caveats") or []
    lo_out = float(stats.get("lo_out") or 0.0)
    ref_off = 0.0
    offsets = stats.get("offsets") or []
    ref = stats.get("ref_angle") or 0
    if ref < len(offsets):
        ref_off = float(offsets[ref])

    def times(t_shared):
        return {"t_shared": round(t_shared, 2),
                "t_out": round(t_shared - lo_out, 2),
                "t_file": round(t_shared - ref_off, 2)}

    bullets: list[dict] = []
    sentences: list[str] = []

    conf = stats.get("team_confidence")
    conf_pct = f", team identification confidence {round(conf * 100)}%" \
        if conf is not None else ""
    n_angles = len(offsets) or 1
    sentences.append(
        f"{_cap(nameA)} vs {_cap(nameB)} — estimated from "
        f"{n_angles} angles' footage{conf_pct}.")

    # ---- score line: confirmed goals only ---------------------------
    ga = (totals.get("A") or {}).get("goals", 0)
    gb = (totals.get("B") or {}).get("goals", 0)
    if ga or gb:
        sentences.append(
            f"Score: {_cap(nameA)} {ga} - {gb} {_cap(nameB)} "
            f"(confirmed goals only).")
    else:
        sentences.append("No goals confirmed yet — confirm them in "
                         "Review to fill in the score.")

    # ---- confirmed events -------------------------------------------
    for e in events:
        label = f"{e.get('kind', 'shot')} ({e.get('team') or 'unassigned'})"
        if e.get("attribution") == "low":
            label += " · low confidence"
        bullets.append({**times(float(e["t_shared"])), "label": label})
    goals_txt = [f"{_cap(teams.get(e['team'], {}).get('name') or e['team'])} "
                 f"{e['kind']} at {e.get('mmss')}"
                 for e in events if e.get("kind") == "goal"]
    if goals_txt:
        sentences.append("Goals: " + "; ".join(goals_txt) + ".")
    shots_conf = sum(1 for e in events if e.get("kind") == "shot")
    if shots_conf:
        sentences.append(
            f"{shots_conf} confirmed shot"
            f"{'s' if shots_conf != 1 else ''} on record.")

    # ---- territory dominance per half --------------------------------
    for h in halves:
        th = (h.get("teams") or {})
        ta = (th.get("A") or {}).get("territory") or {}
        tb = (th.get("B") or {}).get("territory") or {}
        if ta.get("att") is None or tb.get("att") is None:
            continue
        half_name = "first" if h.get("index") == 1 else "second"
        aa, ab = ta["att"] * 100, tb["att"] * 100
        if aa - ab >= 8:
            sentences.append(
                f"{_cap(nameA)} dominated territory in the {half_name} "
                f"half ({aa:.0f}% of the ball-in-play time in their "
                f"attacking third).")
        elif ab - aa >= 8:
            sentences.append(
                f"{_cap(nameB)} dominated territory in the {half_name} "
                f"half ({ab:.0f}% attacking-third share).")
        else:
            sentences.append(
                f"Territory was even in the {half_name} half "
                f"({aa:.0f}% vs {ab:.0f}% attacking-third share).")

    # ---- biggest momentum swing --------------------------------------
    swing = max((m for m in momentum if m.get("value") is not None),
                key=lambda m: abs(m["value"]), default=None)
    if swing is not None and abs(swing["value"]) >= 0.25:
        who = nameA if swing["value"] > 0 else nameB
        sentences.append(
            f"Biggest spell of pressure: {_cap(who)} around "
            f"{mmss(swing['t_start_out'])}.")
    if len(halves) == 2:
        split_t = float(halves[1].get("start") or 0.0)
        bullets.append({**times(split_t), "label": "half-time split"})

    # ---- player distance ----------------------------------------------
    ta, tb = totals.get("A") or {}, totals.get("B") or {}
    if ta.get("distance_m_est") or tb.get("distance_m_est"):
        sentences.append(
            f"Rough player totals: {_cap(nameA)} covered ~"
            f"{ta.get('distance_m_est', 0):.0f} m "
            f"({ta.get('sprints', 0)} sprints), {_cap(nameB)} ~"
            f"{tb.get('distance_m_est', 0):.0f} m "
            f"({tb.get('sprints', 0)} sprints) — per-team totals over "
            f"tracked players only, positional estimates.")

    # closing caveat, then pad from stats.caveats to reach 6 sentences
    if caveats:
        sentences.append(_cap(caveats[0]) +
                         ("" if caveats[0].endswith(".") else "."))
    for c in caveats[1:]:
        if len(sentences) >= 6:
            break
        sentences.append(_cap(c) + ("" if c.endswith(".") else "."))
    while len(sentences) < 6:
        sentences.append("Treat every figure here as an estimate.")

    return {"text": " ".join(sentences[:12]), "bullets": bullets,
            "generated_at": round(time.time(), 1)}
