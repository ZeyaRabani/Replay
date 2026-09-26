"""Deterministic prose summary from a match_stats dict.

`build_summary` maps stats -> {text, bullets, generated_at}; the same
stats always produce the same text (generated_at aside). Sentences are
emitted in a fixed order and padded from stats.caveats to stay inside
the 6..12 sentence range.
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
    shots = stats.get("shots") or []
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

    for h in halves:
        th = (h.get("teams") or {})
        pa = (th.get("A") or {}).get("possession_pct")
        pb = (th.get("B") or {}).get("possession_pct")
        if pa is None or pb is None:
            continue
        half_name = "first" if h.get("index") == 1 else "second"
        if pa - pb >= 8:
            txt = f"{_cap(nameA)} dominated possession in the {half_name} half ({pa:.0f}%)"
        elif pb - pa >= 8:
            txt = f"{_cap(nameB)} dominated possession in the {half_name} half ({pb:.0f}%)"
        else:
            txt = (f"Possession was even in the {half_name} half "
                   f"({pa:.0f}-{pb:.0f})")
        sentences.append(txt + ".")

    ta, tb = totals.get("A") or {}, totals.get("B") or {}
    if ta or tb:
        sentences.append(
            f"{_cap(nameA)} had {ta.get('shots', 0)} shots and "
            f"{ta.get('attacking_third_s', 0)} s in the attacking third; "
            f"{_cap(nameB)} managed {tb.get('shots', 0)} shots and "
            f"{tb.get('attacking_third_s', 0)} s.")
        ga = ta.get("goals_confirmed", 0)
        gb = tb.get("goals_confirmed", 0)
        ea = ta.get("goals_estimated", 0)
        eb = tb.get("goals_estimated", 0)
        if ga or gb or ea or eb:
            sentences.append(
                f"Goals: {_cap(nameA)} {ga} confirmed "
                f"({ea} estimated), {_cap(nameB)} {gb} confirmed "
                f"({eb} estimated).")

    top = sorted((s for s in shots if s.get("confidence") is not None),
                 key=lambda s: (-float(s["confidence"]),
                                float(s.get("t_shared", 0.0))))[:3]
    if top:
        phrases = []
        for s in top:
            phrases.append(
                f"{mmss(s['t_shared'])} ({mmss(s['t_file'])} on the "
                f"main camera)")
            bullets.append({**times(float(s["t_shared"])),
                            "label": f"{s['type']} "
                                     f"({s.get('team') or 'unassigned'})"})
        sentences.append("The best chances came at " +
                         " and ".join(phrases) + ".")

    if len(halves) == 2:
        split_t = float(halves[1].get("start") or 0.0)
        bullets.append({**times(split_t), "label": "half-time split"})
    for s in shots:
        if s.get("type") == "goal" and s.get("status") == "confirmed":
            bullets.append({**times(float(s["t_shared"])),
                            "label": f"goal ({s.get('team') or 'unassigned'})"})

    if ta.get("distance_m_est") or tb.get("distance_m_est"):
        sentences.append(
            f"Rough player totals: {_cap(nameA)} covered ~"
            f"{ta.get('distance_m_est', 0):.0f} m "
            f"({ta.get('sprints', 0)} sprints), {_cap(nameB)} ~"
            f"{tb.get('distance_m_est', 0):.0f} m "
            f"({tb.get('sprints', 0)} sprints) — positional estimates only.")

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
