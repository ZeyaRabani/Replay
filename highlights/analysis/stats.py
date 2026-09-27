"""Per-half match stats from the teams pass + review candidates.

`compute_match_stats` is a pure function: it takes the teams.json dict
(per-shared-second team-A/B player x positions + ball), the reference
angle's features_1s.json (optional, for 2-D distance), and candidates
already converted to shared-T, and returns the match_stats.json dict.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from highlights.pipeline.stats import _quietest_stretch

BALL_OK = 0.35
SHOT_TYPES = {"shot", "goal", "goalmouth"}
PITCH_LEN_M = {5: 40, 7: 55, 9: 70, 11: 100}
TRACK_STEP = 0.05
SPRINT_STEP = 0.03


def _f(x: Any, nd: int = 4) -> float:
    x = float(x)
    if not math.isfinite(x):
        return 0.0
    return round(x, nd)


def mmss(t: float) -> str:
    t = max(0, round(t))
    return f"{t // 60}:{t % 60:02d}"


def _teams_rows(teams: dict) -> dict[int, dict]:
    """teams.json -> per-shared-second dict."""
    cols = teams.get("columns") or []
    idx = {c: i for i, c in enumerate(cols)}
    out = {}
    for r in teams.get("rows") or []:
        sec = r[idx["t_shared"]] if isinstance(r[idx["t_shared"]], int) else int(r[idx["t_shared"]])
        out[sec] = {
            "ball_x": float(r[idx["ball_x"]]),
            "ball_y": float(r[idx["ball_y"]]),
            "ball_conf": float(r[idx["ball_conf"]]),
            "A": [float(x) for x in (r[idx["teamA_xs"]] or [])],
            "B": [float(x) for x in (r[idx["teamB_xs"]] or [])],
            "n": int(r[idx["teamA_n"]]) + int(r[idx["teamB_n"]]),
        }
    return out


def _majority5(owner: dict[int, str | None], secs: list[int]) -> dict[int, str | None]:
    """Rolling 5 s majority vote over the raw owner sequence (ties keep raw)."""
    sm = {}
    for s in secs:
        votes = [owner.get(k) for k in range(s - 2, s + 3)]
        a = votes.count("A")
        b = votes.count("B")
        if a > b:
            sm[s] = "A"
        elif b > a:
            sm[s] = "B"
        else:
            sm[s] = owner.get(s)
    return sm


def _track_distance(xss: list[list[float]]) -> dict:
    """Greedy 1-D nearest-neighbour tracking over per-second x lists.

    Each second's points are matched to active tracks (|dx| <= TRACK_STEP,
    nearest first); unmatched points start new tracks."""
    tracks: dict[int, float] = {}
    next_id = 0
    dist = 0.0
    sprints = 0
    player_seconds = 0
    for xs in xss:
        used = set()
        for x in xs:
            best, best_d = None, TRACK_STEP + 1e-9
            for tid, tx in tracks.items():
                d = abs(x - tx)
                if d < best_d:
                    best, best_d = tid, d
            if best is not None and best not in used:
                step = abs(x - tracks[best])
                dist += step
                if step >= SPRINT_STEP:
                    sprints += 1
                tracks[best] = x
                used.add(best)
            else:
                tracks[next_id] = x
                next_id += 1
            player_seconds += 1
    return {"distance_norm": dist, "sprints": sprints,
            "tracked_player_seconds": player_seconds}


def compute_match_stats(teams: dict, ref_features: dict | None,
                        candidates: list[dict], *,
                        match_window: tuple[float, float],
                        lo_out: float, offsets: list[float],
                        ref_angle: int, pitch_type: int | None = None) -> dict:
    lo, hi = float(match_window[0]), float(match_window[1])
    rows = _teams_rows(teams)
    caveats: list[str] = ["Estimated from footage — not official statistics"]

    def times(t_shared: float) -> dict:
        return {
            "t_shared": _f(t_shared, 2),
            "t_out": _f(t_shared - lo_out, 2),
            "t_file": _f(t_shared - (offsets[ref_angle]
                                     if ref_angle < len(offsets) else 0.0), 2),
        }

    secs = sorted(s for s in rows if lo <= s <= hi)
    if not secs:
        secs = list(range(int(lo), int(hi) + 1))
    t_arr = np.array(secs, dtype=float)

    # ---- halves -----------------------------------------------------
    level = np.array([rows.get(s, {}).get("n", 0) for s in secs], dtype=float)
    mid = (lo + hi) / 2.0
    span = hi - lo
    middle = (t_arr >= lo + 0.35 * span) & (t_arr <= lo + 0.65 * span)
    split = mid
    split_note = None
    if middle.any() and level.mean() > 0:
        qs, qe = _quietest_stretch(t_arr, level, middle, win_s=90)
        qmask = (t_arr >= qs) & (t_arr <= qe)
        if qmask.any() and level[qmask].mean() < 0.5 * level.mean():
            split = (qs + qe) / 2.0
            split_note = (f"half-time break detected from a quiet stretch "
                          f"at {mmss(split - lo)}")
    halves = [{"index": 1, "start": lo, "end": split},
              {"index": 2, "start": split, "end": hi}]
    if split_note:
        caveats.append(split_note)
    else:
        caveats.append("halves split at the midpoint of the match window")

    def in_half(h, s):
        return h["start"] <= s < h["end"] if h["index"] == 1 else h["start"] <= s <= h["end"]

    # ---- attacking direction ----------------------------------------
    # Per half each team defends the x-side where its most extreme player
    # (nearest its own goal — usually the keeper) sits most seconds;
    # ends[h][team] = True when that team attacks toward +x.
    ends: dict[int, dict] = {}
    for h in halves:
        sides: dict[str, list[bool]] = {"A": [], "B": []}
        for s in secs:
            if not in_half(h, s):
                continue
            r = rows.get(s) or {}
            for t in "AB":
                xs = r.get(t) or []
                if xs:
                    x = max(xs, key=lambda v: abs(v - 0.5))
                    sides[t].append(x < 0.5)   # True = own goal on the left
        ends[h["index"]] = {}
        for t in "AB":
            if not sides[t]:
                ends[h["index"]][t] = None
                continue
            left_share = sum(sides[t]) / len(sides[t])
            if 0.45 <= left_share <= 0.55:
                ends[h["index"]][t] = None      # ambiguous
            else:
                ends[h["index"]][t] = left_share > 0.5   # own goal left -> attacks right
    e1, e2 = ends[1], ends[2]
    if e1["A"] is None or e2["A"] is None:
        ends_swapped = False
        caveats.append("teams' ends could not be determined — attacking-third "
                       "numbers are direction-free")
    else:
        ends_swapped = e1["A"] != e2["A"]
        if not ends_swapped:
            caveats.append("teams did not appear to swap ends between halves "
                           "(check the half-time split)")

    def half_of(s: float) -> int:
        return 1 if s < split else 2

    # ---- possession --------------------------------------------------
    owner_raw: dict[int, str | None] = {}
    for s in secs:
        r = rows.get(s)
        if not r or r["ball_conf"] < BALL_OK:
            owner_raw[s] = None
            continue
        dA = min((abs(x - r["ball_x"]) for x in r["A"]), default=math.inf)
        dB = min((abs(x - r["ball_x"]) for x in r["B"]), default=math.inf)
        owner_raw[s] = "A" if dA < dB else ("B" if dB < dA else None)
    owner = _majority5(owner_raw, secs)

    half_stats: dict[int, dict] = {}
    for h in halves:
        hsecs = [s for s in secs if in_half(h, s)]
        ball_secs = [s for s in hsecs
                     if rows.get(s, {}).get("ball_conf", 0.0) >= BALL_OK]
        n_ball = len(ball_secs)
        owned = [s for s in ball_secs if owner.get(s)]
        nA = sum(1 for s in owned if owner[s] == "A")
        nB = len(owned) - nA
        att = {"A": 0, "B": 0}
        for s in owned:
            r = rows[s]
            o = owner[s]
            ar = ends[h["index"]].get(o)
            if (ar is True and r["ball_x"] > 2 / 3) or (ar is False and r["ball_x"] < 1 / 3):
                att[o] += 1
        # territory: share of ball-in-play seconds in each third, along the
        # team's own attacking direction (att = the third holding the
        # opponent's goal)
        territory: dict[str, dict | None] = {"A": None, "B": None}
        for t in "AB":
            ar = ends[h["index"]].get(t)
            if ar is None or not ball_secs:
                continue
            cnt = {"def": 0, "mid": 0, "att": 0}
            for s in ball_secs:
                bx = rows[s]["ball_x"]
                if ar:     # attacks +x: def = [0,1/3), att = (2/3,1]
                    zone = ("def" if bx < 1 / 3 else "att" if bx > 2 / 3
                            else "mid")
                else:      # attacks -x: mirrored
                    zone = ("att" if bx < 1 / 3 else "def" if bx > 2 / 3
                            else "mid")
                cnt[zone] += 1
            territory[t] = {k: _f(v / len(ball_secs), 3)
                            for k, v in cnt.items()}
        denom = nA + nB
        half_stats[h["index"]] = {
            "secs": len(hsecs), "ball_secs": n_ball, "nA": nA, "nB": nB,
            "poss": {"A": nA / denom * 100 if denom else 0.0,
                     "B": nB / denom * 100 if denom else 0.0},
            "attacking": att,
            "territory": territory,
            "ball_visible_pct": n_ball / len(hsecs) * 100 if hsecs else 0.0,
            "contested_pct": (n_ball - len(owned)) / n_ball * 100 if n_ball else 0.0,
        }

    # ---- events -------------------------------------------------------
    # Headline shots/goals come ONLY from candidates confirmed in Review.
    # Pending/rejected ones are kept out of the counts and listed as
    # unreviewed AI chances.
    def _ball_x_at(tc: float) -> float | None:
        """Ball x at shared second tc; falls back to the mean over
        t-2..t+2 when the ball confidence there is low."""
        near = min(secs, key=lambda s: abs(s - tc), default=None)
        if near is not None and abs(near - tc) <= 1.0 and \
                rows[near]["ball_conf"] >= BALL_OK:
            return rows[near]["ball_x"]
        xs = [rows[s]["ball_x"] for s in secs
              if tc - 2 <= s <= tc + 2 and rows[s]["ball_conf"] > 0]
        return float(np.mean(xs)) if xs else None

    def _attribute(tc: float, hid: int) -> tuple[str | None, str]:
        """Which team the event belongs to, + 'high'/'low' confidence.

        Ball in the third nearest team X's own goal -> attributed to the
        other team. Middle third / unknown ends -> more players in the
        ball's half, marked low."""
        bx = _ball_x_at(tc)
        if bx is None:
            return None, "low"
        e = ends.get(hid) or {}
        if bx < 1 / 3 or bx > 2 / 3:
            # goal on the ball's side belongs to the team defending it:
            # attacks_right=True -> own goal on the left
            if bx < 1 / 3:
                defender = ("A" if e.get("A") is True else
                            "B" if e.get("B") is True else None)
            else:
                defender = ("A" if e.get("A") is False else
                            "B" if e.get("B") is False else None)
            if defender is None:
                return None, "low"
            return ("B" if defender == "A" else "A"), "high"
        # middle third: more of the team's players in the ball's half
        near = min(secs, key=lambda s: abs(s - tc), default=None)
        if near is None:
            return None, "low"
        r = rows[near]
        half_side = bx < 0.5
        nA = sum(1 for x in r["A"] if (x < 0.5) == half_side)
        nB = sum(1 for x in r["B"] if (x < 0.5) == half_side)
        if nA == nB:
            return None, "low"
        return ("A" if nA > nB else "B"), "low"

    events: list[dict] = []
    unreviewed: list[dict] = []
    for c in candidates or []:
        typ = str(c.get("type", ""))
        status = str(c.get("status", "pending"))
        if typ not in SHOT_TYPES:
            continue
        tc = float(c.get("t_shared", c.get("t", 0.0)))
        if not (lo <= tc <= hi):
            continue
        base = {**times(tc), "mmss": mmss(tc - lo_out),
                "type": typ,
                "confidence": _f(c.get("confidence", 0.0), 3),
                "status": status,
                "candidate_id": c.get("id") or c.get("candidate_id"),
                "half": half_of(tc)}
        if status == "confirmed":
            team, attr = _attribute(tc, base["half"])
            events.append({**base,
                           "kind": "goal" if typ == "goal" else "shot",
                           "team": team, "attribution": attr})
        elif status != "rejected":
            unreviewed.append(base)
    events.sort(key=lambda e: e["t_shared"])
    unreviewed.sort(key=lambda e: (-e["confidence"], e["t_shared"]))
    unreviewed = unreviewed[:10]

    # ---- momentum: per 5-min bin, ball share in A's attacking half
    # minus share in B's attacking half, in [-1, 1] --------------------
    momentum: list[dict] = []
    BIN = 300.0
    b0 = lo
    while b0 < hi:
        b1 = min(b0 + BIN, hi)
        vals = []
        for s in secs:
            if not (b0 <= s < b1):
                continue
            r = rows.get(s)
            if not r or r["ball_conf"] < BALL_OK:
                continue
            e = ends.get(half_of(s)) or {}
            if e.get("A") is None:
                continue
            a_att = r["ball_x"] > 0.5 if e["A"] else r["ball_x"] < 0.5
            b_att = (r["ball_x"] > 0.5 if e.get("B")
                     else r["ball_x"] < 0.5)
            if a_att:
                vals.append(1.0)
            elif b_att:
                vals.append(-1.0)
            else:
                vals.append(0.0)
        momentum.append({"t_start_shared": _f(b0, 1),
                         "t_start_out": _f(b0 - lo_out, 1),
                         "t_start_file": _f(b0 - (offsets[ref_angle]
                            if ref_angle < len(offsets) else 0.0), 1),
                         "value": _f(float(np.mean(vals)), 3) if vals
                                  else None,
                         "n": len(vals)})
        b0 = b1

    # ---- player distance / sprints (rough, 1-D) -----------------------
    lens = PITCH_LEN_M.get(pitch_type, 100)
    dist = {}
    for team, key in (("A", "A"), ("B", "B")):
        xss = [rows[s][key] for s in secs if rows.get(s)]
        tr = _track_distance(xss)
        dist[team] = {
            **tr,
            "distance_norm": _f(tr["distance_norm"], 3),
            "distance_m_est": _f(tr["distance_norm"] * lens, 0),
        }
    caveats.append("distance in metres assumes the frame spans the pitch "
                   "length; rough")

    dist_2d = None
    if ref_features:
        cols = ref_features.get("columns") or []
        if "players_xy" in cols:
            pi = cols.index("players_xy")
            ti = cols.index("t")
            seq = []
            for r in ref_features.get("rows") or []:
                ft = float(r[ti])
                v = r[pi]
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except Exception:
                        v = []
                seq.append((ft, v or []))
            seq.sort()
            prev: list[list[float]] = []
            total = 0.0
            for _ft, pts in seq:
                used = set()
                for p in pts:
                    best_d, best = math.inf, None
                    for j, q in enumerate(prev):
                        d = math.hypot(p[0] - q[0], p[1] - q[1])
                        if d < best_d:
                            best_d, best = d, j
                    if best is not None and best_d <= TRACK_STEP and best not in used:
                        total += best_d
                        used.add(best)
                prev = pts
            dist_2d = _f(total, 3)

    if (teams.get("team_confidence") or 0.0) < 0.5:
        caveats.append("team identification confidence is low — shirt-colour "
                       "split may mix teams or include the referee")
    caveats.append("shots and goals count only candidates confirmed in "
                   "Review; unconfirmed AI chances are listed separately")

    infoA = (teams.get("teams") or {}).get("A") or {}
    infoB = (teams.get("teams") or {}).get("B") or {}

    halves_out = []
    for h in halves:
        hs = half_stats[h["index"]]
        e = ends[h["index"]]
        ev_in = [ev for ev in events if ev["half"] == h["index"]]

        def team_block(t, hs=hs, ev_in=ev_in, e=e):
            return {
                "possession_pct": _f(hs["poss"][t], 1),
                "attacking_third_s": int(hs["attacking"][t]),
                "shots": sum(1 for s in ev_in
                             if s["team"] == t and s["kind"] == "shot"),
                "goals": sum(1 for s in ev_in
                             if s["team"] == t and s["kind"] == "goal"),
                "territory": hs["territory"][t],
                "attacks_right": e[t],
            }

        halves_out.append({
            "index": h["index"],
            "start": _f(h["start"], 1), "end": _f(h["end"], 1),
            "start_out": _f(h["start"] - lo_out, 1),
            "start_file": _f(h["start"] - (offsets[ref_angle]
                                           if ref_angle < len(offsets) else 0.0), 1),
            "ball_visible_pct": _f(hs["ball_visible_pct"], 1),
            "contested_pct": _f(hs["contested_pct"], 1),
            "teams": {"A": team_block("A"), "B": team_block("B")},
        })

    def totals(t):
        pos = [hs["poss"][t] * hs["secs"] for hs in half_stats.values()]
        secs_tot = [hs["secs"] for hs in half_stats.values()]
        terr = [hs["territory"][t] for hs in half_stats.values()]
        bsec = [hs["ball_secs"] for hs in half_stats.values()]
        territory = None
        if all(x is not None for x in terr) and sum(bsec):
            territory = {
                k: _f(sum(ter[k] * b for ter, b in zip(terr, bsec))
                      / sum(bsec), 3)
                for k in ("def", "mid", "att")}
        return {
            "possession_pct": _f(sum(pos) / sum(secs_tot) if sum(secs_tot) else 0.0, 1),
            "attacking_third_s": int(sum(hs["attacking"][t]
                                         for hs in half_stats.values())),
            "shots": sum(1 for s in events
                         if s["team"] == t and s["kind"] == "shot"),
            "goals": sum(1 for s in events
                         if s["team"] == t and s["kind"] == "goal"),
            "territory": territory,
            "distance_norm": dist[t]["distance_norm"],
            "distance_m_est": dist[t]["distance_m_est"],
            "sprints": dist[t]["sprints"],
            "tracked_player_seconds": dist[t]["tracked_player_seconds"],
        }

    out = {
        "generated_at": round(time.time(), 1),
        "match_window": [_f(lo, 1), _f(hi, 1)],
        "lo_out": _f(lo_out, 1),
        "ref_angle": ref_angle,
        "offsets": offsets,
        "pitch_type": pitch_type,
        "teams": {"A": {"name": infoA.get("name"), "hex": infoA.get("hex")},
                  "B": {"name": infoB.get("name"), "hex": infoB.get("hex")}},
        "team_confidence": teams.get("team_confidence"),
        "halves": halves_out,
        "ends_swapped": ends_swapped,
        "totals": {"A": totals("A"), "B": totals("B")},
        "events": events,
        "unreviewed": unreviewed,
        "n_unreviewed": sum(1 for c in candidates or []
                            if str(c.get("type", "")) in SHOT_TYPES
                            and str(c.get("status", "pending")) not in
                            ("confirmed", "rejected")
                            and lo <= float(c.get("t_shared", c.get("t", 0.0)))
                            <= hi),
        "momentum": momentum,
        "caveats": caveats,
    }
    if dist_2d is not None:
        out["distance_norm_2d_total"] = dist_2d
    return out
