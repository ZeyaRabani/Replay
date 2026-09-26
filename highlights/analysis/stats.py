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

    # ---- ends --------------------------------------------------------
    ends = {}
    for h in halves:
        ax = [x for s in secs if in_half(h, s) for x in rows.get(s, {}).get("A", [])]
        bx = [x for s in secs if in_half(h, s) for x in rows.get(s, {}).get("B", [])]
        meanA = float(np.mean(ax)) if ax else 0.5
        meanB = float(np.mean(bx)) if bx else 0.5
        if abs(meanA - meanB) < 0.03:
            ends[h["index"]] = {"A": None, "B": None}
        elif meanA < meanB:
            ends[h["index"]] = {"A": True, "B": False}
        else:
            ends[h["index"]] = {"A": False, "B": True}
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
        denom = nA + nB
        half_stats[h["index"]] = {
            "secs": len(hsecs), "ball_secs": n_ball, "nA": nA, "nB": nB,
            "poss": {"A": nA / denom * 100 if denom else 0.0,
                     "B": nB / denom * 100 if denom else 0.0},
            "attacking": att,
            "ball_visible_pct": n_ball / len(hsecs) * 100 if hsecs else 0.0,
            "contested_pct": (n_ball - len(owned)) / n_ball * 100 if n_ball else 0.0,
        }

    # ---- shots / goals -----------------------------------------------
    shots = []
    goals = {1: {"A": [0, 0], "B": [0, 0], None: [0, 0]},
             2: {"A": [0, 0], "B": [0, 0], None: [0, 0]}}
    for c in candidates or []:
        typ = str(c.get("type", ""))
        status = str(c.get("status", "pending"))
        if status == "rejected" or typ not in SHOT_TYPES:
            continue
        tc = float(c.get("t_shared", c.get("t", 0.0)))
        if not (lo <= tc <= hi):
            continue
        hid = half_of(tc)
        # attribution: ball travel direction in [t-4, t]
        bx = [(s, rows[s]["ball_x"]) for s in secs
              if tc - 4 <= s <= tc
              and rows.get(s, {}).get("ball_conf", 0.0) >= BALL_OK]
        team = None
        if len(bx) >= 2:
            dx = bx[-1][1] - bx[0][1]
            if abs(dx) > 0.05:
                for t, ar in ends[hid].items():
                    if ar is not None and ar == (dx > 0):
                        team = t
        if team is None:
            near = min(secs, key=lambda s: abs(s - tc), default=None)
            if near is not None and abs(near - tc) <= 2:
                team = owner.get(near)
        shots.append({**times(tc), "type": typ,
                      "confidence": _f(c.get("confidence", 0.0), 3),
                      "status": status, "team": team, "half": hid})
        if typ == "goal":
            conf = float(c.get("confidence", 0.0))
            if status == "confirmed":
                goals[hid][team][0] += 1
            if conf >= 0.8:
                goals[hid][team][1] += 1

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
    caveats.append("goals are candidate-based unless confirmed in Review")

    infoA = (teams.get("teams") or {}).get("A") or {}
    infoB = (teams.get("teams") or {}).get("B") or {}

    halves_out = []
    for h in halves:
        hs = half_stats[h["index"]]
        e = ends[h["index"]]
        g = goals[h["index"]]
        shot_in = [s for s in shots if s["half"] == h["index"]]

        def team_block(t, hs=hs, shot_in=shot_in, g=g, e=e):
            return {
                "possession_pct": _f(hs["poss"][t], 1),
                "attacking_third_s": int(hs["attacking"][t]),
                "shots": sum(1 for s in shot_in if s["team"] == t),
                "goals_confirmed": g[t][0],
                "goals_estimated": g[t][1],
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
        return {
            "possession_pct": _f(sum(pos) / sum(secs_tot) if sum(secs_tot) else 0.0, 1),
            "attacking_third_s": int(sum(hs["attacking"][t]
                                         for hs in half_stats.values())),
            "shots": sum(1 for s in shots if s["team"] == t),
            "goals_confirmed": sum(goals[h][t][0] for h in goals),
            "goals_estimated": sum(goals[h][t][1] for h in goals),
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
        "shots": shots,
        "caveats": caveats,
    }
    if dist_2d is not None:
        out["distance_norm_2d_total"] = dist_2d
    return out
