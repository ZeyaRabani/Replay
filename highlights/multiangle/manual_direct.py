"""You-direct: busiest-stretch suggestion + comparison vs the AI director.

Time conventions (shared timeline):
- fused_candidates.json event "t" values are shared seconds.
- director.json segments index seconds 0..T relative to the cut's shared
  range lo (meta.range[0]) — shared_t = range_lo + seg.t_start.
- angle file time for shared second s is file_t = s - offsets[a], the
  same convention render.py uses (t_file = shared_t - offsets[a]).
"""

from __future__ import annotations

import itertools

PRE_S = 45.0     # seconds before the candidate peak
POST_S = 30.0    # seconds after


def pick_candidate(fused: dict, window: tuple[float, float]) -> dict | None:
    """Busiest stretch = the event a user most wants to switch through:
    prefer confirmed, then goals, then highest confidence, inside the
    shared window."""
    lo, hi = window
    cands = [c for c in (fused.get("candidates") or fused.get("events")
                         or []) if lo <= float(c.get("t", 0)) <= hi]
    if not cands:
        return None

    def rank(c):
        return (
            1 if str(c.get("status")) == "confirmed" else 0,
            1 if str(c.get("type")) == "goal" else 0,
            float(c.get("confidence", 0.0)),
        )

    return max(cands, key=rank)


def suggest_stretch(candidate_t: float, window: tuple[float, float]
                    ) -> tuple[float, float]:
    """[t-PRE_S, t+POST_S] clamped to the coverage window."""
    lo, hi = window
    return (max(lo, candidate_t - PRE_S), min(hi, candidate_t + POST_S))


def director_rows(segments: list[dict], range_lo: float,
                  t_start: float, t_end: float) -> list[dict]:
    """Per-second director view for shared seconds [t_start, t_end):
    [{t (shared), angle, rule}]. Segments index the range-relative
    timeline, so shared = range_lo + seg.t_start."""
    rows = []
    for s in range(int(t_start), int(t_end)):
        rel = s - range_lo
        seg = next((g for g in segments
                    if g["t_start"] <= rel < g["t_end"]), None)
        if seg is not None:
            rows.append({"t": float(s), "angle": int(seg["angle"]),
                         "rule": str(seg.get("rule", ""))})
    return rows


def compare(choices: dict[int, int], rows: list[dict]) -> dict:
    """Compare a user's per-second angle choices against the director.

    choices maps shared second -> angle index. Gaps are filled with the
    last choice; seconds before the first choice get user=null and are
    excluded from the agreement denominator.
    """
    filled: dict[int, int | None] = {}
    last: int | None = None
    for r in rows:
        t = int(r["t"])
        if t in choices:
            last = choices[t]
        filled[t] = last

    agree_n = 0
    total = 0
    by_rule: dict[str, list[int]] = {}
    by_pair: dict[str, int] = {}
    out_rows = []
    for r in rows:
        t = int(r["t"])
        u = filled[t]
        agree = u is not None and u == r["angle"]
        if u is not None:
            total += 1
            if agree:
                agree_n += 1
            else:
                by_rule.setdefault(r["rule"], [0, 0])
                by_rule[r["rule"]][0] += 1
                key = f"you:{u}->dir:{r['angle']}"
                by_pair[key] = by_pair.get(key, 0) + 1
        if u is not None:
            by_rule.setdefault(r["rule"], [0, 0])[1] += 1
        out_rows.append({"t": r["t"], "user": u,
                         "director": r["angle"], "rule": r["rule"],
                         "agree": agree})

    user_seq = [filled[r["t"]] for r in rows]
    user_switches = sum(
        1 for a, b in itertools.pairwise(user_seq)
        if a is not None and b is not None and a != b)
    dir_seq = [r["angle"] for r in rows]
    director_switches = sum(1 for a, b in itertools.pairwise(dir_seq) if a != b)

    return {
        "agreement_pct": round(agree_n / total * 100, 1) if total else None,
        "n_seconds": total,
        "user_switches": user_switches,
        "director_switches": director_switches,
        "disagree_by_rule": {
            k: {"n": v[0], "of": v[1]} for k, v in sorted(by_rule.items())},
        "disagree_by_pair": dict(
            sorted(by_pair.items(), key=lambda kv: -kv[1])),
        "rows": out_rows,
    }
