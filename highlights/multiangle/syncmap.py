"""Piecewise file<->shared time maps.

Phones stall and skip while recording, so an angle's offset vs the
reference is constant for stretches then jumps. sync.json now carries
"timemap": [[{"file_lo","file_hi","offset"}] per angle] — file-time
half-open intervals with the offset to ADD for shared T.

  shared T = file_t + seg.offset   for file_t in [file_lo, file_hi)
  file_t   = T - seg.offset        for T in [file_lo+off, file_hi+off)

Consecutive segments share no shared edge: a forward file skip leaves a
shared-time gap (no segment covers T), a backward one an overlap (T may
map in two segments — the later segment wins in shared_to_file).

Every consumer accepts either a timemap or a legacy flat offsets list —
timemap_from_sync synthesises single-segment maps for old sync.json.
"""

from __future__ import annotations

_SEG_KEYS = ("file_lo", "file_hi", "offset")


def _seg(file_lo: float, file_hi: float, offset: float) -> dict:
    return {"file_lo": float(file_lo), "file_hi": float(file_hi),
            "offset": float(offset)}


def _is_timemap(x) -> bool:
    return bool(x) and isinstance(x[0], (list, tuple, dict))


def tm_from_offsets(offsets: list[float],
                    durations: list[float] | None = None
                    ) -> list[list[dict]]:
    """Legacy flat offsets -> single-segment timemap per angle."""
    out = []
    for i, o in enumerate(offsets):
        dur = float("inf")
        if durations and i < len(durations) and durations[i]:
            dur = float(durations[i])
        out.append([_seg(0.0, dur, float(o))])
    return out


def timemap_from_sync(sync: dict,
                      durations: list[float] | None = None
                      ) -> list[list[dict]]:
    """sync.json -> [[seg] per angle]; falls back to flat offsets +
    durations for files written before "timemap" existed."""
    tm = sync.get("timemap") or []
    offs = [float(o) for o in sync.get("offsets") or []]
    durs = durations or sync.get("durations") or []
    n = max(len(tm), len(offs), len(durs))
    out: list[list[dict]] = []
    for i in range(n):
        segs = tm[i] if i < len(tm) else None
        if segs:
            out.append([_seg(s["file_lo"], s["file_hi"], s["offset"])
                        for s in segs])
        else:
            off = offs[i] if i < len(offs) else 0.0
            dur = (float(durs[i])
                   if i < len(durs) and durs[i] else float("inf"))
            out.append([_seg(0.0, dur, off)])
    return out


def as_tm(x, durations: list[float] | None = None) -> list[list[dict]]:
    """Accept a timemap or a flat offsets list; always return a timemap."""
    if _is_timemap(x):
        return [[_seg(s["file_lo"], s["file_hi"], s["offset"])
                 for s in angle] for angle in x]
    return tm_from_offsets([float(o) for o in (x or [])], durations)


def _segs(tm, i: int) -> list[dict]:
    """Angle i's segments; `tm` may be a timemap, a flat offsets list, or
    a single segment list."""
    if isinstance(tm, (int, float)):
        return tm_from_offsets([float(tm)])[0]
    if _is_timemap(tm):
        if isinstance(tm[0], dict):       # already one angle's segs
            return tm
        return tm[i]
    return as_tm(tm)[i]


def _edges(s: dict) -> tuple[float, float]:
    """(shared_lo, shared_hi) covered by one segment."""
    return s["file_lo"] + s["offset"], s["file_hi"] + s["offset"]


def file_to_shared(tm, i: int, t: float) -> float:
    """File second on angle i -> shared T. t outside the mapped file
    range clamps to the nearest segment edge."""
    segs = _segs(tm, i)
    for s in segs:
        if s["file_lo"] <= t <= s["file_hi"]:
            return t + s["offset"]
    nearest = min(segs, key=lambda s: min(
        abs(t - s["file_lo"]), abs(t - s["file_hi"])))
    edge = (nearest["file_lo"]
            if abs(t - nearest["file_lo"])
            <= abs(t - nearest["file_hi"])
            else nearest["file_hi"])
    return edge + nearest["offset"]


def shared_to_file(tm, i: int, T: float,
                   clamp: bool = False) -> float | None:
    """Shared T -> angle i's file second. None when T falls in a gap
    (unless clamp -> nearest segment edge)."""
    segs = _segs(tm, i)
    for s in reversed(segs):      # overlapping T: later segment wins
        lo, hi = _edges(s)
        if lo <= T <= hi:
            return T - s["offset"]
    if not clamp:
        return None
    edges = [(f, abs(T - (f + s["offset"])))
             for s in segs for f in (s["file_lo"], s["file_hi"])]
    return min(edges, key=lambda e: e[1])[0]


def boundaries_shared(tm, i: int) -> list[float]:
    """Shared-T positions where angle i's mapping jumps — both edges of
    each adjacent segment pair (end of the old map, start of the new)."""
    segs = _segs(tm, i)
    out = []
    from itertools import pairwise
    for a, b in pairwise(segs):
        out.append(_edges(a)[1])
        out.append(_edges(b)[0])
    return sorted(out)


def file_range_for_shared(tm, i: int, T0: float, T1: float
                          ) -> tuple[float, float]:
    """File-time range of angle i covering shared [T0, T1]: min/max over
    the mapped endpoints and every interior boundary, edge-clamped."""
    pts = [T0, T1] + [b for b in boundaries_shared(tm, i)
                      if T0 <= b <= T1]
    fs = [shared_to_file(tm, i, T, clamp=True) for T in pts]
    return min(fs), max(fs)


def scalar_offsets(tm) -> list[float]:
    """timemap -> legacy flat offsets (longest segment per angle)."""
    tm = as_tm(tm)
    out = []
    for segs in tm:
        longest = max(segs, key=lambda s: s["file_hi"] - s["file_lo"])
        out.append(longest["offset"])
    return out
