"""Match statistics (Contract 5) from a per-second feature frame + candidate events.

`compute_stats` is a pure function shared with the app backend, which calls it on
the committed fusion outputs for the demo match. Column names follow
highlights/fusion/build_features.py; any missing column is treated as zeros.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

BIN_S = 30
MOTION_COL = "motion_total"
AUDIO_COL = "rms_db"
EXCITEMENT_COLS = ("z300_speech", "z300_rms", "motion_goal_roi")
EVENT_TYPES = ("goal", "shot", "chance", "excitement", "other")

PIPELINE_INFO = {
    "model": "audio_motion_lr v1",
    "auroc_reference": None,
    "notes": "",
}


def _col(df: pd.DataFrame, name: str, n: int) -> np.ndarray:
    if name in df.columns:
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(n, dtype=float)


def _norm01(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise `v` using the 1st/99th percentiles over `mask`."""
    ref = v[mask] if mask.any() else v
    if ref.size == 0:
        return np.zeros_like(v)
    lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
    if hi - lo <= 1e-12:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def _f(x: Any, nd: int = 4) -> float:
    x = float(x)
    if not math.isfinite(x):
        return 0.0
    return round(x, nd)


def compute_stats(
    features_df: pd.DataFrame,
    candidates_events: Sequence[dict[str, Any]],
    duration: float,
    match_window: Sequence[float] | None,
    halves: Sequence[dict[str, float]] | None,
    whistles: Sequence[float] | None,
    pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the Contract 5 stats dict.

    features_df: one row per second with a `t` column (or the index) and the
        fusion column names (motion_total, rms_db, z300_speech, ...). Values may be
        raw or z-scored; everything is re-normalised to 0-1 over the match window.
    candidates_events: Contract 4 events ({type, t, confidence, ...}).
    duration: video duration in seconds.
    match_window: [lo, hi] detected play window (None -> [0, duration]).
    halves: [{"start", "end"}, ...] or None/[].
    whistles: whistle times in seconds (or dicts with t / t_start,t_end).
    """
    duration = float(duration or 0.0)
    df = features_df.copy()
    if "t" in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    else:
        t = df.index.to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        t = np.arange(math.ceil(duration) or 1, dtype=float)
        n = len(t)
        df = pd.DataFrame({"t": t})
    if duration <= 0 and n:
        duration = float(t.max() + 1)

    if match_window and len(match_window) == 2 and match_window[1] > match_window[0]:
        lo, hi = float(match_window[0]), float(match_window[1])
    else:
        lo, hi = 0.0, duration
    lo = max(0.0, lo)
    hi = min(duration, hi) if duration > 0 else hi
    in_match = (t >= lo) & (t <= hi)
    if not in_match.any():
        in_match = np.ones(n, dtype=bool)

    motion = _col(df, MOTION_COL, n)
    audio = _col(df, AUDIO_COL, n)
    exc_parts = [_norm01(_col(df, c, n), in_match) for c in EXCITEMENT_COLS if c in df.columns]
    excitement = np.mean(exc_parts, axis=0) if exc_parts else np.zeros(n)
    motion_n = _norm01(motion, in_match)
    audio_n = _norm01(audio, in_match)
    excitement_n = _norm01(excitement, in_match)

    events = [e for e in candidates_events if e.get("cross_validation") != "rejected"]
    ev_t = np.array([float(e.get("t", 0.0)) for e in events], dtype=float)

    # timeline
    n_bins = max(1, math.ceil(duration / BIN_S)) if duration > 0 else 1
    bin_idx = np.clip((t // BIN_S).astype(int), 0, n_bins - 1)
    timeline = []
    for b in range(n_bins):
        m = bin_idx == b
        b_lo, b_hi = b * BIN_S, (b + 1) * BIN_S
        n_ev = int(((ev_t >= b_lo) & (ev_t < b_hi)).sum()) if ev_t.size else 0
        timeline.append({
            "t": b * BIN_S,
            "motion": _f(motion_n[m].mean()) if m.any() else 0.0,
            "audio": _f(audio_n[m].mean()) if m.any() else 0.0,
            "excitement": _f(excitement_n[m].mean()) if m.any() else 0.0,
            "events": n_ev,
        })

    # counts
    events_by_type: dict[str, int] = {}
    for e in events:
        k = str(e.get("type", "other"))
        events_by_type[k] = events_by_type.get(k, 0) + 1

    n_ten = max(1, math.ceil(duration / 600)) if duration > 0 else 1
    per10 = []
    for b in range(n_ten):
        row: dict[str, Any] = {"t": b * 600}
        for k in EVENT_TYPES:
            row[k] = 0
        for e in events:
            if b * 600 <= float(e.get("t", 0.0)) < (b + 1) * 600:
                k = str(e.get("type", "other"))
                row[k] = row.get(k, 0) + 1
        per10.append(row)

    # top moments
    type_rank = {"goal": 0, "shot": 1, "chance": 2, "excitement": 3, "other": 4}
    ordered = sorted(
        events,
        key=lambda e: (-float(e.get("confidence", 0.0)), type_rank.get(str(e.get("type")), 5), float(e.get("t", 0.0))),
    )
    top_moments = []
    for e in ordered[:10]:
        reason = e.get("notes") or ""
        if not reason:
            sig = e.get("signals") or {}
            top = sig.get("top_signals") if isinstance(sig, dict) else None
            reason = ", ".join(map(str, top)) if top else f"{e.get('type', 'event')} candidate"
        top_moments.append({
            "t": _f(e.get("t", 0.0), 2),
            "type": str(e.get("type", "other")),
            "confidence": _f(e.get("confidence", 0.0), 3),
            "reason": str(reason),
        })

    # whistles
    w_out: list[float] = []
    for w in whistles or []:
        if isinstance(w, dict):
            if "t" in w:
                w_out.append(_f(w["t"], 2))
            elif "t_start" in w:
                w_out.append(_f((float(w["t_start"]) + float(w.get("t_end", w["t_start"]))) / 2, 2))
        else:
            w_out.append(_f(w, 2))
    w_out.sort()

    # activity
    mm = motion.copy()
    mm[~in_match] = -np.inf
    peak_motion_t = float(t[int(np.argmax(mm))]) if n else 0.0
    aa = audio.copy()
    aa[~in_match] = -np.inf
    loudest_t = float(t[int(np.argmax(aa))]) if n else 0.0
    quiet = _quietest_stretch(t, motion_n + audio_n, in_match)
    activity = {
        "mean_motion": _f(motion_n[in_match].mean()) if in_match.any() else 0.0,
        "peak_motion_t": _f(peak_motion_t, 1),
        "loudest_t": _f(loudest_t, 1),
        "quietest_stretch": quiet,
    }

    halves_out = [
        {"start": _f(h["start"], 1), "end": _f(h["end"], 1)}
        for h in (halves or [])
        if isinstance(h, dict) and "start" in h and "end" in h
    ]

    info = dict(PIPELINE_INFO)
    if pipeline:
        info.update(pipeline)

    return {
        "duration_s": _f(duration, 3),
        "match_window": [_f(lo, 1), _f(hi, 1)],
        "halves": halves_out,
        "bin_s": BIN_S,
        "timeline": timeline,
        "events_by_type": events_by_type,
        "events_per_10min": per10,
        "top_moments": top_moments,
        "whistles": w_out,
        "activity": activity,
        "pipeline": info,
    }


def _quietest_stretch(t: np.ndarray, level: np.ndarray, mask: np.ndarray, win_s: int = 60) -> list[float]:
    """Start/end of the `win_s`-second window with the lowest mean level inside `mask`."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return [0.0, 0.0]
    v = level[idx]
    if v.size <= win_s:
        return [_f(t[idx[0]], 1), _f(t[idx[-1]], 1)]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    sums = cs[win_s:] - cs[:-win_s]
    i = int(np.argmin(sums))
    return [_f(t[idx[i]], 1), _f(t[idx[i + win_s - 1]], 1)]
