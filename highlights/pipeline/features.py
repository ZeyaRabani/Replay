"""Generic feature-table builder + match-window heuristic.

Same columns/z-scoring as highlights/fusion/build_features.py, but sources
are only audio + motion (tracking/spotting columns are zero-filled so the
frame has the same schema). Also detects the in-match window from whistles
and the audio half-time gap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from highlights.audio.events import half_time_gap

AUDIO_COLS = ["rms_db", "z60_rms", "z300_rms", "z300_speech",
              "onset_density", "whistle_frac"]
MOTION_COLS = ["motion_total", "motion_goal_roi", "motion_far",
               "near_frac", "net_disturbance"]

# same optional columns as fusion/build_features.py SOURCES (zero-filled here)
SPOTTING_COLS = ["p_ball_out_of_play", "p_shots_on_target",
                 "p_shots_off_target", "p_goal", "p_foreground_max"]
TRACKING_COLS = ["n_players", "n_near_box", "n_far_box", "n_centre",
                 "rush_near_3s", "rush_far_3s", "restart_flag",
                 "centre_cluster_flag", "mean_speed", "spread",
                 "frac_near_third", "frac_far_third", "keeper_x"]

# model feature columns = the 11 audio+motion cols + whistle
MODEL_COLS = MOTION_COLS + AUDIO_COLS + ["whistle"]

FEATURE_COLS = MOTION_COLS + AUDIO_COLS + SPOTTING_COLS + TRACKING_COLS + ["whistle"]


def load(path: str | Path, cols: list[str]) -> pd.DataFrame:
    """Same loader as fusion/build_features.py."""
    with open(path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["rows"], columns=d["columns"])
    df["t"] = df["t"].astype(int)
    keep = ["t"] + [c for c in cols if c in df.columns]
    return df[keep].set_index("t")


def build_features(audio_json: str | Path, motion_json: str | Path,
                   whistles_json: str | Path | None, duration: float,
                   match_window: tuple[float, float]) -> pd.DataFrame:
    """Per-second feature frame for t in [0, ceil(duration))."""
    t_max = math.ceil(duration) - 1
    t_max = max(t_max, 0)
    idx = pd.Index(range(t_max + 1), name="t")
    df = pd.DataFrame(index=idx)
    df = df.join(load(motion_json, MOTION_COLS))
    df = df.join(load(audio_json, AUDIO_COLS))
    for c in SPOTTING_COLS + TRACKING_COLS:
        df[c] = 0.0

    # whistle flag: any whistle interval overlapping [t, t+1)
    w = np.zeros(t_max + 1, dtype=int)
    if whistles_json and Path(whistles_json).exists():
        with open(whistles_json) as f:
            whistles = json.load(f).get("whistles", [])
        for ev in whistles:
            lo = int(np.floor(ev["t_start"]))
            hi = int(np.floor(ev["t_end"]))
            for t in range(lo, hi + 1):
                if 0 <= t <= t_max and ev["t_start"] < t + 1 and ev["t_end"] > t:
                    w[t] = 1
    df["whistle"] = w

    lo, hi = match_window
    df["in_match"] = ((df.index >= lo) & (df.index <= hi)).astype(int)

    # raw copies (before z-scoring) for debugging / re-normalisation
    df["motion_total_raw"] = df["motion_total"].astype(float).fillna(0.0)
    df["rms_db_raw"] = df["rms_db"].astype(float).fillna(0.0)

    for c in FEATURE_COLS:
        s = df[c].astype(float)
        mu, sd = s.mean(), s.std()
        df[c] = (s - mu) / sd if sd > 0 else s * 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    df["in_match"] = df["in_match"].fillna(0).astype(int)
    return df.reset_index()


def _whistle_mids(whistles_json: str | Path | None, min_dur: float = 0.4) -> list[float]:
    if not whistles_json or not Path(whistles_json).exists():
        return []
    with open(whistles_json) as f:
        segs = json.load(f).get("whistles", [])
    return sorted((s["t_start"] + s["t_end"]) / 2 for s in segs
                  if s.get("duration_s", 0) >= min_dur)


def _strong_whistle_mids(whistles_json: str | Path | None) -> list[float]:
    """Mids of the top 20% of whistles by duration (all when <10 segments)."""
    if not whistles_json or not Path(whistles_json).exists():
        return []
    with open(whistles_json) as f:
        segs = json.load(f).get("whistles", [])
    if len(segs) >= 10:
        cutoff = np.percentile([s.get("duration_s", 0) for s in segs], 80)
        segs = [s for s in segs if s.get("duration_s", 0) >= cutoff]
    return sorted((s["t_start"] + s["t_end"]) / 2 for s in segs)


def _runs(mask: np.ndarray, t: np.ndarray, min_len: float) -> list[tuple[float, float]]:
    """Contiguous True runs of a per-second boolean mask, as (t_start, t_end)."""
    runs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if t[j - 1] - t[i] >= min_len:
                runs.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return runs


def _nearest(mids: list[float], x: float, within: float = 120.0) -> float | None:
    best = min(mids, key=lambda m: abs(m - x), default=None)
    return best if best is not None and abs(best - x) <= within else None


def detect_match_window(audio_json: str | Path, whistles_json: str | Path | None,
                        duration: float, motion_json: str | Path | None = None):
    """Motion-first kickoff/full-time window.

    act = 300 s centred rolling mean of far-region motion (motion_far when
    present, else motion_total): on fixed wide-angle footage, warmups happen
    camera-side and inflate motion_total, while real play uses the far side
    of the pitch. Active stretches are those above 0.3 x the 90th
    percentile. Kickoff/full-time are the edges of the first/last active
    stretch >= 600 s, refined to the nearest strong whistle mid within
    120 s. Halftime comes from the audio gap inside the window, else the
    longest inactive stretch in the window's middle.

    Returns (lo, hi, halves, warning|None). lo/hi include a 30 s margin.
    Falls back to the full video for short clips, missing motion data, or a
    detected window covering <30% of the video.
    """
    fallback = (0.0, float(duration), [], "match window not detected; using full video")
    duration = float(duration)
    if duration < 600:
        return fallback

    # motion-first: sustained activity stretches
    if not motion_json or not Path(motion_json).exists():
        return fallback
    mdf = load(motion_json, ["motion_far", "motion_total"])
    col = "motion_far" if "motion_far" in mdf.columns else "motion_total"
    mt = mdf[col].astype(float)
    idx = np.arange(math.ceil(duration))
    act_s = mt.reindex(idx).fillna(0.0)
    act = act_s.rolling(300, center=True, min_periods=1).mean().to_numpy()
    thr = 0.3 * float(np.percentile(act, 90))
    runs = _runs(act > thr, idx.astype(float), min_len=600)
    if not runs:
        return fallback
    kickoff, fulltime = runs[0][0], runs[-1][1]

    # refine edges to the nearest strong whistle within 120 s
    mids = _strong_whistle_mids(whistles_json)
    near_k, near_f = _nearest(mids, kickoff), _nearest(mids, fulltime)
    if near_k is not None:
        kickoff = near_k
    # fulltime: accept a whistle snap only inward — play often trails the
    # final whistle, so the detected activity end is the better edge
    if near_f is not None and near_f <= fulltime:
        fulltime = near_f

    if fulltime - kickoff < 0.3 * duration:
        return fallback

    # halves: audio half-time gap inside the window, else the longest
    # low-activity stretch centred in the window's middle 30-70%
    gap = None
    with open(audio_json) as f:
        d = json.load(f)
    cols = d["columns"]
    M = np.array(d["rows"], dtype=float)
    t = M[:, cols.index("t")]
    rms_db = M[:, cols.index("rms_db")]
    onset_density = M[:, cols.index("onset_density")]
    sel = (t >= kickoff) & (t <= fulltime)
    if sel.sum() >= 300:
        gap = half_time_gap(t[sel], rms_db[sel], onset_density[sel])
    if gap is None and (fulltime - kickoff) > 60 * 60:
        # halftime needn't cross the global activity threshold (players mill
        # about camera-side); use a window-relative quiet level instead
        sel_act = act[(idx >= kickoff) & (idx <= fulltime)]
        q_thr = float(np.percentile(sel_act, 45))
        quiet = _runs(act <= q_thr, idx.astype(float), min_len=240)

        def centre(r: tuple[float, float]) -> float:
            return (r[0] + r[1]) / 2

        span = fulltime - kickoff
        quiet = [r for r in quiet
                 if kickoff + 0.3 * span <= centre(r) <= kickoff + 0.7 * span]
        if quiet:
            gap = max(quiet, key=lambda r: r[1] - r[0])

    halves = ([{"start": kickoff, "end": gap[0]},
               {"start": gap[1], "end": fulltime}] if gap is not None else [])
    lo = max(0.0, kickoff - 30)
    hi = min(duration, fulltime + 30)
    return lo, hi, halves, None
