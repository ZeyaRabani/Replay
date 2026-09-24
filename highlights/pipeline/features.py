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


def detect_match_window(audio_json: str | Path, whistles_json: str | Path | None,
                        duration: float):
    """Heuristic kickoff/full-time window.

    Returns (lo, hi, halves, warning|None). lo/hi include a 30 s margin.
    Falls back to the full video for short clips or when the detected
    window covers <30% of the video.
    """
    fallback = (0.0, float(duration), [], "match window not detected; using full video")
    duration = float(duration)
    if duration < 600:
        return fallback

    with open(audio_json) as f:
        d = json.load(f)
    cols = d["columns"]
    M = np.array(d["rows"], dtype=float)
    t = M[:, cols.index("t")]
    rms_db = M[:, cols.index("rms_db")]
    onset_density = M[:, cols.index("onset_density")]

    mids = _whistle_mids(whistles_json)
    first40 = [m for m in mids if m < 0.4 * duration]
    last40 = [m for m in mids if m > 0.6 * duration]
    kickoff = first40[0] if first40 else 0.0
    fulltime = last40[-1] if last40 else duration

    halves: list[dict] = []
    gap = half_time_gap(t, rms_db, onset_density)
    if gap is not None:
        halves = [{"start": kickoff, "end": gap[0]},
                  {"start": gap[1], "end": fulltime}]

    if fulltime - kickoff < 0.3 * duration:
        return fallback
    lo = max(0.0, kickoff - 30)
    hi = min(duration, fulltime + 30)
    return lo, hi, halves, None
