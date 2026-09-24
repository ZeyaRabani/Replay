"""Contract 4 candidate extraction from a scored feature frame.

Peaks: greedy argmax over the 3-s smoothed learned probability, 12 s NMS,
top 60, restricted to in-match seconds. Type heuristic: "shot" when the
robust z of motion_goal_roi at the peak exceeds 2 (strong motion in the
goal ROI), else "chance". Confidence = 0.9 * (0.5*prob + 0.5*rank decay)
so candidates are spread and monotonic rather than saturated at 0.9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from highlights.fusion.score import RULE_COLS, pick_peaks, robust_z

NMS = 12
TOP_N = 60
SHOT_Z = 2.0


def _top_signals(zrow: np.ndarray, cols: list[str], k: int = 3) -> list[str]:
    if len(cols) == 0:
        return []
    return [cols[i] for i in np.argsort(zrow)[::-1][:k]]


def make_candidates(df: pd.DataFrame, learned: np.ndarray, rule: np.ndarray,
                    in_match: np.ndarray, duration: float) -> dict:
    t = df["t"].to_numpy(dtype=float) if "t" in df.columns \
        else df.index.to_numpy(dtype=float)
    in_match = np.asarray(in_match, dtype=bool)

    mask = in_match if in_match.any() else np.ones(len(df), dtype=bool)
    roi_z = np.zeros(len(df))
    if "motion_goal_roi" in df.columns and float(df["motion_goal_roi"].astype(float).var()) > 0:
        roi_z = robust_z(df, ["motion_goal_roi"], mask)["motion_goal_roi"].to_numpy()

    zcols = [c for c in RULE_COLS if c in df.columns
             and float(df[c].astype(float).var()) > 0]
    z = robust_z(df, zcols, mask).clip(-1, 4).fillna(0.0) if zcols \
        else pd.DataFrame(index=df.index)

    peak_idx = pick_peaks(np.asarray(learned, dtype=float), t, mask,
                          nms=NMS, top=TOP_N)

    lo = float(t[mask].min()) if mask.any() else 0.0
    hi = float(t[mask].max()) if mask.any() else float(duration)

    events = []
    for rank, i in enumerate(peak_idx, start=1):
        peak_t = float(t[i])
        prob = float(learned[i])
        etype = "shot" if roi_z[i] > SHOT_Z else "chance"
        conf = 0.9 * (0.5 * prob + 0.5 * (1 - (rank - 1) / max(1, len(peak_idx))))
        zrow = z.iloc[i].to_numpy() if zcols else np.zeros(0)
        sig = {
            "learned_prob": round(prob, 4),
            "rule_score": round(float(rule[i]) if len(rule) else 0.0, 4),
            "motion_goal_roi_z": round(float(roi_z[i]), 2),
            "z300_rms": round(float(df["z300_rms"].iloc[i]), 2)
            if "z300_rms" in df.columns else 0.0,
            "top_signals": _top_signals(zrow, zcols),
        }
        notes = (f"learned p={prob:.2f}"
                 + (f"; goal-ROI motion z={roi_z[i]:.1f} -> shot"
                    if etype == "shot" else ""))
        events.append({
            "id": f"event_{rank:03d}",
            "rank": rank,
            "type": etype,
            "t": peak_t,
            "t_start": max(0.0, peak_t - 8.0),
            "t_end": min(float(duration), peak_t + 6.0),
            "confidence": round(conf, 3),
            "signals": sig,
            "notes": notes,
            "cross_validation": "pipeline_only",
            "status": "pending",
        })

    return {
        "source": "pipeline",
        "video_duration_s": float(duration),
        "match_window": [lo, hi],
        "events": events,
        "candidates": events,
    }
