"""Contract 4 candidate extraction from a scored feature frame.

Peaks: greedy argmax over the 3-s smoothed learned probability, 12 s NMS,
a duration-scaled cap (~0.7/min, 15..80) trimmed to MIN_N by relative
probability, restricted to in-match seconds. Type heuristic (first match wins):
goal (roi spike + net disturbance + crowd + restart lull), shot (roi
spike), goalmouth (roi z>=1), crowd (audio z>=2 without a near-goal spike),
attack (anything else). Confidence = 0.9 * (0.5*prob + 0.5*rank decay)
so candidates are spread and monotonic rather than saturated at 0.9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from highlights.fusion.score import RULE_COLS, pick_peaks, robust_z

NMS = 12
PER_MIN = 0.7    # target candidates per in-match minute
MIN_N = 15
MAX_N = 80
REL_PROB = 0.25  # drop peaks (beyond MIN_N) below 25% of the top probability
SHOT_Z = 2.0
GOALMOUTH_Z = 1.0
NET_Z = 2.0
GOAL_AUD_Z = 1.5
CROWD_Z = 2.0
LULL_RATIO = 0.6
LULL_POST = (15.0, 45.0)   # seconds after the peak
LULL_PRE = (-30.0, 0.0)    # seconds before the peak

ACT = 0.45        # activity floor for extending a clip window
MAX_PRE = 12.0
MAX_POST = 12.0
GOAL_MAX_POST = 15.0
MIN_LEN = 4.0
MAX_LEN = 30.0
DEFAULT_PRIOR = {"goal": {"pre": 5.0, "post": 5.0},
                 "other": {"pre": 3.0, "post": 3.0}}


def dynamic_window(i: int, t: np.ndarray, learned: np.ndarray,
                   motion: np.ndarray, duration: float, etype: str,
                   prior: dict | None = None) -> tuple[float, float]:
    """Clip window around peak i, extended while activity stays >= ACT of
    the peak (one single-second dip allowed). prior = {"goal"/"other":
    {"pre","post"}} sets the minimum padding on each side."""
    prior = prior or DEFAULT_PRIOR
    p = prior.get("goal" if etype == "goal" else "other",
                  DEFAULT_PRIOR["other"])
    min_pre, min_post = float(p["pre"]), float(p["post"])
    max_post = GOAL_MAX_POST if etype == "goal" else MAX_POST

    li = learned[i] if learned[i] > 0 else 1e-9
    mi = motion[i] if motion[i] > 0 else 1e-9
    act = np.maximum(learned / li, motion / mi)
    act[i] = 1.0

    def _walk(step: int, limit: float) -> int:
        j_last, dips = i, 0
        j = i + step
        while 0 <= j < len(t) and abs(t[j] - t[i]) < limit:
            if act[j] >= ACT:
                j_last, dips = j, 0
            else:
                dips += 1
                if dips > 1:
                    break
            j += step
        return j_last

    jb = _walk(-1, MAX_PRE)
    jf = _walk(+1, max_post)
    t_start = min(float(t[jb]), float(t[i]) - min_pre)
    t_end = max(float(t[jf]), float(t[i]) + min_post)
    t_start, t_end = max(0.0, t_start), min(float(duration), t_end)
    if duration > 0 and t_end - t_start > MAX_LEN:
        mid = float(t[i])
        t_start = max(0.0, mid - MAX_LEN / 2)
        t_end = min(float(duration), mid + MAX_LEN / 2)
    if t_end - t_start < MIN_LEN:
        t_start = max(0.0, float(t[i]) - MIN_LEN / 2)
        t_end = t_start + MIN_LEN
        if duration > 0 and t_end > duration:
            t_end = float(duration)
            t_start = max(0.0, t_end - MIN_LEN)
    return t_start, t_end


def _top_signals(zrow: np.ndarray, cols: list[str], k: int = 3) -> list[str]:
    if len(cols) == 0:
        return []
    return [cols[i] for i in np.argsort(zrow)[::-1][:k]]


def make_candidates(df: pd.DataFrame, learned: np.ndarray, rule: np.ndarray,
                    in_match: np.ndarray, duration: float, *,
                    vp: np.ndarray | None = None,
                    model_version: str | None = None,
                    window_prior: dict | None = None) -> dict:
    t = df["t"].to_numpy(dtype=float) if "t" in df.columns \
        else df.index.to_numpy(dtype=float)
    in_match = np.asarray(in_match, dtype=bool)

    mask = in_match if in_match.any() else np.ones(len(df), dtype=bool)
    def _z(col: str) -> np.ndarray:
        if col in df.columns and float(df[col].astype(float).var()) > 0:
            return robust_z(df, [col], mask)[col].to_numpy()
        return np.zeros(len(df))

    roi_z = _z("motion_goal_roi")
    net_z = _z("net_disturbance")
    aud_z = _z("z60_rms")
    motion = (df["motion_total"].to_numpy(dtype=float)
              if "motion_total" in df.columns else np.zeros(len(df)))

    zcols = [c for c in RULE_COLS if c in df.columns
             and float(df[c].astype(float).var()) > 0]
    z = robust_z(df, zcols, mask).clip(-1, 4).fillna(0.0) if zcols \
        else pd.DataFrame(index=df.index)

    lo = float(t[mask].min()) if mask.any() else 0.0
    hi = float(t[mask].max()) if mask.any() else float(duration)

    peak_idx = pick_peaks(np.asarray(learned, dtype=float), t, mask,
                          nms=NMS,
                          top=int(np.clip(round((hi - lo) / 60 * PER_MIN),
                                          MIN_N, MAX_N)))
    # drop weak peaks beyond the MIN_N floor
    peak_idx = np.asarray(peak_idx)
    if len(peak_idx) > MIN_N:
        floor = REL_PROB * float(np.max(np.asarray(learned)[peak_idx]))
        peak_idx = np.concatenate([
            peak_idx[:MIN_N],
            peak_idx[MIN_N:][np.asarray(learned)[peak_idx[MIN_N:]] >= floor]])

    def _lull(i: int) -> bool:
        post = (t >= t[i] + LULL_POST[0]) & (t < t[i] + LULL_POST[1])
        pre = (t >= t[i] + LULL_PRE[0]) & (t < t[i] + LULL_PRE[1])
        if not post.any() or not pre.any():
            return False
        return float(motion[post].mean()) < LULL_RATIO * float(motion[pre].mean())

    events = []
    for rank, i in enumerate(peak_idx, start=1):
        peak_t = float(t[i])
        prob = float(learned[i])
        lull = _lull(i)
        if roi_z[i] > SHOT_Z and net_z[i] >= NET_Z and aud_z[i] >= GOAL_AUD_Z and lull:
            etype = "goal"
        elif roi_z[i] > SHOT_Z:
            etype = "shot"
        elif roi_z[i] >= GOALMOUTH_Z:
            etype = "goalmouth"
        elif aud_z[i] >= CROWD_Z:
            etype = "crowd"
        else:
            etype = "attack"
        conf = 0.9 * (0.5 * prob + 0.5 * (1 - (rank - 1) / max(1, len(peak_idx))))
        zrow = z.iloc[i].to_numpy() if zcols else np.zeros(0)
        sig = {
            "window": "dynamic",
            "learned_prob": round(prob, 4),
            "rule_score": round(float(rule[i]) if len(rule) else 0.0, 4),
            "motion_goal_roi_z": round(float(roi_z[i]), 2),
            "net_disturbance_z": round(float(net_z[i]), 2),
            "z60_rms_z": round(float(aud_z[i]), 2),
            "restart_lull": bool(lull),
            "z300_rms": round(float(df["z300_rms"].iloc[i]), 2)
            if "z300_rms" in df.columns else 0.0,
            "top_signals": _top_signals(zrow, zcols),
        }
        if vp is not None:
            sig["verdict_prob"] = round(float(vp[i]), 4)
            sig["model"] = model_version
        parts = [f"goal-ROI motion z={roi_z[i]:.1f}"]
        if etype in ("goal", "shot", "goalmouth"):
            parts += [f"net disturbance z={net_z[i]:.1f}",
                      f"crowd z={aud_z[i]:.1f}"]
            if lull:
                parts.append("restart lull")
        else:
            parts.append(f"crowd z={aud_z[i]:.1f}")
        notes = ", ".join(parts) + f" -> {etype}"
        t_start, t_end = dynamic_window(i, t, np.asarray(learned, dtype=float),
                                        motion, duration, etype, window_prior)
        events.append({
            "id": f"event_{rank:03d}",
            "rank": rank,
            "type": etype,
            "t": peak_t,
            "t_start": t_start,
            "t_end": t_end,
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
