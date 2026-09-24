"""Stats computation (Contract 5): timeline bins, event counts, activity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FUSION_OUT = REPO_ROOT / "highlights" / "fusion" / "outputs"

BIN_S = 30


def _col(df: pd.DataFrame, names: list[str], prefix: str | None = None) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    if prefix:
        for c in df.columns:
            if c.startswith(prefix):
                return df[c]
    return pd.Series(0.0, index=df.index)


def _norm(vals: list[float], in_window: list[bool]) -> list[float]:
    wv = [v for v, w in zip(vals, in_window, strict=True) if w and not math.isnan(v)]
    lo = min(wv) if wv else 0.0
    hi = max(wv) if wv else 0.0
    span = hi - lo
    out = []
    for v in vals:
        if math.isnan(v) or span <= 0:
            out.append(0.0)
        else:
            out.append(round(min(1.0, max(0.0, (v - lo) / span)), 4))
    return out


def compute_stats(
    features_df: pd.DataFrame,
    candidates_events: list[dict],
    duration: float,
    match_window: list | None,
    halves: list | None,
    whistles: list | None,
    *,
    model: str = "unknown",
    auroc_reference: float | None = None,
    notes: str = "",
) -> dict:
    duration = float(duration)
    window = list(match_window) if match_window else [0.0, duration]
    halves = halves or []
    whistles = whistles or []

    n_bins = max(1, math.ceil(duration / BIN_S))
    bin_starts = [i * BIN_S for i in range(n_bins)]
    motion_col = _col(features_df, ["motion_total"], prefix="motion")
    audio_col = _col(features_df, ["rms_db", "z300_rms", "z60_rms"])
    t_col = features_df["t"] if "t" in features_df.columns else pd.Series(dtype=float)

    events = [e for e in candidates_events if e.get("cross_validation") != "rejected"]

    motion_means: list[float] = []
    audio_means: list[float] = []
    for b in bin_starts:
        mask = (t_col >= b) & (t_col < b + BIN_S)
        motion_means.append(float(motion_col[mask].mean()) if mask.any() else float("nan"))
        audio_means.append(float(audio_col[mask].mean()) if mask.any() else float("nan"))

    # fill empty bins with the per-bin min
    def _fill(vals: list[float]) -> list[float]:
        real = [v for v in vals if not math.isnan(v)]
        m = min(real) if real else 0.0
        return [v if not math.isnan(v) else m for v in vals]

    in_window = [window[0] <= b < window[1] for b in bin_starts]
    motion_n = _norm(_fill(motion_means), in_window)
    audio_n = _norm(_fill(audio_means), in_window)

    timeline = []
    for i, b in enumerate(bin_starts):
        exc = min(1.0, max(0.0, 0.5 * motion_n[i] + 0.5 * audio_n[i]))
        n_ev = sum(1 for e in events if b <= float(e.get("t", 0.0)) < b + BIN_S)
        timeline.append(
            {
                "t": b,
                "motion": round(float(motion_n[i]), 4),
                "audio": round(float(audio_n[i]), 4),
                "excitement": round(float(exc), 4),
                "events": int(n_ev),
            }
        )

    events_by_type: dict[str, int] = {}
    for e in events:
        ty = str(e.get("type", "other"))
        events_by_type[ty] = events_by_type.get(ty, 0) + 1

    types = ["goal", "shot", "chance"] + sorted({str(e.get("type", "other")) for e in events} - {"goal", "shot", "chance"})
    n10 = max(1, math.ceil(duration / 600.0))
    per10 = []
    for i in range(n10):
        start = i * 600
        row = {"t": start}
        counts: dict[str, int] = {}
        for e in events:
            if start <= float(e.get("t", 0.0)) < start + 600:
                ty = str(e.get("type", "other"))
                counts[ty] = counts.get(ty, 0) + 1
        for ty in types:
            row[ty] = int(counts.get(ty, 0))
        per10.append(row)

    top = sorted(events, key=lambda e: (-float(e.get("confidence", 0.0)), float(e.get("t", 0.0))))[:10]
    top_moments = [
        {
            "t": float(e.get("t", 0.0)),
            "type": str(e.get("type", "other")),
            "confidence": float(e.get("confidence", 0.0)),
            "reason": e.get("notes") or f"{e.get('cross_validation')}, confidence {float(e.get('confidence', 0.0)):.2f}",
        }
        for e in top
    ]

    win_idx = [i for i, w in enumerate(in_window) if w]
    if win_idx:
        mean_motion = round(sum(motion_n[i] for i in win_idx) / len(win_idx), 4)
        peak_motion_t = int(bin_starts[max(win_idx, key=lambda i: motion_n[i])])
        loudest_t = int(bin_starts[max(win_idx, key=lambda i: audio_n[i])])
        if len(win_idx) >= 10:
            best_a, best_mean = win_idx[0], None
            for j in range(len(win_idx) - 9):
                chunk = win_idx[j : j + 10]
                if chunk != list(range(chunk[0], chunk[0] + 10)):
                    continue  # not consecutive bins
                m = sum(motion_n[i] + audio_n[i] for i in chunk) / 10
                if best_mean is None or m < best_mean:
                    best_mean, best_a = m, chunk[0]
            quietest = [int(bin_starts[best_a]), int(bin_starts[best_a] + 300)]
        else:
            a = bin_starts[win_idx[0]]
            quietest = [int(a), int(a + 300)]
    else:
        mean_motion = 0.0
        peak_motion_t = loudest_t = 0
        quietest = [0, 300]

    return {
        "duration_s": duration,
        "match_window": [float(window[0]), float(window[1])],
        "halves": [{"start": float(h["start"]), "end": float(h["end"])} for h in halves],
        "bin_s": BIN_S,
        "timeline": timeline,
        "events_by_type": {k: int(v) for k, v in events_by_type.items()},
        "events_per_10min": per10,
        "top_moments": top_moments,
        "whistles": sorted(round(float(w), 2) for w in whistles),
        "activity": {
            "mean_motion": float(mean_motion),
            "peak_motion_t": peak_motion_t,
            "loudest_t": loudest_t,
            "quietest_stretch": quietest,
        },
        "pipeline": {
            "model": model,
            "auroc_reference": auroc_reference,
            "notes": notes,
        },
    }


def demo_stats() -> dict:
    """Stats for the bundled demo match from committed fusion outputs."""
    df = pd.read_parquet(FUSION_OUT / "features_1s.parquet")
    cf = json.loads((FUSION_OUT / "candidates.json").read_text())
    whistles: list[float] = []
    wpath = REPO_ROOT / "highlights" / "audio" / "outputs" / "whistles.json"
    if wpath.is_file():
        whistles = [w["t_start"] for w in json.loads(wpath.read_text()).get("whistles", [])]
    return compute_stats(
        df,
        cf["events"],
        float(cf["video_duration_s"]),
        cf.get("match_window"),
        [],  # half-time break is cut from the demo video
        whistles,
        model="fusion learned+rule (committed outputs)",
        auroc_reference=0.78,
        notes="Precomputed from highlights/fusion/outputs for the demo match; halves not detected (break cut from video)",
    )
