"""Score a per-second feature frame with the trained audio+motion model.

learned_prob: same rolling mean/max features as fusion/score.py (b) ->
StandardScaler -> LogisticRegression. Final per-second score = learned
probability smoothed with a 3 s box filter. The rule score (robust-z mean
of the fusion RULE_COLS that have non-zero variance in this frame, clip
[-1,4], 3 s box) is kept as a signal only.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from highlights.fusion.score import RULE_COLS, pick_peaks, robust_z

MODEL_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "models" / "audio_motion_lr.joblib"


def load_model(path: str | Path | None = None) -> dict:
    return joblib.load(path or MODEL_PATH)


def rolling_X(df: pd.DataFrame, columns: list[str], window: int) -> np.ndarray:
    dfw = df[columns].astype(float)
    mean_w = dfw.rolling(2 * window + 1, center=True, min_periods=1).mean()
    max_w = dfw.rolling(2 * window + 1, center=True, min_periods=1).max()
    return np.nan_to_num(np.hstack([mean_w.to_numpy(), max_w.to_numpy()]), nan=0.0)


def learned_prob(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Raw per-second P(event) (unsmoothed)."""
    X = rolling_X(df, list(model["columns"]), int(model["window"]))
    return model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]


def smooth3(v: np.ndarray) -> np.ndarray:
    return np.convolve(v, np.ones(3) / 3, mode="same")


def _rule_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in RULE_COLS if c in df.columns
            and float(df[c].astype(float).var()) > 0]


def rule_z(df: pd.DataFrame, in_match: np.ndarray) -> pd.DataFrame:
    """Robust-z frame over the usable RULE_COLS (zero-variance cols excluded)."""
    cols = _rule_cols(df)
    if not cols:
        return pd.DataFrame(index=df.index)
    return robust_z(df, cols, np.asarray(in_match, dtype=bool)).clip(-1, 4).fillna(0.0)


def rule_score(df: pd.DataFrame, in_match: np.ndarray) -> np.ndarray:
    z = rule_z(df, in_match)
    if z.shape[1] == 0:
        return np.zeros(len(df))
    return smooth3(z.mean(axis=1).to_numpy())


def score_frame(df: pd.DataFrame, model: dict, in_match: np.ndarray) -> pd.DataFrame:
    """Return a frame with columns t, learned (3-s smoothed), rule."""
    t = df["t"].to_numpy(dtype=float) if "t" in df.columns else df.index.to_numpy(dtype=float)
    learned = smooth3(learned_prob(df, model))
    rule = rule_score(df, in_match)
    return pd.DataFrame({"t": t, "learned": learned, "rule": rule})


__all__ = [
    "MODEL_PATH",
    "RULE_COLS",
    "learned_prob",
    "load_model",
    "pick_peaks",
    "robust_z",
    "rule_score",
    "rule_z",
    "score_frame",
    "smooth3",
]
