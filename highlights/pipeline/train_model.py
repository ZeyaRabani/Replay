"""Train the audio+motion logistic model on the committed fusion outputs.

Same learned-features setup as highlights/fusion/score.py (b): rolling
mean & max over a +/-3 s window of each of the 11 audio/motion columns +
whistle (24 features), positives |t-event|<=4 s, negatives in-match
seconds >=15 s from any event, temporal 2-fold CV split at t=3020 for the
held-out AUROC. The final scaler+classifier is fitted on ALL eval rows and
saved to highlights/pipeline/models/audio_motion_lr.joblib.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from highlights.pipeline.features import MODEL_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
FUSION_OUT = os.path.join(HERE, "..", "fusion", "outputs")
FEAT = os.path.join(FUSION_OUT, "features_1s.parquet")
LABELS = os.path.join(FUSION_OUT, "labels.json")
MODELS = os.path.join(HERE, "models")

MATCH_LO, MATCH_HI = 1050, 4990
CV_SPLIT = 3020
WINDOW = 3


def rolling_X(df: pd.DataFrame, columns: list[str], window: int = WINDOW) -> np.ndarray:
    dfw = df[columns].astype(float)
    mean_w = dfw.rolling(2 * window + 1, center=True, min_periods=1).mean()
    max_w = dfw.rolling(2 * window + 1, center=True, min_periods=1).max()
    return np.nan_to_num(np.hstack([mean_w.to_numpy(), max_w.to_numpy()]), nan=0.0)


def main() -> float:
    df = pd.read_parquet(FEAT).set_index("t")
    with open(LABELS) as f:
        labels = json.load(f)
    ev_t = np.array([e["t"] for e in labels])
    t = df.index.to_numpy(dtype=float)
    in_match = (t >= MATCH_LO) & (t <= MATCH_HI)

    X = rolling_X(df, MODEL_COLS)
    dist = np.abs(t[:, None] - ev_t[None, :]).min(axis=1) if len(ev_t) \
        else np.full(len(t), np.inf)
    y_pos = dist <= 4
    y_neg = (~y_pos) & (dist >= 15) & in_match
    eval_mask = y_pos | y_neg
    y = y_pos.astype(int)

    # temporal 2-fold held-out AUROC, same as fusion/score.py
    held = np.full(len(t), np.nan)
    for tr_mask, te_mask in [(t < CV_SPLIT, t >= CV_SPLIT),
                             (t >= CV_SPLIT, t < CV_SPLIT)]:
        tr = tr_mask & eval_mask
        te = te_mask & in_match
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        held[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    le_mask = eval_mask & np.isfinite(held)
    auroc = roc_auc_score(y[le_mask], held[le_mask])
    print(f"labels: {len(ev_t)} events; eval rows: {int(eval_mask.sum())}")
    print(f"audio+motion held-out AUROC = {auroc:.3f}")

    # final model on ALL eval rows
    sc = StandardScaler().fit(X[eval_mask])
    clf = LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000)
    clf.fit(sc.transform(X[eval_mask]), y[eval_mask])

    os.makedirs(MODELS, exist_ok=True)
    meta = {
        "columns": MODEL_COLS,
        "window": WINDOW,
        "auroc_heldout": float(auroc),
        "trained_on": "5qj_nsQSzvQ",
        "version": "audio_motion_lr v1",
    }
    joblib_path = os.path.join(MODELS, "audio_motion_lr.joblib")
    joblib.dump({"scaler": sc, "clf": clf, **meta}, joblib_path)
    json_path = os.path.join(MODELS, "audio_motion_lr.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {joblib_path} and {json_path}")
    return float(auroc)


if __name__ == "__main__":
    main()
