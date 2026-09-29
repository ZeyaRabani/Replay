"""Verdict learning: collect labelled examples from reviewed candidates,
train a logistic model, and apply it in stage_candidates.

Training data lives in ``{workdir}/learning`` (Oracle: /data/learning) so
deleted projects keep contributing. ``HL_LEARN_DIR`` overrides the dir
(tests). Fully deterministic; no network.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from highlights.pipeline.features import MODEL_COLS
from highlights.pipeline.score import rolling_X

VERSION = "verdict_lr v1"
MIN_ROWS = 20

_train_lock = threading.Lock()


def learn_dir() -> Path:
    env = os.environ.get("HL_LEARN_DIR")
    if env:
        return Path(env)
    from highlights.app.backend.store import workdir
    return workdir() / "learning"


def _examples_path() -> Path:
    return learn_dir() / "examples.parquet"


def collect_project_examples(p) -> pd.DataFrame:
    """Labelled rows from a project's reviewed candidates."""
    feat_path = p.pipeline_dir / "features_1s.parquet"
    if not feat_path.is_file() or p.video is None:
        return pd.DataFrame()
    duration = float(p.video.duration_s)
    feats = pd.read_parquet(feat_path)
    if "t" not in feats.columns:
        feats["t"] = feats.index
    ft = feats["t"].to_numpy(dtype=float)
    X = rolling_X(feats, MODEL_COLS, window=3)
    sc_path = p.pipeline_dir / "scores.parquet"
    learned = (
        pd.read_parquet(sc_path)["learned"].to_numpy(dtype=float)
        if sc_path.is_file() else np.zeros(len(feats))
    )

    from highlights.app.backend.store import default_clip_window
    now = time.time()
    rows = []
    for c in p.candidates:
        if c.status not in ("confirmed", "rejected"):
            continue
        if duration > 0 and c.t > duration + 1:
            continue
        i = int(np.argmin(np.abs(ft - c.t))) if len(ft) else 0
        pre = c.t - c.clip_start
        post = c.clip_end - c.t
        d_pre, d_post = default_clip_window(c.t, c.type, duration)
        edited = (abs(pre - (c.t - d_pre)) > 0.25
                  or abs(post - (d_post - c.t)) > 0.25)
        row = {
            "project_id": p.id, "cand_id": c.id, "t": c.t,
            "label": 1 if c.status == "confirmed" else 0,
            "is_goal": bool(c.status == "confirmed" and c.type == "goal"),
            "ai_type": c.signals.get("ai_type") or c.type,
            "pre": pre, "post": post, "edited_window": edited,
            "verdict_at": now,
            "learned_prob": float(learned[i]) if len(learned) > i else 0.0,
        }
        for k in range(24):
            row[f"f{k:02d}"] = float(X[i, k]) if len(X) > i else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_examples(df: pd.DataFrame) -> None:
    """Upsert into examples.parquet keyed on (project_id, cand_id)."""
    if df.empty:
        return
    path = _examples_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["project_id", "cand_id"], keep="last")
    df.to_parquet(path, index=False)


_FEATURE_COLS = [f"f{i:02d}" for i in range(24)] + ["learned_prob"]


def _window_prior(df: pd.DataFrame) -> dict:
    pos = df[df["label"] == 1]

    def pick(mask, fb_pre, fb_post):
        sub = pos[mask]
        if len(sub) < 3:
            return {"pre": fb_pre, "post": fb_post}
        return {"pre": float(sub["pre"].median()),
                "post": float(sub["post"].median())}

    return {"goal": pick(pos["is_goal"], 5.0, 5.0),
            "other": pick(~pos["is_goal"], 3.0, 3.0)}


def train(examples: pd.DataFrame) -> dict:
    """Fit scaler+LR on f00..f23 + learned_prob; write model + meta."""
    if len(examples) < MIN_ROWS:
        return {"trained": False, "reason": f"need {MIN_ROWS} examples, have {len(examples)}"}
    y = examples["label"].to_numpy(dtype=int)
    if len(set(y.tolist())) < 2:
        return {"trained": False, "reason": "need both confirmed and rejected examples"}
    X = examples[_FEATURE_COLS].to_numpy(dtype=float)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # leave-one-project-out AUROC
    auroc = None
    projects = examples["project_id"].unique()
    if len(projects) >= 2:
        aucs = []
        for pid in projects:
            te = (examples["project_id"] == pid).to_numpy()
            tr = ~te
            if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        if aucs:
            auroc = float(np.mean(aucs))

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000)
    clf.fit(sc.transform(X), y)

    ld = learn_dir()
    ld.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "clf": clf, "columns": _FEATURE_COLS,
                 "window": 3, "version": VERSION}, ld / "verdict_lr.joblib")
    meta = {
        "trained_at": time.time(),
        "n_examples": len(examples),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_goals": int(examples[examples["label"] == 1]["is_goal"].sum()),
        "projects": {
            pid: {"pos": int(((examples["project_id"] == pid) & (y == 1)).sum()),
                  "neg": int(((examples["project_id"] == pid) & (y == 0)).sum())}
            for pid in projects
        },
        "auroc_lopo": auroc,
        "window_prior": _window_prior(examples),
        "version": VERSION,
        "trained": True,
    }
    (ld / "verdict_lr.json").write_text(json.dumps(meta, indent=1))
    return meta


def train_from_registry(reg, project=None) -> dict:
    """Collect verdict examples (given project + every other project that
    has any) and retrain. Non-blocking: returns busy if already training."""
    if not _train_lock.acquire(blocking=False):
        return {"trained": False, "reason": "busy"}
    try:
        for p in reg.list_projects():
            if any(c.status != "pending" for c in p.candidates):
                # one bad project must not lose the rest
                with contextlib.suppress(Exception):
                    save_examples(collect_project_examples(p))
        path = _examples_path()
        if not path.exists():
            return {"trained": False, "reason": "no examples"}
        return train(pd.read_parquet(path))
    finally:
        _train_lock.release()


def load_verdict_model() -> dict | None:
    """joblib model + json meta merged, or None."""
    ld = learn_dir()
    jb, js = ld / "verdict_lr.joblib", ld / "verdict_lr.json"
    if not jb.is_file():
        return None
    try:
        vm = joblib.load(jb)
        if js.is_file():
            vm.update(json.loads(js.read_text()))
        return vm
    except Exception:
        return None


def verdict_prob(df: pd.DataFrame, learned: np.ndarray, vm: dict) -> np.ndarray:
    """Per-second P(user confirms) from rolling features + base learned prob."""
    X = rolling_X(df, MODEL_COLS, window=int(vm.get("window", 3)))
    X = np.hstack([X, np.asarray(learned, dtype=float)[:, None]])
    return vm["clf"].predict_proba(vm["scaler"].transform(X))[:, 1]
