#!/usr/bin/env python3
"""Score each second for near-goal action and pick highlight peaks.

(a) rule: robust z (median/MAD over in-match seconds) of six signals,
    clipped to [-1, 4], averaged; 3 s box smoothing; greedy top-60 peaks
    with 12 s NMS restricted to in-match seconds.
(b) learned: [mean, max] aggregates over [t-3, t+3] of every feature
    column -> StandardScaler -> LogisticRegression(class_weight balanced,
    C=0.3). Positives: |t - event| <= 4 s; negatives: in-match seconds
    >=15 s from any event. Temporal 2-fold CV (split at t=3020) gives a
    held-out score for every second.

Writes peaks_rule.json / peaks_learned.json and prints precision/recall
at K in {10,20,30,40} (all events and confidence>=0.6) plus AUROC.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "outputs", "features_1s.parquet")
LABELS = os.path.join(HERE, "outputs", "labels.json")
OUT_RULE = os.path.join(HERE, "outputs", "peaks_rule.json")
OUT_LEARN = os.path.join(HERE, "outputs", "peaks_learned.json")

MATCH_LO, MATCH_HI = 1050, 4990
CV_SPLIT = 3020
NMS = 12
TOP_N = 60

RULE_COLS = ["motion_goal_roi", "net_disturbance", "n_near_box",
             "rush_near_3s", "p_ball_out_of_play", "z300_rms"]


def robust_z(df, cols, mask):
    """Median/MAD z-score over the masked (in-match) seconds."""
    z = pd.DataFrame(index=df.index)
    for c in cols:
        v = df[c].astype(float)
        med = v[mask].median()
        mad = (v[mask] - med).abs().median()
        scale = mad * 1.4826 if mad > 0 else (v[mask].std() or 1.0)
        z[c] = (v - med) / (scale if scale > 0 else 1.0)
    return z


def pick_peaks(score, t, valid_mask, nms=NMS, top=TOP_N):
    """Greedy argmax with nms-second exclusion over valid seconds."""
    work = np.where(valid_mask, score, -np.inf)
    idxs = []
    for _ in range(top):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]):
            break
        idxs.append(i)
        lo, hi = max(0, i - nms), min(len(work), i + nms + 1)
        work[lo:hi] = -np.inf
    return sorted(idxs, key=lambda i: -score[i])


def evaluate(peaks_t, events, ks=(10, 20, 30, 40)):
    res = {}
    for k in ks:
        pk = peaks_t[:k]
        hits = sum(1 for p in pk if min(abs(p - e) for e in events) <= 6) \
            if events else 0
        rec = sum(1 for e in events if min(abs(e - p) for p in pk) <= 6) \
            if events else 0
        res[k] = (hits / k, rec / len(events) if events else 0.0)
    return res


def top_signals(zrow, cols, k=3):
    return [cols[i] for i in np.argsort(zrow)[::-1][:k]]


def main():
    df = pd.read_parquet(FEAT).set_index("t")
    with open(LABELS) as f:
        labels = json.load(f)
    ev_t = np.array([e["t"] for e in labels])
    ev_conf = np.array([e.get("confidence", 0) for e in labels])
    t = df.index.to_numpy(dtype=float)
    in_match = ((t >= MATCH_LO) & (t <= MATCH_HI)).astype(bool)

    # ---------------- (a) rule score ----------------
    z = robust_z(df, RULE_COLS, in_match).clip(-1, 4).fillna(0.0)
    rule = z.mean(axis=1).to_numpy()
    rule_s = np.convolve(rule, np.ones(3) / 3, mode="same")
    peak_idx = pick_peaks(rule_s, t, in_match)
    peaks_rule = [{
        "t": float(t[i]),
        "score": float(rule_s[i]),
        "rank": r + 1,
        "top_signals": top_signals(z.iloc[i].to_numpy(), RULE_COLS),
    } for r, i in enumerate(peak_idx)]
    with open(OUT_RULE, "w") as f:
        json.dump(peaks_rule, f, indent=1)

    # ---------------- (b) learned score ----------------
    feat_cols = [c for c in df.columns if c != "in_match"]
    win = 3
    dfw = df[feat_cols].astype(float)
    mean_w = dfw.rolling(2 * win + 1, center=True, min_periods=1).mean()
    max_w = dfw.rolling(2 * win + 1, center=True, min_periods=1).max()
    X = np.hstack([mean_w.to_numpy(), max_w.to_numpy()])
    X = np.nan_to_num(X, nan=0.0)

    dist = np.abs(t[:, None] - ev_t[None, :]).min(axis=1) if len(ev_t) \
        else np.full(len(t), np.inf)
    y_pos = dist <= 4
    y_neg = (~y_pos) & (dist >= 15) & in_match
    eval_mask = (y_pos | y_neg)
    y = y_pos.astype(int)

    held = np.full(len(t), np.nan)
    for tr_mask, te_mask in [(t < CV_SPLIT, t >= CV_SPLIT),
                             (t >= CV_SPLIT, t < CV_SPLIT)]:
        tr = tr_mask & eval_mask
        te = te_mask & in_match
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight="balanced", C=0.3,
                                 max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        held[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    le_mask = eval_mask & np.isfinite(held)
    auroc = roc_auc_score(y[le_mask], held[le_mask])
    learn_s = np.convolve(np.nan_to_num(held), np.ones(3) / 3, mode="same")
    peak_idx_l = pick_peaks(learn_s, t, in_match & np.isfinite(held))
    peaks_learn = [{
        "t": float(t[i]),
        "score": float(learn_s[i]),
        "rank": r + 1,
        "top_signals": top_signals(z.iloc[i].to_numpy(), RULE_COLS),
    } for r, i in enumerate(peak_idx_l)]
    with open(OUT_LEARN, "w") as f:
        json.dump(peaks_learn, f, indent=1)

    # ---------------- metrics ----------------
    hi = ev_t[ev_conf >= 0.6]
    print(f"labels: {len(ev_t)} events, {len(hi)} with confidence>=0.6")
    print(f"learned held-out AUROC = {auroc:.3f}")
    for name, pk in [("rule", [p["t"] for p in peaks_rule]),
                     ("learned", [p["t"] for p in peaks_learn])]:
        print(f"\n== {name} ==")
        print("K    prec@K   rec@K   | prec@K conf>=.6  rec@K conf>=.6")
        ra = evaluate(pk, list(ev_t))
        rh = evaluate(pk, list(hi))
        for k in ra:
            print(f"{k:<4} {ra[k][0]:.3f}    {ra[k][1]:.3f}   | "
                  f"{rh[k][0]:.3f}           {rh[k][1]:.3f}")
    print(f"\nwrote {OUT_RULE} and {OUT_LEARN}")


if __name__ == "__main__":
    main()
