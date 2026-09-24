#!/usr/bin/env python3
"""Build the fused per-second feature table.

Joins the 1 s feature JSONs from the motion, audio, spotting and tracking
tracks on integer second t in [0, 5337], adds a `whistle` overlap flag and
an `in_match` window flag, z-scores every numeric feature column and fills
NaN with 0. Writes highlights/fusion/outputs/features_1s.parquet.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HL = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs", "features_1s.parquet")

T_MAX = 5337
MATCH_LO, MATCH_HI = 1050, 4990  # kickoff ~1105, FT ~4986, with margin

SOURCES = {
    "motion": (os.path.join(HL, "motion", "outputs", "features_1s.json"),
               ["motion_total", "motion_goal_roi", "motion_far",
                "near_frac", "net_disturbance"]),
    "audio": (os.path.join(HL, "audio", "outputs", "features_1s.json"),
              ["rms_db", "z60_rms", "z300_rms", "z300_speech",
               "onset_density", "whistle_frac"]),
    "spotting": (os.path.join(HL, "spotting", "outputs", "features_1s.json"),
                 ["p_ball_out_of_play", "p_shots_on_target",
                  "p_shots_off_target", "p_goal", "p_foreground_max"]),
    "tracking": (os.path.join(HL, "tracking", "outputs", "features_1s.json"),
                 ["n_players", "n_near_box", "n_far_box", "n_centre",
                  "rush_near_3s", "rush_far_3s", "restart_flag",
                  "centre_cluster_flag", "mean_speed", "spread",
                  "frac_near_third", "frac_far_third", "keeper_x"]),
}


def load(path, cols):
    with open(path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["rows"], columns=d["columns"])
    df["t"] = df["t"].astype(int)
    keep = ["t"] + [c for c in cols if c in df.columns]
    return df[keep].set_index("t")


def main():
    idx = pd.Index(range(T_MAX + 1), name="t")
    df = pd.DataFrame(index=idx)
    for name, (path, cols) in SOURCES.items():
        df = df.join(load(path, cols))
        print(f"{name}: {len(cols)} cols", flush=True)

    # whistle flag: any whistle interval overlapping [t, t+1)
    with open(os.path.join(HL, "audio", "outputs", "whistles.json")) as f:
        whistles = json.load(f)["whistles"]
    w = np.zeros(T_MAX + 1, dtype=int)
    for ev in whistles:
        lo = int(np.floor(ev["t_start"]))
        hi = int(np.floor(ev["t_end"]))
        for t in range(lo, hi + 1):
            # whistle [s,e) overlaps bin [t, t+1)?
            if ev["t_start"] < t + 1 and ev["t_end"] > t:
                w[t] = 1
    df["whistle"] = w

    df["in_match"] = ((df.index >= MATCH_LO) & (df.index <= MATCH_HI)).astype(int)

    feat_cols = [c for c in df.columns if c not in ("in_match",)]
    # z-score then NaN -> 0
    for c in feat_cols:
        s = df[c].astype(float)
        mu, sd = s.mean(), s.std()
        df[c] = (s - mu) / sd if sd > 0 else s * 0.0
    df[feat_cols] = df[feat_cols].fillna(0.0)
    df["in_match"] = df["in_match"].fillna(0).astype(int)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.reset_index().to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {df.shape[0]} rows x {df.shape[1]} cols", flush=True)


if __name__ == "__main__":
    main()
