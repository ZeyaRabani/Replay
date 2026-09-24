"""Candidate events from features_1s.json (shared schema).

Rules (all thresholds in metres / players, see README):
  chance  : players rush toward a goal (mean per-track x displacement over
            3 s >= RUSH_M) and the box at that end is crowded (>= BOX_N)
            within a few seconds. Near-goal chances get higher confidence
            than far-goal ones (far end is low precision).
  goal    : a chance followed within GOAL_WINDOW_S by a centre-cluster
            (kickoff) formation. Very conservative confidence.
  other   : centre-cluster / stationary-cluster restart patterns not preceded
            by a chance (signals.kind = "restart_centre" / "restart_static"),
            and *everything* before WARMUP_END_S is re-labelled
            type="other", signals.kind="warmup" (original type kept in
            signals.warmup_type) so the parent can exclude it.
"""

import argparse
import json

import numpy as np

WARMUP_END_S = 1020.0
RUSH_M = 6.0
BOX_N = 3.0
FAR_BOX_N = 2.0
MERGE_GAP_S = 12.0
GOAL_WINDOW_S = (10.0, 75.0)
KICKOFF_MIN_S = 2


def runs(mask, t):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return out


def merge(runs_, gap):
    merged = []
    for a, b in runs_:
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def smooth(x, k=3):
    x = np.where(np.isfinite(x), x, 0.0)
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, required=True)
    args = ap.parse_args()

    with open(args.features) as fh:
        d = json.load(fh)
    cols = d["columns"]
    R = np.array([[np.nan if v is None else v for v in r] for r in d["rows"]],
                 dtype=float)
    c = {n: R[:, i] for i, n in enumerate(cols)}
    t = c["t"]
    n = len(t)

    def win(arr, a, b, fn=np.nanmax):
        m = (t >= a) & (t <= b)
        v = arr[m]
        v = v[np.isfinite(v)]
        return float(fn(v)) if len(v) else float("nan")

    events = []

    for end, rush_col, box_col, box_n, conf_scale in (
        ("near", "rush_near_3s", "n_near_box", BOX_N, 1.0),
        ("far", "rush_far_3s", "n_far_box", FAR_BOX_N, 0.55),
    ):
        rush = smooth(c[rush_col])
        box = smooth(c[box_col])
        # rush at s, crowded box within [s-2, s+6]
        crowded = np.zeros(n, bool)
        for i in range(n):
            lo, hi = max(0, i - 2), min(n, i + 7)
            crowded[i] = np.nanmax(box[lo:hi]) >= box_n
        cand = (rush >= RUSH_M) & crowded & (c["n_players"] >= 5)
        for a, b in merge(runs(cand, t), MERGE_GAP_S):
            t0, t1 = a - 3.0, b + 8.0
            m = (t >= t0) & (t <= t1)
            peak_i = int(np.where(m)[0][np.nanargmax(box[m])])
            r_max = win(rush, a, b)
            b_max = win(box, t0, t1)
            conf = 0.25 + 0.06 * min(b_max, 6) + 0.015 * min(r_max, 20)
            conf = float(min(0.9, conf) * conf_scale)
            events.append({
                "type": "chance",
                "t": float(t[peak_i]),
                "t_start": float(max(0.0, t0)),
                "t_end": float(min(args.duration, t1)),
                "confidence": round(conf, 3),
                "signals": {
                    "goal_end": end,
                    "rush_max_m_3s": round(r_max, 2),
                    "box_players_max": round(b_max, 2),
                    "n_players_mean": round(win(c["n_players"], t0, t1, np.nanmean), 2),
                    "spread_min_m": round(win(c["spread"], t0, t1, np.nanmin), 2),
                    "mean_speed_max": round(win(c["mean_speed"], t0, t1), 2),
                    "keeper_x_min_m": round(win(c["keeper_x"], t0, t1, np.nanmin), 2),
                },
                "notes": (f"{end}-goal attack: cluster moved {r_max:.1f} m toward the "
                          f"{end} goal in 3 s, {b_max:.1f} players in the {end} box"),
            })

    # kickoff / restart patterns
    centre_runs = merge(runs(c["centre_cluster_flag"] == 1, t), 6.0)
    centre_runs = [(a, b) for a, b in centre_runs if b - a + 1 >= KICKOFF_MIN_S]
    static_runs = merge(runs(c["restart_flag"] == 1, t), 6.0)

    chances = sorted([e for e in events if e["type"] == "chance"], key=lambda e: e["t"])
    used_centre = set()
    for e in chances:
        for a, b in centre_runs:
            dt = a - e["t"]
            if GOAL_WINDOW_S[0] <= dt <= GOAL_WINDOW_S[1]:
                used_centre.add((a, b))
                n_c = win(c["n_centre"], a, b)
                conf = float(min(0.6, 0.15 + 0.5 * e["confidence"] + 0.03 * (b - a + 1)))
                events.append({
                    "type": "goal",
                    "t": e["t"],
                    "t_start": e["t_start"],
                    "t_end": float(b),
                    "confidence": round(conf, 3),
                    "signals": dict(e["signals"], kickoff_t_start=a, kickoff_t_end=b,
                                    kickoff_delay_s=round(dt, 1),
                                    kickoff_centre_players_max=round(n_c, 2)),
                    "notes": (f"{e['signals']['goal_end']}-goal attack followed after "
                              f"{dt:.0f} s by a centre-cluster formation "
                              f"({b - a + 1:.0f} s, {n_c:.1f} players near centre)"),
                })
                break

    for a, b in centre_runs:
        if (a, b) in used_centre:
            continue
        n_c = win(c["n_centre"], a, b)
        events.append({
            "type": "other",
            "t": float(a),
            "t_start": float(a),
            "t_end": float(b),
            "confidence": round(min(0.5, 0.15 + 0.05 * (b - a + 1)), 3),
            "signals": {"kind": "restart_centre",
                        "centre_players_max": round(n_c, 2),
                        "mean_speed_min": round(win(c["mean_speed"], a, b, np.nanmin), 2)},
            "notes": "centre-cluster formation without a preceding detected attack "
                     "(kickoff / half start / missed far-goal attack?)",
        })
    for a, b in static_runs:
        if b - a + 1 < 2:
            continue
        events.append({
            "type": "other",
            "t": float(a),
            "t_start": float(a),
            "t_end": float(b),
            "confidence": 0.2,
            "signals": {"kind": "restart_static",
                        "cx_m": round(win(c["cx"], a, b, np.nanmean), 2),
                        "spread_min_m": round(win(c["spread"], a, b, np.nanmin), 2)},
            "notes": "players clustered and near-stationary (restart / stoppage)",
        })

    # near-box crowding without a detected rush (weak cue; also what fires on
    # the warm-up shooting drills)
    covered = [(e["t_start"], e["t_end"]) for e in events if e["type"] in ("chance", "goal")]
    crowd = smooth(c["n_near_box"]) >= BOX_N
    for a, b in merge(runs(crowd, t), MERGE_GAP_S):
        if b - a + 1 < 4 or any(a <= t1 and b >= t0 for t0, t1 in covered):
            continue
        b_max = win(c["n_near_box"], a, b)
        m = (t >= a) & (t <= b)
        peak_i = int(np.where(m)[0][np.nanargmax(c["n_near_box"][m])])
        events.append({
            "type": "other",
            "t": float(t[peak_i]),
            "t_start": float(a),
            "t_end": float(b),
            "confidence": round(min(0.35, 0.1 + 0.03 * min(b_max, 6)), 3),
            "signals": {"kind": "near_box_crowd", "box_players_max": round(b_max, 2),
                        "n_players_mean": round(win(c["n_players"], a, b, np.nanmean), 2)},
            "notes": f"{b_max:.1f} players in the near box for {b - a + 1:.0f} s, no cluster rush",
        })

    for e in events:
        if e["t"] < WARMUP_END_S:
            e["signals"] = dict(e["signals"], kind="warmup", warmup_type=e["type"],
                                warmup_kind=e["signals"].get("kind", e["type"]))
            e["type"] = "other"
            e["notes"] = "WARM-UP window (not a match event): " + e["notes"]

    events.sort(key=lambda e: (e["t"], e["type"]))
    out = {"source": "tracking", "video_duration_s": args.duration, "events": events}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    kinds = {}
    for e in events:
        k = e["type"] if e["signals"].get("kind") != "warmup" else "warmup"
        kinds[k] = kinds.get(k, 0) + 1
    print(f"wrote {len(events)} events -> {args.out}  {kinds}")


if __name__ == "__main__":
    main()
