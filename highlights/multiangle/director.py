"""Director: broadcast-style per-second angle selection. Pure numpy — no IO.

Rules per second over the shared timeline:
  BALL    — if any available angle saw a ball (ball_conf >= BALL_OK within
            ±1 s), eligible angles = those with a ball; score = ball_size.
  CLUSTER — else score = cluster_score (player bunch close to that camera).
  HOLD    — no available angle has any data -> keep current.

Switching: a candidate is *proposed* when its 3 s median score beats the
current angle's by MARGIN (25% ball, 40% cluster) for CONFIRM consecutive
seconds; the cut then lands on the lowest summed-motion second inside
[t-2, t+2]. MIN_HOLD = 8 s (ball may override once hold >= 4 s). A hard cut
fires when the current angle leaves coverage.
"""

from __future__ import annotations

import numpy as np

BALL_OK = 0.35
MIN_HOLD = 8
BALL_MIN_HOLD = 4
CONFIRM = 3
MARGIN_BALL = 0.25
MARGIN_CLUSTER = 0.40
MEDIAN_W = 3


def _median3(x: np.ndarray) -> np.ndarray:
    """3 s centred median (edge-safe)."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        out[i] = np.median(x[lo:hi])
    return out


def per_second(track: list[dict], available: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-second best candidate (angle, score, rule_id) over available angles.

    track[i] = dict of per-second arrays: ball_conf, ball_size, cluster.
    available[i, t] bool. Returns best_angles (-1 none), scores, rules
    (0 hold, 1 cluster, 2 ball).
    """
    n_angles = len(track)
    T = available.shape[1]
    best_a = np.full(T, -1)
    best_s = np.zeros(T)
    best_r = np.zeros(T, dtype=int)

    ball_conf = np.stack([track[i]["ball_conf"] for i in range(n_angles)])
    ball_seen = np.zeros((n_angles, T), dtype=bool)
    for i in range(n_angles):
        bc = ball_conf[i] >= BALL_OK
        ball_seen[i] = bc | np.roll(bc, 1) | np.roll(bc, -1)  # ±1 s window
        ball_seen[i, 0] |= bc[1] if T > 1 else False
        ball_seen[i, -1] |= bc[-2] if T > 1 else False

    ball_size = np.stack([track[i]["ball_size"] for i in range(n_angles)])
    cluster = np.stack([track[i]["cluster"] for i in range(n_angles)])

    for t in range(T):
        av = available[:, t]
        if not av.any():
            continue
        elig = av & ball_seen[:, t]
        if elig.any():
            s = np.where(elig, ball_size[:, t], -1.0)
            j = int(np.argmax(s))
            best_a[t], best_s[t], best_r[t] = j, s[j], 2
        else:
            s = np.where(av, cluster[:, t], -np.inf)
            j = int(np.argmax(s))
            if np.isfinite(s[j]) and s[j] > 0:
                best_a[t], best_s[t], best_r[t] = j, s[j], 1
            else:
                # angles available but no data (dead time)
                best_a[t] = int(np.argmax(av.astype(int)))
                best_s[t] = 0.0
    return best_a, best_s, best_r


def cut_director(track: list[dict], available: np.ndarray,
                 motion: list[np.ndarray]) -> dict:
    """Full decision. track[i]: {"ball_conf","ball_size","cluster"} 1 Hz arrays
    on the shared timeline; available[i, t]; motion[i] shared-timeline motion.
    Returns the director.json dict."""
    T = available.shape[1]
    n_angles = len(track)
    cand_a, cand_s, cand_r = per_second(track, available)
    sm = [ _median3(cand_s * (cand_a == i)) for i in range(n_angles) ]
    sm = np.stack(sm)  # smoothed score per angle (0 when not the candidate)
    mot = np.stack([np.where(available[i], m, np.inf) for i, m in enumerate(motion)])

    segs: list[dict] = []
    rule_counts = {"ball": 0, "cluster": 0, "hold": 0, "coverage": 0}
    cur = int(cand_a[0]) if cand_a[0] >= 0 else 0
    seg_start = 0
    hold = 0
    streak = 0
    propose = -1

    def close_seg(end_t: int, rule: str, scores=None, runner=None):
        seg = {"t_start": float(seg_start), "t_end": float(end_t),
               "angle": cur, "rule": rule}
        if scores is not None:
            seg["score"] = round(float(scores), 4)
        if runner is not None:
            seg["runner_up"] = runner
        segs.append(seg)

    for t in range(1, T):
        hold += 1
        avail_cur = available[cur, t]
        if not avail_cur:
            # hard cut: angle left coverage
            j = int(np.argmax(available[:, t].astype(int)))
            rule_counts["coverage"] += 1
            close_seg(t, "coverage", cand_s[t])
            cur, seg_start, hold, streak, propose = j, t, 0, 0, -1
            continue

        j = cand_a[t]
        if j < 0:
            rule_counts["hold"] += 1
            continue  # no data anywhere -> hold
        rule_counts[{2: "ball", 1: "cluster", 0: "coverage"}[cand_r[t]]] += 1
        if j == cur:
            streak = 0
            propose = -1
            continue
        margin = MARGIN_BALL if cand_r[t] == 2 else MARGIN_CLUSTER
        cur_score = sm[cur, t]
        better = sm[j, t] > cur_score * (1 + margin) or cur_score <= 0
        if better and j == propose:
            streak += 1
        elif better:
            propose, streak = j, 1
        else:
            streak = 0
            propose = -1
        need_hold = BALL_MIN_HOLD if cand_r[t] == 2 else MIN_HOLD
        if propose == j and streak >= CONFIRM and hold >= need_hold:
            # cut at min summed motion within [t-2, t+2]
            lo, hi = max(seg_start, t - 2), min(T - 1, t + 2)
            msum = mot[:, lo:hi + 1].sum(axis=0)
            cut_t = lo + int(np.argmin(msum))
            runner = {"angle": cur, "score": round(float(cur_score), 4)}
            close_seg(cut_t, {2: "ball", 1: "cluster"}[cand_r[t]], sm[j, t], runner)
            cur, hold, streak, propose = j, max(0, t - cut_t), 0, -1
            seg_start = cut_t
    close_seg(T - 1, "end", cand_s[-1])

    tot = sum(rule_counts.values()) or 1
    cuts = len(segs) - 1
    angle_share = {str(i): 0.0 for i in range(n_angles)}
    total_dur = 0.0
    for s in segs:
        d = s["t_end"] - s["t_start"]
        total_dur += d
        angle_share[str(s["angle"])] = angle_share.get(str(s["angle"]), 0.0) + d
    if total_dur > 0:
        angle_share = {k: round(v / total_dur, 4) for k, v in angle_share.items()}
    return {
        "segments": segs,
        "per_second_rule": rule_counts,
        "ratios": {k: round(v / tot, 4) for k, v in rule_counts.items()},
        "n_cuts": cuts,
        "mean_hold_s": round(total_dur / max(1, len(segs)), 2),
        "angle_share": angle_share,
    }
