"""Director: broadcast-style per-second angle selection. Pure numpy — no IO.

Per-second candidate rules (on the shared timeline):
  EVENT   — any available angle whose event channel (shared-timeline
            candidate-event confidence, 0 when none) is > 0 wins outright;
            score = that confidence. Cuts happen immediately (no margin,
            no confirm, no hold, raw unsmoothed score, cut at the event
            second itself so the event window's pre-roll is kept).
  BALL    — any available angle whose ball_conf >= BALL_OK in >= 2 of the
            5 s window [t-2, t+2] is eligible; score = max ball_size over
            that window. Eligible angles only.
  CLUSTER — else score = cluster_score / per-angle 90th-percentile over the
            common span ("how close to this camera's best view is it now"),
            so wider cameras don't dominate by baseline alone.
  HOLD    — no available angle has any data -> keep current.

Smoothing: per-angle 5 s median then 9 s centred rolling mean.

Switching: a challenger is proposed when its smoothed score beats the
incumbent's by MARGIN (25% ball, 50% cluster) — or the incumbent's smoothed
score has been 0 for >= 6 consecutive seconds while some other available
angle's smoothed score is > 0.2 (dead-feed recovery, margin waived) — for
CONFIRM consecutive seconds (6 cluster / 3 ball); the cut lands on the
lowest summed-motion second inside [t-2, t+2]. MIN_HOLD = 20 s
(ball overrides at 10 s). A hard "coverage" cut fires when the current
angle leaves coverage.

Segment semantics: each segment describes the angle SHOWN in
[t_start, t_end); `rule` is the rule that selected it ("start", "coverage",
"ball", "cluster", "event"), `score` the winning smoothed score at selection time,
`runner_up` the angle it displaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BALL_OK = 0.35
BALL_WINDOW = 5          # seconds, centred
BALL_MIN_SIGHTINGS = 2
MIN_HOLD = 20
BALL_MIN_HOLD = 10
CONFIRM_CLUSTER = 6
CONFIRM_BALL = 3
MARGIN_BALL = 0.25
MARGIN_CLUSTER = 0.50
DEAD_SCORE_S = 6         # consecutive zero-score seconds before recovery
DEAD_CHALLENGER = 0.2    # challenger smoothed score threshold for recovery
EVENT_PRE = 3            # event window: seconds before the candidate peak
EVENT_POST = 6           # seconds after
EVENT_TYPES = ("goal", "shot", "goalmouth")
SMOOTH_MEDIAN = 5
SMOOTH_MEAN = 9
BASELINE_Q = 90
ZONE_LINGER = 3        # seconds an angle stays zone-eligible after last hit
ZONE_MIN_HOLD = 2      # min hold before cutting TO a zone angle


@dataclass(frozen=True)
class Style:
    """Switching-policy knobs. 'normal' is broadcast-style holds; 'fast'
    follows the ball with quick cuts."""
    min_hold: int
    ball_min_hold: int
    confirm_cluster: int
    confirm_ball: int
    margin_ball: float
    margin_cluster: float
    smooth_mean: int
    dead_score_s: int


STYLES = {
    "normal": Style(20, 10, 6, 3, 0.25, 0.50, 9, DEAD_SCORE_S),
    "fast": Style(2, 1, 1, 1, 0.02, 0.10, 1, 2),
}


def _smooth(x: np.ndarray, mean: int = SMOOTH_MEAN) -> np.ndarray:
    from scipy.ndimage import median_filter, uniform_filter1d
    return uniform_filter1d(
        median_filter(x, size=SMOOTH_MEDIAN, mode="nearest"),
        size=mean, mode="nearest")


def _cluster_baselines(track: list[dict], available: np.ndarray) -> np.ndarray:
    """Per-angle BASELINE_Q-quantile of cluster_score over the seconds where
    every angle is available (so cameras are compared on the same span; a
    camera that ran through an empty pre-match isn't inflated). Falls back
    to each angle's own seconds when there is no common span."""
    n = len(track)
    base = np.ones(n)
    common = available.all(axis=0)
    for i in range(n):
        span = common if common.any() else available[i]
        vals = np.asarray(track[i]["cluster"])[span]
        vals = vals[vals > 0]
        m = float(np.percentile(vals, BASELINE_Q)) if len(vals) else 1.0
        base[i] = m if m > 0 else 1.0
    return base


def _in_poly(pts: np.ndarray, poly: list[list[float]]) -> np.ndarray:
    """Ray-casting point-in-polygon for Nx2 pts vs an [[x,y],...] polygon."""
    x = pts[:, 0]
    y = pts[:, 1]
    px = np.asarray(poly, dtype=float)
    inside = np.zeros(len(pts), dtype=bool)
    n = len(px)
    for k in range(n):
        x1, y1 = px[k]
        x2, y2 = px[(k + 1) % n]
        cross = (y1 > y) != (y2 > y)
        xint = x1 + (y - y1) / (y2 - y1 + 1e-12) * (x2 - x1)
        inside ^= cross & (x < xint)
    return inside


def _zone_eligible(track: list[dict], available: np.ndarray,
                   zones: list[list[list[list[float]]]],
                   zone_ok: np.ndarray | None = None) -> np.ndarray:
    """[n_angles, T] bool: ball detected inside one of the angle's zones,
    extended ZONE_LINGER seconds after the last in-zone sighting."""
    n = len(track)
    T = available.shape[1]
    hits = np.zeros((n, T), dtype=bool)
    for i in range(n):
        if not zones[i]:
            continue
        bx = np.asarray(track[i].get("ball_x", np.zeros(T)), dtype=float)
        by = np.asarray(track[i].get("ball_y", np.zeros(T)), dtype=float)
        bc = np.asarray(track[i].get("ball_conf", np.zeros(T)), dtype=float)
        seen = available[i] & (bc >= BALL_OK) & (bx > 0) & (by > 0)
        if not seen.any():
            continue
        pts = np.stack([bx, by], axis=1)
        in_any = np.zeros(T, dtype=bool)
        for poly in zones[i]:
            if len(poly) >= 3:
                in_any |= _in_poly(pts, poly)
        hits[i] = seen & in_any
    # linger: eligible at t if any hit in [t-ZONE_LINGER, t]
    elig = hits.copy()
    for s in range(1, ZONE_LINGER + 1):
        elig[:, s:] |= hits[:, :T - s]
    if zone_ok is not None:
        elig &= zone_ok
    return elig


def per_second(track: list[dict], available: np.ndarray,
               zones: list | None = None,
               zone_ok: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-second best candidate + full score matrix over available angles.

    Returns best_angles (-1 none), scores, rules (0 hold, 1 cluster,
    2 ball, 3 event, 4 zone),
    S[n_angles, T] (each angle's score under the active rule), and the
    per-angle cluster baselines used for normalisation.
    """
    from scipy.ndimage import maximum_filter

    n_angles = len(track)
    T = available.shape[1]
    best_a = np.full(T, -1)
    best_s = np.zeros(T)
    best_r = np.zeros(T, dtype=int)
    S = np.zeros((n_angles, T))

    event = np.stack([
        np.asarray(track[i].get("event", np.zeros(T)), dtype=float)
        for i in range(n_angles)])
    ball_conf = np.stack([track[i]["ball_conf"] for i in range(n_angles)])
    ball_size = np.stack([track[i]["ball_size"] for i in range(n_angles)])
    cluster = np.stack([track[i]["cluster"] for i in range(n_angles)])
    baselines = _cluster_baselines(track, available)
    cluster_n = cluster / baselines[:, None]
    zone_elig = (_zone_eligible(track, available, zones, zone_ok)
                 if zones else None)

    # sighting: >= BALL_MIN_SIGHTINGS hits of ball_conf >= BALL_OK in the
    # centred 5 s window; score = max ball_size in that window
    hits = np.stack([
        np.convolve((ball_conf[i] >= BALL_OK).astype(float),
                    np.ones(BALL_WINDOW), mode="same")
        for i in range(n_angles)])
    ball_seen = hits >= BALL_MIN_SIGHTINGS
    ball_score = np.stack([
        maximum_filter(ball_size[i], size=BALL_WINDOW, mode="nearest")
        for i in range(n_angles)])

    for t in range(T):
        av = available[:, t]
        if not av.any():
            continue
        ev = np.where(av, event[:, t], 0.0)
        if ev.max() > 0:
            S[:, t] = ev
            j = int(np.argmax(S[:, t]))
            best_a[t], best_s[t], best_r[t] = j, S[j, t], 3
            continue
        if zone_elig is not None:
            ze = zone_elig[:, t]
            if ze.any():
                S[:, t] = np.where(ze, ball_conf[:, t], 0.0)
                j = int(np.argmax(S[:, t]))
                if S[j, t] > 0:
                    best_a[t], best_s[t], best_r[t] = j, S[j, t], 4
                    continue
        elig = av & ball_seen[:, t]
        if elig.any():
            S[:, t] = np.where(elig, ball_score[:, t], 0.0)
            j = int(np.argmax(S[:, t]))
            best_a[t], best_s[t], best_r[t] = j, S[j, t], 2
        else:
            S[:, t] = np.where(av, cluster_n[:, t], 0.0)
            j = int(np.argmax(S[:, t]))
            if S[j, t] > 0:
                best_a[t], best_s[t], best_r[t] = j, S[j, t], 1
            else:
                best_a[t] = int(np.argmax(av.astype(int)))
                best_s[t] = 0.0
    return best_a, best_s, best_r, S, baselines


def cut_director(track: list[dict], available: np.ndarray,
                 motion: list[np.ndarray], style: str = "normal",
                 zones: list | None = None,
                 zone_ok: np.ndarray | None = None) -> dict:
    """Full decision. track[i]: {"ball_conf","ball_size","cluster",
    "ball_x","ball_y"} 1 Hz arrays on the shared timeline;
    available[i, t]; motion[i] shared-timeline motion. zones (optional):
    per-angle lists of normalised polygons; a ball inside a zone cuts to
    that angle immediately. Returns the director.json dict."""
    sty = STYLES[style]
    T = available.shape[1]
    n_angles = len(track)
    cand_a, _cand_s, cand_r, S, baselines = per_second(
        track, available, zones=zones, zone_ok=zone_ok)
    sm = np.stack([_smooth(S[i], sty.smooth_mean) for i in range(n_angles)])
    mot = np.stack([np.where(available[i], m, np.inf) for i, m in enumerate(motion)])

    rule_counts = {"event": 0, "zone": 0, "ball": 0, "cluster": 0,
                   "hold": 0, "coverage": 0}
    cur = int(cand_a[0]) if cand_a[0] >= 0 else int(np.argmax(available[:, 0]))
    segs: list[dict] = [
        {"t_start": 0.0, "t_end": float(T - 1), "angle": cur, "rule": "start",
         "score": round(float(sm[cur, 0]), 4),
         "runner_up": None}]
    hold = 0
    streak = 0
    zero_run = 0
    propose = -1

    def open_seg(start_t: int, angle: int, rule: str, score: float,
                 runner: dict | None):
        segs.append({"t_start": float(start_t), "t_end": float(T - 1),
                     "angle": int(angle), "rule": rule,
                     "score": round(float(score), 4),
                     "runner_up": runner})

    for t in range(1, T):
        hold += 1
        # hard cut: incumbent left coverage (hold if nothing is available)
        if not available[cur, t]:
            if not available[:, t].any():
                rule_counts["hold"] += 1
                continue
            j = int(np.argmax(available[:, t].astype(int)))
            rule_counts["coverage"] += 1
            segs[-1]["t_end"] = float(t)
            open_seg(t, j, "coverage", float(sm[j, t]),
                     {"angle": int(cur), "score": round(float(sm[cur, t]), 4)})
            cur, hold, streak, zero_run, propose = j, 0, 0, 0, -1
            continue

        j = cand_a[t]
        if j < 0:
            rule_counts["hold"] += 1
            zero_run = zero_run + 1 if sm[cur, t] <= 0 else 0
            continue
        rule_counts[{4: "zone", 3: "event", 2: "ball", 1: "cluster",
                     0: "coverage"}[cand_r[t]]] += 1
        if j == cur:
            streak = 0
            propose = -1
            zero_run = zero_run + 1 if sm[cur, t] <= 0 else 0
            continue

        if cand_r[t] == 3:
            # event rule: cut to the camera that sees the shot — immediate,
            # raw score, no margin/confirm/hold, cut at the event second
            # itself (the window's pre-roll is built into S)
            segs[-1]["t_end"] = float(t)
            open_seg(t, j, "event", float(S[j, t]),
                     {"angle": int(cur), "score": round(float(sm[cur, t]), 4)})
            cur, hold, streak, zero_run, propose = j, 0, 0, 0, -1
            continue

        if cand_r[t] == 4:
            # zone rule: ball inside a painted zone -> cut to that camera
            # immediately like an event, after a short min hold; while the
            # hold is short just keep waiting (never use the margin path)
            if hold >= ZONE_MIN_HOLD:
                segs[-1]["t_end"] = float(t)
                open_seg(t, j, "zone", float(S[j, t]),
                         {"angle": int(cur), "score": round(float(sm[cur, t]), 4)})
                cur, hold, streak, zero_run, propose = j, 0, 0, 0, -1
            continue

        cur_score = sm[cur, t]
        # dead-feed recovery: incumbent scoreless for >= DEAD_SCORE_S while a
        # challenger shows real signal -> margin waived
        zero_run = zero_run + 1 if cur_score <= 0 else 0
        dead_recovery = (zero_run >= sty.dead_score_s
                         and sm[j, t] > DEAD_CHALLENGER)
        margin = sty.margin_ball if cand_r[t] == 2 else sty.margin_cluster
        better = dead_recovery or sm[j, t] > cur_score * (1 + margin)

        if better and j == propose:
            streak += 1
        elif better:
            propose, streak = j, 1
        else:
            streak = 0
            propose = -1
        confirm = sty.confirm_ball if cand_r[t] == 2 else sty.confirm_cluster
        need_hold = sty.ball_min_hold if cand_r[t] == 2 else sty.min_hold
        if propose == j and streak >= confirm and hold >= need_hold:
            # cut at min summed motion within [t-2, t+2]
            lo, hi = max(segs[-1]["t_start"], t - 2), min(T - 1, t + 2)
            lo, hi = int(lo), int(hi)
            msum = mot[:, lo:hi + 1].sum(axis=0)
            cut_t = lo + int(np.argmin(msum))
            segs[-1]["t_end"] = float(cut_t)
            open_seg(cut_t, j, {2: "ball", 1: "cluster"}.get(cand_r[t], "cluster"),
                     sm[j, t],
                     {"angle": int(cur), "score": round(float(cur_score), 4)})
            cur, hold, streak, zero_run, propose = j, max(0, t - cut_t), 0, 0, -1

    durs = [s["t_end"] - s["t_start"] for s in segs]
    total_dur = sum(durs)
    angle_share = {str(i): 0.0 for i in range(n_angles)}
    for s in segs:
        angle_share[str(s["angle"])] += s["t_end"] - s["t_start"]
    if total_dur > 0:
        angle_share = {k: round(v / total_dur, 4) for k, v in angle_share.items()}
    tot = sum(rule_counts.values()) or 1
    span_min = max(total_dur / 60.0, 1e-9)
    return {
        "style": style,
        "zones_used": zones is not None and any(len(z) for z in zones),
        "segments": segs,
        "per_second_rule": rule_counts,
        "ratios": {k: round(v / tot, 4) for k, v in rule_counts.items()},
        "n_cuts": len(segs) - 1,
        "mean_hold_s": round(total_dur / max(1, len(segs)), 2),
        "median_hold_s": round(float(np.median(durs)), 2) if durs else 0.0,
        "min_hold_s": round(float(min(durs)), 2) if durs else 0.0,
        "cuts_per_10min": round((len(segs) - 1) / span_min * 10, 1),
        "angle_share": angle_share,
        "cluster_baseline": [round(float(b), 4) for b in baselines],
    }
