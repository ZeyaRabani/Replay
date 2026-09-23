"""Ball track linking and per-bin ball signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calib import GoalZone, Space, in_zone


@dataclass
class BallTrack:
    t: np.ndarray          # seconds, one per processed frame
    u: np.ndarray          # pixel or pitch x
    v: np.ndarray          # pixel or pitch y
    conf: np.ndarray
    valid: np.ndarray      # False where interpolated
    fps_eff: float


def link_ball(dets_per_frame: list[list[dict]], fps_eff: float, frame_w: float,
              max_jump_frac: float = 0.08, min_tracklet: int = 3,
              interp_gap_s: float = 0.5, smooth_window: int = 7) -> BallTrack:
    """Greedy nearest-to-prediction linking of top-K per-frame detections."""
    n = len(dets_per_frame)
    centres = np.full((n, 2), np.nan)
    conf = np.zeros(n)
    max_jump = max_jump_frac * frame_w

    pos = None           # last valid position
    vel = np.zeros(2)
    age_gap = 0          # processed frames since last valid
    run_len = 0
    keep_runs: list[tuple[int, int]] = []  # (start_idx, length) of accepted runs
    run_start = -1

    for i, dets in enumerate(dets_per_frame):
        cands = np.array([[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2] for d in dets]) \
            if dets else np.zeros((0, 2))
        confs = np.array([d["conf"] for d in dets]) if dets else np.zeros(0)
        chosen = -1
        if len(cands):
            if pos is None:
                chosen = int(np.argmax(confs))
            else:
                pred = pos + vel * (age_gap + 1)
                d = np.linalg.norm(cands - pred, axis=1)
                if d.min() <= max_jump * (age_gap + 1):
                    chosen = int(np.argmin(d))
        if chosen >= 0:
            new = cands[chosen]
            if pos is not None:
                vel = (new - pos) / (age_gap + 1)
            centres[i] = new
            conf[i] = confs[chosen]
            pos = new
            if run_start < 0:
                run_start = i
            run_len += 1
            age_gap = 0
        else:
            age_gap += 1
            if run_start >= 0 and run_len >= min_tracklet:
                keep_runs.append((run_start, run_len))
            run_start, run_len = -1, 0
            pos = None
            vel = np.zeros(2)
    if run_start >= 0 and run_len >= min_tracklet:
        keep_runs.append((run_start, run_len))

    valid = np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for s, l in keep_runs:
        keep[s:s + l] = True
    t = np.arange(n) / fps_eff

    # interpolate short gaps inside kept runs
    u = centres[:, 0].copy()
    v = centres[:, 1].copy()
    good = np.isfinite(u) & keep
    valid = good.copy()
    idx = np.where(keep)[0]
    if len(idx) >= 2:
        u[keep] = np.interp(t[keep], t[good], u[good])
        v[keep] = np.interp(t[keep], t[good], v[good])
        valid[keep] = good[keep]
        # drop interpolations spanning gaps > interp_gap_s
        good_t = t[good]
        gaps = np.diff(good_t)
        big = np.where(gaps > interp_gap_s)[0]
        for gi in big:
            gap_mask = keep & (t > good_t[gi]) & (t < good_t[gi + 1])
            keep[gap_mask] = False
            valid[gap_mask] = False
    u[~keep] = np.nan
    v[~keep] = np.nan

    # Savitzky-Golay smoothing on the kept samples
    if smooth_window >= 3:
        from scipy.signal import savgol_filter

        for arr in (u, v):
            m = np.isfinite(arr)
            if m.sum() >= smooth_window:
                win = min(smooth_window, m.sum() if m.sum() % 2 else m.sum() - 1)
                if win >= 3:
                    arr[m] = savgol_filter(arr[m], win, 2)
    return BallTrack(t, u, v, conf, valid, fps_eff)


def project_ball(track: BallTrack, cal, space: Space) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (x, y, speed): pitch metres + m/s, or pixels + px/s."""
    dt = 1.0 / track.fps_eff
    if space is Space.PITCH:
        m = np.isfinite(track.u)
        xy = cal.project(np.c_[track.u[m], track.v[m]])
        x = np.full(len(track.t), np.nan)
        y = np.full(len(track.t), np.nan)
        x[m], y[m] = xy[:, 0], xy[:, 1]
    else:
        x, y = track.u, track.v
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.hypot(vx, vy)
    return x, y, speed


def ball_signals(track: BallTrack, zones: dict[str, GoalZone], space: Space, bin_s: float,
                 v_shot: float, cal=None, lost_s: float = 1.0) -> dict[str, np.ndarray]:
    """Per-bin attack/lost/seen signals for both goals."""
    n_bins = max(1, int(np.ceil((track.t[-1] + 1.0 / track.fps_eff) / bin_s))) if len(track.t) else 1
    out: dict[str, np.ndarray] = {k: np.zeros(n_bins) for k in
                                  ("ball_attack_A", "ball_attack_B", "ball_lost_A", "ball_lost_B", "ball_seen")}
    if len(track.t) == 0:
        return out
    x, y, speed = project_ball(track, cal, space)
    seen = np.isfinite(x)
    bin_idx = np.clip((track.t / bin_s).astype(int), 0, n_bins - 1)
    np.add.at(out["ball_seen"], bin_idx[seen], 1.0)
    counts = np.zeros(n_bins)
    np.add.at(counts, bin_idx, 1.0)
    out["ball_seen"] = np.divide(out["ball_seen"], np.maximum(counts, 1))

    dx = np.gradient(x, 1.0 / track.fps_eff)
    for g, z in zones.items():
        poly = np.array(z.poly)
        inside = np.zeros(len(track.t), dtype=bool)
        inside[seen] = in_zone(poly, np.c_[x[seen], y[seen]])
        toward = np.sign(dx) == np.sign(z.attack_dir)
        shot = inside & seen & toward & (speed > v_shot)
        att = np.zeros(n_bins)
        np.add.at(att, bin_idx[shot], 1.0)
        att_counts = np.zeros(n_bins)
        np.add.at(att_counts, bin_idx[inside & seen], 1.0)
        out[f"ball_attack_{g}"] = np.divide(att, np.maximum(att_counts, 1))

        # lost: was inside, then unseen for >= lost_s
        lost = np.zeros(n_bins)
        i = 0
        while i < len(track.t):
            if inside[i]:
                j = i
                while j + 1 < len(track.t) and inside[j + 1] and seen[j + 1]:
                    j += 1
                k = j + 1
                while k < len(track.t) and not seen[k] and (track.t[k] - track.t[j]) < lost_s:
                    k += 1
                if k < len(track.t) and (track.t[k] - track.t[j]) >= lost_s:
                    lost[bin_idx[min(k, len(track.t) - 1)]] = 1.0
                elif k >= len(track.t) and (track.t[-1] - track.t[j]) >= lost_s:
                    lost[bin_idx[j]] = 1.0
                i = k
            else:
                i += 1
        out[f"ball_lost_{g}"] = lost
    return out
