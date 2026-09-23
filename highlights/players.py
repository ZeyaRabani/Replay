"""Per-player tracks -> per-bin attack / cluster / restart signals.

Pixel space is pseudo-metric: speeds and distances are converted with
``m_per_px = 1.75 / box_h`` so pitch-space thresholds apply in both spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calib import GoalZone, Space, in_zone


@dataclass
class PlayerSigCfg:
    v_run: float            # m/s sprint threshold (both spaces)
    cluster_radius: float   # m
    cluster_min_players: int
    cluster_slow: float     # m/s
    cluster_min_dur_s: float
    restart_half_width: float
    restart_centre_r: float
    v_sanity: float = 12.0  # m/s; faster implied velocities are dropped
    gap_s: float = 1.0      # no velocity across detection gaps longer than this


def frame_scales(frame: list[dict]) -> dict[int, float]:
    """player id -> metres-per-pixel from that frame's box height."""
    out = {}
    for d in frame:
        box_h = max(20.0, d["box"][3] - d["box"][1])
        out[d["id"]] = 1.75 / box_h
    return out


def median_scale(frame: list[dict]) -> float | None:
    s = frame_scales(frame)
    return float(np.median(list(s.values()))) if s else None


def _project(frames: list[list[dict]], cal, space: Space, frame_w: float) -> dict[int, dict]:
    """id -> {fi, x, y, scale} arrays; pitch metres, or pixels + per-obs m_per_px."""
    obs: dict[int, list] = {}
    for fi, fr in enumerate(frames):
        sc = frame_scales(fr)
        for d in fr:
            if d["id"] < 0:
                continue
            cx = (d["box"][0] + d["box"][2]) / 2
            cy = d["box"][3]
            obs.setdefault(d["id"], []).append((fi, cx, cy, sc.get(d["id"], np.nan)))
    out = {}
    if space is Space.PITCH and obs:
        for pid, rows in obs.items():
            arr = np.array(rows)
            xy = cal.project(arr[:, 1:3])
            out[pid] = {"fi": arr[:, 0].astype(int), "x": xy[:, 0], "y": xy[:, 1],
                        "scale": np.ones(len(arr))}
    else:
        for pid, rows in obs.items():
            arr = np.array(rows)
            out[pid] = {"fi": arr[:, 0].astype(int), "x": arr[:, 1], "y": arr[:, 2],
                        "scale": arr[:, 3]}
    return out


def player_signals(frames: list[list[dict]], fps_eff: float, cal, space: Space,
                   zones: dict[str, GoalZone], frame_w: float, frame_h: float,
                   pitch, cfg: PlayerSigCfg, bin_s: float) -> dict[str, np.ndarray]:
    n = len(frames)
    n_bins = max(1, int(np.ceil(n / fps_eff / bin_s)))
    out = {k: np.zeros(n_bins) for k in
           ("attack_A", "attack_B", "cluster", "cluster_A", "cluster_B", "cluster_size",
            "restart", "n_players")}
    if n == 0:
        return out

    tracks = _project(frames, cal, space, frame_w)
    dt = 1.0 / fps_eff
    half_x = (pitch.length / 2) if space is Space.PITCH else frame_w / 2

    per_frame: list[list[dict]] = [[] for _ in range(n)]
    for pid, tr in tracks.items():
        if len(tr["fi"]) < 3:
            continue
        dfi = np.diff(tr["fi"])
        dpos = np.hypot(np.diff(tr["x"]), np.diff(tr["y"]))
        v = np.zeros(len(tr["fi"]))
        ok = (dfi * dt <= cfg.gap_s)
        v[1:][ok] = dpos[ok] / (dfi[ok] * dt)
        for k, fi in enumerate(tr["fi"]):
            scale = tr["scale"][k] if np.isfinite(tr["scale"][k]) else np.nan
            v_m = v[k] * (scale if space is Space.PIXEL else 1.0)
            if v_m > cfg.v_sanity:
                v_m = 0.0  # id collision or jitter
            vx = 0.0
            if k and ok[k - 1]:
                vx = (tr["x"][k] - tr["x"][k - 1]) / (dfi[k - 1] * dt) * (scale if space is Space.PIXEL else 1.0)
                if abs(vx) > cfg.v_sanity:
                    vx = 0.0
            per_frame[fi].append({"id": pid, "x": tr["x"][k], "y": tr["y"][k],
                                  "vx": vx, "v": v_m, "mpp": scale})

    cluster_runs = np.zeros(n, dtype=bool)
    cluster_zone = np.zeros(n, dtype=int)
    cluster_sz = np.zeros(n, dtype=int)
    for fi, pl in enumerate(per_frame):
        b = min(n_bins - 1, int(fi / fps_eff / bin_s))
        out["n_players"][b] = max(out["n_players"][b], len(pl))
        for g, z in zones.items():
            fast = [p for p in pl if p["v"] > cfg.v_run and np.sign(p["vx"]) == np.sign(z.attack_dir)
                    and ((p["x"] <= half_x) if z.attack_dir < 0 else (p["x"] > half_x))]
            out[f"attack_{g}"][b] = max(out[f"attack_{g}"][b], min(1.0, len(fast) / 8.0))
        if len(pl) >= cfg.cluster_min_players:
            mpp = median_scale(frames[fi]) if space is Space.PIXEL else 1.0
            if mpp is None:
                mpp = frame_h / 1.75  # whole-frame fallback ~ treat as metres
            pts = np.array([[p["x"], p["y"]] for p in pl]) * mpp
            vs = np.array([p["v"] for p in pl])
            r = cfg.cluster_radius
            d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
            for i in range(len(pts)):
                nb = np.where(d[i] <= r)[0]
                if len(nb) >= cfg.cluster_min_players:
                    cen = pts[nb].mean(0)
                    if np.linalg.norm(pts[nb] - cen, axis=1).max() <= r and vs[nb].mean() < cfg.cluster_slow:
                        cluster_runs[fi] = True
                        cluster_sz[fi] = len(nb)
                        cen_px = cen / mpp
                        for zi, (g, z) in enumerate(zones.items(), start=1):
                            if in_zone(np.array(z.poly), cen_px[None, :])[0]:
                                cluster_zone[fi] = zi
                        break

    # sustained runs >= min_dur, and only when >= 10 s of no cluster preceded (novelty)
    min_len = int(cfg.cluster_min_dur_s * fps_eff)
    gap_len = int(10.0 * fps_eff)
    runs = []
    run = 0
    for fi in range(n + 1):
        v = cluster_runs[fi] if fi < n else False
        if v:
            run += 1
        else:
            if run >= min_len:
                runs.append((fi - run, fi - 1))
            run = 0
    prev_end = -gap_len - 1
    for s, e in runs:
        if s - prev_end < gap_len + 1:
            continue
        b0 = int(s / fps_eff / bin_s)
        b1 = min(n_bins - 1, int(e / fps_eff / bin_s))
        out["cluster"][b0:b1 + 1] = 1.0
        out["cluster_size"][b0:b1 + 1] = np.maximum(out["cluster_size"][b0:b1 + 1],
                                                  cluster_sz[s:e + 1].max() / max(cfg.cluster_min_players, 1))
        for zi in set(cluster_zone[s:e + 1]) - {0}:
            g = list(zones)[zi - 1]
            out[f"cluster_{g}"][b0:b1 + 1] = 1.0
        prev_end = e

    if space is Space.PITCH:
        cx, cy = pitch.length / 2, pitch.width / 2
        for fi, pl in enumerate(per_frame):
            b = min(n_bins - 1, int(fi / fps_eff / bin_s))
            near_line = [p for p in pl if abs(p["x"] - cx) <= cfg.restart_half_width]
            near_ctr = [p for p in pl if np.hypot(p["x"] - cx, p["y"] - cy) <= cfg.restart_centre_r
                        and p["v"] < cfg.cluster_slow]
            if len(pl):
                out["restart"][b] = max(out["restart"][b],
                                        (len(near_line) / len(pl)) if len(near_ctr) >= 2 else 0.0)
    return out



