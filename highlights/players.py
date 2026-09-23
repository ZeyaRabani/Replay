"""Per-player tracks -> per-bin attack / cluster / restart signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calib import GoalZone, Space


@dataclass
class PlayerSigCfg:
    v_run: float            # m/s (pitch) or frame_h/s (pixel) sprint threshold
    cluster_radius: float   # m or fraction of frame width
    cluster_min_players: int
    cluster_slow: float
    cluster_min_dur_s: float
    restart_half_width: float
    restart_centre_r: float


def _project(frames: list[list[dict]], cal, space: Space, frame_w: float) -> dict[int, dict]:
    """id -> {t, x, y} arrays of box centre-bottom, projected if pitch space."""
    obs: dict[int, list] = {}
    for fi, fr in enumerate(frames):
        for d in fr:
            if d["id"] < 0:
                continue
            cx = (d["box"][0] + d["box"][2]) / 2
            cy = d["box"][3]
            obs.setdefault(d["id"], []).append((fi, cx, cy))
    out = {}
    if space is Space.PITCH and obs:
        all_ids = list(obs)
        for pid in all_ids:
            arr = np.array(obs[pid])
            xy = cal.project(arr[:, 1:3])
            out[pid] = {"fi": arr[:, 0].astype(int), "x": xy[:, 0], "y": xy[:, 1]}
    else:
        for pid, rows in obs.items():
            arr = np.array(rows)
            out[pid] = {"fi": arr[:, 0].astype(int), "x": arr[:, 1], "y": arr[:, 2]}
    return out


def player_signals(frames: list[list[dict]], fps_eff: float, cal, space: Space,
                   zones: dict[str, GoalZone], frame_w: float, frame_h: float,
                   pitch, cfg: PlayerSigCfg, bin_s: float) -> dict[str, np.ndarray]:
    n = len(frames)
    n_bins = max(1, int(np.ceil(n / fps_eff / bin_s)))
    out = {k: np.zeros(n_bins) for k in
           ("attack_A", "attack_B", "cluster", "cluster_A", "cluster_B", "restart", "n_players")}
    if n == 0:
        return out

    tracks = _project(frames, cal, space, frame_w)
    # per-frame speed and toward-goal flags
    dt = 1.0 / fps_eff
    speed_scale = 1.0 if space is Space.PITCH else 1.0 / frame_h  # normalise pixel speeds to frame_h/s
    dist_scale = 1.0 if space is Space.PITCH else frame_w         # cluster_radius stored as fraction
    half_x = (pitch.length / 2) if space is Space.PITCH else frame_w / 2

    per_frame: list[list[dict]] = [[] for _ in range(n)]
    for pid, tr in tracks.items():
        if len(tr["fi"]) < 3:
            continue
        vx = np.gradient(tr["x"], dt) * speed_scale
        vy = np.gradient(tr["y"], dt) * speed_scale
        spd = np.hypot(vx, vy)
        for k, fi in enumerate(tr["fi"]):
            per_frame[fi].append({"id": pid, "x": tr["x"][k], "y": tr["y"][k],
                                  "vx": vx[k], "v": spd[k]})

    cluster_runs = np.zeros(n, dtype=bool)
    cluster_zone = np.zeros(n, dtype=int)  # 0 none, 1 A, 2 B
    for fi, pl in enumerate(per_frame):
        b = min(n_bins - 1, int(fi / fps_eff / bin_s))
        out["n_players"][b] = max(out["n_players"][b], len(pl))
        for g, z in zones.items():
            fast = [p for p in pl if p["v"] > cfg.v_run and np.sign(p["vx"]) == np.sign(z.attack_dir)
                    and ((p["x"] <= half_x) if z.attack_dir < 0 else (p["x"] > half_x))]
            out[f"attack_{g}"][b] = max(out[f"attack_{g}"][b], min(1.0, len(fast) / 8.0))
        # cluster: >= min_players within radius of a common centroid, mean slow
        if len(pl) >= cfg.cluster_min_players:
            pts = np.array([[p["x"], p["y"]] for p in pl])
            vs = np.array([p["v"] for p in pl])
            r = cfg.cluster_radius * dist_scale
            d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
            for i in range(len(pts)):
                nb = np.where(d[i] <= r)[0]
                if len(nb) >= cfg.cluster_min_players:
                    cen = pts[nb].mean(0)
                    if np.linalg.norm(pts[nb] - cen, axis=1).max() <= r and vs[nb].mean() < cfg.cluster_slow:
                        cluster_runs[fi] = True
                        from .calib import in_zone
                        for zi, (g, z) in enumerate(zones.items(), start=1):
                            if in_zone(np.array(z.poly), cen[None, :])[0]:
                                cluster_zone[fi] = zi
                        break
    # sustained cluster >= min_dur -> bins covering the run
    min_len = int(cfg.cluster_min_dur_s * fps_eff)
    run = 0
    for fi in range(n + 1):
        v = cluster_runs[fi] if fi < n else False
        if v:
            run += 1
        else:
            if run >= min_len:
                b0 = int((fi - run) / fps_eff / bin_s)
                b1 = min(n_bins - 1, int((fi - 1) / fps_eff / bin_s))
                out["cluster"][b0:b1 + 1] = 1.0
                zones_hit = set(cluster_zone[fi - run:fi]) - {0}
                for zi in zones_hit:
                    g = list(zones)[zi - 1]
                    out[f"cluster_{g}"][b0:b1 + 1] = 1.0
            run = 0

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
