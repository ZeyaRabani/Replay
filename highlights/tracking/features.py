"""Per-second player-cluster features in pitch coordinates.

Input: tracks.npz from detect_track.py and calib.json from calibrate.py.
Output: features_1s.json in the shared schema
    {"source": "tracking", "step_s": 1.0, "columns": [...], "rows": [[t, ...]]}

World frame: origin at the near-goal centre, x toward the far goal (0..L),
y across the pitch. All distances in metres (scale depends on the assumed
focal length; treat as approximate, far-end precision is low).
"""

import argparse
import json

import numpy as np

from .calibrate import GroundModel, load_tracks

COLUMNS = [
    "t", "n_players", "n_lime", "n_orange", "n_raw", "stab_ok",
    "cx", "cy", "spread", "mean_speed",
    "frac_near_third", "frac_mid_third", "frac_far_third",
    "n_near_box", "n_far_box", "n_centre",
    "rush_near_3s", "rush_far_3s", "restart_flag", "centre_cluster_flag",
    "keeper_x",
]

BOX_DEPTH_M = 12.0
BOX_HALF_W_M = 9.0
CENTRE_RADIUS_M = 9.0


def team_of(h, s, v):
    """Coarse bib colour from torso HSV (OpenCV H in 0..180)."""
    lime = (s > 110) & (v > 120) & (h >= 26) & (h <= 42)
    orange = (s > 130) & (v > 120) & (h <= 22)
    return lime, orange


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--half-width", type=float, default=16.0)
    args = ap.parse_args()

    det, frames, _Hs, feet, heads = load_tracks(args.tracks)
    with open(args.calib) as fh:
        cal = json.load(fh)
    g = GroundModel.from_json(cal["model"])
    L = g.length
    W = g.world(feet, heads)
    fps = frames[1, 1] - frames[0, 1]
    fps = 1.0 / fps

    t_det = det[:, 1]
    tid = det[:, 2].astype(int)
    lime, orange = team_of(det[:, 8], det[:, 9], det[:, 10])
    on_pitch = (np.isfinite(W).all(1) & (W[:, 0] > -1.5) & (W[:, 0] < L + 4)
                & (np.abs(W[:, 1]) < args.half_width))

    # per-track speed from consecutive frames (m/s), in world coords
    speed = np.full(len(det), np.nan)
    order = np.lexsort((det[:, 0], tid))
    ds = det[order]
    Ws = W[order]
    ts = tid[order]
    same = (ts[1:] == ts[:-1]) & (ts[1:] >= 0) & (ds[1:, 0] - ds[:-1, 0] <= 2)
    dt = (ds[1:, 1] - ds[:-1, 1])
    d = np.linalg.norm(Ws[1:] - Ws[:-1], axis=1)
    sp = np.where(same, d / np.maximum(dt, 1e-3), np.nan)
    speed[order[1:]] = sp

    duration = args.duration or float(frames[-1, 1]) + 1.0 / fps
    n_sec = int(np.ceil(duration))
    sec = np.floor(t_det).astype(int)
    frame_sec = np.floor(frames[:, 1]).astype(int)

    rows = []
    track_x_hist = []  # per second: {track_id: median x}
    for s in range(n_sec):
        m = (sec == s) & on_pitch
        fm = frame_sec == s
        n_frames = max(int(fm.sum()), 1)
        n_raw = float((sec == s).sum()) / n_frames
        stab = float(frames[fm, 3].mean()) if fm.any() else 0.0
        if m.sum() == 0:
            track_x_hist.append({})
            rows.append([float(s), 0, 0, 0, n_raw, stab] + [np.nan] * 4
                        + [0, 0, 0, 0, 0, 0, np.nan, np.nan, 0, 0, np.nan])
            continue
        Wm = W[m]
        n_players = float(m.sum()) / n_frames
        n_lime = float((m & lime).sum()) / n_frames
        n_orange = float((m & orange).sum()) / n_frames
        cx, cy = Wm.mean(0)
        spread = float(np.linalg.norm(Wm - Wm.mean(0), axis=1).mean())
        sp = speed[m]
        sp = sp[np.isfinite(sp) & (sp < 12)]
        mean_speed = float(np.median(sp)) if len(sp) else np.nan
        x = Wm[:, 0]
        frac_near = float((x < L / 3).mean())
        frac_far = float((x > 2 * L / 3).mean())
        frac_mid = 1.0 - frac_near - frac_far
        near_box = (x < BOX_DEPTH_M) & (np.abs(Wm[:, 1]) < BOX_HALF_W_M)
        far_box = (x > L - BOX_DEPTH_M) & (np.abs(Wm[:, 1]) < BOX_HALF_W_M)
        centre = np.linalg.norm(Wm - np.array([L / 2, 0]), axis=1) < CENTRE_RADIUS_M
        n_near_box = float(near_box.sum()) / n_frames
        n_far_box = float(far_box.sum()) / n_frames
        n_centre = float(centre.sum()) / n_frames
        keeper_x = float(np.min(x))
        ids = tid[m]
        tx = {int(i): float(np.median(x[ids == i])) for i in np.unique(ids) if i >= 0}
        track_x_hist.append(tx)
        # rush: mean per-track x displacement over the last 3 s, using only
        # tracks present in both seconds (robust to players entering/leaving)
        rush_near = rush_far = np.nan
        if len(track_x_hist) >= 4:
            prev = track_x_hist[-4]
            shared = [i for i in tx if i in prev]
            if len(shared) >= 4:
                dx = float(np.mean([tx[i] - prev[i] for i in shared]))
                rush_near, rush_far = -dx, dx
        restart = int(n_players >= 5 and spread < 8.0
                      and np.isfinite(mean_speed) and mean_speed < 1.5)
        centre_cluster = int(n_players >= 6 and n_centre >= 4
                             and abs(cx - L / 2) < 8.0
                             and np.isfinite(mean_speed) and mean_speed < 2.5)
        rows.append([float(s), n_players, n_lime, n_orange, n_raw, stab,
                     float(cx), float(cy), spread, mean_speed,
                     frac_near, frac_mid, frac_far,
                     n_near_box, n_far_box, n_centre,
                     rush_near, rush_far, restart, centre_cluster, keeper_x])

    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return round(float(v), 3) if isinstance(v, float) else v

    out = {"source": "tracking", "step_s": 1.0, "columns": COLUMNS,
           "pitch_length_m": round(L, 2),
           "rows": [[clean(v) for v in r] for r in rows]}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
