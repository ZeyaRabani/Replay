"""Fuse per-angle hi-res detections into pitch-space tracks (v2).

Per 0.5 s step:
  1. project each angle's footpoints through its calib H -> pitch metres;
     drop boxes < 12 px tall and points > 3 m outside the pitch;
  2. cross-view merge: greedy agglomerate detections from different
     angles within 5 m into one conf-weighted observation;
  3. constant-velocity Kalman + Hungarian assignment (gate 4 m,
     6 m after misses; 8 s of misses ends a track; >= 4 s to keep);
     team conflict never assigned when both sides know the team;
  4. tracks.json + summary.json + up to 6 1080p crops per track;
     ball = per-angle features_1s ball projected through H, conf-merged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from highlights.io import write_json_atomic

STEP = 0.5
MIN_BOX_H_PX = 12
EDGE_MARGIN_M = 3.0
MERGE_M = 5.0
GATE_M = 4.0
GATE_MISS_M = 6.0
MAX_MISS_S = 8.0
MIN_LEN_S = 4.0
FILL_GAP_S = 2.0
OWNERSHIP = True          # drop detections a farther camera saw better
CROPS_PER_TRACK = 6
BALL_CONF = 0.35
SPRINT_MS = 6.5           # same threshold as players.py? kept local
V2_ID_BASE = 100001       # v2 track ids never collide with tracklet ids


def _kalman_new(x: float, y: float) -> dict:
    return {"x": np.array([x, y, 0.0, 0.0]),
            "P": np.eye(4) * 10.0}


def _kalman_pred(tr: dict, dt: float) -> None:
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    tr["x"] = F @ tr["x"]
    tr["P"] = F @ tr["P"] @ F.T + np.eye(4) * (0.5 * dt)


def _kalman_upd(tr: dict, mx: float, my: float) -> None:
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R = np.eye(2) * 1.0
    z = np.array([mx, my])
    y = z - H @ tr["x"]
    S = H @ tr["P"] @ H.T + R
    K = tr["P"] @ H.T @ np.linalg.inv(S)
    tr["x"] = tr["x"] + K @ y
    tr["P"] = (np.eye(4) - K @ H) @ tr["P"]


def _majority(teams: list[str]) -> str | None:
    a = sum(1 for t in teams if t == "A")
    b = sum(1 for t in teams if t == "B")
    if a == b == 0:
        return None
    return "A" if a > b else "B" if b > a else None


def _merge_obs(points: list[dict]) -> list[dict]:
    """Greedy merge: detections from *different* angles within MERGE_M
    collapse into one conf-weighted observation."""
    obs: list[dict] = []
    for p in sorted(points, key=lambda p: -p["conf"]):
        tgt = None
        for o in obs:
            if p["angle"] in o["angles"]:
                continue
            if math.hypot(p["x"] - o["x"], p["y"] - o["y"]) <= MERGE_M:
                tgt = o
                break
        if tgt is None:
            obs.append({"x": p["x"], "y": p["y"], "w": p["conf"],
                        "angles": {p["angle"]: p["conf"]},
                        "teams": [p["team"]] if p["team"] else [],
                        "refs": [p["ref"]]})
        else:
            w = tgt["w"] + p["conf"]
            tgt["x"] = (tgt["x"] * tgt["w"] + p["x"] * p["conf"]) / w
            tgt["y"] = (tgt["y"] * tgt["w"] + p["y"] * p["conf"]) / w
            tgt["w"] = w
            tgt["angles"][p["angle"]] = p["conf"]
            if p["team"]:
                tgt["teams"].append(p["team"])
            tgt["refs"].append(p["ref"])
    return obs


def load_dets(path: Path) -> dict:
    """npz -> {'t': [...file s], 'foot': [(fx,fy)], 'conf', 'team',
    'box': [(x1,y1,x2,y2)]} filtered to box height >= MIN_BOX_H_PX."""
    z = np.load(path)
    keep = (z["y2"] - z["y1"]) >= MIN_BOX_H_PX
    meta = _load_json(path.with_suffix(".meta.json")) or {}
    w = meta.get("w") or 1.0
    h = meta.get("h") or 1.0
    return {"t": z["t"][keep],
            "box": np.stack([z["x1"], z["y1"], z["x2"], z["y2"]],
                            axis=1)[keep],
            "conf": z["conf"][keep],
            "team": z["team"][keep],
            "w": w, "h": h,
            "foot": np.stack([(z["x1"] + z["x2"]) / 2 / w,
                              z["y2"] / h], axis=1)[keep]}


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fuse(dets: dict[int, dict], Hs: dict[int, list], *,
         window: tuple[float, float], offsets: list[float],
         pitch: tuple[float, float], t0: float | None = None,
         crops_for=None, ownership: bool = OWNERSHIP,
         log=print) -> dict:
    """dets: {angle: load_dets() output}. Hs: {angle: 3x3 frame->pitch}.
    window: (lo,hi) shared seconds. Returns the tracks.json doc."""
    from .calib import apply_h
    lo, hi = window
    L, W = pitch
    n_steps = max(1, round((hi - lo) / STEP) + 1)

    # camera ownership: each camera's pitch position is its image
    # bottom-centre ((0.5, 1.0) normalized — feet are stored normalized
    # by meta w/h). Angles without meta w/h never own and are never
    # filtered; "nearest" is compared among cameras with a position.
    cam_xy: dict[int, tuple[float, float]] = {}
    if ownership:
        for a, d in dets.items():
            H = Hs.get(a)
            if H is None or not d.get("w") or not d.get("h"):
                continue
            cx, cy = apply_h(H, 0.5, 1.0)
            if math.isfinite(cx) and math.isfinite(cy):
                cam_xy[a] = (cx, cy)
    # step-indexed footpoints per angle
    per_step: dict[int, list[dict]] = {}
    for a, d in dets.items():
        H = Hs.get(a)
        if H is None:
            continue
        off = offsets[a] if a < len(offsets) else 0.0
        for i in range(len(d["t"])):
            ts = float(d["t"][i]) + off
            k = round((ts - lo) / STEP)
            if not (0 <= k < n_steps):
                continue
            fx, fy = float(d["foot"][i][0]), float(d["foot"][i][1])
            x, y = apply_h(H, fx, fy)
            if (len(cam_xy) >= 2 and a in cam_xy
                    and any(math.hypot(x - ox, y - oy)
                            < math.hypot(x - cam_xy[a][0],
                                         y - cam_xy[a][1])
                            for o, (ox, oy) in cam_xy.items() if o != a)):
                continue
            if not (-EDGE_MARGIN_M <= x <= L + EDGE_MARGIN_M
                    and -EDGE_MARGIN_M <= y <= W + EDGE_MARGIN_M):
                continue
            x = min(max(x, 0.0), L)
            y = min(max(y, 0.0), W)
            tm = str(d["team"][i]) if d["team"][i] in ("A", "B") else None
            per_step.setdefault(k, []).append(
                {"x": x, "y": y, "conf": float(d["conf"][i]),
                 "team": tm, "angle": a,
                 "ref": {"angle": a, "box": [float(v) for v in d["box"][i]],
                         "t_file": float(d["t"][i])}})

    tracks: list[dict] = []
    next_id = V2_ID_BASE
    vis_hist = np.zeros(n_steps, dtype=int)
    for k in range(n_steps):
        pts = _merge_obs(per_step.get(k, []))
        vis_hist[k] = len(pts)
        alive = [t for t in tracks if t["miss_s"] < MAX_MISS_S]
        if alive and pts:
            from scipy.optimize import linear_sum_assignment
            cost = np.full((len(alive), len(pts)), 1e6)
            for ti, tr in enumerate(alive):
                _kalman_pred(tr, STEP)
                px, py = float(tr["x"][0]), float(tr["x"][1])
                gate = GATE_M if tr["miss_s"] <= 0 else GATE_MISS_M
                for oi, o in enumerate(pts):
                    d = math.hypot(px - o["x"], py - o["y"])
                    if d > gate:
                        continue
                    o_team = _majority(o["teams"])
                    if o_team and tr["team"] and o_team != tr["team"]:
                        continue
                    cost[ti, oi] = d
            rows, cols = linear_sum_assignment(cost)
            used = set()
            for ti, oi in zip(rows, cols):
                if cost[ti, oi] >= 1e6:
                    continue
                tr, o = alive[ti], pts[oi]
                _kalman_upd(tr, o["x"], o["y"])
                tr["xy"].append([o["x"], o["y"]])
                tr["step_end"] = k
                tr["miss_s"] = 0.0
                tr["all_teams"].extend(o["teams"])
                tr["refs"].append((k, o["refs"]))
                o_team = _majority(o["teams"])
                if o_team and not tr["team"]:
                    tr["team"] = o_team
                used.add(oi)
            missed = set(range(len(alive))) - {
                int(ti) for ti, oi in zip(rows, cols)
                if cost[ti, oi] < 1e6}
            for ti in missed:
                alive[ti]["miss_s"] += STEP
                alive[ti]["xy"].append([np.nan, np.nan])
            pts = [o for oi, o in enumerate(pts) if oi not in used]
        elif alive:
            for tr in alive:
                _kalman_pred(tr, STEP)
                tr["miss_s"] += STEP
                tr["xy"].append([np.nan, np.nan])
        # suppress unmerged duplicates: no new track within the gate of
        # an existing track's prediction
        pts = [o for o in pts if not any(
            math.hypot(float(tr["x"][0]) - o["x"],
                       float(tr["x"][1]) - o["y"]) < GATE_M
            for tr in alive)]
        for o in pts:
            tr = _kalman_new(o["x"], o["y"])
            tr.update({"id": next_id, "team": _majority(o["teams"]),
                       "step_start": k, "step_end": k, "miss_s": 0.0,
                       "xy": [[o["x"], o["y"]]], "all_teams": o["teams"][:],
                       "refs": [(k, o["refs"])]})
            next_id += 1
            tracks.append(tr)
    return _finish(tracks, lo, pitch, vis_hist, t0 if t0 is not None else lo,
                   n_steps, crops_for, log, ownership, cam_xy)


def _finish(tracks, lo, pitch, vis_hist, t0, n_steps, crops_for, log,
            ownership, cam_xy):
    out = []
    for tr in tracks:
        dur = (tr["step_end"] - tr["step_start"] + 1) * STEP
        if dur < MIN_LEN_S:
            continue
        # fill NaN gaps <= FILL_GAP_S linearly
        xy = np.array(tr["xy"], dtype=float)
        for i in range(len(xy)):
            if not np.isnan(xy[i, 0]):
                continue
            j = i
            while j < len(xy) and np.isnan(xy[j, 0]):
                j += 1
            gap_s = (j - i) * STEP
            if i > 0 and j < len(xy) and gap_s <= FILL_GAP_S:
                for m in range(i, j):
                    f = (m - i + 1) / (j - i + 1)
                    xy[m] = xy[i - 1] + (xy[j] - xy[i - 1]) * f
        dist = 0.0
        sprints = 0
        for i in range(1, len(xy)):
            if np.isnan(xy[i]).any() or np.isnan(xy[i - 1]).any():
                continue
            d = float(np.hypot(*(xy[i] - xy[i - 1])))
            dist += d
            if d / STEP >= SPRINT_MS:
                sprints += 1
        team = _majority(tr["all_teams"])
        out.append({"id": tr["id"], "team": team,
                    "start": round(lo + tr["step_start"] * STEP, 3),
                    "end": round(lo + tr["step_end"] * STEP, 3),
                    "xy": [[None if np.isnan(v) else round(float(v), 2)
                            for v in p] for p in xy],
                    "dist_m": round(dist, 1), "sprints": sprints,
                    "crops": (crops_for(tr) if crops_for else [])})
    lens = [(o["end"] - o["start"]) for o in out]
    med_vis = float(np.median(vis_hist)) if len(vis_hist) else 0.0
    doc = {"step": STEP, "t0": t0, "tracks": out,
           "summary": {"n_tracks": len(out),
                       "mean_len_s": round(float(np.mean(lens)), 1)
                       if lens else 0.0,
                       "median_visible": med_vis,
                       "visible_hist": vis_hist.tolist(),
                       "ownership": bool(ownership),
                       "cam_xy": {str(a): [round(c, 1) for c in xy]
                                  for a, xy in cam_xy.items()}}}
    log(f"fuse: {len(out)} tracks, median visible {med_vis:.0f}/step")
    return doc


def run_fuse(project_dir: Path, players_v2: Path, log=print,
             status=None, crops_for=None) -> dict:
    """Load calib + det_a*.npz, fuse over the shared window, write
    tracks.json + summary.json into players_v2."""
    from .run import _load_json, resolve_context
    ctx = resolve_context(project_dir)
    calib = _load_json(project_dir / "multiangle" / "calib.json") or {}
    angles = calib.get("angles") or {}
    n = int(ctx["n_angles"])
    missing = [a for a in range(n) if str(a) not in angles]
    if missing:
        from highlights.pipeline.errors import PipelineError
        raise PipelineError(
            f"calibration missing for angle(s) {missing} — set "
            "landmarks first")
    pitch = calib.get("pitch") or {}
    L = float(pitch.get("len_m") or 100.0)
    W = float(pitch.get("wid_m") or 64.0)
    Hs = {int(k): v["H"] for k, v in angles.items()}
    dets = {}
    for a in range(n):
        f = players_v2 / f"det_a{a}.npz"
        if f.exists():
            dets[a] = load_dets(f)
    doc = fuse(dets, Hs, window=(ctx["window"][0], ctx["window"][1]),
               offsets=ctx["offsets"], pitch=(L, W),
               crops_for=crops_for, log=log)
    # ball: per-angle features_1s ball through H, conf-weighted
    ball = _fuse_ball(project_dir, ctx, Hs, L, W)
    if ball:
        doc["ball"] = ball
    write_json_atomic(players_v2 / "tracks.json", doc, indent=0)
    write_json_atomic(players_v2 / "summary.json",
                      doc["summary"], indent=1)
    return doc


def _fuse_ball(project_dir, ctx, Hs, L, W):
    from .calib import apply_h
    from .run import _load_json
    rows_by_t: dict[int, list] = {}
    for a, H in Hs.items():
        feats = _load_json(project_dir / "angles" / f"a{a}" / "track"
                           / "features_1s.json") or {}
        ci = {k: i for i, k in enumerate(feats.get("columns") or [])}
        if not {"ball_conf", "ball_x", "ball_y"} <= set(ci):
            continue
        off = ctx["offsets"][a] if a < len(ctx["offsets"]) else 0.0
        for i, r in enumerate(feats.get("rows") or []):
            bc, bx, by = r[ci["ball_conf"]], r[ci["ball_x"]], r[ci["ball_y"]]
            if bc < BALL_CONF or bx <= 0 or by <= 0:
                continue
            x, y = apply_h(H, bx, by)
            if not (0 <= x <= L and 0 <= y <= W):
                continue
            k = round(i + off)
            rows_by_t.setdefault(k, []).append((bc, x, y))
    out = []
    for k in sorted(rows_by_t):
        pts = rows_by_t[k]
        wsum = sum(c for c, _, _ in pts)
        out.append([float(k),
                    round(sum(c * x for c, x, _ in pts) / wsum, 2),
                    round(sum(c * y for c, _, y in pts) / wsum, 2)])
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="highlights.analysis.fuse_tracks")
    ap.add_argument("--project-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    run_fuse(args.project_dir,
             args.project_dir / "analysis" / "players_v2")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
