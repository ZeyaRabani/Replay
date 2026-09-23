"""Near-goal candidate extraction from detect_neargoal chunk outputs.

Merges ball detections and MOG2 blobs into observations, links them into
multi-tracks in full-frame pixel space, and emits goal/chance candidates:
net entry after a mouth crossing, or a net-motion spike timed by a fast
track; fast tracks ending near the mouth are chances.
"""

from __future__ import annotations

import numpy as np

from .combine import Candidate
from .config import Config


def _in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    """Ray casting point-in-polygon."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _seg_near_poly(p0: np.ndarray, p1: np.ndarray, poly: np.ndarray, tol: float) -> bool:
    """Segment crosses the polygon or passes within tol of any vertex/edge."""
    for pt in np.linspace(0, 1, 50)[:, None] * (p1 - p0) + p0:
        if _in_poly(poly, pt[0], pt[1]):
            return True
        if np.linalg.norm(poly - pt, axis=1).min() <= tol:
            return True
    return False


def _observations(chunks: list[dict], max_blob_area: float = 100.0,
                  max_obs_per_frame: int = 30) -> list[tuple[int, float, float, str]]:
    """Merge ball dets + blobs per frame -> [(frame_idx, x, y, kind)].

    MOG2 yields thousands of grass/crowd blobs per frame; keep only ball-sized
    blobs (area <= max_blob_area at half-res), cap per frame preferring smallest,
    and drop blobs in cells that fire in >70% of frames (static background churn).
    """
    per_frame: dict[int, list] = {}
    for ch in chunks:
        for fi, cx, cy, _w, _h, conf in ch["ball"]:
            per_frame.setdefault(fi, []).append((cx, cy, "ball", conf))
    blob_cells: dict[tuple, int] = {}
    n_frames = 0
    for ch in chunks:
        n_frames += len({b[0] for b in ch["blobs"]} | {b[0] for b in ch["ball"]})
        for fi, cx, cy, area in ch["blobs"]:
            key = (int(cx) // 8, int(cy) // 8)
            blob_cells[key] = blob_cells.get(key, 0) + 1
    for ch in chunks:
        for fi, cx, cy, area in ch["blobs"]:
            if area > max_blob_area:
                continue
            if blob_cells[(int(cx) // 8, int(cy) // 8)] > 0.7 * n_frames:
                continue  # static churn
            balls = [o for o in per_frame.get(fi, []) if o[2] == "ball"]
            if any(np.hypot(cx - bx, cy - by) <= 40.0 for bx, by, *_ in balls):
                continue  # blob duplicates a ball det
            per_frame.setdefault(fi, []).append((cx, cy, "blob", area))
    out = []
    for fi in sorted(per_frame):
        obs = per_frame[fi]
        if len(obs) > max_obs_per_frame:
            obs.sort(key=lambda o: (o[2] != "ball", o[3]))  # ball dets first, then smallest blobs
            obs = obs[:max_obs_per_frame]
        for x, y, kind, conf in obs:
            out.append((fi, x, y, kind))
    return out


def link_tracks(obs: list[tuple[int, float, float, str]], max_jump: float = 120.0,
                min_len: int = 4, max_gap: int = 6) -> list[list[tuple]]:
    """Greedy constant-velocity multi-track linker."""
    by_frame: dict[int, list] = {}
    for o in obs:
        by_frame.setdefault(o[0], []).append(o)
    tracks: list[list[tuple]] = []
    active: list[dict] = []
    for fi in sorted(by_frame):
        remaining = list(by_frame[fi])
        for tr in active:
            gap = fi - tr["last_fi"]
            if gap > max_gap:
                continue
            pred = tr["pos"] + tr["vel"] * gap
            if not remaining:
                break
            d = [np.hypot(o[1] - pred[0], o[2] - pred[1]) for o in remaining]
            j = int(np.argmin(d))
            if d[j] <= max_jump * gap:
                o = remaining.pop(j)
                tr["vel"] = (np.array([o[1], o[2]]) - tr["pos"]) / gap
                tr["pos"] = np.array([o[1], o[2]])
                tr["obs"].append(o)
                tr["last_fi"] = fi
        for o in remaining:
            active.append({"pos": np.array([o[1], o[2]]), "vel": np.zeros(2),
                           "obs": [o], "last_fi": fi})
        # retire stale tracks
        still = []
        for tr in active:
            (still if fi - tr["last_fi"] <= max_gap else tracks).append(tr)
        active = still
    tracks.extend(active)
    return [t["obs"] for t in tracks if len(t["obs"]) >= min_len]


def _track_metrics(track: list[tuple], fps: float) -> dict:
    fi = np.array([o[0] for o in track], float)
    xy = np.array([[o[1], o[2]] for o in track])
    dt = np.diff(fi) / fps
    dt[dt == 0] = np.nan
    speed = np.hypot(*np.diff(xy, axis=0).T) / dt
    n_ball = sum(1 for o in track if o[3] == "ball")
    return {"fis": fi, "xy": xy, "speed": speed, "vmax": float(np.nanmax(speed)) if len(speed) else 0.0,
            "n_ball": n_ball, "t0": fi[0] / fps, "t1": fi[-1] / fps,
            "dur": (fi[-1] - fi[0]) / fps}


def neargoal_candidates(chunks: list[dict], audio_bins: np.ndarray, bin_s: float,
                        cfg: Config, t_offset: float, duration_s: float) -> list[Candidate]:
    """Emit Candidates (anchor='neargoal', goal='A') from cached chunk dicts."""
    net = np.array(cfg.net_poly, float)
    mouth = np.array(cfg.mouth_poly, float)
    obs = _observations(chunks)
    tracks = link_tracks(obs)
    fps = chunks[0]["fps"]

    # net_motion time series -> absolute-time spikes
    nm: dict[int, float] = {}
    for ch in chunks:
        for fi, v in ch["net_motion"]:
            nm[fi] = max(nm.get(fi, 0.0), v)
    fis = np.array(sorted(nm))
    vals = np.array([nm[f] for f in fis])
    w = int(20.0 * fps)
    spikes: list[tuple[float, int, int]] = []  # (t_start, i0, i1)
    med = np.array([np.median(vals[max(0, i - w):i + 1]) for i in range(len(vals))])
    run = -1
    for i in range(len(vals) + 1):
        on = i < len(vals) and vals[i] > 5.0 * med[i] and vals[i] > 0
        if on and run < 0:
            run = i
        elif not on and run >= 0:
            if i - run >= 3:
                spikes.append((fis[run] / fps, run, i - 1))
            run = -1

    def audio_boost(t_abs: float) -> float:
        b0 = int((t_abs - t_offset) / bin_s)
        b1 = int((t_abs + 6.0 - t_offset) / bin_s)
        return float(audio_bins[max(0, b0):b1].max()) if len(audio_bins) and b1 > 0 else 0.0

    cands: list[Candidate] = []

    def _emit(t_ev: float, typ: str, base: float, m: dict | None = None) -> None:
        conf = base
        ab = audio_boost(t_ev)
        if ab > 0.3:
            conf += 0.2
        if m and m["vmax"] > 1200:
            conf += 0.1
        if m and m["n_ball"] >= 3:
            conf += 0.1
        roll = cfg.roll_goal_s if typ == "goal" else cfg.roll_chance_s
        cands.append(Candidate("", 0, typ, round(min(conf, 1.0), 3), "A", round(t_ev, 2),
                               round(max(0.0, t_ev - roll), 2), round(min(t_ev + roll, duration_s), 2),
                               {"speed": round(m["vmax"], 0) if m else 0.0,
                                "n_ball_obs": float(m["n_ball"]) if m else 0.0,
                                "audio": round(ab, 3)}, anchor="neargoal"))

    for tr in tracks:
        m = _track_metrics(tr, fps)
        in_net = np.array([_in_poly(net, x, y) for x, y in m["xy"]])
        in_mouth = np.array([_in_poly(mouth, x, y) for x, y in m["xy"]])
        net_entry = np.where(in_net)[0]
        if len(net_entry) >= 2 and in_mouth[:net_entry[0]].any():
            _emit(m["fis"][net_entry[0]] / fps, "goal", 0.6, m)
            continue
        # shot: fast track whose path approaches the mouth
        if m["vmax"] > 600 and m["dur"] >= 0.3:
            i_v = int(np.nanargmax(m["speed"]))
            p0 = m["xy"][i_v]
            v_hat = np.diff(m["xy"], axis=0)[i_v]
            nrm = np.linalg.norm(v_hat)
            if nrm > 0:
                p1 = p0 + v_hat / nrm * 800.0
                if _seg_near_poly(p0, p1, mouth, 150.0):
                    _emit(m["t0"], "chance", 0.5, m)
    # net-motion spikes timed by a fast track ending <=1 s before
    for t0_rel, i0, i1 in spikes:
        t_spike = t0_rel  # fis are absolute frame idx -> /fps is absolute t
        for tr in tracks:
            m = _track_metrics(tr, fps)
            if m["vmax"] > 600 and 0 <= t_spike - (m["t1"]) <= 1.0:
                _emit(t_spike, "goal", 0.5, m)
                break

    # merge within 8 s keeping max conf, union signals
    merged: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.t_event):
        if merged and c.t_event - merged[-1].t_event < 8.0:
            prev = merged[-1]
            keep = c if c.confidence > prev.confidence else prev
            other = prev if keep is c else c
            keep.signals = {k: max(keep.signals.get(k, 0.0), other.signals.get(k, 0.0))
                            for k in keep.signals.keys() | other.signals.keys()}
            keep.start = min(prev.start, c.start)
            keep.end = max(prev.end, c.end)
            merged[-1] = keep
        else:
            merged.append(c)
    merged.sort(key=lambda c: -c.confidence)
    for rank, c in enumerate(merged, start=1):
        c.rank = rank
        c.id = f"c{rank:02d}"
    return merged
