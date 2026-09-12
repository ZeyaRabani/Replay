"""Post-processing helpers for tracking quality and team visualizations."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.signal import savgol_filter

from .pitch import PitchModel


def filter_static_tracks(timeline: list[dict], pitch: PitchModel, fps: float, min_disp_m: float = 1.5,
                         min_duration_s: float = 2.0, band_m: float = 3.0) -> tuple[list[dict], list[int]]:
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for fr in timeline:
        for p in fr["players"]:
            tracks[p["id"]].append((fr["frame"], p["x"], p["y"]))
    dropped = set()
    for gid, points in tracks.items():
        duration = (points[-1][0] - points[0][0] + 1) / max(fps, 1e-9)
        xy = np.array([[p[1], p[2]] for p in points])
        median = np.median(xy, axis=0)
        if (duration >= min_duration_s and np.max(np.linalg.norm(xy - median, axis=1)) < min_disp_m
                and not pitch.contains(median[0], median[1], 0.0)
                and pitch.contains(median[0], median[1], band_m)):
            dropped.add(gid)
    out = [{**fr, "players": [p for p in fr["players"] if p["id"] not in dropped]} for fr in timeline]
    return out, sorted(dropped)


def smooth_tracks(timeline: list[dict], fps: float, window: int = 9, polyorder: int = 2,
                  max_gap: int = 10, min_frames: int = 15) -> list[dict]:
    tracks: dict[int, dict[int, dict]] = defaultdict(dict)
    for fr in timeline:
        for p in fr["players"]:
            tracks[p["id"]][fr["frame"]] = p
    keep = {gid for gid, frames in tracks.items() if len(frames) >= min_frames}
    processed: dict[int, dict[int, dict]] = {}
    for gid in keep:
        frames = tracks[gid]
        all_frames = dict(frames)
        keys = sorted(frames)
        for a, b in pairwise(keys):
            if 1 < b - a <= max_gap + 1:
                pa, pb = frames[a], frames[b]
                for f in range(a + 1, b):
                    r = (f - a) / (b - a)
                    all_frames[f] = {
                        "id": gid, "x": pa["x"] + r * (pb["x"] - pa["x"]), "y": pa["y"] + r * (pb["y"] - pa["y"]),
                        "conf": 0.0, "cameras": [], "detections": [], "interpolated": True,
                    }
        ordered = sorted(all_frames)
        x = np.array([all_frames[f]["x"] for f in ordered], dtype=float)
        y = np.array([all_frames[f]["y"] for f in ordered], dtype=float)
        for start in range(len(ordered)):
            end = start
            while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
                end += 1
            n = end - start + 1
            xs, ys = x[start:end + 1], y[start:end + 1]
            if n >= 3:
                w = min(window, n if n % 2 else n - 1)
                w = max(3, w)
                if w > n:
                    w = n if n % 2 else n - 1
                po = min(polyorder, w - 1)
                xs, ys = savgol_filter(xs, w, po), savgol_filter(ys, w, po)
            vx = np.gradient(xs) * fps if n >= 3 else np.zeros(n)
            vy = np.gradient(ys) * fps if n >= 3 else np.zeros(n)
            for i, f in enumerate(ordered[start:end + 1]):
                all_frames[f]["x"] = round(float(xs[i]), 2)
                all_frames[f]["y"] = round(float(ys[i]), 2)
                all_frames[f]["vx"] = round(float(vx[i]), 2)
                all_frames[f]["vy"] = round(float(vy[i]), 2)
                all_frames[f]["speed"] = round(float(np.hypot(vx[i], vy[i])), 2)
        processed[gid] = all_frames
    out = []
    for fr in timeline:
        players = [processed[p["id"]][fr["frame"]] for p in fr["players"] if p["id"] in processed]
        for gid, frames in processed.items():
            if fr["frame"] in frames and not any(p["id"] == gid for p in players):
                players.append(frames[fr["frame"]])
        out.append({**fr, "players": sorted(players, key=lambda p: p["id"])})
    return out


def jersey_feature(frame_bgr: np.ndarray, box: list[float]) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, x1 + int(0.2 * bw)), min(w, x2 - int(0.2 * bw))
    y1, y2 = max(0, y1 + int(0.15 * bh)), min(h, y1 + int(0.5 * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    keep = ~(((hsv[:, 0] >= 35) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 60)) | (hsv[:, 2] < 40))
    hsv = hsv[keep]
    if len(hsv) < 20:
        return None
    hue = hsv[:, 0].astype(float) * 2 * np.pi / 180
    sat = hsv[:, 1].astype(float)
    val = hsv[:, 2].astype(float)
    return np.median(np.stack([sat * np.cos(hue), sat * np.sin(hue), sat, val], axis=1), axis=0)


def collect_jersey_features(timeline: list[dict], synced_clips: list[Path], n_samples: int = 40) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = defaultdict(list)
    if not timeline:
        return out
    indices = np.linspace(0, len(timeline) - 1, min(n_samples, len(timeline)), dtype=int)
    caps = [cv2.VideoCapture(str(path)) for path in synced_clips]
    try:
        for f in sorted(set(indices)):
            for cam, cap in enumerate(caps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, image = cap.read()
                if not ok:
                    continue
                for p in timeline[f]["players"]:
                    for det in p.get("detections", []):
                        if det["camera"] == cam:
                            feature = jersey_feature(image, det["box"])
                            if feature is not None:
                                out[p["id"]].append(feature)
                            break
    finally:
        for cap in caps:
            cap.release()
    return out


def cluster_teams(features_by_id: dict[int, np.ndarray | list[np.ndarray]], k: int = 3,
                  min_team_frac: float = 0.15) -> dict[int, tuple[int | None, str]]:
    ids = sorted(features_by_id)
    medians = []
    for gid in ids:
        values = np.asarray(features_by_id[gid])
        medians.append(np.median(values, axis=0) if values.ndim > 1 else values)
    if not ids:
        return {}
    data = np.asarray(medians, dtype=float)
    n_clusters = min(k, len(ids))
    if n_clusters == 1:
        labels = np.zeros(len(ids), dtype=int)
        centres = np.array([data.mean(axis=0)])
    else:
        centres, labels = kmeans2(data, n_clusters, minit="++", seed=0)
    counts = np.bincount(labels, minlength=n_clusters)
    big = [i for i in range(n_clusters) if counts[i] > max(2, min_team_frac * len(ids))]
    if len(big) < 2:
        big = sorted(range(n_clusters), key=lambda i: counts[i], reverse=True)[:2]
    big = sorted(big, key=lambda i: math.atan2(centres[i][1], centres[i][0]))
    result = {}
    for pos, cluster in enumerate(big[:2]):
        for gid, label in zip(ids, labels):
            if label == cluster:
                result[gid] = (pos, "player")
    for gid, label in zip(ids, labels):
        if gid in result:
            continue
        if len(big) >= 2 and counts[label] <= max(2, min_team_frac * len(ids)):
            result[gid] = (None, "other")
        else:
            nearest = min(big, key=lambda i: np.linalg.norm(centres[i] - centres[label]))
            result[gid] = (big.index(nearest), "player")
    return result


def assign_teams(timeline: list[dict], synced_clips: list[Path], pitch: PitchModel, n_samples: int = 40
                 ) -> tuple[list[dict], dict]:
    features = collect_jersey_features(timeline, synced_clips, n_samples)
    assignments = cluster_teams(features)
    all_ids = sorted({p["id"] for fr in timeline for p in fr["players"]})
    for gid in all_ids:
        assignments.setdefault(gid, (None, "other"))
    medians = {}
    for gid in all_ids:
        pts = [(p["x"], p["y"]) for fr in timeline for p in fr["players"] if p["id"] == gid]
        medians[gid] = np.median(pts, axis=0) if pts else (0, 0)
    updated = []
    for fr in timeline:
        players = []
        for p in fr["players"]:
            team, role = assignments[p["id"]]
            if role == "other":
                role = "goalkeeper" if medians[p["id"]][0] < 8 or medians[p["id"]][0] > pitch.length - 8 else "referee"
            players.append({**p, "team": team, "role": role})
        updated.append({**fr, "players": players})
    team_sizes = {str(team): sum(1 for gid, (t, role) in assignments.items() if t == team and role == "player")
                  for team in (0, 1)}
    colours = []
    for team in (0, 1):
        vals = [np.asarray(features[gid]) for gid, (t, role) in assignments.items() if t == team and gid in features]
        if vals:
            c = np.median(np.concatenate(vals), axis=0)
            hue = (math.atan2(c[1], c[0]) % (2 * math.pi)) * 180 / math.pi / 2
            hsv = np.uint8([[[hue, np.clip(c[2], 0, 255), np.clip(c[3], 0, 255)]]])
            colours.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist())
        else:
            colours.append([255, 255, 255])
    info = {"team_sizes": team_sizes, "team_colours_bgr": colours,
            "other_ids": sorted(gid for gid, (team, role) in assignments.items() if role == "other")}
    return updated, info


def id_stability(timeline: list[dict], fps: float | None = None) -> dict:
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for fr in timeline:
        for p in fr["players"]:
            tracks[p["id"]].append((fr["frame"], p["x"], p["y"]))
    starts = {gid: points[0] for gid, points in tracks.items()}
    ends = {gid: points[-1] for gid, points in tracks.items()}
    switches = 0
    for gid, (f0, x0, y0) in starts.items():
        if f0 <= 0:
            continue
        if any(other != gid and f0 - 30 <= end[0] < f0 and np.hypot(end[1] - x0, end[2] - y0) <= 2.0
               for other, end in ends.items()):
            switches += 1
    if fps is None and len(timeline) > 1:
        dt = timeline[1]["t"] - timeline[0]["t"]
        fps = 1 / dt if dt else None
    duration_min = (len(timeline) / fps / 60) if fps else None
    lengths = [len(points) for points in tracks.values()]
    counts = [len(fr["players"]) for fr in timeline]
    return {"id_switches_est": switches, "id_switches_per_min": switches / duration_min if duration_min else None,
            "fragmentation": len(tracks) / max(1, np.mean(counts) if counts else 1),
            "mean_track_len_frames": float(np.mean(lengths)) if lengths else 0.0,
            "short_tracks_lt15": sum(length < 15 for length in lengths)}


def render_team_snapshot(timeline: list[dict], pitch: PitchModel, frame_idx: int, out_path: Path,
                         team_colours_bgr: list[list[int]]) -> None:
    from .viz import PitchCanvas

    canvas = PitchCanvas(pitch)
    image = canvas.base()
    frame = timeline[min(frame_idx, len(timeline) - 1)]
    for p in frame["players"]:
        pt = canvas.to_px(p["x"], p["y"])
        team = p.get("team")
        colour = tuple(team_colours_bgr[team]) if team in (0, 1) else (0, 255, 255)
        if p.get("interpolated"):
            cv2.circle(image, pt, 8, colour, 2)
        else:
            cv2.circle(image, pt, 7, colour, -1)
        cv2.putText(image, str(p["id"]), (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    if frame.get("ball"):
        b = frame["ball"]
        cv2.circle(image, canvas.to_px(b["x"], b["y"]), 5, (255, 255, 255), -1)
    cv2.imwrite(str(out_path), image)


def render_jersey_sheet(synced_clip: Path, timeline: list[dict], cam: int, frame_idx: int, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(synced_clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return
    crops = []
    for p in timeline[min(frame_idx, len(timeline) - 1)]["players"]:
        for det in p.get("detections", []):
            if det["camera"] != cam:
                continue
            x1, y1, x2, y2 = map(int, det["box"])
            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                crop = cv2.resize(crop, (64, 128))
                cv2.putText(crop, f"{p['id']} T{p.get('team', '-')}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 255), 1)
                crops.append(crop)
            break
    if crops:
        cv2.imwrite(str(out_path), np.hstack(crops))
