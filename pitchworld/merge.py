"""Project per-camera tracks to pitch coordinates and fuse them into one set of
players per synced frame.

Input per camera: ``frames[i] = [{"id", "conf", "box": [x1,y1,x2,y2]}, ...]``
(same frame indexing across cameras because clips were re-encoded to a common
CFR timeline). Output: per frame, a list of global players.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibrate import CameraCalibration
from .pitch import PitchModel


@dataclass
class Obs:
    cam: int
    tid: int
    xy: np.ndarray
    conf: float
    box: list[float]
    weight: float  # larger = trust more (closer to camera)


def project_frame(dets: list[dict], cal: CameraCalibration, pitch: PitchModel, cam: int,
                  margin: float = 3.0, min_box_h: int = 12) -> list[Obs]:
    if not dets:
        return []
    boxes = np.array([d["box"] for d in dets], dtype=float)
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    xy = cal.project(feet)
    out = []
    for d, p, b in zip(dets, xy, boxes):
        h = b[3] - b[1]
        if h < min_box_h or not np.all(np.isfinite(p)) or not pitch.contains(p[0], p[1], margin):
            continue  # off-pitch (spectators, adjacent pitch) or numerically behind the horizon
        out.append(Obs(cam=cam, tid=int(d["id"]), xy=p, conf=float(d["conf"]), box=list(b), weight=float(h)))
    return out


def _cluster_frame(obs: list[Obs], merge_dist: float) -> list[list[Obs]]:
    """Group observations from *different* cameras that are within ``merge_dist`` m.

    Pairwise Hungarian assignment between each camera pair, then union-find. A
    cluster never contains two observations from the same camera.
    """
    by_cam: dict[int, list[Obs]] = defaultdict(list)
    for o in obs:
        by_cam[o.cam].append(o)
    cams = sorted(by_cam)
    parent = list(range(len(obs)))
    index = {id(o): i for i, o in enumerate(obs)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def cam_set(root):
        return {obs[i].cam for i in range(len(obs)) if find(i) == root}

    for a_i in range(len(cams)):
        for b_i in range(a_i + 1, len(cams)):
            A, B = by_cam[cams[a_i]], by_cam[cams[b_i]]
            D = np.linalg.norm(np.array([o.xy for o in A])[:, None, :] - np.array([o.xy for o in B])[None, :, :], axis=-1)
            rows, cols = linear_sum_assignment(D)
            for r, c in zip(rows, cols):
                if D[r, c] <= merge_dist:
                    ra, rb = find(index[id(A[r])]), find(index[id(B[c])])
                    if ra != rb and not (cam_set(ra) & cam_set(rb)):
                        parent[rb] = ra
    groups: dict[int, list[Obs]] = defaultdict(list)
    for i, o in enumerate(obs):
        groups[find(i)].append(o)
    return list(groups.values())


class GlobalIdentity:
    """Maps (camera, track id) -> stable global player id, learning links from co-clustering."""

    def __init__(self):
        self.link: dict[tuple[int, int], int] = {}
        self.next_id = 1
        self.last_pos: dict[int, np.ndarray] = {}
        self.last_seen: dict[int, int] = {}

    def assign(self, groups: list[list[Obs]], frame: int, reacquire_dist: float = 1.5, max_gap: int = 30) -> list[tuple[int, list[Obs]]]:
        out = []
        used: set[int] = set()
        pending = []
        for g in groups:
            votes = Counter(self.link[(o.cam, o.tid)] for o in g if (o.cam, o.tid) in self.link)
            gid = None
            for cand, _ in votes.most_common():
                if cand not in used:
                    gid = cand
                    break
            if gid is None:
                pending.append(g)
                continue
            used.add(gid)
            out.append((gid, g))
        # New/unlinked groups: try to re-acquire a recently lost global id nearby, else mint one
        for g in pending:
            pos = np.average(np.array([o.xy for o in g]), axis=0, weights=[o.weight for o in g])
            best, best_d = None, reacquire_dist
            for gid, lp in self.last_pos.items():
                if gid in used or frame - self.last_seen.get(gid, -10**9) > max_gap:
                    continue
                d = float(np.linalg.norm(lp - pos))
                if d < best_d:
                    best, best_d = gid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            out.append((best, g))
        for gid, g in out:
            for o in g:
                self.link[(o.cam, o.tid)] = gid
            pos = np.average(np.array([o.xy for o in g]), axis=0, weights=[o.weight for o in g])
            self.last_pos[gid] = pos
            self.last_seen[gid] = frame
        return out


def fuse(per_cam_frames: list[list[list[dict]]], calibrations: list[CameraCalibration], pitch: PitchModel,
         fps: float, merge_dist: float = 2.0, smooth_alpha: float = 0.6) -> list[dict]:
    n_frames = min(len(f) for f in per_cam_frames)
    ident = GlobalIdentity()
    smoothed: dict[int, np.ndarray] = {}
    timeline = []
    for f in range(n_frames):
        obs: list[Obs] = []
        for cam, (frames, cal) in enumerate(zip(per_cam_frames, calibrations)):
            obs += project_frame(frames[f], cal, pitch, cam)
        groups = _cluster_frame(obs, merge_dist)
        players = []
        for gid, g in ident.assign(groups, f):
            pos = np.average(np.array([o.xy for o in g]), axis=0, weights=[o.weight for o in g])
            if gid in smoothed:
                pos = smooth_alpha * pos + (1 - smooth_alpha) * smoothed[gid]
            smoothed[gid] = pos
            players.append({
                "id": gid,
                "x": round(float(pos[0]), 2), "y": round(float(pos[1]), 2),
                "conf": round(float(np.mean([o.conf for o in g])), 3),
                "cameras": sorted(o.cam for o in g),
                "detections": [{"camera": o.cam, "track_id": o.tid, "box": [round(v, 1) for v in o.box]} for o in g],
            })
        players.sort(key=lambda p: p["id"])
        timeline.append({"frame": f, "t": round(f / fps, 4), "players": players})
    return timeline


def summarize(timeline: list[dict]) -> dict:
    ids = Counter()
    multi = 0
    total = 0
    for fr in timeline:
        for p in fr["players"]:
            ids[p["id"]] += 1
            total += 1
            multi += len(p["cameras"]) > 1
    return {"unique_ids": len(ids), "mean_players_per_frame": round(total / max(1, len(timeline)), 2),
            "frac_multi_camera": round(multi / max(1, total), 3),
            "ids_seen_over_half": sum(1 for c in ids.values() if c > len(timeline) / 2)}
