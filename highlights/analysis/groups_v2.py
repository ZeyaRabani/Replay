"""Group fused players-v2 tracks into likely-same-player groups.

Reuses groups.py's fingerprint + constrained clustering on the fused
pitch-space tracks (tracks.json). Fused tracks are already
cross-camera, so the same cannot-link rule applies verbatim: two tracks
overlapping >=1 s in time are two different people.

    build_groups_v2(players_v2_dir) -> writes groups.json, returns doc
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from highlights.io import write_json_atomic

from .fuse_tracks import SPRINT_MS, STEP
from .groups import DIST_THRESH, FEAT_DIM, MAX_PER_TEAM, group_tracklets, tracklet_fingerprint

TEAM_ORDER = {"A": 0, "B": 1, None: 2}


def _xy_at(track: dict, t: float) -> list | None:
    """[x,y] pitch metres at shared second t, or None."""
    i = round((t - float(track["start"])) / STEP)
    xy = track.get("xy") or []
    if 0 <= i < len(xy) and xy[i][0] is not None:
        return xy[i]
    return None


def _is_dup(a: dict, b: dict, *, max_gap_m: float, min_overlap_s: float,
            max_offset_std_m: float) -> bool:
    """Two fused tracks are the same player seen from two cameras:
    compatible teams, overlap >= min_overlap_s, >=4 shared valid steps,
    mean distance <= max_gap_m, and a *constant* offset — the
    cross-camera homography error displaces a duplicate by a fixed
    vector, so std of the offset vectors stays small."""
    ta, tb = a.get("team"), b.get("team")
    if ta is not None and tb is not None and ta != tb:
        return False
    lo = max(float(a["start"]), float(b["start"]))
    hi = min(float(a["end"]), float(b["end"]))
    if hi - lo < min_overlap_s:
        return False
    offs = []
    t = lo
    while t <= hi + 1e-6:
        pa, pb = _xy_at(a, t), _xy_at(b, t)
        if pa is not None and pb is not None:
            offs.append((pa[0] - pb[0], pa[1] - pb[1]))
        t += STEP
    if len(offs) < 4:
        return False
    o = np.asarray(offs)
    mean_gap = float(np.hypot(o[:, 0], o[:, 1]).mean())
    off_std = float(np.sqrt(o[:, 0].var() + o[:, 1].var()))
    return bool(mean_gap <= max_gap_m
                and off_std <= max_offset_std_m)


def _merge_stats(xy: list) -> tuple[float, int]:
    """dist_m + sprint count from a merged xy track (fuse_tracks logic)."""
    dist = 0.0
    sprints = 0
    for i in range(1, len(xy)):
        p, q = xy[i - 1], xy[i]
        if p is None or q is None or p[0] is None or q[0] is None:
            continue
        d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        dist += d
        if d / STEP >= SPRINT_MS:
            sprints += 1
    return round(dist, 1), sprints


def merge_duplicates(tracks: list[dict], *, max_gap_m: float = 8.0,
                     min_overlap_s: float = 1.0,
                     max_offset_std_m: float = 1.5) -> list[dict]:
    """Union-find merge of cross-camera duplicate tracks (greedy sweep
    by start time, only interval-overlapping pairs compared). Each
    super-track: id = longest member's id, member_ids, majority team,
    per-step mean xy over the union interval, dist/sprints recomputed,
    crops = members' crops longest-first."""
    ordered = sorted(tracks, key=lambda t: float(t["start"]))
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if float(b["start"]) > float(a["end"]):
                break
            if find(i) != find(j) and _is_dup(
                    a, b, max_gap_m=max_gap_m, min_overlap_s=min_overlap_s,
                    max_offset_std_m=max_offset_std_m):
                parent[find(j)] = find(i)

    comps: dict[int, list[dict]] = {}
    for i, t in enumerate(ordered):
        comps.setdefault(find(i), []).append(t)

    out = []
    for members in comps.values():
        longest = max(members,
                      key=lambda t: float(t["end"]) - float(t["start"]))
        teams = [t.get("team") for t in members if t.get("team")]
        team = (max(set(teams), key=teams.count) if teams else None)
        start = min(float(t["start"]) for t in members)
        end = max(float(t["end"]) for t in members)
        xy = []
        t = start
        while t <= end + 1e-6:
            pts = [p for m in members if (p := _xy_at(m, t)) is not None]
            xy.append([round(float(np.mean([p[0] for p in pts])), 2),
                       round(float(np.mean([p[1] for p in pts])), 2)]
                      if pts else [None, None])
            t += STEP
        dist, sprints = _merge_stats(xy)
        crops = [c for t2 in sorted(
            members, key=lambda t: -(float(t["end"]) - float(t["start"])))
            for c in t2.get("crops") or []]
        out.append({"id": int(longest["id"]),
                    "member_ids": sorted(int(t["id"]) for t in members),
                    "team": team, "start": round(start, 3),
                    "end": round(end, 3), "xy": xy,
                    "dist_m": dist, "sprints": sprints,
                    "crops": crops})
    return out


def build_groups_v2(v2_dir: Path, *, dist_thresh: float = DIST_THRESH,
                    **merge_kw) -> dict:
    """Cluster tracks.json -> groups.json (one card per player).
    merge_kw is forwarded to merge_duplicates."""
    v2_dir = Path(v2_dir)
    doc = json.loads((v2_dir / "tracks.json").read_text())
    tracks = doc.get("tracks") or []
    merged = merge_duplicates(tracks, **merge_kw)
    tracklets = [dict(t, t_start=float(t["start"]), t_end=float(t["end"]))
                 for t in merged]
    crops_dir = v2_dir / "crops"
    feats = np.stack([
        tracklet_fingerprint(
            [crops_dir / c for c in (t.get("crops") or [])])
        for t in tracklets]) if tracklets else np.zeros((0, FEAT_DIM))
    clusters = group_tracklets(tracklets, feats, max_per_team=MAX_PER_TEAM,
                               dist_thresh=dist_thresh)
    by_id = {int(t["id"]): t for t in merged}
    groups = []
    for c in clusters:
        members = sorted(
            (by_id[i] for i in c["tracklet_ids"]),
            key=lambda t: -(float(t["end"]) - float(t["start"])))
        crops: list[str] = []
        for t in members:
            for cp in t.get("crops") or []:
                if len(crops) < 8:
                    crops.append(cp)
        groups.append({
            "team": c["team"],
            # original track ids so roster naming/hiding still works
            "track_ids": sorted(i for t in members
                                for i in t["member_ids"]),
            "n_members": sum(len(t["member_ids"]) for t in members),
            "minutes": round(sum(float(t["end"]) - float(t["start"])
                                 for t in members) / 60.0, 2),
            "dist_m": round(sum(float(t.get("dist_m") or 0.0)
                                for t in members), 1),
            "sprints": sum(int(t.get("sprints") or 0) for t in members),
            "crops": crops,
            "start": min(float(t["start"]) for t in members),
            "end": max(float(t["end"]) for t in members),
        })
    groups.sort(key=lambda g: (TEAM_ORDER.get(g["team"], 3),
                               -g["minutes"]))
    for i, g in enumerate(groups):
        g["id"] = f"g{i + 1}"
    out = {
        "groups": groups,
        "n_tracks": len(tracks),
        "n_merged": len(merged),
        "n_grouped": sum(len(g["track_ids"]) for g in groups),
        "generated_at": time.time(),
    }
    print(f"groups_v2: {out['n_tracks']} tracks -> {out['n_merged']} "
          f"merged -> {len(groups)} groups")
    write_json_atomic(v2_dir / "groups.json", out, indent=1)
    return out
