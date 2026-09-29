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

from .groups import FEAT_DIM, MAX_PER_TEAM, group_tracklets, tracklet_fingerprint

TEAM_ORDER = {"A": 0, "B": 1, None: 2}


def build_groups_v2(v2_dir: Path) -> dict:
    """Cluster tracks.json -> groups.json (one card per player)."""
    v2_dir = Path(v2_dir)
    doc = json.loads((v2_dir / "tracks.json").read_text())
    tracks = doc.get("tracks") or []
    tracklets = [dict(t, t_start=float(t["start"]), t_end=float(t["end"]))
                 for t in tracks]
    crops_dir = v2_dir / "crops"
    feats = np.stack([
        tracklet_fingerprint(
            [crops_dir / c for c in (t.get("crops") or [])])
        for t in tracklets]) if tracklets else np.zeros((0, FEAT_DIM))
    clusters = group_tracklets(tracklets, feats, max_per_team=MAX_PER_TEAM)
    by_id = {int(t["id"]): t for t in tracks}
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
            "track_ids": c["tracklet_ids"],
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
        "n_grouped": sum(len(g["track_ids"]) for g in groups),
        "generated_at": time.time(),
    }
    write_json_atomic(v2_dir / "groups.json", out, indent=1)
    return out
