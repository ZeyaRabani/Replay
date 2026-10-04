"""User-placed player anchors: named clicks on camera stills.

The user clicks every visible player at three moments (start / middle /
end of the fused window), giving team + player name. Each click is
projected through the camera homography and resolved to the nearest
same-team v2 track active at that moment. Resolved anchors become hard
constraints for link_identities (same name = same player, different
names = never the same player).

anchors.json lives in analysis/players_v2/:

    {"moments": [{"id": "start"|"mid"|"end", "t": <shared s>}],
     "clicks":  [{"id", "moment", "angle", "fx", "fy", "team", "label",
                  "track_id", "dist_m", "xy", "note"}],
     "updated_at": <epoch>}
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from highlights.io import write_json_atomic

from .calib import apply_h
from .fuse_tracks import STEP

MAX_DIST_M = 3.0          # a click further than this from any track stays unresolved


def _load(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def path(v2_dir: Path) -> Path:
    return Path(v2_dir) / "anchors.json"


def load(v2_dir: Path) -> dict | None:
    return _load(path(v2_dir))


def save(v2_dir: Path, doc: dict) -> None:
    write_json_atomic(path(v2_dir), doc, indent=1)


def window_of(v2_doc: dict) -> tuple[float, float]:
    """(t0, hi) shared seconds from tracks.json (visible_hist length)."""
    t0 = float(v2_doc.get("t0") or 0.0)
    hist = (v2_doc.get("summary") or {}).get("visible_hist") or []
    if hist:
        return t0, t0 + STEP * (len(hist) - 1)
    return t0, max((float(t["end"]) for t in v2_doc.get("tracks") or []),
                   default=t0)


def default_moments(v2_doc: dict) -> list[dict]:
    """start = t0+90, mid = middle, end = hi-90 (clamped inside)."""
    t0, hi = window_of(v2_doc)
    return [{"id": "start", "t": round(min(t0 + 90.0, hi), 1)},
            {"id": "mid", "t": round((t0 + hi) / 2.0, 1)},
            {"id": "end", "t": round(max(hi - 90.0, t0), 1)}]


def _track_xy_at(tr: dict, t: float) -> tuple[float, float] | None:
    """Track position at shared t, or None when inactive/unobserved."""
    if not (float(tr["start"]) <= t <= float(tr["end"])):
        return None
    xy = tr.get("xy") or []
    i = round((t - float(tr["start"])) / STEP)
    if not (0 <= i < len(xy)):
        return None
    p = xy[i]
    if p is None or p[0] is None:
        return None
    return float(p[0]), float(p[1])


def resolve_clicks(doc: dict, tracks_doc: dict, calib: dict) -> dict:
    """Project each click to the pitch and attach it to the nearest
    same-team track active at its moment. Per-moment greedy global
    assignment so two clicks never resolve to the same track; clicks
    further than MAX_DIST_M stay unresolved with a note. Returns a new
    doc (moments kept, clicks annotated)."""
    doc = dict(doc)
    clicks = [dict(c) for c in doc.get("clicks") or []]
    tracks = tracks_doc.get("tracks") or []
    moments = {str(m.get("id")): float(m["t"])
               for m in doc.get("moments") or []}
    angles = (calib or {}).get("angles") or {}
    for c in clicks:
        c.update(track_id=None, dist_m=None, xy=None, note=None)

    # duplicate (moment, team, label): only the first counts
    seen: set[tuple] = set()
    for c in clicks:
        key = (c.get("moment"), c.get("team"),
               str(c.get("label") or "").strip().lower())
        if key in seen:
            c["note"] = "duplicate name"
        seen.add(key)

    live = [c for c in clicks if c["note"] is None]
    for c in live:
        H = (angles.get(str(c.get("angle"))) or {}).get("H")
        if not H:
            c["note"] = "camera not calibrated"
            continue
        c["xy"] = list(apply_h(H, float(c.get("fx") or 0.0),
                               float(c.get("fy") or 0.0)))
    live = [c for c in live if c["note"] is None]

    for mid in {c.get("moment") for c in live}:
        group = [c for c in live if c.get("moment") == mid]
        t = moments.get(mid)
        if t is None:
            continue
        pairs: list[tuple[float, int, int]] = []
        for ci, c in enumerate(group):
            xy = c.get("xy")
            if xy is None or math.isinf(xy[0]):
                continue
            for tr in tracks:
                if tr.get("team") not in (c.get("team"), None):
                    continue
                p = _track_xy_at(tr, t)
                if p is None:
                    continue
                d = math.hypot(p[0] - xy[0], p[1] - xy[1])
                pairs.append((d, ci, int(tr["id"])))
        pairs.sort(key=lambda x: (x[0], x[1], x[2]))
        nearest: dict[int, float] = {}
        used_c: set[int] = set()
        used_t: set[int] = set()
        for d, ci, tid in pairs:
            nearest.setdefault(ci, d)
            if ci in used_c or tid in used_t or d > MAX_DIST_M:
                continue
            used_c.add(ci)
            used_t.add(tid)
            group[ci]["track_id"] = tid
            group[ci]["dist_m"] = round(d, 2)
        for ci, c in enumerate(group):
            if c["track_id"] is None and c["note"] is None:
                d = nearest.get(ci)
                c["note"] = (
                    f"no {c.get('team')} player within {MAX_DIST_M:.0f} m"
                    + (f" (nearest {d:.1f} m)" if d is not None else ""))

    # conflicting labels on one track void both clicks
    by_track: dict[int, set[str]] = {}
    for c in clicks:
        if c.get("track_id") is not None:
            by_track.setdefault(int(c["track_id"]), set()).add(
                str(c.get("label") or "").strip().lower())
    for c in clicks:
        tid = c.get("track_id")
        if tid is not None and len(by_track.get(int(tid)) or set()) > 1:
            c["track_id"] = None
            c["dist_m"] = None
            c["note"] = "conflicting names"

    doc["clicks"] = clicks
    doc["updated_at"] = time.time()
    return doc


def constraints(doc: dict) -> dict[int, tuple[str, str]]:
    """track_id -> (team, label) for resolved clicks; a track carrying
    two different labels is dropped entirely."""
    by_track: dict[int, set[str]] = {}
    raw: dict[int, str] = {}
    team_of: dict[int, str] = {}
    for c in doc.get("clicks") or []:
        if c.get("track_id") is None:
            continue
        tid = int(c["track_id"])
        by_track.setdefault(tid, set()).add(
            str(c.get("label") or "").strip().lower())
        raw.setdefault(tid, str(c.get("label") or "").strip())
        team_of[tid] = str(c["team"])
    return {tid: (team_of[tid], raw[tid])
            for tid, labs in by_track.items() if len(labs) == 1}
