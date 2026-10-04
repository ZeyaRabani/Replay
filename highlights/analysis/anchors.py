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

import numpy as np

from highlights.io import write_json_atomic

from .calib import apply_h
from .fuse_tracks import STEP, load_dets

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


def dets_at(players_v2: Path, angle: int, file_t: float,
            tol: float = 0.6) -> list[dict]:
    """Detection boxes (normalized 0-1) on `angle` at the det frame
    nearest `file_t` (file seconds); [] when none within tol."""
    p = Path(players_v2) / f"det_a{angle}.npz"
    if not p.is_file():
        return []
    d = load_dets(p)
    if not len(d["t"]):
        return []
    i = int(np.abs(d["t"] - file_t).argmin())
    if abs(float(d["t"][i]) - file_t) > tol:
        return []
    sel = np.flatnonzero(d["t"] == d["t"][i])
    w, h = float(d.get("w") or 1.0), float(d.get("h") or 1.0)
    out = []
    for j in sel:
        x1, y1, x2, y2 = (float(v) for v in d["box"][j])
        tm = d["team"][j]
        out.append({"x1": x1 / w, "y1": y1 / h, "x2": x2 / w,
                    "y2": y2 / h,
                    "team": str(tm) if tm in ("A", "B") else ""})
    return out


def _pick_box(boxes: list[dict], fx: float, fy: float) -> dict | None:
    """Box containing (fx,fy), else the nearest box whose centre lies
    within one box-height of the click."""
    best, best_d = None, None
    for b in boxes:
        if b["x1"] <= fx <= b["x2"] and b["y1"] <= fy <= b["y2"]:
            return b
        h = max(1e-6, b["y2"] - b["y1"])
        d = math.hypot(fx - (b["x1"] + b["x2"]) / 2.0,
                       fy - (b["y1"] + b["y2"]) / 2.0)
        if d <= h and (best_d is None or d < best_d):
            best, best_d = b, d
    return best


def _project(H, stab, ft: float | None, px: float, py: float) -> list:
    """Stab-warp (px,py) onto the calibration frame, then apply_h."""
    if stab is not None and ft is not None:
        from .stabilize import warp_at
        px, py = warp_at(stab, ft, px, py)
    return list(apply_h(H, px, py))


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


def resolve_clicks(doc: dict, tracks_doc: dict, calib: dict,
                   dets_by: dict | None = None, *,
                   stabs: dict | None = None,
                   offsets: list | None = None) -> dict:
    """Project each click to the pitch and attach it to the nearest
    track active at its moment. dets_by: {(angle, moment): [boxes]} — a
    click inside/near a detection projects that box's foot point
    (torsos don't lie on the ground plane); stabs/offsets warp the foot
    from the click's frame onto the calibration frame (shared t minus
    offsets[angle]). The click's team is a hint: same-team (or unteamed)
    tracks are always preferred, other-team tracks are only taken when
    no hint-matching candidate is within MAX_DIST_M. Per-moment greedy
    global assignment so two clicks never resolve to the same track.
    Returns a new doc (moments kept, clicks annotated)."""
    dets_by = dets_by or {}
    stabs = stabs or {}
    doc = dict(doc)
    clicks = [dict(c) for c in doc.get("clicks") or []]
    tracks = tracks_doc.get("tracks") or []
    moments = {str(m.get("id")): float(m["t"])
               for m in doc.get("moments") or []}
    angles = (calib or {}).get("angles") or {}
    for c in clicks:
        c.update(track_id=None, dist_m=None, xy=None, note=None,
                 box=None)

    # duplicate (moment, team, label, angle): only the first counts —
    # clicking the same player in a different camera is valid
    seen: set[tuple] = set()
    for c in clicks:
        key = (c.get("moment"), c.get("team"), c.get("angle"),
               str(c.get("label") or "").strip().lower())
        if key in seen:
            c["note"] = "duplicate name"
        seen.add(key)

    live = [c for c in clicks if c["note"] is None]
    nodet: set[int] = set()
    for c in live:
        c["track_team"] = None
    for c in live:
        H = (angles.get(str(c.get("angle"))) or {}).get("H")
        if not H:
            c["note"] = "camera not calibrated"
            continue
        fx = float(c.get("fx") or 0.0)
        fy = float(c.get("fy") or 0.0)
        a_idx = int(c.get("angle") or 0)
        stab = stabs.get(a_idx)
        ft = None
        if stab is not None and offsets is not None:
            mt = moments.get(str(c.get("moment")))
            if mt is not None:
                ft = mt - (float(offsets[a_idx])
                           if a_idx < len(offsets) else 0.0)
        box = _pick_box(dets_by.get((a_idx,
                                    str(c.get("moment")))) or [],
                        fx, fy)
        if box is not None:
            c["box"] = [box["x1"], box["y1"], box["x2"], box["y2"]]
            c["xy"] = _project(H, stab, ft,
                               (box["x1"] + box["x2"]) / 2.0, box["y2"])
        else:
            nodet.add(id(c))
            c["xy"] = _project(H, stab, ft, fx, fy)
    live = [c for c in live if c["note"] is None]

    for mid in {c.get("moment") for c in live}:
        group = [c for c in live if c.get("moment") == mid]
        t = moments.get(mid)
        if t is None:
            continue
        pairs: list[tuple] = []
        nearest: dict[int, float] = {}
        for ci, c in enumerate(group):
            xy = c.get("xy")
            if xy is None or math.isinf(xy[0]):
                continue
            for tr in tracks:
                p = _track_xy_at(tr, t)
                if p is None:
                    continue
                d = math.hypot(p[0] - xy[0], p[1] - xy[1])
                if d < nearest.get(ci, np.inf):
                    nearest[ci] = d
                if d > MAX_DIST_M:
                    continue
                hint = tr.get("team") in (c.get("team"), None)
                cost = d if hint else d + 1.5
                pairs.append((0 if hint else 1, cost, d, ci,
                              int(tr["id"])))
        pairs.sort()
        team_by_id = {int(tr["id"]): tr.get("team") for tr in tracks}
        used_c: set[int] = set()
        used_t: set[int] = set()
        for _flag, _cost, d, ci, tid in pairs:
            if ci in used_c or tid in used_t:
                continue
            used_c.add(ci)
            used_t.add(tid)
            group[ci]["track_id"] = tid
            group[ci]["dist_m"] = round(d, 2)
            group[ci]["track_team"] = team_by_id.get(tid)
        for ci, c in enumerate(group):
            if c["track_id"] is None and c["note"] is None:
                if id(c) in nodet:
                    c["note"] = "no detection under click"
                else:
                    d = nearest.get(ci)
                    c["note"] = (
                        f"no {c.get('team')} player within "
                        f"{MAX_DIST_M:.0f} m"
                        + (f" (nearest {d:.1f} m)"
                           if d is not None else ""))

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
        team_of[tid] = str(c.get("track_team") or c["team"])
    return {tid: (team_of[tid], raw[tid])
            for tid, labs in by_track.items() if len(labs) == 1}
