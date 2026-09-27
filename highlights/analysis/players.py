"""Per-player pass: ByteTrack tracklets, team votes, crops + roster/stats.

Decodes the reference-angle video (480p proxy when present) at 2 fps,
tracks persons with YOLO + ByteTrack, votes each tracklet's team from
torso colour (centroids from analysis/teams.json), and keeps three
review crops per kept tracklet. Roster/scorer bookkeeping and the
per-player stat aggregation are pure functions here — the web API in
app/backend/players_api.py only does IO.

    python -m highlights.analysis.players_run --project-dir P [--force]
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np

from highlights.io import write_json_atomic
from highlights.multiangle.trackfeat import CONF, FRAME_W, _ensure_model, _frame_reader

from .stats import PITCH_LEN_M
from .teams import descriptor_features, torso_descriptor

FPS = 2.0
MIN_TRACKLET_S = 6.0
SPRINT_MPS = 5.5        # speed threshold for a sprint step (m/s)
SPRINT_MIN_S = 1.0      # minimum duration of a sprint run
MAX_STEP_MPS = 12.0     # faster steps are ID-switch noise — dropped
GAP_S = 3.0             # finalize a track unseen for longer than this
CROP_MAX_H = 160
CROP_PAD = 0.15

TEAM_MAX_DIST = 0.35    # feature-space distance cap for a team vote
TEAM_MIN_RATIO = 0.85   # nearest/other distance ratio above -> ambiguous


def build_tracker(model_path=None, imgsz: int = 960):
    """YOLO + ByteTrack tracker(frame) -> list[(tid:int, xyxy)]."""
    import os

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "0")) or
                          (os.cpu_count() or 1))
    model = YOLO(str(_ensure_model(
        model_path or
        Path(__file__).parent.parent / "multiangle" / "models" / "yolov8n.pt")))

    def tracker(frame):
        res = model.track(frame, imgsz=imgsz, conf=CONF, classes=[0],
                          tracker="bytetrack.yaml", persist=True,
                          verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None or not len(boxes):
            return []
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        return [(int(tid), xyxy[k]) for k, tid in enumerate(ids)]

    return tracker


def assign_team(desc: np.ndarray, teams: dict) -> str | None:
    """Vote 'A'/'B'/None for one torso descriptor against teams.json.

    Centroids are rebuilt in descriptor_features space from each team's
    representative hsv; the nearest wins unless it is too far or the two
    are too close to tell apart."""
    centroids = {}
    for k in ("A", "B"):
        hsv = (teams.get("teams") or {}).get(k, {}).get("hsv")
        if hsv is None:
            continue
        centroids[k] = descriptor_features(np.array(list(hsv) + [0]))
    if not centroids:
        return None
    feat = descriptor_features(desc)
    dists = {k: float(np.linalg.norm(feat - c)) for k, c in centroids.items()}
    near = min(dists, key=dists.get)
    if dists[near] > TEAM_MAX_DIST:
        return None
    if len(dists) > 1:
        other = min(d for k, d in dists.items() if k != near)
        if other > 1e-9 and dists[near] / other > TEAM_MIN_RATIO:
            return None
    return near


def _crop(frame: np.ndarray, box) -> np.ndarray | None:
    """Box padded by CROP_PAD, clipped, resized to height <= CROP_MAX_H."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    px, py = CROP_PAD * bw, CROP_PAD * bh
    cx1 = int(max(0, x1 - px))
    cy1 = int(max(0, y1 - py))
    cx2 = int(min(w, x2 + px))
    cy2 = int(min(h, y2 + py))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    ch = crop.shape[0]
    if ch > CROP_MAX_H:
        crop = cv2.resize(crop, (round(crop.shape[1] * CROP_MAX_H / ch),
                                 CROP_MAX_H), interpolation=cv2.INTER_AREA)
    return crop


def _track_stats(samples: list[list[float]], len_m: float) -> tuple[float, int]:
    """Distance (m) and sprint count over [[t, x_norm, y_norm], ...]."""
    dist = 0.0
    sprints = 0
    run_s = 0.0
    for (t0, x0, y0), (t1, x1, y1) in itertools.pairwise(samples):
        dt = t1 - t0
        if dt <= 0:
            continue
        if dt > GAP_S:
            run_s = 0.0
            continue
        step = math.hypot((x1 - x0) * len_m, (y1 - y0) * len_m * 0.6)
        speed = step / dt
        if speed > MAX_STEP_MPS:
            run_s = 0.0
            continue
        dist += step
        if speed > SPRINT_MPS:
            run_s += dt
        else:
            if run_s >= SPRINT_MIN_S:
                sprints += 1
            run_s = 0.0
    if run_s >= SPRINT_MIN_S:
        sprints += 1
    return dist, sprints


def _finish_track(tid: int, tr: dict, crops_dir: Path, len_m: float,
                  tracklets: list[dict]) -> None:
    t_end = tr["last_t"]
    if t_end - tr["t_start"] < MIN_TRACKLET_S:
        return
    votes = tr["votes"]
    team = None
    if votes["A"] > 0 or votes["B"] > 0:
        team = "A" if votes["A"] >= votes["B"] else "B"
    dist, sprints = _track_stats(tr["samples"], len_m)
    crops = []
    import cv2
    for i, key in enumerate(("first", "best", "last")):
        img = tr["crops"].get(key)
        if img is None:
            continue
        name = f"{tid}_{i}.jpg"
        crops_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crops_dir / name), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        crops.append(name)
    tracklets.append({
        "id": int(tid),
        "team": team,
        "t_start": round(tr["t_start"], 3),
        "t_end": round(t_end, 3),
        "n_samples": len(tr["samples"]),
        "distance_m": round(dist, 1),
        "sprints": sprints,
        "crops": crops,
        "path": [[round(s[0], 3), round(s[1], 3), round(s[2], 3)]
                 for s in tr["samples"]],
    })


def run_players_pass(video, out_dir: Path, *,
                     window_file: tuple[float, float],
                     proxy_offset: float = 0.0,
                     shared_offset: float = 0.0,
                     teams: dict,
                     pitch_type=None,
                     fps: float = FPS,
                     log=print,
                     status=None,
                     tracker=None,
                     frames=None) -> dict:
    """Track persons over window_file (ref-angle file seconds); write
    crops + tracklets.json into out_dir (times on the shared-T axis)."""
    if tracker is None:
        tracker = build_tracker()
    if frames is None:
        frames = _frame_reader(str(video), fps, FRAME_W,
                               max(0.0, window_file[0] - proxy_offset),
                               window_file[1] - proxy_offset,
                               t_base=window_file[0])
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    try:
        len_m = float(PITCH_LEN_M.get(int(pitch_type), 100))
    except (TypeError, ValueError):
        len_m = 100.0

    live: dict[int, dict] = {}
    tracklets: list[dict] = []
    n_frames = 0
    window_s = max(1e-6, window_file[1] - window_file[0])
    for t_file, frame in frames:
        t_shared = float(t_file) + shared_offset
        seen = set()
        h, w = frame.shape[:2]
        for tid, box in tracker(frame):
            tid = int(tid)
            seen.add(tid)
            fx = float((box[0] + box[2]) / 2 / w)
            fy = float(box[3] / h)
            desc = torso_descriptor(frame, box)
            vote = assign_team(desc, teams) if desc is not None else None
            img = _crop(frame, box)
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            tr = live.get(tid)
            if tr is None:
                tr = live[tid] = {
                    "t_start": t_shared, "last_t": t_shared,
                    "samples": [], "votes": {"A": 0, "B": 0, None: 0},
                    "crops": {}, "best_area": -1.0,
                }
                if img is not None:
                    tr["crops"]["first"] = img
            tr["last_t"] = t_shared
            tr["samples"].append([t_shared, fx, fy])
            tr["votes"][vote] = tr["votes"].get(vote, 0) + 1
            if img is not None:
                tr["crops"]["last"] = img
                if area > tr["best_area"]:
                    tr["best_area"] = area
                    tr["crops"]["best"] = img
        # finalize tracks not seen for > GAP_S of video time
        for tid in [k for k, tr in live.items()
                    if k not in seen and t_shared - tr["last_t"] > GAP_S]:
            _finish_track(tid, live.pop(tid), crops_dir, len_m, tracklets)
        n_frames += 1
        if n_frames % 60 == 0:
            log(f"players @{t_file:.0f}s")
        if status is not None and n_frames % 15 == 0:
            prog = 0.05 + 0.85 * (float(t_file) - window_file[0]) / window_s
            status.update(progress=min(0.9, max(0.05, prog)),
                          message=f"players @{float(t_file):.0f}s")
    for tid, tr in list(live.items()):
        _finish_track(tid, tr, crops_dir, len_m, tracklets)
    tracklets.sort(key=lambda tr: tr["t_start"])

    doc = {
        "fps": fps,
        "window_shared": [round(window_file[0] + shared_offset, 3),
                          round(window_file[1] + shared_offset, 3)],
        "shared_offset": shared_offset,
        "pitch_len_m": len_m,
        "n_frames": n_frames,
        "generated_at": round(time.time(), 1),
        "tracklets": tracklets,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "tracklets.json", doc, indent=0)
    log(f"players: {len(tracklets)} tracklets over {n_frames} frames "
        f"-> {out_dir / 'tracklets.json'}")
    return doc


def strip_for_api(doc: dict) -> dict:
    """tracklets.json minus the heavy per-sample `path` arrays."""
    return {**doc, "tracklets": [
        {k: v for k, v in tr.items() if k != "path"} |
        {"duration_s": round(tr["t_end"] - tr["t_start"], 3)}
        for tr in doc.get("tracklets") or []]}


# ---------- roster + per-player stats ----------


def default_roster() -> dict:
    return {"players": [], "scorers": {}}


def validate_roster(roster, tracklet_ids: set[int],
                    candidate_ids: set[str]) -> dict:
    """Validate + normalize a roster; raises ValueError with a message."""
    if not isinstance(roster, dict):
        raise ValueError("roster must be an object")
    players = roster.get("players")
    if not isinstance(players, list):
        raise ValueError("roster.players must be a list")
    seen_ids: set[str] = set()
    tid_owner: dict[int, str] = {}
    out_players = []
    for i, pl in enumerate(players):
        if not isinstance(pl, dict):
            raise ValueError(f"players[{i}] must be an object")
        pid = pl.get("id")
        name = pl.get("name")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"players[{i}] missing string id")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError(f"players[{i}] name must be 1-60 chars")
        if pid in seen_ids:
            raise ValueError(f"duplicate player id {pid!r}")
        seen_ids.add(pid)
        team = pl.get("team")
        if team not in ("A", "B", None):
            raise ValueError(f"players[{i}] team must be A, B or null")
        tids = pl.get("tracklet_ids") or []
        if not isinstance(tids, list):
            raise ValueError(f"players[{i}] tracklet_ids must be a list")
        norm_tids = []
        for tid in tids:
            if not isinstance(tid, int) or isinstance(tid, bool):
                raise ValueError(f"players[{i}] tracklet ids must be ints")
            if tid not in tracklet_ids:
                raise ValueError(f"unknown tracklet id {tid}")
            if tid in tid_owner:
                raise ValueError(
                    f"tracklet {tid} assigned to two players")
            tid_owner[tid] = pid
            norm_tids.append(tid)
        out_players.append({"id": pid, "name": name.strip(), "team": team,
                            "tracklet_ids": norm_tids})
    scorers = roster.get("scorers") or {}
    if not isinstance(scorers, dict):
        raise ValueError("roster.scorers must be an object")
    out_scorers = {}
    for cid, pid in scorers.items():
        if cid not in candidate_ids:
            raise ValueError(f"unknown candidate id {cid!r}")
        if pid not in seen_ids:
            raise ValueError(f"scorer {pid!r} is not a known player id")
        out_scorers[str(cid)] = pid
    return {"players": out_players, "scorers": out_scorers}


def players_stats(roster: dict, tracklets: list[dict],
                  candidates: list[dict], teams) -> dict:
    """Aggregate per-player + per-team stats from the roster."""
    by_tid = {int(t["id"]): t for t in tracklets or []}
    cand_status = {str(c.get("id")): str(c.get("status")) for c in candidates or []}
    goals_by_player: dict[str, int] = {}
    for cid, pid in (roster.get("scorers") or {}).items():
        if cand_status.get(str(cid)) == "confirmed":
            goals_by_player[pid] = goals_by_player.get(pid, 0) + 1
    assigned: set[int] = set()
    team_order = {"A": 0, "B": 1, None: 2}
    out_players = []
    for pl in roster.get("players") or []:
        rows = [by_tid[t] for t in pl.get("tracklet_ids") or [] if t in by_tid]
        assigned.update(t["id"] for t in rows)
        tracked = sum(t["t_end"] - t["t_start"] for t in rows)
        dist = sum(float(t.get("distance_m") or 0.0) for t in rows)
        spr = sum(int(t.get("sprints") or 0) for t in rows)
        out_players.append({
            "id": pl["id"], "name": pl["name"], "team": pl.get("team"),
            "n_tracklets": len(rows),
            "tracked_s": round(tracked, 1),
            "distance_m": round(dist, 0),
            "sprints": spr,
            "goals": goals_by_player.get(pl["id"], 0),
        })
    out_players.sort(key=lambda p: (team_order.get(p["team"], 2),
                                    -p["distance_m"]))
    out_teams = {}
    for k in ("A", "B"):
        members = [p for p in out_players if p["team"] == k]
        out_teams[k] = {
            "n_players": len(members),
            "tracked_s": round(sum(p["tracked_s"] for p in members), 1),
            "distance_m": round(sum(p["distance_m"] for p in members), 0),
            "sprints": sum(p["sprints"] for p in members),
            "goals": sum(p["goals"] for p in members),
        }
    rest = [t for t in tracklets or [] if int(t["id"]) not in assigned]
    unassigned = {
        "n_tracklets": len(rest),
        "tracked_s": round(sum(t["t_end"] - t["t_start"] for t in rest), 1),
        "distance_m": round(sum(float(t.get("distance_m") or 0.0)
                                for t in rest), 0),
        "sprints": sum(int(t.get("sprints") or 0) for t in rest),
    }
    return {
        "players": out_players,
        "teams": out_teams,
        "unassigned": unassigned,
        "caveat": "Estimates from tracked time on the main camera only — "
                  "not official statistics.",
    }
