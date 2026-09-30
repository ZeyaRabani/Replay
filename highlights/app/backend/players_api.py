"""Per-player analysis endpoints for multi-angle projects.

POST /analyse/players spawns `python -m highlights.analysis.players_run`
as a detached subprocess writing analysis/players/{status.json,log.txt};
GET /analysis/players returns the reconciled status plus tracklets,
roster and per-player stats. Mirrors analysis_api.py's
launch/reconcile pattern.
"""

# NOTE: no `from __future__ import annotations` — the ScopedP Depends
# annotation inside make_router must evaluate eagerly against the factory
# argument, not be resolved later from module globals.
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from highlights.io import write_json_atomic

from . import history, pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]

_procs: dict[int, subprocess.Popen] = {}

CROP_NAME_RE = re.compile(r"^\d+_[0-2]\.jpg$")


class AnalysePlayersPut(BaseModel):
    force: bool = False


class IdentityNamePut(BaseModel):
    name: str | None = None


IDENTITY_ID_RE = re.compile(r"^[AB]\d{1,3}$")


def _players_dir(p) -> Path:
    return p.root / "analysis" / "players"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path)


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    proc = _procs.get(pid)
    if proc is not None:
        return proc.poll() is None
    return pipeline.pid_alive(pid)


def _status_alive(status: dict) -> bool:
    pid = status.get("pid")
    if pid is None:
        last = status.get("updated_at") or status.get("started_at") or 0
        return time.time() - last < 15
    return _pid_alive(pid)


def _runner_cmd() -> list[str]:
    env = os.environ.get("HL_PLAYERS_CMD")
    if env:
        return shlex.split(env)
    return [sys.executable, "-m", "highlights.analysis.players_run"]


def _reconciled_status(p) -> dict | None:
    path = _players_dir(p) / "status.json"
    status = _read_json(path)
    if status and status.get("state") in ("queued", "running") and not _status_alive(status):
        status["state"] = "failed"
        status["error"] = "interrupted"
        status["message"] = "interrupted"
        status["finished_at"] = time.time()
        status["updated_at"] = time.time()
        _write_status(path, status)
    return status


def _lo_out(p) -> float:
    from highlights.analysis.run import resolve_context
    try:
        return float(resolve_context(p.root)["window"][0])
    except Exception:
        return 0.0


def _estimate_min(p) -> int:
    from highlights.analysis.players import FPS
    from highlights.analysis.run import resolve_context
    try:
        ctx = resolve_context(p.root)
        window_s = ctx["window"][1] - ctx["window"][0]
    except Exception:
        window_s = float(p.video.duration_s) if p.video else 0.0
        if window_s <= 0:
            window_s = 600.0
    return max(1, math.ceil(window_s * FPS / 4 / 60))


def _candidate_dicts(p) -> list[dict]:
    return [{"id": str(c.id), "status": str(c.status), "type": str(c.type)}
            for c in p.candidates]


def _payload(p) -> dict:
    """Shared GET response body (status reconciled by the caller)."""
    from highlights.analysis.players import (
        default_roster,
        players_stats,
        strip_for_api,
    )
    pdir = _players_dir(p)
    teams_doc = _read_json(p.root / "analysis" / "teams.json") or {}
    teams = teams_doc.get("teams")
    teams_out = None
    if isinstance(teams, dict):
        teams_out = {k: {"name": (teams.get(k) or {}).get("name"),
                         "hex": (teams.get(k) or {}).get("hex")}
                     for k in ("A", "B") if teams.get(k)}
    doc = _read_json(pdir / "tracklets.json")
    lo = _lo_out(p)
    roster = _read_json(pdir / "roster.json") or default_roster()
    named_tids = {tid for pl in roster.get("players") or []
                  for tid in pl.get("tracklet_ids") or []}
    hidden_tids = set(roster.get("hidden_tracklet_ids") or [])
    groups_doc = _read_json(pdir / "groups.json")
    if groups_doc is None and doc and doc.get("tracklets"):
        try:
            from highlights.analysis.groups import build_groups
            groups_doc = build_groups(pdir)
        except Exception:
            groups_doc = None
    groups_out = None
    if groups_doc:
        groups_out = [g for g in groups_doc.get("groups") or []
                      if not hidden_tids.intersection(g["tracklet_ids"])]
    all_tracklets = []
    if doc:
        for tr in strip_for_api(doc)["tracklets"]:
            all_tracklets.append({
                **tr,
                "t_start_out": round(tr["t_start"] - lo, 3),
                "t_end_out": round(tr["t_end"] - lo, 3),
            })
    # keep the naming UI usable: the >=30 s tracks, longest first, at
    # most 40 per team; roster-assigned tracklets are always included
    shown: list[dict] = []
    per_team: dict[str, int] = {}
    for tr in sorted(all_tracklets,
                     key=lambda t: -t.get("duration_s", 0.0)):
        team = tr.get("team") or "-"
        if tr["id"] in named_tids or (
                tr.get("duration_s", 0.0) >= 30.0
                and per_team.get(team, 0) < 40):
            shown.append(tr)
            per_team[team] = per_team.get(team, 0) + 1
    stats = players_stats(roster, doc.get("tracklets") if doc else [],
                          _candidate_dicts(p), teams_doc)
    return {
        "teams": teams_out,
        "tracklets": shown,
        "n_tracklets_total": len(all_tracklets),
        "n_shown": len(shown),
        "roster": roster,
        "players_stats": stats,
        "groups": groups_out,
        "estimate_min": _estimate_min(p),
    }


def _ball_path(features: dict, shared_offset: float) -> list[list[float]]:
    """Ref-angle ball positions at 1 fps on the shared timeline.

    features is a track features_1s.json doc: {columns: [...], rows: [[...]]}.
    Keeps rows with ball_conf >= 0.35 and a positive ball position."""
    ci = {k: i for i, k in enumerate(features.get("columns") or [])}
    if not {"ball_conf", "ball_x", "ball_y"} <= set(ci):
        return []
    out = []
    for i, row in enumerate(features.get("rows") or []):
        bc = row[ci["ball_conf"]]
        bx = row[ci["ball_x"]]
        by = row[ci["ball_y"]]
        if bc >= 0.35 and bx > 0 and by > 0:
            out.append([round(i + shared_offset, 3),
                        round(float(bx), 3), round(float(by), 3)])
    return out


def make_router(ScopedP, PublicP) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}")

    @router.post("/analyse/players")
    def post_analyse_players(p: ScopedP, body: AnalysePlayersPut | None = None) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        if p.pipeline_state != "done":
            raise HTTPException(409, "pipeline is not done")
        if not (p.root / "analysis" / "teams.json").is_file():
            raise HTTPException(409, "run Team analysis first")
        pdir = _players_dir(p)
        status_path = pdir / "status.json"
        status = _read_json(status_path)
        if status and status.get("state") in ("queued", "running") and _status_alive(status):
            raise HTTPException(409, "player analysis already running")

        force = bool(body and body.force)
        now = time.time()
        status = {
            "state": "queued",
            "stage": "players",
            "progress": 0.0,
            "stage_progress": 0.0,
            "message": "queued",
            "error": None,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "pid": None,
        }
        _write_status(status_path, status)

        argv = _runner_cmd() + ["--project-dir", str(p.root)]
        if force:
            argv.append("--force")
        pdir.mkdir(parents=True, exist_ok=True)
        log = open(pdir / "log.txt", "ab")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                argv, cwd=REPO_ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        finally:
            log.close()
        _procs[proc.pid] = proc

        cur = _read_json(status_path)
        if cur and cur.get("pid") is None:
            cur["pid"] = proc.pid
            _write_status(status_path, cur)
            status = cur
        elif cur:
            status = cur
        return status

    def _v2_dir(p) -> Path:
        return p.root / "analysis" / "players_v2"

    @router.post("/analysis/players/v2/run")
    def post_players_v2(p: ScopedP, body: AnalysePlayersPut | None = None
                        ) -> dict:
        """Spawn the multi-view hi-res pass (players_run --v2): detect
        per angle then fuse into pitch-space tracks."""
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        if p.pipeline_state != "done":
            raise HTTPException(409, "pipeline is not done")
        if not (p.root / "analysis" / "teams.json").is_file():
            raise HTTPException(409, "run Team analysis first")
        calib = _read_json(p.multiangle_dir / "calib.json") or {}
        if not calib.get("angles"):
            raise HTTPException(409, "set pitch calibration landmarks "
                                     "first (/analysis/calib)")
        status_path = _players_dir(p) / "status.json"
        status = _read_json(status_path)
        if status and status.get("state") in ("queued", "running") \
                and _status_alive(status):
            raise HTTPException(409, "player analysis already running")
        now = time.time()
        status = {
            "state": "queued", "stage": "players v2",
            "progress": 0.0, "stage_progress": 0.0,
            "message": "queued", "error": None,
            "started_at": now, "updated_at": now,
            "finished_at": None, "pid": None,
        }
        _write_status(status_path, status)
        argv = _runner_cmd() + ["--project-dir", str(p.root), "--v2"]
        if body and body.force:
            argv.append("--force")
        _players_dir(p).mkdir(parents=True, exist_ok=True)
        log = open(_players_dir(p) / "log.txt", "ab")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                argv, cwd=REPO_ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        finally:
            log.close()
        _procs[proc.pid] = proc
        status["pid"] = proc.pid
        _write_status(status_path, status)
        return status

    @router.get("/analysis/players/v2/tracks")
    def get_players_v2(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        doc = _read_json(_v2_dir(p) / "tracks.json")
        if not doc:
            raise HTTPException(404, "players v2 has not run yet")
        roster = _read_json(_players_dir(p) / "roster.json") or {}
        owner = {tid: pl["id"] for pl in roster.get("players") or []
                 for tid in pl.get("tracklet_ids") or []}
        hidden = set(roster.get("hidden_tracklet_ids") or [])
        def crop_url(c: str) -> str:
            return (f"/api/projects/{p.id}/analysis/players/"
                    f"v2/crops/{c}")
        tracks = [{
            "id": t["id"], "team": t.get("team"),
            "start": t["start"], "end": t["end"],
            "dist_m": t.get("dist_m"), "sprints": t.get("sprints"),
            "player_id": owner.get(t["id"]),
            "hidden": t["id"] in hidden,
            "crops": [crop_url(c) for c in t.get("crops") or []],
        } for t in doc.get("tracks") or []]
        s = doc.get("summary") or {}
        gdoc = _read_json(_v2_dir(p) / "groups.json") or {}
        groups = [{
            "id": g["id"], "team": g.get("team"),
            "track_ids": g.get("track_ids") or [],
            "minutes": g.get("minutes"), "dist_m": g.get("dist_m"),
            "sprints": g.get("sprints"),
            "start": g.get("start"), "end": g.get("end"),
            "player_id": next((owner[tid]
                               for tid in g.get("track_ids") or []
                               if tid in owner), None),
            "hidden": bool(g.get("track_ids")) and all(
                tid in hidden for tid in g["track_ids"]),
            "crops": [crop_url(c) for c in g.get("crops") or []],
        } for g in gdoc.get("groups") or []]
        return {"tracks": tracks, "groups": groups,
                "n_merged": gdoc.get("n_merged"),
                "summary": {"n_tracks": s.get("n_tracks"),
                            "median_visible": s.get("median_visible"),
                            "mean_len_s": s.get("mean_len_s")}}

    @router.get("/analysis/players/v2/crops/{name}")
    def get_v2_crop(p: PublicP, name: str) -> FileResponse:
        if not re.match(r"^v2_\d+_\d+\.jpg$", name):
            raise HTTPException(404, "not found")
        path = _v2_dir(p) / "crops" / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="image/jpeg")

    def _identities_payload(p, doc: dict) -> dict:
        def crop_url(c: str) -> str:
            return (f"/api/projects/{p.id}/analysis/players/"
                    f"v2/crops/{c}")
        idents = [{**{k: v for k, v in i.items() if k != "crops"},
                   "crops": [crop_url(c) for c in i.get("crops") or []]}
                  for i in doc.get("identities") or []]
        teams = (_read_json(p.root / "analysis" / "teams.json") or {}
                 ).get("teams") or {}
        return {"identities": idents,
                "unassigned_track_ids": doc.get("unassigned_track_ids") or [],
                "quality": doc.get("quality") or {},
                "window": doc.get("window"),
                "generated_at": doc.get("generated_at"),
                "teams": {k: {"name": v.get("name"), "hex": v.get("hex")}
                          for k, v in teams.items()}}

    @router.get("/players/identities")
    def get_identities(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        doc = _read_json(_v2_dir(p) / "identities.json")
        if not doc:
            raise HTTPException(404, "player identities have not been "
                                     "linked yet")
        return _identities_payload(p, doc)

    @router.post("/players/identities/rebuild")
    def post_identities_rebuild(p: ScopedP) -> dict:
        """Re-link identities offline from the saved tracks.json."""
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        v2d = _v2_dir(p)
        if not (v2d / "tracks.json").is_file():
            raise HTTPException(409, "players v2 has not run yet")
        from highlights.analysis.identity import build_identities
        try:
            doc = build_identities(v2d, log=lambda _m: None)
        except Exception as e:
            raise HTTPException(500, f"build_identities failed: {e}") from e
        return _identities_payload(p, doc)

    @router.put("/players/identities/{iid}")
    def put_identity_name(p: ScopedP, iid: str,
                          body: IdentityNamePut) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        if not IDENTITY_ID_RE.match(iid):
            raise HTTPException(404, "unknown identity")
        name = (body.name or "").strip()
        if len(name) > 60:
            raise HTTPException(422, "name too long (max 60)")
        from highlights.analysis.identity import set_name
        try:
            ident = set_name(_v2_dir(p), iid, name or None)
        except FileNotFoundError as e:
            raise HTTPException(404, "player identities have not been "
                                     "linked yet") from e
        except KeyError as e:
            raise HTTPException(404, "unknown identity") from e
        return {"id": ident["id"], "name": ident["name"]}

    @router.get("/analysis/players")
    def get_players_analysis(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        status = _reconciled_status(p)
        if (status and status.get("state") == "done"
                and not status.get("history_logged")):
            try:
                doc = _read_json(_players_dir(p) / "tracklets.json") or {}
                history.log(p.id, "players_analysed",
                            n_tracklets=len(doc.get("tracklets") or []))
            except Exception:
                pass
            status["history_logged"] = True
            _write_status(_players_dir(p) / "status.json", status)
        return {"status": status, **_payload(p)}

    @router.put("/analysis/players/roster")
    def put_roster(p: ScopedP, body: dict) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        from highlights.analysis.players import players_stats, validate_roster
        pdir = _players_dir(p)
        doc = _read_json(pdir / "tracklets.json") or {}
        tracklet_ids = {int(t["id"]) for t in doc.get("tracklets") or []}
        v2 = _read_json(_v2_dir(p) / "tracks.json") or {}
        tracklet_ids |= {int(t["id"]) for t in v2.get("tracks") or []}
        candidate_ids = {str(c.id) for c in p.candidates}
        try:
            roster = validate_roster(body, tracklet_ids, candidate_ids)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        pdir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(pdir / "roster.json", roster, indent=1)
        teams_doc = _read_json(p.root / "analysis" / "teams.json") or {}
        return {"roster": roster,
                "players_stats": players_stats(
                    roster, doc.get("tracklets") or [],
                    _candidate_dicts(p), teams_doc)}

    @router.post("/analysis/players/groups/rebuild")
    def post_groups_rebuild(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        pdir = _players_dir(p)
        if not (pdir / "tracklets.json").is_file():
            raise HTTPException(409, "player analysis has not run yet")
        try:
            from highlights.analysis.groups import build_groups
            out = build_groups(pdir)
        except Exception as e:
            raise HTTPException(500, f"build_groups failed: {e}") from e
        return {"groups": out["groups"], "n_tracklets": out["n_tracklets"],
                "n_grouped": out["n_grouped"]}

    @router.post("/analysis/players/v2/groups/rebuild")
    def post_v2_groups_rebuild(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        v2d = _v2_dir(p)
        if not (v2d / "tracks.json").is_file():
            raise HTTPException(409, "players v2 has not run yet")
        try:
            from highlights.analysis.groups_v2 import build_groups_v2
            out = build_groups_v2(v2d)
        except Exception as e:
            raise HTTPException(500, f"build_groups_v2 failed: {e}") from e
        return out

    @router.get("/analysis/players/paths")
    def get_player_paths(p: ScopedP) -> dict:
        """Radar replay data: tracklet positions downsampled to 1/s on
        the shared timeline + the ref-angle ball track."""
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        pdir = _players_dir(p)
        roster = _read_json(pdir / "roster.json") or {}
        owner = {tid: pl["id"] for pl in roster.get("players") or []
                 for tid in pl.get("tracklet_ids") or []}
        hidden = set(roster.get("hidden_tracklet_ids") or [])

        # v2 pitch-space tracks take precedence when present
        v2 = _read_json(_v2_dir(p) / "tracks.json")
        if v2 and v2.get("tracks"):
            idoc = _read_json(_v2_dir(p) / "identities.json") or {}
            ident_of = {tid: i["id"] for i in idoc.get("identities") or []
                        for tid in i.get("track_ids") or []}
            step = float(v2.get("step") or 0.5)
            t0 = float(v2.get("t0") or 0.0)
            tracks = []
            for t in v2["tracks"]:
                xys = t.get("xy") or []
                start = float(t["start"])
                pts = [[round(start + i * step, 3), p0, p1]
                       for i, (p0, p1) in enumerate(xys)
                       if p0 is not None]
                tracks.append({
                    "id": int(t["id"]), "team": t.get("team"),
                    "player_id": owner.get(t["id"]),
                    "identity_id": ident_of.get(int(t["id"])),
                    "hidden": t["id"] in hidden,
                    "pts": pts})
            cal = _read_json(p.multiangle_dir / "calib.json") or {}
            pitch = cal.get("pitch") or {}
            return {
                "space": "pitch", "t0": t0, "step": step,
                "pitch": {"len_m": pitch.get("len_m", 100.0),
                          "wid_m": pitch.get("wid_m", 64.0)},
                "visible_hist": (v2.get("summary") or {})
                .get("visible_hist") or [],
                "ball": v2.get("ball") or [],
                "tracks": tracks,
                "ref_angle": 0, "frame_t": None,
                "window_shared": [t0, t0 + step * len(
                    (v2.get("summary") or {}).get("visible_hist") or [0])],
                "fps": 1, "pitch_len_m": pitch.get("len_m"),
            }

        doc = _read_json(pdir / "tracklets.json")
        if not doc or not doc.get("tracklets"):
            raise HTTPException(404, "player analysis has not run yet")
        tracks = []
        for tr in doc["tracklets"]:
            tid = int(tr["id"])
            pts = []
            last_s = None
            for t, fx, fy in tr.get("path") or []:
                s = round(t)
                if s == last_s:
                    continue
                last_s = s
                pts.append([round(t, 3), round(fx, 3), round(fy, 3)])
            tracks.append({
                "id": tid, "team": tr.get("team"),
                "player_id": owner.get(tid),
                "hidden": tid in hidden,
                "pts": pts,
            })
        # ball path on the ref angle: features_1s.json, conf >= 0.35
        ball: list[list[float]] = []
        ref = 0
        mid_t = None
        try:
            from highlights.analysis.run import resolve_context
            ctx = resolve_context(p.root)
            ref = int(ctx.get("ref_angle") or 0)
            mid_t = (ctx["window_file"][0] + ctx["window_file"][1]) / 2
            f = (p.root / "angles" / f"a{ref}" / "track"
                 / "features_1s.json")
            feats = _read_json(f) or {}
            off = float(ctx.get("shared_offset") or 0.0)
            ball = _ball_path(feats, off)
        except Exception:
            pass
        return {
            "ref_angle": ref,
            "frame_t": mid_t,
            "window_shared": doc.get("window_shared"),
            "fps": 1,
            "pitch_len_m": doc.get("pitch_len_m"),
            "tracks": tracks,
            "ball": ball,
        }

    @router.get("/analysis/radar/pitch")
    def get_radar_pitch(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        doc = _read_json(p.root / "analysis" / "radar_pitch.json")
        return doc or {"corners": None, "t": None}

    @router.put("/analysis/radar/pitch")
    def put_radar_pitch(p: ScopedP, body: dict) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        corners = body.get("corners")
        if (not isinstance(corners, list) or len(corners) != 4
                or any(not isinstance(pt, list) or len(pt) != 2
                       or not all(isinstance(v, (int, float))
                                and -1.0 <= v <= 2.0 for v in pt)
                       for pt in corners)):
            raise HTTPException(422, "corners must be 4 [fx, fy] points "
                                   "in [-1, 2] (near-left, near-right, "
                                   "far-right, far-left)")
        doc = {"corners": [[round(float(x), 4), round(float(y), 4)]
                           for x, y in corners],
               "t": body.get("t")}
        pdir = p.root / "analysis"
        pdir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(pdir / "radar_pitch.json", doc, indent=1)
        return doc

    def _pitch_dims(p) -> tuple[float, float]:
        """(len_m, wid_m): players pass estimate, pitch_type table, 100x64."""
        len_m = ((_read_json(_players_dir(p) / "tracklets.json") or {})
                 .get("pitch_len_m"))
        if not len_m:
            from highlights.analysis.stats import PITCH_LEN_M
            try:
                pt = int((p.meta or {}).get("pitch_type") or 0)
            except (TypeError, ValueError):
                pt = 0
            len_m = PITCH_LEN_M.get(pt, 100.0)
        len_m = float(len_m or 100.0)
        return len_m, round(len_m * 0.64, 3)

    def _pitch_doc(p) -> dict:
        """Full pitch dict {len_m, wid_m, template, goal_w_m,
        d_radius_m} — stored calib pitch wins, else defaults."""
        len_m, wid_m = _pitch_dims(p)
        doc = _read_json(p.multiangle_dir / "calib.json") or {}
        stored = doc.get("pitch") or {}
        return {
            "len_m": float(stored.get("len_m") or len_m),
            "wid_m": float(stored.get("wid_m") or wid_m),
            "template": stored.get("template") or "full",
            "goal_w_m": float(stored.get("goal_w_m") or 0) or None,
            "d_radius_m": float(stored.get("d_radius_m") or 0) or None,
        }

    _PITCH_RANGES = {"len_m": (30, 130), "wid_m": (20, 90),
                     "goal_w_m": (2, 8), "d_radius_m": (3, 20)}

    def _validate_pitch(body: dict) -> dict:
        """Validate a PUT pitch object -> clean dict (422 on bad)."""
        if not isinstance(body, dict):
            raise HTTPException(422, "pitch must be an object")
        template = body.get("template", "full")
        if template not in ("full", "small"):
            raise HTTPException(422, "pitch.template must be 'full' or 'small'")
        out = {"template": template}
        for key, (lo, hi) in _PITCH_RANGES.items():
            v = body.get(key)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(
                    422, f"pitch.{key} must be a number") from None
            if not lo <= v <= hi:
                raise HTTPException(
                    422, f"pitch.{key} must be in [{lo}, {hi}]")
            out[key] = v
        return out

    @router.get("/analysis/calib/landmarks")
    def get_calib_landmarks(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        from highlights.analysis.calib import landmarks_for
        pitch = _pitch_doc(p)
        return {"pitch": pitch,
                "landmarks": landmarks_for(pitch)}

    @router.get("/analysis/calib")
    def get_calib(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        doc = _read_json(p.multiangle_dir / "calib.json")
        doc = doc or {"angles": {}}
        doc["pitch"] = _pitch_doc(p)
        doc.setdefault("cameras", {})
        return doc

    @router.put("/analysis/calib/cameras")
    def put_calib_cameras(p: ScopedP, body: dict) -> dict:
        """Replace camera placements: {"cameras": {"<i>":
        {"x_m","y_m","dir_deg"}}} — pitch metres, may sit up to 30 m
        outside the pitch; dir_deg 0 = +x, 90 = +y."""
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        cams_in = body.get("cameras")
        if not isinstance(cams_in, dict):
            raise HTTPException(422, "cameras must be an object")
        len_m, wid_m = _pitch_dims(p)
        cams: dict[str, dict] = {}
        for key, val in cams_in.items():
            try:
                akey = str(int(key))
            except (TypeError, ValueError):
                raise HTTPException(
                    422, f"angle key {key!r} must be an int") from None
            if not isinstance(val, dict):
                raise HTTPException(422, f"camera {akey}: must be an object")
            try:
                x_m = float(val["x_m"])
                y_m = float(val["y_m"])
                dir_deg = float(val["dir_deg"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(
                    422, f"camera {akey}: needs numeric x_m, y_m, dir_deg"
                ) from None
            if not (-30 <= x_m <= len_m + 30 and
                    -30 <= y_m <= wid_m + 30):
                raise HTTPException(
                    422, f"camera {akey}: x_m/y_m out of range "
                         f"(±30 m around the pitch)")
            if not (-360 <= dir_deg <= 360):
                raise HTTPException(
                    422, f"camera {akey}: dir_deg must be in [-360, 360]")
            cams[akey] = {"x_m": x_m, "y_m": y_m, "dir_deg": dir_deg}
        doc = _read_json(p.multiangle_dir / "calib.json") or {}
        doc.setdefault("angles", {})
        doc["pitch"] = {"len_m": len_m, "wid_m": wid_m}
        doc["cameras"] = cams
        write_json_atomic(p.multiangle_dir / "calib.json",
                          doc, indent=1)
        return doc

    @router.put("/analysis/calib")
    def put_calib(p: ScopedP, body: dict) -> dict:
        """Set landmark picks: {"angles": {"<i>": {"pts":
        [{"name","fx","fy"} ...]}}} — solves each angle's H (422 when
        any given angle has <4 valid pts); "pts": [] clears an angle."""
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        angles_in = body.get("angles")
        pitch_in = body.get("pitch")
        if pitch_in is None and (not isinstance(angles_in, dict)
                                 or not angles_in):
            raise HTTPException(422, "angles must be a non-empty object")
        doc = _read_json(p.multiangle_dir / "calib.json") or {}
        doc.setdefault("angles", {})
        if pitch_in is not None:
            base = _pitch_doc(p)
            base.update(_validate_pitch(pitch_in))
            if base["template"] == "small":
                base["goal_w_m"] = base["goal_w_m"] or 3.66
                base["d_radius_m"] = base["d_radius_m"] or 9.0
            doc["pitch"] = base
            write_json_atomic(p.multiangle_dir / "calib.json",
                              doc, indent=1)
        pitch = _pitch_doc(p)
        len_m, wid_m = pitch["len_m"], pitch["wid_m"]
        if angles_in is None:
            doc["pitch"] = pitch
            return doc
        from highlights.analysis.calib import solve_homography
        solved = {}
        for key, val in angles_in.items():
            try:
                akey = str(int(key))
            except (TypeError, ValueError):
                raise HTTPException(
                    422, f"angle key {key!r} must be an int") from None
            pts = (val or {}).get("pts")
            if not isinstance(pts, list):
                raise HTTPException(422, f"angle {akey}: pts must be a list")
            if not pts:
                solved[akey] = None
                continue
            try:
                solved[akey] = solve_homography(
                    pts, len_m, wid_m, pitch) + (pts,)
            except ValueError as e:
                raise HTTPException(422, f"angle {akey}: {e}") from e
        for akey, res in solved.items():
            if res is None:
                doc["angles"].pop(akey, None)
                continue
            H, rms, pts = res
            doc["angles"][akey] = {
                "pts": [{"name": str(q["name"]),
                         "fx": float(q["fx"]), "fy": float(q["fy"])}
                        for q in pts],
                "H": H, "rms_m": round(rms, 4)}
        doc["pitch"] = pitch
        write_json_atomic(p.multiangle_dir / "calib.json",
                          doc, indent=1)
        return doc

    @router.get("/analysis/players/crops/{name}")
    def get_crop(p: PublicP, name: str) -> FileResponse:
        if not CROP_NAME_RE.match(name):
            raise HTTPException(404, "not found")
        path = _players_dir(p) / "crops" / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="image/jpeg")

    return router
