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
        candidate_ids = {str(c.id) for c in p.candidates}
        try:
            roster = validate_roster(body, tracklet_ids, candidate_ids)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
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

    @router.get("/analysis/players/crops/{name}")
    def get_crop(p: PublicP, name: str) -> FileResponse:
        if not CROP_NAME_RE.match(name):
            raise HTTPException(404, "not found")
        path = _players_dir(p) / "crops" / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="image/jpeg")

    return router
