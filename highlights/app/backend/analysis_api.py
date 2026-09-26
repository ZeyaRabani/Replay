"""Analyse-match endpoints for multi-angle projects.

POST /analyse spawns `python -m highlights.analysis.run` as a detached
subprocess writing analysis/status.json + analysis/log.txt; GET
/analysis returns the reconciled status plus stats/summary JSON.
Mirrors pipeline.py's launch/reconcile pattern.
"""

# NOTE: no `from __future__ import annotations` — the ScopedP Depends
# annotation inside make_router must evaluate eagerly against the factory
# argument, not be resolved later from module globals.
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]

_procs: dict[int, subprocess.Popen] = {}


class AnalysePut(BaseModel):
    force: bool = False


def _analysis_dir(p) -> Path:
    return p.root / "analysis"


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
    env = os.environ.get("HL_ANALYSIS_CMD")
    if env:
        return shlex.split(env)
    return [sys.executable, "-m", "highlights.analysis.run"]


def _reconciled_status(p) -> dict | None:
    path = _analysis_dir(p) / "status.json"
    status = _read_json(path)
    if status and status.get("state") in ("queued", "running") and not _status_alive(status):
        status["state"] = "failed"
        status["error"] = "interrupted"
        status["message"] = "interrupted"
        status["finished_at"] = time.time()
        status["updated_at"] = time.time()
        _write_status(path, status)
    return status


def _estimate_min(p) -> int:
    from highlights.analysis.run import estimate_minutes, resolve_context
    try:
        ctx = resolve_context(p.root)
        return estimate_minutes(ctx["window"][1] - ctx["window"][0])
    except Exception:
        dur = 0.0
        if p.video is not None:
            dur = float(getattr(p.video, "duration_s", 0.0) or 0.0)
        if dur <= 0:
            dur = 600.0
        return estimate_minutes(dur)


def make_router(ScopedP) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}")

    @router.post("/analyse")
    def post_analyse(p: ScopedP, body: AnalysePut | None = None) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        if p.pipeline_state != "done":
            raise HTTPException(409, "pipeline is not done")
        adir = _analysis_dir(p)
        status_path = adir / "status.json"
        status = _read_json(status_path)
        if status and status.get("state") in ("queued", "running") and _status_alive(status):
            raise HTTPException(409, "analysis already running")

        force = bool(body and body.force)
        now = time.time()
        status = {
            "state": "queued",
            "stage": "teams",
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
        adir.mkdir(parents=True, exist_ok=True)
        log = open(adir / "log.txt", "ab")  # noqa: SIM115
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

    @router.get("/analysis")
    def get_analysis(p: ScopedP) -> dict:
        if not p.is_multiangle:
            raise HTTPException(404, "not a multi-angle project")
        adir = _analysis_dir(p)
        return {
            "status": _reconciled_status(p),
            "stats": _read_json(adir / "match_stats.json"),
            "summary": _read_json(adir / "summary.json"),
            "estimate_min": _estimate_min(p),
        }

    return router
