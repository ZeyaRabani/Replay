"""Detached pipeline subprocess management.

The pipeline runs as a detached Popen (start_new_session) writing
pipeline/status.json + pipeline/log.txt under the project dir. The backend
survives restarts: dead queued/running statuses are reconciled lazily.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import ffmpeg as fx
from .schemas import CandidatesFile, VideoInfo
from .store import ProjectStore

REPO_ROOT = Path(__file__).resolve().parents[3]

_procs: dict[int, subprocess.Popen] = {}


class PipelineBusy(RuntimeError):
    pass


def runner_cmd() -> list[str]:
    env = os.environ.get("HL_PIPELINE_CMD")
    if env:
        return shlex.split(env)
    return [sys.executable, "-m", "highlights.pipeline.run"]


def multiangle_runner_cmd() -> list[str]:
    env = os.environ.get("HL_MULTIANGLE_CMD")
    if env:
        return shlex.split(env)
    return [sys.executable, "-m", "highlights.multiangle.run"]


MULTIANGLE_STAGES = ["download", "angles", "sync", "track", "director",
                     "render", "fuse", "stats"]
MULTIANGLE_FROM_SYNC = MULTIANGLE_STAGES[2:]


def read_status(p: ProjectStore) -> dict | None:
    try:
        return json.loads(p.status_path.read_text())
    except Exception:
        return None


def write_status(p: ProjectStore, status: dict) -> None:
    p.status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(p.status_path)


def pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    proc = _procs.get(pid)
    if proc is not None:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _status_alive(status: dict) -> bool:
    """True if a queued/running status should block a new spawn or survive reconcile.

    A None pid is the tiny pre-Popen window in spawn() — treat as alive for a
    short grace period based on the status timestamp.
    """
    pid = status.get("pid")
    if pid is None:
        last = status.get("updated_at") or status.get("started_at") or 0
        return time.time() - last < 15
    return pid_alive(pid)


def spawn(
    p: ProjectStore,
    *,
    youtube_url: str | None = None,
    video: str | None = None,
    stages: list[str] | None = None,
    force: bool = False,
    cookies: str | None = None,
) -> dict:
    argv = runner_cmd() + ["--project-dir", str(p.root)]
    if youtube_url:
        argv += ["--youtube-url", youtube_url]
    elif video:
        argv += ["--video", video]
    if stages:
        argv += ["--stages", ",".join(stages)]
    if force:
        argv += ["--force"]
    ck = cookies or os.environ.get("HL_YT_COOKIES")
    if ck:
        argv += ["--cookies", ck]
    return _launch(p, argv, initial_stage="download" if youtube_url else "probe")


def spawn_multiangle(
    p: ProjectStore,
    *,
    stages: list[str] | None = None,
    force: bool = False,
    cookies: str | None = None,
    offsets: list[float] | None = None,
) -> dict:
    argv = multiangle_runner_cmd() + ["--project-dir", str(p.root)]
    if stages:
        argv += ["--stages", ",".join(stages)]
    if force:
        argv += ["--force"]
    ck = cookies or os.environ.get("HL_YT_COOKIES")
    if ck:
        argv += ["--cookies", ck]
    if offsets is not None:
        argv += ["--offsets", ",".join(f"{o:g}" for o in offsets)]
    return _launch(p, argv, initial_stage="download")


def _launch(p: ProjectStore, argv: list[str], *, initial_stage: str) -> dict:
    status = read_status(p)
    if status and status.get("state") in ("queued", "running") and _status_alive(status):
        raise PipelineBusy("pipeline already running for this project")

    p.status_path.parent.mkdir(parents=True, exist_ok=True)
    # write queued BEFORE Popen so a fast runner's running/done status is
    # never clobbered back to queued; pid is filled in just after spawn
    now = time.time()
    status = {
        "state": "queued",
        "stage": initial_stage,
        "progress": 0.0,
        "stage_progress": 0.0,
        "message": "queued",
        "error": None,
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "pid": None,
        "video_path": None,
        "video": None,
        "download": None,
    }
    write_status(p, status)
    p.set_pipeline_state("queued")

    log = open(p.log_path, "ab")  # noqa: SIM115
    try:
        proc = subprocess.Popen(
            argv, cwd=REPO_ROOT,
            stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    finally:
        log.close()
    _procs[proc.pid] = proc

    # fill in the pid only if the runner hasn't already written its own status
    cur = read_status(p)
    if cur and cur.get("pid") is None:
        cur["pid"] = proc.pid
        write_status(p, cur)
        status = cur
    elif cur:
        status = cur
    return status


def cancel(p: ProjectStore) -> dict:
    status = read_status(p)
    now = time.time()
    if status is None:
        return {
            "state": "failed",
            "stage": None,
            "progress": 0.0,
            "stage_progress": 0.0,
            "message": "no pipeline run to cancel",
            "error": "cancelled",
            "started_at": None,
            "updated_at": now,
            "finished_at": now,
            "pid": None,
            "video_path": None,
            "video": None,
            "download": None,
        }
    pid = status.get("pid")
    if pid_alive(pid):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        deadline = time.time() + 3.0
        while time.time() < deadline and pid_alive(pid):
            time.sleep(0.05)
        if pid_alive(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
    status.update(
        {
            "state": "failed",
            "message": "cancelled by user",
            "error": "cancelled",
            "finished_at": now,
            "updated_at": now,
        }
    )
    write_status(p, status)
    p.set_pipeline_state("failed")
    return status


def reconcile(p: ProjectStore) -> dict | None:
    """Fail queued/running statuses whose pid is gone."""
    status = read_status(p)
    if status and status.get("state") in ("queued", "running") and not _status_alive(status):
        status["state"] = "failed"
        status["message"] = "interrupted; click Re-run to resume"
        status["error"] = status["message"]
        status["finished_at"] = time.time()
        status["updated_at"] = time.time()
        write_status(p, status)
        p.set_pipeline_state("failed")
    return status


def refresh(p: ProjectStore) -> dict | None:
    """Lazy sync: reconcile dead pids, import results when done."""
    status = reconcile(p)
    if status is None:
        return None
    state = status.get("state")
    if state != p.pipeline_state:
        p.set_pipeline_state(state)
    if state == "done":
        vp = status.get("video_path")
        if not vp and p.is_multiangle:
            ma_video = p.root / "match.mp4"
            if ma_video.is_file():
                vp = str(ma_video)
        if vp:
            resolved = str(Path(vp).resolve())
            if (p.video is None or p.video.path != resolved) and Path(resolved).is_file():
                if p.video is not None:
                    p.invalidate_video()
                try:
                    info = fx.probe(resolved)
                except RuntimeError:
                    info = status.get("video") or {"duration_s": 0.0, "width": 0, "height": 0, "fps": 0.0}
                p.set_video(VideoInfo(path=resolved, registered_at=time.time(), **info))
        if (p.candidates_version == 0 or not p.candidates) and (p.pipeline_dir / "candidates.json").is_file():
            try:
                cf = CandidatesFile(**json.loads((p.pipeline_dir / "candidates.json").read_text()))
                p.load_candidates(cf)
            except Exception as e:
                print(f"warning: could not import pipeline candidates for {p.id}: {e}")
    return status


def tail_log(p: ProjectStore, n: int = 50) -> list[str]:
    if not p.log_path.is_file():
        return []
    try:
        lines = p.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def status_with_log(p: ProjectStore) -> dict:
    status = refresh(p)
    if status is None:
        status = {
            "state": p.pipeline_state or "none",
            "stage": None,
            "progress": 0.0,
            "stage_progress": 0.0,
            "message": "",
            "error": None,
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
            "pid": None,
            "video_path": None,
            "video": None,
            "download": None,
        }
    status["log"] = tail_log(p)
    return status
