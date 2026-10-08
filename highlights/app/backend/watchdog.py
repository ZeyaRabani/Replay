"""Self-healing watchdog: auto-resume dead or transiently-failed pipelines.

One daemon thread (started from the app lifespan, HL_WATCHDOG=0 disables)
walks every project on a fixed interval:

- queued/running with a dead pid -> re-spawn the same kind of run WITHOUT
  --force; the stage-skip logic in the runners makes it resume where it
  stopped. Logged to history as "auto_resumed".
- failed with a transient error (YouTube 5xx / network) or the
  "interrupted; click Re-run" reconcile marker -> re-spawn with a growing
  backoff persisted per project; gives up after 12 attempts.
- failed because the saved cookies were rejected / bot check -> never on a
  timer; re-spawn once the user's (or shared) cookies file is newer than
  the mtime last tried.
- failed with a KNOWN_STALE_ERROR (a run poisoned by a since-removed
  stage) -> retried once immediately; the re-run skips everything and
  marks the project done in seconds.

Cancelled runs and non-transient PipelineErrors are never retried, and a
busy job slot (PipelineBusy) means leave it alone.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path

from highlights.pipeline.download import is_cookie_failure, is_transient_failure

INTERVAL_S = 60.0
FIRST_PASS_S = 10.0
MAX_ATTEMPTS = 12
BACKOFF_S = [120, 300, 900, 1800, 3600]     # then every 2h
INTERRUPTED_MSG = "interrupted; click Re-run to resume"
GAVE_UP_PREFIX = f"gave up after {MAX_ATTEMPTS} automatic retries"
# errors left behind by stages that no longer exist/are optional — one
# plain re-run flushes them to done immediately
KNOWN_STALE_ERRORS = ("scoreboard.json missing",)

_kick = threading.Event()
_thread: threading.Thread | None = None


def kick() -> None:
    """Trigger an immediate pass (e.g. fresh cookies were just saved)."""
    _kick.set()


def start() -> None:
    global _thread
    if os.environ.get("HL_WATCHDOG") == "0" or _thread is not None:
        return
    _thread = threading.Thread(
        target=_loop, name="pipeline-watchdog", daemon=True)
    _thread.start()


def _loop() -> None:
    _kick.wait(FIRST_PASS_S)
    while True:
        _kick.clear()
        try:
            pass_once()
        except Exception as e:  # watchdog must never die
            print(f"watchdog: pass failed ({type(e).__name__}: {e})")
        _kick.wait(INTERVAL_S)


def _delay_s(n: int) -> float:
    """Delay after n failed attempts (0-indexed)."""
    return BACKOFF_S[n] if n < len(BACKOFF_S) else 7200.0


# ---------- per-project retry bookkeeping ----------

# The runner rewrites status.json wholesale on every spawn, so the
# counters can't live there across runs — auto_retry.json is canonical
# and mirrored into status["auto_retry"] for the UI.

def _state_path(p) -> Path:
    return p.pipeline_dir / "auto_retry.json"


def _load_state(p) -> dict:
    try:
        return json.loads(_state_path(p).read_text())
    except Exception:
        return {}


def _save_state(p, st: dict) -> None:
    try:
        _state_path(p).parent.mkdir(parents=True, exist_ok=True)
        _state_path(p).write_text(json.dumps(st, indent=1))
    except OSError:
        pass


def _cancelled(err: str) -> bool:
    return "cancel" in err.lower()


def pass_once() -> None:
    """One sweep over all projects. Lazily imports main to avoid the
    circular import (main imports watchdog for kick/start)."""
    from . import main as m
    for p in m.get_registry().list_projects():
        try:
            _check(p, m)
        except Exception as e:
            print(f"watchdog: {p.id}: {type(e).__name__}: {e}")


def _check(p, m) -> None:
    from . import pipeline as pl

    status = pl.reconcile(p)  # dead queued/running pids -> failed/interrupted
    if not status:
        return
    st = status.get("state")
    err = status.get("error") or ""
    if _cancelled(err):
        return
    ar = _load_state(p)

    if st in ("queued", "running"):
        if pl._status_alive(status):
            return
        _respawn(p, m, pl, ar, status)
        return
    if st != "failed":
        return

    # mirror the retry bookkeeping into status for the project list UI
    view = {"n": int(ar.get("n", 0)),
            "next_at": float(ar.get("next_at", 0))}
    if status.get("auto_retry") != view:
        status["auto_retry"] = view
        pl.write_status(p, status)

    if is_cookie_failure(err):
        # retry only when a saved cookies file is newer than what we
        # tried — compare the freshest mtime across all distinct sources
        srcs = m._cookies_sources(p.owner)
        if not srcs:
            return
        mtime = max(f.stat().st_mtime for f in srcs)
        if mtime <= float(ar.get("cookies_mtime_tried", 0)):
            return
        ar["cookies_mtime_tried"] = mtime
        _respawn(p, m, pl, ar, status)
        return

    if any(k in err for k in KNOWN_STALE_ERRORS):
        if ar.get("stale_done"):
            return
        ar["stale_done"] = True
        _respawn(p, m, pl, ar, status)
        return

    if is_transient_failure(err) or err == INTERRUPTED_MSG:
        n = int(ar.get("n", 0))
        if n >= MAX_ATTEMPTS:
            if not err.startswith(GAVE_UP_PREFIX):
                status["error"] = f"{GAVE_UP_PREFIX}: {err}"
                status["message"] = status["error"]
                pl.write_status(p, status)
            return
        if time.time() < float(ar.get("next_at", 0)):
            return
        ar["n"] = n + 1
        ar["next_at"] = time.time() + _delay_s(n)
        _respawn(p, m, pl, ar, status)


def _respawn(p, m, pl, ar: dict, status: dict) -> None:
    """Re-spawn the pipeline exactly like POST /pipeline/run (no force).
    PipelineBusy means something else got there first — leave it."""
    try:
        m._respawn_project(p, p.owner)
    except pl.PipelineBusy:
        return
    except Exception as e:
        print(f"watchdog: {p.id}: respawn failed ({type(e).__name__}: {e})")
        return
    _save_state(p, ar)
    with contextlib.suppress(Exception):
        m._hist(p, "auto_resumed")
