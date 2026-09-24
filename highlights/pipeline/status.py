"""Atomic, throttled status.json writer for the pipeline run."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _now() -> float:
    return time.time()


def default_status() -> dict:
    now = _now()
    return {
        "state": "queued",
        "stage": None,
        "progress": 0.0,
        "stage_progress": 0.0,
        "message": "",
        "error": None,
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "pid": os.getpid(),
        "video_path": None,
        "video": None,
        "download": None,
    }


class StatusWriter:
    """Merges update(**fields) into a status dict and writes it atomically.

    Writes are throttled to at most one per `min_interval` seconds unless
    force=True is passed (used for stage transitions / done / failed).
    """

    def __init__(self, path: str | Path, min_interval: float = 1.0):
        self.path = Path(path)
        self.min_interval = min_interval
        self.status = default_status()
        self._last_write = 0.0

    def update(self, force: bool = False, **fields) -> dict:
        self.status.update(fields)
        self.status["updated_at"] = _now()
        if force or (self.status["updated_at"] - self._last_write) >= self.min_interval:
            self._write()
        return self.status

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status, indent=1))
        os.replace(tmp, self.path)
        self._last_write = _now()


def load_status(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
