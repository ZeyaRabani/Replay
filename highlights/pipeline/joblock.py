"""Global pipeline job lock — caps concurrent match pipelines host-wide.

Each runner grabs one of `<workdir>/.jobs/slot{k}.lock` via fcntl.flock
(LOCK_EX|LOCK_NB, k in range(max_jobs)); with none free it marks its
status "queued — waiting for another match to finish" and retries every
5 s. The fd is held for the with-block, so the OS releases it on exit or
kill — no stale-lock cleanup needed.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

DEFAULT_MAX_JOBS = 1


@contextlib.contextmanager
def job_slot(workdir: Path, status=None, log=None,
             max_jobs: int | None = None):
    """Hold one of `max_jobs` flock slots for the block's duration."""
    if max_jobs is None:
        max_jobs = int(os.environ.get("HL_MAX_JOBS", str(DEFAULT_MAX_JOBS)))
    slot_dir = Path(workdir) / ".jobs"
    slot_dir.mkdir(parents=True, exist_ok=True)
    announced = False
    fh = None
    while fh is None:
        for k in range(max_jobs):
            # fd intentionally held for the with-block (released on exit)
            f = open(slot_dir / f"slot{k}.lock", "a+b")  # noqa: SIM115
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            fh = f
            break
        if fh is None:
            if not announced:
                if status is not None:
                    status.update(state="queued",
                                  message="waiting for another match to finish",
                                  force=True)
                if log is not None:
                    log("job lock: all slots busy — queued")
                announced = True
            time.sleep(5)
    if announced and log is not None:
        log("job lock: slot acquired — starting")
    try:
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def workdir_for(project_dir: Path) -> Path:
    """HL_WORKDIR for the projects/<id> layout."""
    return Path(project_dir).resolve().parent.parent
