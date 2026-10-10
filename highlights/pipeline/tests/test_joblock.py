"""Global job lock: second contender waits for the first to release."""

import threading
import time

from highlights.pipeline.joblock import job_slot, workdir_for


class _Status:
    def __init__(self):
        self.calls = []

    def update(self, **kw):
        self.calls.append(kw)


def test_job_slot_serializes_two_contenders(tmp_path):
    order = []

    def worker(name, delay, hold, status=None):
        time.sleep(delay)
        with job_slot(tmp_path, status=status, max_jobs=1):
            order.append(name)
            time.sleep(hold)

    s2 = _Status()
    t1 = threading.Thread(target=worker, args=("first", 0.0, 0.4))
    t2 = threading.Thread(target=worker, args=("second", 0.1, 0.0),
                          kwargs={"status": s2})
    t1.start(); t2.start(); t1.join(); t2.join()
    assert order == ["first", "second"]
    # the waiter announced itself queued exactly once
    queued = [c for c in s2.calls if c.get("state") == "queued"]
    assert len(queued) == 1 and "waiting" in queued[0]["message"]


def test_job_slot_two_slots_run_concurrently(tmp_path):
    order = []

    def worker(name, delay, hold):
        time.sleep(delay)
        with job_slot(tmp_path, max_jobs=2):
            order.append(name)
            time.sleep(hold)

    t1 = threading.Thread(target=worker, args=("a", 0.0, 0.3))
    t2 = threading.Thread(target=worker, args=("b", 0.1, 0.0))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert order == ["a", "b"]  # b didn't wait for a to finish


def test_workdir_for(tmp_path):
    proj = tmp_path / "projects" / "abc123"
    proj.mkdir(parents=True)
    assert workdir_for(proj) == tmp_path
