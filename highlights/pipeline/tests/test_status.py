import json
import os
import time

from highlights.pipeline.status import StatusWriter, load_status


def test_atomic_write_and_defaults(tmp_path):
    p = tmp_path / "status.json"
    w = StatusWriter(p)
    w.update(stage="probe", message="hi", force=True)
    d = json.loads(p.read_text())
    assert d["stage"] == "probe"
    assert d["message"] == "hi"
    assert d["state"] == "queued"
    assert d["progress"] == 0.0
    assert d["pid"] == os.getpid()
    assert d["error"] is None
    assert not (tmp_path / "status.json.tmp").exists()


def test_throttle_and_force(tmp_path):
    p = tmp_path / "status.json"
    w = StatusWriter(p, min_interval=60.0)
    w.update(message="first")          # first write always goes through
    assert json.loads(p.read_text())["message"] == "first"
    w.update(message="second")         # throttled
    assert json.loads(p.read_text())["message"] == "first"
    w.update(message="third", force=True)
    assert json.loads(p.read_text())["message"] == "third"


def test_throttle_interval_elapsed(tmp_path):
    p = tmp_path / "status.json"
    w = StatusWriter(p, min_interval=0.05)
    w.update(message="a")
    time.sleep(0.06)
    w.update(message="b")
    assert json.loads(p.read_text())["message"] == "b"


def test_merge_and_load(tmp_path):
    p = tmp_path / "status.json"
    w = StatusWriter(p)
    w.update(stage="motion", force=True)
    w.update(progress=0.5, force=True)
    d = load_status(p)
    assert d["stage"] == "motion" and d["progress"] == 0.5
    assert load_status(tmp_path / "nope.json") is None
