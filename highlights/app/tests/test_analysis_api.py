"""Analyse-match endpoint tests (fake runner via HL_ANALYSIS_CMD)."""

import json
import sys
import time
from pathlib import Path

from conftest import new_project, scoped

FAKE_ANALYSIS = Path(__file__).resolve().parent / "fake_analysis.py"


def _wait_pipeline(client, pid, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(scoped(pid, "")).json()
        if d["pipeline_state"] in ("done", "failed"):
            return d
        time.sleep(0.1)
    return client.get(scoped(pid, "")).json()


def _wait_analysis(client, pid, timeout=10.0):
    deadline = time.time() + timeout
    d = {}
    while time.time() < deadline:
        d = client.get(scoped(pid, "/analysis")).json()
        if (d.get("status") or {}).get("state") in ("done", "failed"):
            return d
        time.sleep(0.1)
    return d


def _done_multiangle(client, sample_video):
    r = client.post("/api/projects/multiangle", json={
        "angles": [{"url": f"https://youtu.be/a{i}", "label": f"C{i}"}
                   for i in range(3)]})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    _wait_pipeline(client, pid)
    return pid


def test_analyse_not_multiangle_404(client, sample_video):
    pid = new_project(client, sample_video)
    r = client.post(scoped(pid, "/analyse"), json={})
    assert r.status_code == 404
    assert r.json()["detail"] == "not a multi-angle project"


def test_analyse_before_done_409(client, sample_video):
    pid = _done_multiangle(client, sample_video)
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    p.set_pipeline_state("running")
    try:
        r = client.post(scoped(pid, "/analyse"), json={})
        assert r.status_code == 409
        assert "not done" in r.json()["detail"]
    finally:
        p.set_pipeline_state("done")


def test_analysis_empty_get(client, sample_video):
    pid = _done_multiangle(client, sample_video)
    r = client.get(scoped(pid, "/analysis"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] is None
    assert d["stats"] is None
    assert d["summary"] is None
    assert d["estimate_min"] >= 1


def test_analyse_run_and_busy(client, sample_video, monkeypatch):
    monkeypatch.setenv("HL_ANALYSIS_CMD", f"{sys.executable} {FAKE_ANALYSIS}")
    monkeypatch.setenv("FAKE_A_SLEEP", "0.4")
    pid = _done_multiangle(client, sample_video)

    r = client.post(scoped(pid, "/analyse"), json={})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["state"] in ("queued", "running")

    # second POST while the runner is alive -> 409
    r2 = client.post(scoped(pid, "/analyse"), json={})
    assert r2.status_code == 409
    assert "already running" in r2.json()["detail"]

    d = _wait_analysis(client, pid)
    assert d["status"]["state"] == "done"
    assert d["stats"]["team_confidence"] == 0.72
    assert d["summary"]["text"]
    assert d["estimate_min"] >= 1

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    argv = json.loads((p.root / "analysis" / "argv.json").read_text())
    assert "--project-dir" in argv
    assert "--force" not in argv

    # done -> a fresh POST is allowed again
    r3 = client.post(scoped(pid, "/analyse"), json={"force": True})
    assert r3.status_code == 200
    for _ in range(50):
        argv = json.loads((p.root / "analysis" / "argv.json").read_text())
        if "--force" in argv:
            break
        time.sleep(0.1)
    assert "--force" in argv
    _wait_analysis(client, pid)


def test_reconcile_interrupted(client, sample_video):
    pid = _done_multiangle(client, sample_video)
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    adir = p.root / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "status.json").write_text(json.dumps({
        "state": "running", "stage": "teams", "progress": 0.4,
        "message": "running", "error": None, "started_at": time.time(),
        "updated_at": time.time(), "finished_at": None,
        "pid": 99999999}))
    d = client.get(scoped(pid, "/analysis")).json()
    assert d["status"]["state"] == "failed"
    assert d["status"]["error"] == "interrupted"
