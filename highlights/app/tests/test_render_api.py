"""Render: 409 while a job is in flight; GET /renders lists output dirs."""

import json
import threading

from conftest import new_project, scoped

import highlights.app.backend.main as m


def _project_with_candidate(client, video, tmp_path):
    pid = new_project(client, video)
    r = client.post(scoped(pid, "/video"), json={"path": str(video)})
    assert r.status_code == 200, r.text
    cf = tmp_path / "cands.json"
    cf.write_text(json.dumps({
        "events": [{"t": 2.0, "t_start": 0.5, "t_end": 3.5, "type": "goal",
                    "confidence": 0.9}]}))
    r = client.post(scoped(pid, "/candidates/load"), json={"path": str(cf)})
    assert r.status_code == 200, r.text
    return pid


def test_render_conflict_while_running(
        client, sample_video, monkeypatch, tmp_path):
    pid = _project_with_candidate(client, sample_video, tmp_path)
    gate = threading.Event()
    monkeypatch.setattr(
        m, "_run_render",
        lambda p, job_id, req, items: gate.wait(10))
    r = client.post(scoped(pid, "/render"), json={"overlay": False})
    assert r.status_code == 200, r.text
    # make the in-flight job visibly running before re-POSTing
    m._jobs[r.json()["job_id"]].state = "running"
    r = client.post(scoped(pid, "/render"), json={"overlay": False})
    assert r.status_code == 409
    gate.set()


def test_list_renders(client, sample_video, tmp_path):
    pid = _project_with_candidate(client, sample_video, tmp_path)
    p = m.get_registry().get(pid)
    d = p.root / "renders" / "abc12345"
    (d / "clips").mkdir(parents=True)
    (d / "reel.mp4").write_bytes(b"v")
    (d / "stats.json").write_text("{}")
    (d / "clips" / "01_goal.mp4").write_bytes(b"c")
    r = client.get(scoped(pid, "/renders"))
    assert r.status_code == 200, r.text
    [entry] = [e for e in r.json() if e["job_id"] == "abc12345"]
    names = {f["name"] for f in entry["files"]}
    assert names == {"reel.mp4", "stats.json", "clips/01_goal.mp4"}
    reel = next(f for f in entry["files"] if f["name"] == "reel.mp4")
    assert reel["url"].endswith("/files/abc12345/reel.mp4")
    fr = client.get(reel["url"])
    assert fr.status_code == 200 and fr.content == b"v"
