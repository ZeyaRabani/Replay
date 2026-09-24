"""Multi-user / multi-project / pipeline tests."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from conftest import new_project, scoped

SUMMARY_KEYS = {
    "id", "title", "created_at", "source", "pipeline_state", "progress",
    "stage", "message", "video", "n_candidates", "n_confirmed", "thumb_url",
}
STATS_KEYS = {
    "duration_s", "match_window", "halves", "bin_s", "timeline",
    "events_by_type", "events_per_10min", "top_moments", "whistles",
    "activity", "pipeline",
}


def _wait_done(client, pid, timeout=10.0, states=("done", "failed")):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(scoped(pid, "")).json()
        if d["pipeline_state"] in states:
            return d
        time.sleep(0.1)
    return client.get(scoped(pid, "")).json()


# ---------- users / auth ----------


def test_users(client):
    r = client.post("/api/users", json={"name": "  alice  "})
    assert r.status_code == 200
    assert r.json()["name"] == "alice"
    # idempotent
    r2 = client.post("/api/users", json={"name": "alice"})
    assert r2.json()["created_at"] == r.json()["created_at"]
    assert client.post("/api/users", json={"name": ""}).status_code == 422
    assert client.post("/api/users", json={"name": "x" * 41}).status_code == 422
    users = client.get("/api/users").json()
    names = {u["name"] for u in users}
    assert {"demo", "tester", "alice"} <= names
    assert all("n_projects" in u for u in users)


def test_auth_required(client):
    r = client.get("/api/projects", headers={"X-User": ""})
    assert r.status_code == 401
    c2 = client
    c2.headers.pop("X-User")
    assert c2.get("/api/projects").status_code == 401
    assert c2.get("/api/projects", headers={"X-User": "ghost"}).status_code == 401
    c2.headers["X-User"] = "tester"


# ---------- project lifecycle ----------


def test_create_path_with_pipeline(client, sample_video, tmp_path):
    r = client.post("/api/projects", json={"path": str(sample_video)})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    d = _wait_done(client, pid)
    assert d["pipeline_state"] == "done"
    assert set(d.keys()) >= SUMMARY_KEYS
    assert d["n_candidates"] == 2
    assert d["video"]["duration_s"] > 0
    # argv: --stages with probe first, no download
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    argv = json.loads((proot / "pipeline" / "argv.json").read_text())
    assert "--project-dir" in argv
    i = argv.index("--stages")
    stages = argv[i + 1].split(",")
    assert stages[0] == "probe"
    assert "download" not in stages
    assert "--youtube-url" not in argv
    pl = client.get(scoped(pid, "/pipeline")).json()
    assert isinstance(pl["log"], list) and len(pl["log"]) > 0


def test_create_upload(client, sample_video):
    files = {"file": ("my clip.mp4", sample_video.read_bytes(), "video/mp4")}
    r = client.post("/api/projects", files=files)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["source"]["kind"] == "upload"
    assert d["source"]["filename"] == "my clip.mp4"
    import highlights.app.backend.main as m
    p = m.get_registry().get(d["id"])
    saved = p.source_dir / "my clip.mp4"
    assert saved.is_file()
    assert saved.stat().st_size == sample_video.stat().st_size


def test_create_youtube(client):
    r = client.post("/api/projects", json={"youtube_url": "https://youtu.be/abc"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["source"]["kind"] == "youtube"
    import highlights.app.backend.main as m
    proot = m.get_registry().get(d["id"]).root
    for _ in range(50):
        argv_p = proot / "pipeline" / "argv.json"
        if argv_p.is_file():
            break
        time.sleep(0.1)
    argv = json.loads(argv_p.read_text())
    assert "--youtube-url" in argv
    assert "--stages" not in argv
    # let it finish so it can't interfere with other tests
    _wait_done(client, d["id"])


def test_create_youtube_with_cookies(client):
    r = client.post("/api/projects", json={
        "youtube_url": "https://youtu.be/abc",
        "cookies_text": "# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\tFALSE\t1\tc\tv\n",
    })
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    ck = proot / "source" / "cookies.txt"
    assert ck.is_file()
    assert ck.read_text().startswith("# Netscape HTTP Cookie File")
    for _ in range(50):
        argv_p = proot / "pipeline" / "argv.json"
        if argv_p.is_file():
            break
        time.sleep(0.1)
    argv = json.loads(argv_p.read_text())
    i = argv.index("--cookies")
    assert argv[i + 1] == str(ck)
    _wait_done(client, pid)
    # rerun reuses the stored cookies file
    r = client.post(scoped(pid, "/pipeline/run"), json={})
    assert r.status_code in (200, 409)
    for _ in range(50):
        argv = json.loads(argv_p.read_text())
        if "--cookies" in argv:
            break
        time.sleep(0.1)
    assert argv[argv.index("--cookies") + 1] == str(ck)
    _wait_done(client, pid)


COOKIES = ("# Netscape HTTP Cookie File\n"
           ".youtube.com\tTRUE\t/\tTRUE\t1\tSID\tabc\n")


def test_user_cookies_roundtrip(client):
    r = client.get("/api/me/youtube-cookies")
    assert r.status_code == 200 and r.json() == {"saved": False, "updated_at": None}
    r = client.put("/api/me/youtube-cookies", json={"cookies_text": COOKIES})
    assert r.status_code == 200 and r.json()["saved"] and r.json()["updated_at"]
    r = client.get("/api/me/youtube-cookies")
    assert r.json()["saved"]
    r = client.delete("/api/me/youtube-cookies")
    assert r.status_code == 204
    assert client.get("/api/me/youtube-cookies").json()["saved"] is False


def test_user_cookies_validation(client):
    r = client.put("/api/me/youtube-cookies", json={"cookies_text": "   "})
    assert r.status_code == 422
    r = client.put("/api/me/youtube-cookies", json={"cookies_text": "# Netscape\n.example.com\tTRUE\t/\tF\t1\ta\tb\n"})
    assert r.status_code == 422


def test_youtube_project_uses_saved_cookies(client, tmp_path):
    client.put("/api/me/youtube-cookies", json={"cookies_text": COOKIES})
    r = client.post("/api/projects", json={"youtube_url": "https://youtu.be/abc"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    ck = proot / "source" / "cookies.txt"
    assert ck.is_file()
    assert ck.read_text() == COOKIES
    ud = Path(m.workdir()) / "users" / "tester" / "youtube_cookies.txt"
    assert ud.is_file() and ud.read_text() == COOKIES
    for _ in range(50):
        argv_p = proot / "pipeline" / "argv.json"
        if argv_p.is_file():
            break
        time.sleep(0.1)
    argv = json.loads(argv_p.read_text())
    assert argv[argv.index("--cookies") + 1] == str(ck)
    _wait_done(client, pid)


def test_cancel_and_409(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_PIPELINE_HANG", "1")
    r = client.post("/api/projects", json={"path": str(sample_video)})
    pid = r.json()["id"]
    # wait until running
    status = None
    for _ in range(100):
        status = client.get(scoped(pid, "/pipeline")).json()
        if status["state"] == "running":
            break
        time.sleep(0.1)
    assert status["state"] == "running"
    run_pid = status["pid"]
    # 409 while running
    r = client.post(scoped(pid, "/pipeline/run"), json={})
    assert r.status_code == 409
    # cancel
    r = client.post(scoped(pid, "/pipeline/cancel"))
    assert r.status_code == 200
    assert r.json()["state"] == "failed"
    assert r.json()["message"] == "cancelled by user"
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.kill(run_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(run_pid, 0)


def test_reconcile_dead_pid(client, sample_video):
    pid = new_project(client, sample_video)
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    # a pid that has already exited
    proc = subprocess.Popen(["true"])
    proc.wait()
    from highlights.app.backend import pipeline
    pipeline.write_status(p, {
        "state": "running", "stage": "probe", "progress": 0.4,
        "stage_progress": 0.4, "message": "running", "error": None,
        "started_at": time.time(), "updated_at": time.time(),
        "finished_at": None, "pid": proc.pid,
        "video_path": None, "video": None, "download": None,
    })
    p.set_pipeline_state("running")
    m.reset_registry()
    d = client.get(scoped(pid, "")).json()
    assert d["pipeline_state"] == "failed"
    assert d["message"] == "interrupted; click Re-run to resume"


def test_rerun_finished(client, sample_video):
    r = client.post("/api/projects", json={"path": str(sample_video)})
    pid = r.json()["id"]
    _wait_done(client, pid)
    r = client.post(scoped(pid, "/pipeline/run"), json={})
    assert r.status_code == 200, r.text
    assert r.json()["state"] in ("queued", "running")
    _wait_done(client, pid)
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    argv = json.loads((proot / "pipeline" / "argv.json").read_text())
    assert "--video" in argv


def test_owner_isolation_and_delete(client, sample_video):
    pid = new_project(client, sample_video)
    client.post("/api/users", json={"name": "other"})
    other = client.get(scoped(pid, ""), headers={"X-User": "other"})
    assert other.status_code == 404
    lst = client.get("/api/projects", headers={"X-User": "other"}).json()
    assert lst == []
    r = client.delete(scoped(pid, ""))
    assert r.status_code == 204
    assert client.get(scoped(pid, "")).status_code == 404


# ---------- legacy / demo ----------


def test_legacy_404_without_demo(client):
    # remove the seeded demo project so legacy routes have nothing to map to
    demo = client.get("/api/projects", headers={"X-User": "demo"}).json()
    for d in demo:
        client.delete(scoped(d["id"], ""), headers={"X-User": "demo"})
    r = client.get("/api/video")
    assert r.status_code == 404
    assert "legacy" in r.json()["detail"]


def test_demo_project(client, sample_video, tmp_path, monkeypatch):
    import highlights.app.backend.main as m
    monkeypatch.setenv("HL_WORKDIR", str(tmp_path / "wd2"))
    monkeypatch.setenv("HL_DEMO_VIDEO", str(sample_video))
    m.reset_registry()
    from fastapi.testclient import TestClient
    c = TestClient(m.app)
    assert any(u["name"] == "demo" for u in c.get("/api/users").json())
    projects = c.get("/api/projects", headers={"X-User": "demo"}).json()
    assert len(projects) == 1
    d = projects[0]
    assert d["title"] == "Demo match (5qj_nsQSzvQ)"
    assert d["n_candidates"] == 47
    assert d["pipeline_state"] == "done"
    pid = d["id"]
    st = c.get(scoped(pid, "/stats"), headers={"X-User": "demo"}).json()
    assert set(st.keys()) == STATS_KEYS
    for row in st["timeline"]:
        assert 0 <= row["motion"] <= 1
        assert 0 <= row["excitement"] <= 1
    n_non_rejected = sum(
        1 for e in json.loads(
            (m.REPO_ROOT / "highlights/fusion/outputs/candidates.json").read_text()
        )["events"] if e["cross_validation"] != "rejected"
    )
    assert sum(st["events_by_type"].values()) == n_non_rejected
    # legacy routes map to the demo project
    cands = c.get("/api/candidates").json()
    assert len(cands) == 47
    # media route works without X-User
    r = c.get(scoped(pid, "/video/source.mp4"))
    assert r.status_code == 200
    # scoped routes still enforce ownership
    assert c.get(scoped(pid, ""), headers={"X-User": "tester"}).status_code == 401


def test_spawn_race_fast_runner(client, sample_video, monkeypatch):
    """A runner that finishes instantly must not be clobbered to queued/failed."""
    monkeypatch.setenv("FAKE_PIPELINE_SLEEP", "0")
    for _ in range(3):
        r = client.post("/api/projects", json={"path": str(sample_video)})
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        d = _wait_done(client, pid)
        assert d["pipeline_state"] == "done"
        assert d["n_candidates"] == 2


def test_spa_fallback(client, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>spa</html>")
    import highlights.app.backend.main as m
    monkeypatch.setattr(m, "DIST", dist)
    r = client.get("/projects/abc")
    assert r.status_code == 200
    assert r.text == "<html>spa</html>"
    assert client.get("/api/nonexistent").status_code == 404
