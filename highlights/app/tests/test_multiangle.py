"""Multi-angle (director cut) project tests."""

import json
import time

from conftest import scoped

COOKIES = ("# Netscape HTTP Cookie File\n"
           ".youtube.com\tTRUE\t/\tTRUE\t1\tSID\tabc\n")


def _wait(client, pid, timeout=10.0, states=("done", "failed", "needs_input")):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(scoped(pid, "")).json()
        if d["pipeline_state"] in states:
            return d
        time.sleep(0.1)
    return client.get(scoped(pid, "")).json()


def _argv(proot):
    p = proot / "multiangle" / "argv.json"
    for _ in range(50):
        if p.is_file():
            return json.loads(p.read_text())
        time.sleep(0.1)
    return json.loads(p.read_text())


def _create(client, n=3, **kw):
    angles = [{"url": f"https://youtu.be/a{i}", "label": f"Cam {i}"}
              for i in range(n)]
    return client.post("/api/projects/multiangle",
                       json={"angles": angles, **kw})


def test_create_multiangle(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    r = _create(client, 3, title="3 cams")
    assert r.status_code == 200, r.text
    d = r.json()
    pid = d["id"]
    assert d["source"]["kind"] == "multiangle"
    assert len(d["source"]["angles"]) == 3
    assert d["source"]["angles"][0]["label"] == "Cam 0"
    assert d["mode"] == "multiangle"
    assert d["n_angles"] == 3

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    assert p.angle_dir(0).is_dir() and p.angle_dir(2).is_dir()

    argv = _argv(p.root)
    assert "--project-dir" in argv
    assert "--youtube-url" not in argv

    d = _wait(client, pid)
    assert d["pipeline_state"] == "done"
    assert d["n_candidates"] == 3
    assert d["video"] is not None and d["video"]["duration_s"] > 0
    assert (p.root / "match.mp4").is_file()

    # list endpoint shows mode/n_angles
    lst = client.get("/api/projects").json()
    row = next(x for x in lst if x["id"] == pid)
    assert row["mode"] == "multiangle" and row["n_angles"] == 3

    # multiangle info endpoint
    info = client.get(scoped(pid, "/multiangle")).json()
    assert info["sync"]["method"] == "xcorr"
    assert len(info["sync"]["pairs"]) == 2
    assert info["director"]["n_cuts"] == 133
    assert "segments" not in info["director"]
    assert len(info["angles"]) == 3
    assert info["angles"][0]["label"] == "Cam 0"
    assert info["angles"][0]["duration"] == 20.0
    assert info["angles"][0]["status"] == "done"
    assert info["score"]["home"]["label"] == "Green end"
    assert info["status"]["state"] == "done"

    # full director.json
    full = client.get(scoped(pid, "/multiangle/director")).json()
    assert len(full["segments"]) == 6

    # single-camera project gets 404 on the multiangle endpoint
    r2 = client.post("/api/projects",
                     json={"path": str(sample_video), "run_pipeline": False})
    other = r2.json()["id"]
    assert client.get(scoped(other, "/multiangle")).status_code == 404


def test_multiangle_validation(client):
    r = _create(client, 1)
    assert r.status_code == 422
    r = _create(client, 5)
    assert r.status_code == 422
    r = client.post("/api/projects/multiangle", json={
        "angles": [{"url": "https://youtu.be/a"}, {"label": "no url"}]})
    assert r.status_code == 422


def test_multiangle_needs_input_offsets(client, monkeypatch):
    monkeypatch.setenv("FAKE_MA_MODE", "needs_input")
    r = _create(client, 3)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    d = _wait(client, pid)
    assert d["pipeline_state"] == "needs_input"
    assert "enter offsets" in d["message"]
    assert d["pipeline_state"] != "failed"

    # offsets validation
    r = client.put(scoped(pid, "/multiangle/offsets"), json={"offsets": [0, 1]})
    assert r.status_code == 422
    r = client.put(scoped(pid, "/multiangle/offsets"),
                   json={"offsets": [1.5, 12.5, -3.2]})
    assert r.status_code == 422

    # rerun from sync with manual offsets; the child reads env at spawn
    monkeypatch.delenv("FAKE_MA_MODE")
    r = client.put(scoped(pid, "/multiangle/offsets"),
                   json={"offsets": [0, 12.5, -3.2]})
    assert r.status_code == 200, r.text

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    # argv.json from the first run still exists; wait for the respawn
    argv = []
    for _ in range(50):
        argv = json.loads((p.root / "multiangle" / "argv.json").read_text())
        if "--offsets" in argv:
            break
        time.sleep(0.1)
    i = argv.index("--offsets")
    assert argv[i + 1] == "0,12.5,-3.2"
    i = argv.index("--stages")
    stages = argv[i + 1].split(",")
    assert stages[0] == "sync"
    assert "download" not in stages

    d = _wait(client, pid, states=("done", "failed"))
    assert d["pipeline_state"] == "done"
    sync = (p.root / "multiangle" / "sync.json")
    assert json.loads(sync.read_text())["method"] == "manual"


def test_multiangle_score_recompute(client):
    r = _create(client, 3)
    pid = r.json()["id"]
    d = _wait(client, pid)
    assert d["pipeline_state"] == "done"

    cands = client.get(scoped(pid, "/candidates")).json()
    goals = [c for c in cands if c["type"] == "goal"]
    assert len(goals) == 2

    # confirm the cross-validated goal (team already in signals)
    r = client.patch(scoped(pid, f"/candidates/{goals[0]['id']}"),
                     json={"status": "confirmed"})
    assert r.status_code == 200
    # a confirmed goal with no team counts as unassigned
    r = client.patch(scoped(pid, f"/candidates/{goals[1]['id']}"),
                     json={"status": "confirmed"})
    assert r.status_code == 200
    st = client.get(scoped(pid, "/stats")).json()
    score = st["multiangle"]["score"]
    assert score["home"]["goals"] == 1 and score["home"]["label"] == "Green end"
    assert score["away"]["goals"] == 0 and score["away"]["label"] == "Blue end"
    assert score["unassigned"] == 1

    # set team on the second goal via patch
    r = client.patch(scoped(pid, f"/candidates/{goals[1]['id']}"),
                     json={"team": "away"})
    assert r.status_code == 200
    assert r.json()["signals"]["team"] == "away"
    st = client.get(scoped(pid, "/stats")).json()
    score = st["multiangle"]["score"]
    assert score["away"]["goals"] == 1
    assert score["unassigned"] == 0

    # rejecting the away goal drops it back to 0
    client.patch(scoped(pid, f"/candidates/{goals[1]['id']}"),
                 json={"status": "rejected"})
    st = client.get(scoped(pid, "/stats")).json()
    assert st["multiangle"]["score"]["away"]["goals"] == 0


def test_multiangle_cookies(client):
    r = _create(client, 2, cookies_text=COOKIES)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    ck = proot / "source" / "cookies.txt"
    assert ck.is_file() and ck.read_text() == COOKIES
    argv = _argv(proot)
    assert argv[argv.index("--cookies") + 1] == str(ck)
    _wait(client, pid)


def test_multiangle_saved_user_cookies(client):
    client.put("/api/me/youtube-cookies", json={"cookies_text": COOKIES})
    r = _create(client, 2)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    import highlights.app.backend.main as m
    proot = m.get_registry().get(pid).root
    ck = proot / "source" / "cookies.txt"
    assert ck.is_file() and ck.read_text() == COOKIES
    argv = _argv(proot)
    assert argv[argv.index("--cookies") + 1] == str(ck)
    _wait(client, pid)


def test_multiangle_upload(client, short_video):
    files = [
        ("files", ("cam0.mp4", short_video.read_bytes(), "video/mp4")),
        ("files", ("cam1.mp4", short_video.read_bytes(), "video/mp4")),
    ]
    data = {"labels": ["Main", "Far side"], "title": "two cams"}
    r = client.post("/api/projects/multiangle/upload", files=files, data=data)
    assert r.status_code == 200, r.text
    d = r.json()
    pid = d["id"]
    assert d["n_angles"] == 2
    assert d["source"]["angles"][1]["label"] == "Far side"

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    a0 = p.angle_dir(0) / "match.mp4"
    a1 = p.angle_dir(1) / "match.mp4"
    assert a0.is_file() and a1.is_file()
    assert a0.stat().st_size == short_video.stat().st_size

    argv = _argv(p.root)
    assert "--youtube-url" not in argv

    d = _wait(client, pid)
    assert d["pipeline_state"] == "done"

    # angle video endpoint streams; out-of-range 404
    r = client.get(scoped(pid, "/multiangle/angle/0/video"))
    assert r.status_code == 200
    assert int(r.headers["content-length"]) > 0
    assert client.get(scoped(pid, "/multiangle/angle/5/video")).status_code == 404


def test_multiangle_pipeline_run_uses_multiangle_runner(client):
    r = _create(client, 2)
    pid = r.json()["id"]
    _wait(client, pid)
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    argv_p = p.root / "multiangle" / "argv.json"
    first = argv_p.read_text()
    mtime = argv_p.stat().st_mtime

    r = client.post(scoped(pid, "/pipeline/run"), json={})
    assert r.status_code == 200, r.text
    for _ in range(50):
        if argv_p.is_file() and (argv_p.stat().st_mtime != mtime
                                 or argv_p.read_text() != first):
            break
        time.sleep(0.1)
    argv = json.loads(argv_p.read_text())
    assert "--force" not in argv
    _wait(client, pid)

    # busy -> 409 while a second run is in flight is covered implicitly;
    # single-camera projects still use HL_PIPELINE_CMD (existing tests)
