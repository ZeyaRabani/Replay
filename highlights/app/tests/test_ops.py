"""Tests for match-window editing, recut, purge-sources, storage, trim."""

import json
import time

from conftest import new_project, scoped

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


def _multi_done(client):
    angles = [{"url": f"https://youtu.be/a{i}", "label": f"Cam {i}"}
              for i in range(2)]
    r = client.post("/api/projects/multiangle",
                    json={"angles": angles})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    _wait(client, pid)
    return pid


def test_match_window_get_put(client, sample_video):
    pid = new_project(client, sample_video)
    r = client.get(scoped(pid, "/match-window"))
    assert r.status_code == 200
    win = r.json()["match_window"]
    assert win[0] == 0.0 and win[1] > 0

    r = client.put(scoped(pid, "/match-window"),
                   json={"start_s": 1.0, "end_s": 3.0})
    assert r.status_code == 200, r.text
    assert r.json()["match_window"] == [1.0, 3.0]
    r = client.get(scoped(pid, "/match-window"))
    assert r.json()["match_window"] == [1.0, 3.0]

    # invalid windows -> 422
    r = client.put(scoped(pid, "/match-window"),
                   json={"start_s": 5.0, "end_s": 3.0})
    assert r.status_code == 422
    r = client.put(scoped(pid, "/match-window"),
                   json={"start_s": -1.0, "end_s": 3.0})
    assert r.status_code == 422
    r = client.put(scoped(pid, "/match-window"),
                   json={"start_s": 0.0, "end_s": 10 ** 9})
    assert r.status_code == 422


def test_recut_spawns_with_style(client, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", "")
    pid = _multi_done(client)

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    argv_p = p.root / "multiangle" / "argv.json"
    first = argv_p.read_text()

    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "fast"})
    assert r.status_code == 200, r.text
    for _ in range(50):
        if argv_p.is_file() and argv_p.read_text() != first:
            break
        time.sleep(0.1)
    argv = json.loads(argv_p.read_text())
    assert argv[argv.index("--style") + 1] == "fast"
    assert argv[argv.index("--stages") + 1] == "director,render,fuse"
    assert "--force" in argv
    assert p.meta["cut_style"] == "fast"
    _wait(client, pid)

    # invalid style -> 422
    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "turbo"})
    assert r.status_code == 422


def test_purge_sources(client):
    pid = _multi_done(client)
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    # give the angles fake video files to delete
    for i in range(2):
        v = p.angle_dir(i) / "match.mp4"
        v.write_bytes(b"fake-video")
    r = client.post(scoped(pid, "/purge-sources"))
    assert r.status_code == 200, r.text
    assert r.json()["sources_purged"] is True
    for i in range(2):
        assert p.angle_video(i) is None
    info = client.get(scoped(pid, "/multiangle")).json()
    assert info["sources_purged"] is True
    # recut now blocked
    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "fast"})
    assert r.status_code == 409


def test_storage_endpoint(client, sample_video):
    pid = new_project(client, sample_video)
    r = client.get("/api/storage")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total_bytes"] > 0 and d["free_bytes"] >= 0
    row = next(x for x in d["per_project"] if x["id"] == pid)
    assert row["bytes"] >= 0 and "sources_bytes" in row


def test_trim_endpoints(client, short_video):
    pid = new_project(client, short_video)
    r = client.post(scoped(pid, "/video/trim"),
                    json={"start_s": 0.5, "end_s": 2.0})
    assert r.status_code == 200, r.text
    # poll until ready (short video -> quick)
    deadline = time.time() + 20
    while time.time() < deadline:
        st = client.get(scoped(pid, "/video/trim/status"),
                        params={"start": 0.5, "end": 2.0}).json()
        if st.get("ready"):
            break
        time.sleep(0.2)
    assert st.get("ready"), st
    r = client.get(scoped(pid, "/video/trimmed.mp4"),
                   params={"start": 0.5, "end": 2.0})
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    # invalid window -> 422
    r = client.post(scoped(pid, "/video/trim"),
                    json={"start_s": 5.0, "end_s": 1.0})
    assert r.status_code == 422


def test_admin_shared_cookies(client):
    # tester is not in the default admin list (REPLAY_ADMINS default "john")
    r = client.put("/api/me/youtube-cookies",
                   json={"cookies_text": COOKIES, "share": True})
    assert r.status_code == 403
    st = client.get("/api/me/youtube-cookies").json()
    assert st["is_admin"] is False and st["shared_available"] is False


def test_zones_roundtrip_and_validation(client):
    pid = _multi_done(client)
    zones = {"angles": [[[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]], []],
             "ref_t": [30.0, None]}
    r = client.put(scoped(pid, "/multiangle/zones"), json=zones)
    assert r.status_code == 200, r.text
    r = client.get(scoped(pid, "/multiangle/zones"))
    assert r.status_code == 200
    assert r.json()["angles"][0][0][0] == [0.1, 0.1]
    assert r.json()["ref_t"][0] == 30.0

    # wrong number of angle entries -> 422
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[]]})
    assert r.status_code == 422
    # polygon with <3 points -> 422
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[[[0, 0], [1, 1]]], []]})
    assert r.status_code == 422
    # coord out of range -> 422
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[[[0, 0], [1.5, 0], [1, 1]]], []]})
    assert r.status_code == 422
