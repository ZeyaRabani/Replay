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


def test_direct_suggest_and_sessions(client, short_video, monkeypatch):
    """You-direct: suggest returns the busiest stretch + director rows;
    sessions save + compare against the fake director."""
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _multi_done(client)

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    fused = {"events": [
        {"t": 10.0, "type": "shot", "status": "pending",
         "confidence": 0.9, "id": "e1"},
        {"t": 15.0, "type": "goal", "status": "confirmed",
         "confidence": 0.5, "id": "e2"}]}
    (p.multiangle_dir / "fused_candidates.json").write_text(
        json.dumps(fused))

    # suggest: confirmed goal at t=15 -> stretch [0,20] clamped (fake
    # director covers shared 0..20, no meta.range)
    r = client.get(scoped(pid, "/multiangle/direct/suggest"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["candidate"]["t"] == 15.0 and d["candidate"]["type"] == "goal"
    assert d["t_start"] == 0.0 and d["t_end"] == 20.0
    assert d["t_start_out"] == 0.0
    assert len(d["offsets"]) == 2
    assert len(d["director"]) == 20          # per-second rows
    assert d["director"][0]["angle"] == 0
    assert d["director"][5]["angle"] == 1    # seg 4..8.5 -> angle 1

    director_path = p.multiangle_dir / "director.json"
    director = json.loads(director_path.read_text())
    director["replays"] = [{
        "t_src_start": 0.0, "t_src_end": 6.0, "speed": 0.5,
        "t_live_at": 2.0, "t_out_start": 2.0, "t_out_end": 14.0,
    }]
    director_path.write_text(json.dumps(director))

    # arbitrary stretch
    r = client.get(scoped(pid, "/multiangle/direct/suggest"),
                   params={"t_start": 4.0, "t_end": 9.0})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["candidate"] is None
    assert d["t_start_out"] == 16.0
    assert [row["t"] for row in d["director"]] == [4, 5, 6, 7, 8]
    assert [row["rule"] for row in d["director"]] == \
        ["ball"] * 5                       # seg boundary at t=4
    # outside the cut span -> 422
    r = client.get(scoped(pid, "/multiangle/direct/suggest"),
                   params={"t_start": 0.0, "t_end": 99.0})
    assert r.status_code == 422

    # save a session: pick angle 0 all stretch -> disagrees where dir!=0
    r = client.post(scoped(pid, "/multiangle/direct/sessions"),
                    json={"t_start": 4.0, "t_end": 9.0,
                          "choices": [{"t": 4, "angle": 0}]})
    assert r.status_code == 200, r.text
    body = r.json()
    cmp = body["comparison"]
    assert cmp["n_seconds"] == 5
    assert cmp["agreement_pct"] == 0.0       # director picked angle 1 all 5 s
    assert cmp["disagree_by_rule"]["ball"]["n"] == 5
    sdir = p.multiangle_dir / "manual_direct"
    saved = list(sdir.glob("*.json"))
    assert len(saved) == 1

    # validation: bad angle, t outside, empty choices
    r = client.post(scoped(pid, "/multiangle/direct/sessions"),
                    json={"t_start": 4.0, "t_end": 9.0,
                          "choices": [{"t": 4, "angle": 7}]})
    assert r.status_code == 422
    r = client.post(scoped(pid, "/multiangle/direct/sessions"),
                    json={"t_start": 4.0, "t_end": 9.0,
                          "choices": [{"t": 99, "angle": 0}]})
    assert r.status_code == 422
    r = client.post(scoped(pid, "/multiangle/direct/sessions"),
                    json={"t_start": 4.0, "t_end": 9.0, "choices": []})
    assert r.status_code == 422

    # list sessions
    r = client.get(scoped(pid, "/multiangle/direct/sessions"))
    assert r.status_code == 200
    sess = r.json()["sessions"]
    assert len(sess) == 1
    assert sess[0]["comparison"]["agreement_pct"] == 0.0
    assert sess[0]["t_start"] == 4.0


def test_direct_suggest_excludes_saved_and_learn(client, short_video, monkeypatch):
    """suggest skips candidates inside saved sessions; learn 409 with
    none, saves params + returns result with a cheap replay."""
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _multi_done(client)

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    fused = {"events": [
        {"t": 10.0, "type": "goal", "status": "confirmed",
         "confidence": 0.9, "id": "e1"},
        {"t": 15.0, "type": "shot", "status": "pending",
         "confidence": 0.5, "id": "e2"}]}
    (p.multiangle_dir / "fused_candidates.json").write_text(
        json.dumps(fused))

    # suggest picks the confirmed goal at t=10 -> stretch [0,20]
    r = client.get(scoped(pid, "/multiangle/direct/suggest"))
    assert r.json()["candidate"]["t"] == 10.0
    assert r.json()["n_sessions_saved"] == 0

    # learn with no sessions -> 409
    r = client.post(scoped(pid, "/multiangle/direct/learn"), json={})
    assert r.status_code == 409

    # save a session covering the goal's stretch -> suggest falls back
    # to the pending shot at t=15 (the only uncovered candidate)
    r = client.post(scoped(pid, "/multiangle/direct/sessions"),
                    json={"t_start": 4.0, "t_end": 12.0,
                          "choices": [{"t": 4, "angle": 0}]})
    assert r.status_code == 200, r.text
    r = client.get(scoped(pid, "/multiangle/direct/suggest"))
    d = r.json()
    assert d["n_sessions_saved"] == 1
    assert d["candidate"]["t"] == 15.0

    # learn with a cheap replay: best overrides saved, result returned
    fake = {"best": {"min_hold": 3},
            "agreement_pct_before": 50.0, "agreement_pct_after": 80.0,
            "n_sessions": 1, "n_seconds": 8, "grid": []}
    monkeypatch.setattr(m, "_run_direct_learn", lambda p_, s_: fake)
    r = client.post(scoped(pid, "/multiangle/direct/learn"), json={})
    assert r.status_code == 200, r.text
    assert r.json()["learn"]["best"]["min_hold"] == 3
    assert json.loads(
        (p.multiangle_dir / "director_params.json").read_text()
    )["style_overrides"] == {"min_hold": 3}
    assert (p.multiangle_dir / "direct_learn.json").exists()
    r = client.get(scoped(pid, "/multiangle/direct/learn"))
    assert r.status_code == 200
    assert r.json()["agreement_pct_after"] == 80.0


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
    poly = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    # v2 body: per-angle keyframes
    zones = {"version": 2,
             "angles": [[{"t": 30.0, "zones": [poly]},
                         {"t": 1500.0, "zones": [poly]}], []]}
    r = client.put(scoped(pid, "/multiangle/zones"), json=zones)
    assert r.status_code == 200, r.text
    r = client.get(scoped(pid, "/multiangle/zones"))
    assert r.status_code == 200
    got = r.json()
    assert got["version"] == 2
    assert got["angles"][0][0]["t"] == 30.0
    assert got["angles"][0][0]["zones"][0][0] == [0.1, 0.1]
    assert len(got["angles"][0]) == 2

    # legacy body (flat polys + ref_t) -> stored as v2
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[poly], []], "ref_t": [45.0, None]})
    assert r.status_code == 200, r.text
    got = client.get(scoped(pid, "/multiangle/zones")).json()
    assert got["version"] == 2
    assert got["angles"][0] == [{"t": 45.0, "zones": [poly]}]

    # wrong number of angle entries -> 422
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[]]})
    assert r.status_code == 422
    # polygon with <3 points -> 422 (inside a v2 keyframe)
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[{"t": 0, "zones": [[[0, 0], [1, 1]]]}],
                                    []]})
    assert r.status_code == 422
    # coord out of range -> 422
    r = client.put(scoped(pid, "/multiangle/zones"),
                   json={"angles": [[{"t": 0, "zones":
                                      [[[0, 0], [1.5, 0], [1, 1]]]}], []]})
    assert r.status_code == 422


def test_recut_window_writes_cut_range(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _multi_done(client)

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    cr = p.multiangle_dir / "cut_range.json"

    # window in current video output time -> absolute shared-T
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [1.0, 4.0]})
    assert r.status_code == 200, r.text
    d = json.loads(cr.read_text())
    # fake sync.json has no coverage.union -> current_lo falls back to 0
    assert d == {"lo": 1.0, "hi": 4.0}
    _wait(client, pid)

    # out-of-duration window -> 422 and file unchanged
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [1.0, 99.0]})
    assert r.status_code == 422
    assert json.loads(cr.read_text()) == {"lo": 1.0, "hi": 4.0}

    # window must fit the CURRENT cut's span (hi-lo = 3 s), not the
    # source video's length: [1, 5] is inside the video but extends
    # past the windowed cut's end -> 422 (was wrongly accepted when
    # dur came from the source video)
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [1.0, 5.0]})
    assert r.status_code == 422
    assert json.loads(cr.read_text()) == {"lo": 1.0, "hi": 4.0}

    # no window = "the whole current video": the existing range is
    # kept so the re-cut covers the same span, not the union
    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "normal"})
    assert r.status_code == 200, r.text
    assert json.loads(cr.read_text()) == {"lo": 1.0, "hi": 4.0}
    _wait(client, pid)

    # with no range active, a no-window re-cut leaves cut_range absent
    cr.unlink()
    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "normal"})
    assert r.status_code == 200, r.text
    assert not cr.exists()


def test_recut_preview_saves_and_restores_range(client, short_video,
                                                monkeypatch):
    """preview=True stashes the prior cut_range in cut_range_base.json;
    the next non-preview recut restores it (or deletes it) and removes
    the base file."""
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _multi_done(client)

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    cr = p.multiangle_dir / "cut_range.json"
    base = p.multiangle_dir / "cut_range_base.json"

    # --- with a pre-existing cut_range --------------------------------
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [1.0, 4.0]})
    assert r.status_code == 200, r.text
    assert json.loads(cr.read_text()) == {"lo": 1.0, "hi": 4.0}
    _wait(client, pid)

    # preview a 2 s stretch of the CURRENT cut ([1,4] shared-T): base
    # remembers {1,4}, cr becomes {1,3}
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [0.0, 2.0],
                          "preview": True})
    assert r.status_code == 200, r.text
    assert json.loads(base.read_text()) == {"lo": 1.0, "hi": 4.0}
    assert json.loads(cr.read_text()) == {"lo": 1.0, "hi": 3.0}
    _wait(client, pid)

    # a second preview does NOT overwrite the base
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [0.0, 1.0],
                          "preview": True})
    assert r.status_code == 200, r.text
    assert json.loads(base.read_text()) == {"lo": 1.0, "hi": 4.0}
    _wait(client, pid)

    # non-preview recut restores the stashed range and removes base
    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "normal"})
    assert r.status_code == 200, r.text
    assert json.loads(cr.read_text()) == {"lo": 1.0, "hi": 4.0}
    assert not base.exists()
    _wait(client, pid)

    # --- without a pre-existing cut_range -----------------------------
    cr.unlink()
    r = client.post(scoped(pid, "/multiangle/recut"),
                    json={"style": "fast", "window": [0.0, 2.0],
                          "preview": True})
    assert r.status_code == 200, r.text
    assert json.loads(base.read_text()) == {"none": True}
    assert json.loads(cr.read_text()) == {"lo": 0.0, "hi": 2.0}
    _wait(client, pid)

    # non-preview recut: base "none" -> cut_range deleted again
    r = client.post(scoped(pid, "/multiangle/recut"), json={"style": "normal"})
    assert r.status_code == 200, r.text
    assert not cr.exists()
    assert not base.exists()
