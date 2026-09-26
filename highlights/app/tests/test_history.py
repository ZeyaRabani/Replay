"""History/archive v1: schema, events, upsert, delete->archive, restart."""

import json

from conftest import new_project

from highlights.app.backend import history


def _evts(client, pid):
    r = client.get(f"/api/history/{pid}/events")
    assert r.status_code == 200, r.text
    return r.json()


def test_schema_idempotent_and_log_order(client):
    history.init_db()
    history.init_db()   # idempotent
    history.log("m1", "created", title="x")
    history.log("m1", "zones_saved")
    history.log("m1", "deleted")
    evts = history.events("m1")
    assert [e["kind"] for e in evts] == ["deleted", "zones_saved", "created"]
    assert evts[-1]["detail"]["title"] == "x"


def test_backfill_and_created_event(client, short_video):
    # create a real project via the API (records created+source_added hooks)
    r = client.post("/api/projects",
                    json={"path": str(short_video), "run_pipeline": False})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    evts = _evts(client, pid)
    kinds = {e["kind"] for e in evts}
    assert {"created", "source_added"} <= kinds
    ms = client.get("/api/history").json()
    mine = [m for m in ms if m["id"] == pid]
    assert mine and mine[0]["deleted"] is False
    assert mine[0]["mode"] == "single"
    # full record via history.get_match
    full = history.get_match(pid)
    assert full["record"]["sources"]["kind"] == "path"


def test_delete_archives_and_streams(client, short_video):
    pid = new_project(client, short_video)
    # give it a kept artefact: a fake reel + the match.json survive
    from highlights.app.backend.main import get_registry
    p = get_registry().get(pid)
    (p.root / "match.mp4").write_bytes(b"fakecut")
    (p.root / "renders").mkdir(exist_ok=True)
    (p.root / "renders" / "reel.mp4").write_bytes(b"fakereel")
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 204
    # match row is deleted-flagged and carries artefacts
    ms = client.get("/api/history?include_deleted=1").json()
    m = next(m for m in ms if m["id"] == pid)
    assert m["deleted"] is True
    assert set(m["artefacts"]) >= {"match.mp4", "project.json", "reel.mp4"}
    # artefact stream
    r = client.get(f"/api/history/{pid}/artefacts/match.mp4")
    assert r.status_code == 200 and r.content == b"fakecut"
    r = client.get(f"/api/history/{pid}/artefacts/../project.json")
    assert r.status_code in (404, 422)
    # deleted event logged
    assert "deleted" in {e["kind"] for e in _evts(client, pid)}


def test_restart_multiangle(client, short_video):
    body = {
        "title": "MA", "angles": [
            {"url": "https://youtu.be/aaa", "label": "A"},
            {"url": "https://youtu.be/bbb", "label": "B"},
        ],
    }
    r = client.post("/api/projects/multiangle", json=body)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    from highlights.app.backend.main import get_registry
    p = get_registry().get(pid)
    import highlights.io
    p.multiangle_dir.mkdir(parents=True, exist_ok=True)
    highlights.io.write_json_atomic(
        p.multiangle_dir / "match_window_src.json",
        {"angle": 0, "start": 10.0, "end": 20.0}, indent=1)
    highlights.io.write_json_atomic(
        p.multiangle_dir / "zones.json",
        {"version": 2,
         "angles": [[{"t": 0.0, "zones": [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]]}],
                    []]}, indent=1)
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    # restart creates a new project with the same sources/window/zones
    r = client.post(f"/api/history/{pid}/restart")
    assert r.status_code == 200, r.text
    new_id = r.json()["id"]
    assert new_id != pid
    p2 = get_registry().get(new_id)
    src = p2.source_info["angles"]
    assert [a["url"] for a in src] == ["https://youtu.be/aaa",
                                     "https://youtu.be/bbb"]
    mw = json.loads((p2.multiangle_dir / "match_window_src.json").read_text())
    assert mw["start"] == 10.0
    z = json.loads((p2.multiangle_dir / "zones.json").read_text())
    assert z["version"] == 2 and z["angles"][0][0]["zones"]
    evs = {e["kind"] for e in _evts(client, new_id)}
    assert "created" in evs


def test_owner_scoping(client, short_video):
    r = client.post("/api/projects",
                    json={"path": str(short_video), "run_pipeline": False})
    pid = r.json()["id"]
    client.post("/api/users", json={"name": "other"})
    c2 = client.__class__(client.app)
    c2.headers["X-User"] = "other"
    assert c2.get(f"/api/history/{pid}/events").status_code == 404
    assert c2.post(f"/api/history/{pid}/restart").status_code == 404
    ids = {m["id"] for m in c2.get("/api/history").json()}
    assert pid not in ids


def test_archive_keeps_active_cut_video_only(client, short_video):
    """Two cut versions: small files archived for both; only the ACTIVE
    version's mp4 survives (as the root match.mp4 hardlink)."""
    import os

    from highlights.app.backend.main import get_registry
    body = {"title": "MA2", "angles": [
        {"url": "https://youtu.be/aaa", "label": "A"},
        {"url": "https://youtu.be/bbb", "label": "B"}]}
    pid = client.post("/api/projects/multiangle", json=body).json()["id"]
    p = get_registry().get(pid)
    cuts = p.multiangle_dir / "cuts"
    for cid, active in (("cutA", False), ("cutB", True)):
        d = cuts / cid
        d.mkdir(parents=True)
        v = d / "match.mp4"
        v.write_bytes(b"video-" + cid.encode())
        for j in ("meta", "director", "probe", "stats"):
            (d / f"{j}.json").write_text(json.dumps({"cut": cid, "j": j}))
    # root match.mp4 = hardlink of the ACTIVE cut's video (same inode)
    (p.root / "match.mp4").unlink(missing_ok=True)
    os.link(cuts / "cutB" / "match.mp4", p.root / "match.mp4")
    (cuts / "active.json").write_text(json.dumps({"id": "cutB"}))
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    arch = history.archive_root() / pid
    # small files kept for both versions
    for cid in ("cutA", "cutB"):
        for j in ("meta", "director", "probe", "stats"):
            assert (arch / "cuts" / cid / f"{j}.json").is_file()
    # only one mp4 kept, and it's the active cut's bytes
    mp4s = list(arch.rglob("*.mp4"))
    assert [f.relative_to(arch).as_posix() for f in mp4s] == ["match.mp4"]
    assert (arch / "match.mp4").read_bytes() == b"video-cutB"
    # record.cuts reflects versions + video retention
    m = history.get_match(pid)
    rec_cuts = {c["id"]: c for c in m["record"]["cuts"]}
    assert rec_cuts["cutB"]["active"] and rec_cuts["cutB"]["archived_video"]
    assert not rec_cuts["cutA"]["archived_video"]
    # artefact_path: one level under cuts/ allowed, traversal blocked
    assert history.artefact_path(pid, "cuts/cutA/meta.json") is not None
    assert history.artefact_path(pid, "cuts/../meta.json") is None
    assert history.artefact_path(pid, "cuts/../../x") is None
    r = client.get(f"/api/history/{pid}/artefacts/cuts/cutA/meta.json")
    assert r.status_code == 200
