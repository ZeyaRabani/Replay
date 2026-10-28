"""Director-cut version snapshots (multiangle/cuts/)."""

import json
import os
import shutil

from conftest import scoped
from test_multiangle import _create, _wait


def test_cuts_snapshot_activate_delete(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    r = _create(client, 2, title="cuts")
    pid = r.json()["id"]
    _wait(client, pid)

    import highlights.app.backend.main as m
    from highlights.multiangle.cuts import snapshot_cut

    p = m.get_registry().get(pid)
    live = p.root / "match.mp4"
    assert live.is_file()

    # legacy project: first GET snapshots the current cut
    res = client.get(scoped(pid, "/multiangle/cuts"))
    assert res.status_code == 200, res.text
    d = res.json()
    assert len(d["cuts"]) == 1
    cut0 = d["cuts"][0]
    assert d["active"] == cut0["id"]
    assert cut0["label"] == "Fast (AI)"
    cdir = p.multiangle_dir / "cuts" / cut0["id"]
    assert os.stat(cdir / "match.mp4").st_ino == os.stat(live).st_ino

    # a second, different cut lands (simulate a re-cut): new video + director
    os.replace(live, live.with_name("match_old.mp4"))
    shutil.copyfile(short_video, live)
    (p.multiangle_dir / "director.json").write_text(
        json.dumps({"n_cuts": 999, "style": "fast", "ratios": {},
                    "angle_share": {}, "segments": []}))
    meta2 = snapshot_cut(p.root, "fast")
    assert meta2 and meta2["label"] == "Fast (AI)"
    # restore live state to the second cut's content (as a real recut would)
    cuts = client.get(scoped(pid, "/multiangle/cuts")).json()
    assert len(cuts["cuts"]) == 2
    assert cuts["active"] == meta2["id"]

    # activating a cut restores the range it was made with (written to
    # multiangle/cut_range.json, not inside cuts/)
    cr = p.multiangle_dir / "cut_range.json"
    cr.write_text('{"lo": 9.0, "hi": 99.0}')
    mpath = cdir / "meta.json"
    mm = json.loads(mpath.read_text()); mm["range"] = [5.0, 10.0]
    mpath.write_text(json.dumps(mm))
    r = client.post(scoped(pid, f"/multiangle/cuts/{cut0['id']}/activate"))
    assert r.status_code == 200, r.text
    assert r.json()["active"] == cut0["id"]
    assert json.loads(cr.read_text()) == {"lo": 5.0, "hi": 10.0}
    assert not (p.multiangle_dir / "cuts" / "cut_range.json").exists()
    assert os.stat(live).st_ino == os.stat(cdir / "match.mp4").st_ino
    dj = json.loads((p.multiangle_dir / "director.json").read_text())
    assert dj["n_cuts"] == 133          # fake runner's original director
    assert p.meta["cut_style"] == "fast"
    assert p.video is not None and p.video.path.endswith("match.mp4")

    # unknown cut -> 404; active cannot be deleted; non-active can
    assert client.post(scoped(pid, "/multiangle/cuts/nope/activate")) \
        .status_code == 404
    assert client.delete(scoped(pid, f"/multiangle/cuts/{cut0['id']}")) \
        .status_code == 409
    r = client.delete(scoped(pid, f"/multiangle/cuts/{meta2['id']}"))
    assert r.status_code == 200, r.text
    assert len(r.json()["cuts"]) == 1

    # download route serves the snapshot with a filename
    r = client.get(scoped(pid, f"/multiangle/cuts/{cut0['id']}/match.mp4"))
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")


def _mkcut(p, cid: str, director: dict | None = None,
           meta: dict | None = None, video: bool = True):
    cdir = p.multiangle_dir / "cuts" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    if video:
        (cdir / "match.mp4").write_bytes(b"vid-" + cid.encode())
    (cdir / "director.json").write_text(json.dumps(
        director or {"n_cuts": 7, "style": "fast", "segments": []}))
    m = {"id": cid, "label": f"L{cid}", "style": "fast",
         "created_at": 1.0}
    m.update(meta or {})
    (cdir / "meta.json").write_text(json.dumps(m))
    return cdir


def test_remove_cut_video_rules(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _create(client, 2, title="rmv").json()["id"]
    _wait(client, pid)
    import highlights.app.backend.main as m
    from highlights.multiangle.cuts import list_cuts, remove_cut_video

    p = m.get_registry().get(pid)
    cuts = p.multiangle_dir / "cuts"
    cuts.mkdir(parents=True, exist_ok=True)
    _mkcut(p, "cA")
    _mkcut(p, "cB")
    (cuts / "active.json").write_text(json.dumps({"id": "cA"}))

    # refuses the active cut
    import pytest
    with pytest.raises(ValueError, match="active"):
        remove_cut_video(p.root, "cA")

    # refuses a cut video that IS the live inode
    cdir_live = _mkcut(p, "cLive")
    (cdir_live / "match.mp4").unlink()
    os.link(p.root / "match.mp4", cdir_live / "match.mp4")
    with pytest.raises(ValueError, match="live match"):
        remove_cut_video(p.root, "cLive")

    # refuses unknown / bad ids
    with pytest.raises(ValueError):
        remove_cut_video(p.root, "nope")
    with pytest.raises(ValueError):
        remove_cut_video(p.root, "../x")

    # normal removal: meta.json written first, then unlink; second call 0
    freed = remove_cut_video(p.root, "cB")
    assert freed == len(b"vid-cB")
    cdir = cuts / "cB"
    assert not (cdir / "match.mp4").exists()
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["video_removed"]["bytes"] == freed
    assert remove_cut_video(p.root, "cB") == 0

    # list_cuts reports has_video/video_bytes
    lc = list_cuts(p.root)
    by_id = {c["id"]: c for c in lc["cuts"]}
    assert by_id["cB"]["has_video"] is False
    assert by_id["cA"]["has_video"] is True
    assert by_id["cA"]["video_bytes"] == len(b"vid-cA")


def test_activate_removed_cut_409(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _create(client, 2, title="act409").json()["id"]
    _wait(client, pid)
    import highlights.app.backend.main as m
    from highlights.multiangle.cuts import activate_cut, remove_cut_video

    p = m.get_registry().get(pid)
    # snapshot the live cut once so there is an active.json + a real cut
    client.get(scoped(pid, "/multiangle/cuts"))
    cuts_active = json.loads(
        (p.multiangle_dir / "cuts" / "active.json").read_text())["id"]
    _mkcut(p, "gone")
    live_dj = (p.multiangle_dir / "director.json").read_text()

    assert remove_cut_video(p.root, "gone") > 0
    assert activate_cut(p.root, "gone") is None
    r = client.post(scoped(pid, "/multiangle/cuts/gone/activate"))
    assert r.status_code == 409, r.text
    assert "rebuild" in r.json()["detail"]
    # live director.json untouched
    assert (p.multiangle_dir / "director.json").read_text() == live_dj
    # video delete route: active cut -> 409, unknown -> 404
    r = client.delete(scoped(
        pid, f"/multiangle/cuts/{cuts_active}/video"))
    assert r.status_code == 409
    assert client.delete(scoped(pid, "/multiangle/cuts/nope/video")) \
        .status_code == 404


def test_rebuild_cut_route(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    pid = _create(client, 2, title="rebuild").json()["id"]
    _wait(client, pid)
    import highlights.app.backend.main as m
    from highlights.app.backend import pipeline
    from highlights.multiangle.cuts import remove_cut_video

    p = m.get_registry().get(pid)
    _mkcut(
        p, "old",
        director={"n_cuts": 42, "style": "normal", "segments": []},
        meta={"range": [10.0, 100.0], "style": "normal"})
    remove_cut_video(p.root, "old")

    # 409 while the video still exists
    _mkcut(p, "keepvid")
    r = client.post(scoped(pid, "/multiangle/cuts/keepvid/rebuild"))
    assert r.status_code == 409
    # 404 unknown
    assert client.post(scoped(pid, "/multiangle/cuts/nope/rebuild")) \
        .status_code == 404

    # 409 when sources_purged
    p.meta["sources_purged"] = True
    r = client.post(scoped(pid, "/multiangle/cuts/old/rebuild"))
    assert r.status_code == 409
    del p.meta["sources_purged"]

    spawn_calls = []

    class FakeJob:
        id = "j1"
        state = "queued"

    def fake_spawn(*a, **k):
        spawn_calls.append(k)
        return {"job_id": "j1"}

    monkeypatch.setattr(pipeline, "spawn_multiangle", fake_spawn)
    r = client.post(scoped(pid, "/multiangle/cuts/old/rebuild"))
    assert r.status_code == 200, r.text
    assert spawn_calls and spawn_calls[-1]["stages"] == ["render", "fuse"]
    assert spawn_calls[-1]["force"] is True
    assert spawn_calls[-1]["style"] == "normal"

    # live director.json now equals the cut's; cut_range restored
    dj = json.loads((p.multiangle_dir / "director.json").read_text())
    assert dj["n_cuts"] == 42
    assert json.loads(
        (p.multiangle_dir / "cut_range.json").read_text()) == \
        {"lo": 10.0, "hi": 100.0}
    rb = json.loads(
        (p.multiangle_dir / "rebuild_of.json").read_text())
    assert rb == {"id": "old"}

    # a cut without range removes cut_range.json
    _mkcut(p, "norange", meta={})
    remove_cut_video(p.root, "norange")
    r = client.post(scoped(pid, "/multiangle/cuts/norange/rebuild"))
    assert r.status_code == 200, r.text
    assert not (p.multiangle_dir / "cut_range.json").exists()


def test_rebuild_snapshot_suffix(tmp_path, monkeypatch):
    """run.main with rebuild_of.json: snapshot label gains ' (rebuilt)',
    meta.rebuilt_from, and the marker file is consumed."""
    import json as _json
    from contextlib import nullcontext

    from highlights.multiangle import run

    proj = tmp_path / "proj"
    pipe = proj / "multiangle"
    (pipe / "cuts").mkdir(parents=True)
    (pipe / "rebuild_of.json").write_text(_json.dumps({"id": "oldcut"}))
    (pipe / "director.json").write_text(_json.dumps(
        {"n_cuts": 3, "style": "fast", "segments": []}))
    (proj / "match.mp4").write_bytes(b"video")
    angles = [{"label": "a0", "url": None,
               "dir": proj / "angles" / "a0"}]
    monkeypatch.setattr(run, "_load_angles", lambda *a, **k: angles)
    monkeypatch.setattr(run, "job_slot", lambda *a, **k: nullcontext())
    monkeypatch.setattr(run, "run_stages", lambda ctx, names: None)

    rc = run.main(["--project-dir", str(proj), "--stages", "render"])
    assert rc == 0
    assert not (pipe / "rebuild_of.json").exists()
    metas = [d for d in (pipe / "cuts").iterdir() if d.is_dir()]
    assert len(metas) == 1
    meta = _json.loads((metas[0] / "meta.json").read_text())
    assert meta["label"].endswith(" (rebuilt)")
    assert meta["rebuilt_from"] == "oldcut"
