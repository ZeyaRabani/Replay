"""Match window (match_window_src.json / cut_range.json) endpoint tests."""

import json

from conftest import scoped
from test_multiangle import _create, _wait


def test_match_window_pre_sync_only_writes_src(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    monkeypatch.setenv("FAKE_MA_SLEEP", "0.5")
    r = _create(client, 3)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    src = p.multiangle_dir / "match_window_src.json"

    # set while the pipeline hasn't reached sync yet
    r = client.put(scoped(pid, "/multiangle/match-window"),
                   json={"angle": 2, "start": 100.0, "end": 200.0})
    assert r.status_code == 200, r.text
    assert json.loads(src.read_text()) == \
        {"angle": 2, "start": 100.0, "end": 200.0}
    assert r.json()["match_window_src"]["angle"] == 2

    # validation
    assert client.put(scoped(pid, "/multiangle/match-window"),
                      json={"start": 100.0, "end": 50.0}).status_code == 422
    assert client.put(scoped(pid, "/multiangle/match-window"),
                      json={"angle": 9, "start": 1.0, "end": 5.0}).status_code == 422
    assert client.put(scoped(pid, "/multiangle/match-window"),
                      json={"start": 10.0}).status_code == 422

    _wait(client, pid)


def test_match_window_post_sync_writes_both(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    r = _create(client, 3)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    d = _wait(client, pid)
    assert d["pipeline_state"] == "done"

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    src = p.multiangle_dir / "match_window_src.json"
    cr = p.multiangle_dir / "cut_range.json"

    # known offsets so the conversion is deterministic
    (p.multiangle_dir / "sync.json").write_text(
        json.dumps({"offsets": [0.0, 5.0, -748.0], "coverage": {"union": [0.0, 6000.0]}}))

    r = client.put(scoped(pid, "/multiangle/match-window"),
                   json={"angle": 2, "start": 900.0, "end": 2000.0})
    assert r.status_code == 200, r.text
    assert json.loads(src.read_text()) == \
        {"angle": 2, "start": 900.0, "end": 2000.0}
    # shared-T = file_t + offset[2] = t - 748
    assert json.loads(cr.read_text()) == {"lo": 152.0, "hi": 1252.0}
    got = client.get(scoped(pid, "/multiangle")).json()
    assert got["match_window"] == [152.0, 1252.0]
    assert got["match_window_src"]["angle"] == 2

    # a changed window must invalidate director/fuse/render outputs —
    # director segments are output-time and would remap onto the new
    # union otherwise
    (p.multiangle_dir / "director.json").write_text("{}")
    (p.multiangle_dir / "fused_candidates.json").write_text("[]")
    (p.pipeline_dir / "candidates.json").write_text("[]")
    r = client.put(scoped(pid, "/multiangle/match-window"),
                   json={"angle": 2, "start": 950.0, "end": 2000.0})
    assert r.status_code == 200
    assert not (p.multiangle_dir / "director.json").exists()
    assert not (p.multiangle_dir / "fused_candidates.json").exists()
    assert not (p.pipeline_dir / "candidates.json").exists()
    # same window again -> no churn (files stay absent, no error)
    (p.multiangle_dir / "director.json").write_text("{}")
    client.put(scoped(pid, "/multiangle/match-window"),
               json={"angle": 2, "start": 950.0, "end": 2000.0})
    assert (p.multiangle_dir / "director.json").exists()
    (p.multiangle_dir / "director.json").unlink()

    # angle omitted with one angle unprobed -> stays null, no cut_range.
    # Make a2 look undownloaded: no status/probe and no video file
    cr.unlink(missing_ok=True)
    a2p = p.angle_dir(2)
    for f in (a2p / "pipeline" / "status.json",
              a2p / "pipeline" / "probe.json"):
        f.unlink(missing_ok=True)
    for f in a2p.glob("match.*"):
        f.unlink()
    r = client.put(scoped(pid, "/multiangle/match-window"),
                   json={"start": 900.0, "end": 2000.0})
    assert r.status_code == 200, r.text
    assert r.json()["match_window_src"]["angle"] is None
    assert r.json()["match_window"] is None
    assert not cr.exists()

    # all durations known -> resolves to the longest angle (a1 below)
    for i, dur in enumerate((5000.0, 6200.0, 4000.0)):
        pr = p.angle_dir(i) / "pipeline"
        pr.mkdir(parents=True, exist_ok=True)
        (pr / "status.json").write_text(json.dumps({"video": {"duration_s": dur}}))
    r = client.put(scoped(pid, "/multiangle/match-window"),
                   json={"start": 900.0, "end": 2000.0})
    assert r.status_code == 200, r.text
    assert r.json()["match_window_src"]["angle"] == 1
    # converted with offset[1] = +5
    assert json.loads(cr.read_text()) == {"lo": 905.0, "hi": 2005.0}

    # clear removes both
    r = client.put(scoped(pid, "/multiangle/match-window"), json={})
    assert r.status_code == 200 and r.json()["match_window"] is None
    assert not src.exists() and not cr.exists()
    assert client.get(scoped(pid, "/multiangle")).json()["match_window"] is None


def test_match_window_at_create(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    r = _create(client, 2, match_window=[60.0, 300.0], match_window_angle=1)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    assert json.loads((p.multiangle_dir / "match_window_src.json").read_text()) \
        == {"angle": 1, "start": 60.0, "end": 300.0}
    assert client.get(scoped(pid, "/multiangle")).json()["match_window_src"]["angle"] == 1

    # angle omitted at create -> stored as null (longest, resolved later)
    r = _create(client, 2, match_window=[60.0, 300.0])
    assert r.status_code == 200, r.text
    pid2 = r.json()["id"]
    p2 = m.get_registry().get(pid2)
    assert json.loads((p2.multiangle_dir / "match_window_src.json").read_text()) \
        == {"angle": None, "start": 60.0, "end": 300.0}

    # bad window / bad angle at create
    assert _create(client, 2, match_window=[300.0, 60.0]).status_code == 422
    assert _create(client, 2, match_window=[1.0, 60.0],
                   match_window_angle=5).status_code == 422


def test_put_angle_url_swaps_and_clears_outputs(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_SLEEP", "0.5")
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    pid = _create(client, 3).json()["id"]
    _wait(client, pid)
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    ma = p.multiangle_dir
    (ma / "sync.json").write_text(json.dumps(
        {"offsets": [0.0, 5.0, -748.0],
         "coverage": {"union": [0.0, 6000.0]}}))
    (ma / "director.json").write_text(json.dumps({"segments": [
        {"t_start": 0.0, "t_end": 5.0, "angle": 2},
        {"t_start": 5.0, "t_end": 10.0, "angle": 0}]}))
    (ma / "fused_candidates.json").write_text("[]")
    (ma / "concat.txt").write_text("x")
    adir = p.angle_dir(2)
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "match.mp4").write_bytes(b"v")
    # the stale cached segment for angle 2, plus one from angle 0
    from highlights.multiangle.render import plan_segments, seg_key
    (ma / "segs").mkdir(exist_ok=True)
    for pl in plan_segments(
            json.loads((ma / "director.json").read_text())["segments"],
            [0.0, 5.0, -748.0], 0.0, 6000.0, []):
        if not pl.get("skip"):
            (ma / "segs" /
             f"{seg_key(pl['angle'], pl['t_file'], pl['dur'])}.mp4"
             ).write_bytes(b"s")
    (ma / "mezz").mkdir(exist_ok=True)
    (ma / "mezz" / "2_abc123.mp4").write_bytes(b"m")
    (ma / "mezz" / "0_def456.mp4").write_bytes(b"m")

    r = client.put(scoped(pid, "/multiangle/angles/2"),
                   json={"url": "https://youtu.be/NEWURL"})
    assert r.status_code == 200, r.text
    assert json.loads((p.root / "project.json").read_text()
                      )["source"]["angles"][2]["url"] \
        == "https://youtu.be/NEWURL"
    assert not adir.exists()
    assert not (ma / "sync.json").exists()
    assert not (ma / "director.json").exists()
    assert not (ma / "fused_candidates.json").exists()
    # the whole seg cache + angle-2's mezzanine were evicted
    assert not (ma / "segs").exists()
    assert not (ma / "mezz" / "2_abc123.mp4").exists()
    assert (ma / "mezz" / "0_def456.mp4").exists()
    # 404 / 409
    assert client.put(scoped(pid, "/multiangle/angles/9"),
                      json={"url": "x"}).status_code == 404
    (p.multiangle_dir / "status.json").write_text(
        json.dumps({"state": "running", "stage": "track"}))
    assert client.put(scoped(pid, "/multiangle/angles/0"),
                      json={"url": "x"}).status_code == 409
