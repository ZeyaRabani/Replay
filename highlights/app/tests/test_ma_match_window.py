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

    # angle omitted -> resolves to the longest angle (fake per-angle
    # durations: a1 is longest below)
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
