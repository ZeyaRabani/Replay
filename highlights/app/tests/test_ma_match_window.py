"""Match window (multiangle/cut_range.json) endpoint tests."""

import json

from conftest import scoped
from test_multiangle import _create


def test_match_window_put_get_clear(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    r = _create(client, 3)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    cr = p.multiangle_dir / "cut_range.json"

    # none set -> match_window null
    assert client.get(scoped(pid, "/multiangle")).json()["match_window"] is None

    # set
    r = client.put(scoped(pid, "/multiangle/match-window"),
                   json={"start": 1740.0, "end": 5640.0})
    assert r.status_code == 200, r.text
    assert r.json()["match_window"] == [1740.0, 5640.0]
    assert json.loads(cr.read_text()) == {"lo": 1740.0, "hi": 5640.0}
    assert client.get(scoped(pid, "/multiangle")).json()["match_window"] \
        == [1740.0, 5640.0]

    # validation
    assert client.put(scoped(pid, "/multiangle/match-window"),
                      json={"start": 100.0, "end": 50.0}).status_code == 422
    assert client.put(scoped(pid, "/multiangle/match-window"),
                      json={"start": -1.0, "end": 50.0}).status_code == 422
    assert client.put(scoped(pid, "/multiangle/match-window"),
                      json={"start": 10.0}).status_code == 422

    # clear
    r = client.put(scoped(pid, "/multiangle/match-window"), json={})
    assert r.status_code == 200 and r.json()["match_window"] is None
    assert not cr.exists()


def test_match_window_at_create(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(sample_video))
    r = _create(client, 2, match_window=[60.0, 300.0])
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    assert json.loads((p.multiangle_dir / "cut_range.json").read_text()) \
        == {"lo": 60.0, "hi": 300.0}
    assert client.get(scoped(pid, "/multiangle")).json()["match_window"] \
        == [60.0, 300.0]

    # bad window at create
    r = _create(client, 2, match_window=[300.0, 60.0])
    assert r.status_code == 422
