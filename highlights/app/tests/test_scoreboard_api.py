"""Score labels endpoint + scoreboard burn request."""

import json

from conftest import scoped
from test_multiangle import _create, _wait


def test_score_put_and_scoreboard(client, short_video, monkeypatch):
    monkeypatch.setenv("FAKE_MA_VIDEO", str(short_video))
    r = _create(client, 2, title="scoreboard")
    pid = r.json()["id"]
    _wait(client, pid)

    import highlights.app.backend.main as m
    from highlights.app.backend import pipeline

    p = m.get_registry().get(pid)
    assert (p.root / "match.mp4").is_file()

    # PUT validates
    assert client.put(scoped(pid, "/multiangle/score"),
                      json={"home_label": "", "away_label": "B"}) \
        .status_code == 422
    assert client.put(scoped(pid, "/multiangle/score"),
                      json={"home_label": "A", "away_label": "B",
                            "home_hex": "zzzzzz"}).status_code == 422
    assert client.put(scoped(pid, "/multiangle/score"),
                      json={"home_label": "x" * 21,
                            "away_label": "B"}).status_code == 422
    r = client.put(scoped(pid, "/multiangle/score"),
                   json={"home_label": "Reds", "away_label": "Blues",
                         "home_hex": "#aa1122", "away_hex": "2233cc"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["home"]["label"] == "Reds" and d["home"]["hex"] == "#aa1122"
    assert d["away"]["hex"] == "#2233cc"      # '#'-less input normalised
    saved = json.loads((p.multiangle_dir / "score.json").read_text())
    assert saved["home"] == {"label": "Reds", "hex": "#aa1122"}

    # confirm the fake goals: c001 home; c002 has no team -> unassigned
    # (c002 at t=12 is beyond the 6 s video so it is filtered from
    # GET /candidates, but PATCH still resolves it)
    r = client.patch(scoped(pid, "/candidates/c001"),
                     json={"status": "confirmed", "team": "home"})
    assert r.status_code == 200, r.text
    r = client.patch(scoped(pid, "/candidates/c002"),
                     json={"status": "confirmed"})
    assert r.status_code == 200, r.text

    # kickoff from pipeline/match_window.json when present
    (p.pipeline_dir / "match_window.json").write_text(
        json.dumps({"match_window": [2.0, 18.0]}))

    spawned = {}

    def fake_spawn(pp, **kw):
        spawned.update(kw)
        return {"state": "queued", "pid": None}

    monkeypatch.setattr(pipeline, "spawn_multiangle", fake_spawn)
    r = client.post(scoped(pid, "/multiangle/scoreboard"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["goals"] == 1 and d["unassigned"] == 1
    assert spawned["stages"] == ["scoreboard"] and spawned["force"] is True
    sb = json.loads((p.multiangle_dir / "scoreboard.json").read_text())
    assert sb["goals"] == [{"t": 5.0, "team": "home"}]
    assert sb["kickoff"] == 2.0
    assert sb["home"] == {"label": "Reds", "hex": "#aa1122"}
    # the pre-scoreboard cut is snapshotted as a version
    assert (p.multiangle_dir / "cuts").is_dir()


def test_scoreboard_409_without_match(client, short_video, monkeypatch):
    # no FAKE_MA_VIDEO -> fake runner finishes without writing match.mp4
    r = _create(client, 2, title="no cut yet")
    pid = r.json()["id"]
    _wait(client, pid)
    r = client.post(scoped(pid, "/multiangle/scoreboard"))
    assert r.status_code == 409, r.text
