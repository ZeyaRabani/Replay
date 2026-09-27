"""players_run orchestrator tests: roster reset on re-run."""

import json
from pathlib import Path

import highlights.analysis.players_run as prun


def _setup(tmp_path, monkeypatch):
    adir = tmp_path / "analysis"
    adir.mkdir()
    (adir / "teams.json").write_text(
        json.dumps({"teams": {"A": {}, "B": {}}}))
    pdir = adir / "players"
    pdir.mkdir()
    ctx = {
        "analysis_dir": adir,
        "video": tmp_path / "v.mp4",
        "window": [0.0, 20.0],
        "window_file": (0.0, 20.0),
        "proxy_offset": 0.0,
        "shared_offset": 0.0,
        "pitch_type": None,
    }
    monkeypatch.setattr(prun, "resolve_context", lambda _p: ctx)
    return pdir


def _fake_pass(out_dir):
    doc = {"tracklets": [], "n_frames": 0, "fps": 1.0}
    (Path(out_dir) / "tracklets.json").write_text(json.dumps(doc))
    return doc


def test_rerun_clears_stale_roster_ids(tmp_path, monkeypatch):
    pdir = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(prun, "run_players_pass",
                        lambda *a, **k: _fake_pass(a[1]))
    roster = {"players": [
        {"id": "p1", "name": "asim", "team": "A", "tracklet_ids": [7, 9]},
        {"id": "p2", "name": "bob", "team": "B", "tracklet_ids": [3]}],
        "scorers": {"c1": "p1"},
        "hidden_tracklet_ids": [12, 40]}
    (pdir / "roster.json").write_text(json.dumps(roster))
    (pdir / "tracklets.json").write_text("{}")   # existing → force needed

    prun.run_players(tmp_path, log=lambda _m: None, force=True)
    out = json.loads((pdir / "roster.json").read_text())
    assert [p["name"] for p in out["players"]] == ["asim", "bob"]
    assert all(p["tracklet_ids"] == [] for p in out["players"])
    assert out["players"][0]["team"] == "A"
    assert out["scorers"] == {"c1": "p1"}
    assert out["hidden_tracklet_ids"] == []


def test_skip_run_keeps_roster(tmp_path, monkeypatch):
    pdir = _setup(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(prun, "run_players_pass",
                        lambda *a, **k: called.append(1) or _fake_pass(a[1]))
    roster = {"players": [{"id": "p1", "name": "asim", "team": "A",
                           "tracklet_ids": [7]}],
              "scorers": {}, "hidden_tracklet_ids": [3]}
    (pdir / "roster.json").write_text(json.dumps(roster))
    (pdir / "tracklets.json").write_text(
        json.dumps({"tracklets": []}))

    prun.run_players(tmp_path, log=lambda _m: None, force=False)
    assert not called
    out = json.loads((pdir / "roster.json").read_text())
    assert out["players"][0]["tracklet_ids"] == [7]
    assert out["hidden_tracklet_ids"] == [3]
