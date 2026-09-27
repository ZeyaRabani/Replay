"""Players-analysis endpoint tests (fake runner via HL_PLAYERS_CMD)."""

import json
import sys
import time
from pathlib import Path

from conftest import new_project, scoped

FAKE_PLAYERS = Path(__file__).resolve().parent / "fake_players.py"

TEAMS = {"teams": {"A": {"name": "orange", "hex": "#f08c00",
                        "hsv": [10, 220, 230]},
                   "B": {"name": "white", "hex": "#ebebeb",
                         "hsv": [0, 25, 235]}}}


def _wait_pipeline(client, pid, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(scoped(pid, "")).json()
        if d["pipeline_state"] in ("done", "failed"):
            return d
        time.sleep(0.1)
    return client.get(scoped(pid, "")).json()


def _wait_players(client, pid, timeout=10.0):
    deadline = time.time() + timeout
    d = {}
    while time.time() < deadline:
        d = client.get(scoped(pid, "/analysis/players")).json()
        if (d.get("status") or {}).get("state") in ("done", "failed"):
            return d
        time.sleep(0.1)
    return d


def _done_multiangle(client):
    r = client.post("/api/projects/multiangle", json={
        "angles": [{"url": f"https://youtu.be/p{i}", "label": f"C{i}"}
                   for i in range(3)]})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    _wait_pipeline(client, pid)
    return pid


def _write_teams(client, pid):
    import highlights.app.backend.main as m
    p = m.get_registry().get(pid)
    adir = p.root / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "teams.json").write_text(json.dumps(TEAMS))
    return p


def test_analyse_players_not_multiangle_404(client, sample_video):
    pid = new_project(client, sample_video)
    r = client.post(scoped(pid, "/analyse/players"), json={})
    assert r.status_code == 404
    assert r.json()["detail"] == "not a multi-angle project"


def test_analyse_players_no_teams_409(client):
    pid = _done_multiangle(client)
    r = client.post(scoped(pid, "/analyse/players"), json={})
    assert r.status_code == 409
    assert "Team analysis" in r.json()["detail"]


def test_players_run_get_put_crops(client, monkeypatch):
    monkeypatch.setenv("HL_PLAYERS_CMD", f"{sys.executable} {FAKE_PLAYERS}")
    pid = _done_multiangle(client)
    p = _write_teams(client, pid)

    r = client.post(scoped(pid, "/analyse/players"), json={})
    assert r.status_code == 200, r.text
    assert r.json()["state"] in ("queued", "running")

    d = _wait_players(client, pid)
    assert d["status"]["state"] == "done"
    assert d["teams"]["A"]["name"] == "orange"
    assert d["teams"]["B"]["hex"] == "#ebebeb"
    # the 5 s tracklet (id 4) is filtered out of the naming list
    assert len(d["tracklets"]) == 3
    assert {t["id"] for t in d["tracklets"]} == {1, 2, 3}
    assert d["n_tracklets_total"] == 4
    assert d["n_shown"] == 3
    # sorted longest-first
    assert [t["id"] for t in d["tracklets"]] == [3, 2, 1]
    t0 = d["tracklets"][0]
    assert "path" not in t0 and t0["duration_s"] > 0
    assert "t_start_out" in t0 and "t_end_out" in t0
    assert d["roster"] == {"players": [], "scorers": {}}
    assert d["players_stats"]["unassigned"]["n_tracklets"] == 4
    assert d["estimate_min"] >= 1

    argv = json.loads((p.root / "analysis" / "players" / "argv.json")
                      .read_text())
    assert "--project-dir" in argv

    # roster PUT: valid
    cand_id = str(p.candidates[0].id) if p.candidates else "c1"
    roster = {"players": [{"id": "p1", "name": "Nine", "team": "A",
                           "tracklet_ids": [1, 2]},
                          {"id": "p2", "name": "Ten", "team": "B",
                           "tracklet_ids": [3, 4]}],
              "scorers": {}}
    r = client.put(scoped(pid, "/analysis/players/roster"), json=roster)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["roster"]["players"][0]["name"] == "Nine"
    assert body["players_stats"]["teams"]["A"]["distance_m"] == 85.0
    assert body["players_stats"]["unassigned"]["n_tracklets"] == 0

    # a roster-assigned tracklet is always shown, even when short
    d2 = client.get(scoped(pid, "/analysis/players")).json()
    assert {t["id"] for t in d2["tracklets"]} == {1, 2, 3, 4}
    assert d2["n_shown"] == 4 and d2["n_tracklets_total"] == 4

    # 422 cases
    bad_dup = {"players": [roster["players"][0], roster["players"][0]]}
    r = client.put(scoped(pid, "/analysis/players/roster"), json=bad_dup)
    assert r.status_code == 422
    bad_tid = {"players": [{"id": "p1", "name": "x", "tracklet_ids": [99]}]}
    r = client.put(scoped(pid, "/analysis/players/roster"), json=bad_tid)
    assert r.status_code == 422
    bad_scorer = {"players": roster["players"],
                  "scorers": {"no-such-candidate": "p1"}}
    r = client.put(scoped(pid, "/analysis/players/roster"), json=bad_scorer)
    assert r.status_code == 422
    if p.candidates:
        bad_scorer2 = {"players": roster["players"],
                       "scorers": {cand_id: "ghost"}}
        r = client.put(scoped(pid, "/analysis/players/roster"),
                       json=bad_scorer2)
        assert r.status_code == 422

    # crops: public, name-restricted
    r = client.get(scoped(pid, "/analysis/players/crops/1_0.jpg"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert client.get(
        scoped(pid, "/analysis/players/crops/1_9.jpg")).status_code == 404
    assert client.get(
        scoped(pid, "/analysis/players/crops/..%2F..%2Fx.jpg")
        ).status_code in (404, 422)
    assert client.get(
        scoped(pid, "/analysis/players/crops/9_0.jpg")).status_code == 404

    # history event
    evts = client.get(f"/api/history/{pid}/events").json()
    kinds = {e["kind"] for e in evts}
    assert "players_analysed" in kinds
