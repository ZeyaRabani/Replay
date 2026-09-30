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
    assert d["roster"] == {"players": [], "scorers": {},
                           "hidden_tracklet_ids": []}
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


def _players_done(client, monkeypatch):
    monkeypatch.setenv("HL_PLAYERS_CMD", f"{sys.executable} {FAKE_PLAYERS}")
    pid = _done_multiangle(client)
    _write_teams(client, pid)
    r = client.post(scoped(pid, "/analyse/players"), json={})
    assert r.status_code == 200, r.text
    d = _wait_players(client, pid)
    assert d["status"]["state"] == "done"
    return pid


def test_player_paths_endpoint(client, monkeypatch):
    pid = _players_done(client, monkeypatch)
    d = client.get(scoped(pid, "/analysis/players/paths")).json()
    assert d["fps"] == 1 and d["window_shared"] == [0.0, 20.0]
    assert d["pitch_len_m"] == 100.0
    tracks = {t["id"]: t for t in d["tracks"]}
    assert set(tracks) == {1, 2, 3, 4}
    assert all(not t["hidden"] and t["player_id"] is None
               for t in tracks.values())
    # 1 sample/s dedup: fake paths have samples 0.5 s apart
    assert [p[0] for p in tracks[1]["pts"]] == [0.0, 1.0, 1.5]
    assert all(len(p) == 3 for t in tracks.values() for p in t["pts"])

    roster = {"players": [{"id": "p1", "name": "Nine", "team": "A",
                           "tracklet_ids": [1]}],
              "scorers": {}, "hidden_tracklet_ids": [4]}
    assert client.put(scoped(pid, "/analysis/players/roster"),
                      json=roster).status_code == 200
    d = client.get(scoped(pid, "/analysis/players/paths")).json()
    tracks = {t["id"]: t for t in d["tracks"]}
    assert tracks[1]["player_id"] == "p1"
    assert tracks[4]["hidden"] is True


def test_radar_pitch_endpoints(client, monkeypatch):
    pid = _players_done(client, monkeypatch)
    d = client.get(scoped(pid, "/analysis/radar/pitch")).json()
    assert d == {"corners": None, "t": None}
    r = client.put(scoped(pid, "/analysis/radar/pitch"),
                   json={"corners": [[0.1, 0.9], [0.9, 0.9],
                                     [0.95, 0.1], [0.05, 0.1]],
                         "t": 12.5})
    assert r.status_code == 200, r.text
    d = client.get(scoped(pid, "/analysis/radar/pitch")).json()
    assert d["corners"][2] == [0.95, 0.1] and d["t"] == 12.5
    for bad in ({"corners": [[0.5, 0.5]] * 3, "t": 0.0},
                {"corners": [[0.0, 0.0], [2.2, 0.5], [0.5, 0.5],
                             [0.1, 0.1]], "t": 0.0}):
        assert client.put(scoped(pid, "/analysis/radar/pitch"),
                          json=bad).status_code == 422


def test_calib_endpoints(client):
    pid = _done_multiangle(client)
    d = client.get(scoped(pid, "/analysis/calib/landmarks")).json()
    assert d["pitch"]["len_m"] == 100.0 and d["pitch"]["wid_m"] == 64.0
    assert d["pitch"]["template"] == "full"
    lm = {l["name"]: [l["x"], l["y"]] for l in d["landmarks"]}
    assert "corner_near_left" in lm
    assert next(l for l in d["landmarks"]
                if l["name"] == "corner_near_left"
                )["label"] == "Corner - near left"
    # 6 landmarks through a known pitch->frame H
    import numpy as np
    Hk = np.array([[400.0, 30.0, 100.0], [20.0, 500.0, 200.0],
                   [0.0005, -0.0002, 1.0]])
    Hi = np.linalg.inv(Hk)

    def fx_fy(name):
        v = Hi @ np.array([*lm[name], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    names = ["corner_near_left", "corner_far_right", "halfway_far",
             "centre_spot", "pen_spot_l", "six_r_edge_near"]
    pts = [{"name": n, "fx": fx_fy(n)[0], "fy": fx_fy(n)[1]}
           for n in names]
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"angles": {"0": {"pts": pts}}})
    assert r.status_code == 200, r.text
    a0 = r.json()["angles"]["0"]
    assert a0["rms_m"] < 1e-4 and len(a0["H"]) == 3
    d2 = client.get(scoped(pid, "/analysis/calib")).json()
    assert d2["angles"]["0"]["rms_m"] == a0["rms_m"]
    # 3 pts -> 422
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"angles": {"1": {"pts": pts[:3]}}})
    assert r.status_code == 422
    # empty pts removes the angle
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"angles": {"0": {"pts": []}}})
    assert "0" not in r.json()["angles"]


def test_calib_pitch_endpoints(client):
    pid = _done_multiangle(client)
    # default pitch echoes full template + dims
    d = client.get(scoped(pid, "/analysis/calib")).json()
    assert d["pitch"]["template"] == "full"
    assert client.get(scoped(pid, "/analysis/calib/landmarks")
                      ).json()["pitch"]["template"] == "full"
    # pitch-only PUT persists and leaves angles untouched
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"pitch": {"template": "small", "len_m": 70,
                                   "wid_m": 45, "goal_w_m": 3.66,
                                   "d_radius_m": 9}})
    assert r.status_code == 200, r.text
    p = r.json()["pitch"]
    assert p["template"] == "small" and p["len_m"] == 70
    assert p["goal_w_m"] == 3.66 and p["d_radius_m"] == 9.0
    # landmarks endpoint now serves the small table
    lm = client.get(scoped(pid, "/analysis/calib/landmarks")).json()
    assert lm["pitch"]["template"] == "small"
    assert "d_l_apex" in {l["name"] for l in lm["landmarks"]}
    # solve with small names through a known H
    import numpy as np
    lmxy = {l["name"]: [l["x"], l["y"]] for l in lm["landmarks"]}
    Hk = np.array([[400.0, 30.0, 100.0], [20.0, 500.0, 200.0],
                   [0.0005, -0.0002, 1.0]])
    Hi = np.linalg.inv(Hk)
    names = ["corner_near_left", "corner_far_right", "halfway_far",
             "centre_spot", "d_l_apex", "d_r_near"]
    pts = []
    for n in names:
        v = Hi @ np.array([*lmxy[n], 1.0])
        pts.append({"name": n, "fx": float(v[0] / v[2]),
                    "fy": float(v[1] / v[2])})
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"angles": {"0": {"pts": pts}}})
    assert r.status_code == 200, r.text
    assert r.json()["angles"]["0"]["rms_m"] < 1e-4
    # pitch survives alongside angles
    assert r.json()["pitch"]["template"] == "small"
    # pitch-only PUT leaves angles intact
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"pitch": {"len_m": 71}})
    assert r.status_code == 200 and "0" in r.json()["angles"]
    assert r.json()["pitch"]["len_m"] == 71.0
    # validation: bad template / out-of-range dims
    for bad in ({"pitch": {"template": "tiny"}},
                {"pitch": {"len_m": 10}},
                {"pitch": {"wid_m": 200}},
                {"pitch": {"d_radius_m": 50}},
                {"pitch": {"goal_w_m": 1}}):
        assert client.put(scoped(pid, "/analysis/calib"),
                          json=bad).status_code == 422
    # full-template landmark rejected while small is saved
    pts.append({"name": "pen_spot_l", "fx": 0.5, "fy": 0.5})
    r = client.put(scoped(pid, "/analysis/calib"),
                   json={"angles": {"1": {"pts": pts}}})
    assert r.status_code == 422


def test_calib_cameras_endpoints(client):
    pid = _done_multiangle(client)
    d = client.get(scoped(pid, "/analysis/calib")).json()
    assert d["cameras"] == {}
    r = client.put(scoped(pid, "/analysis/calib/cameras"),
                   json={"cameras": {
                       "0": {"x_m": 50.0, "y_m": -10.0, "dir_deg": 90.0},
                       "2": {"x_m": -5.0, "y_m": 32.0, "dir_deg": 0.0}}})
    assert r.status_code == 200, r.text
    cams = r.json()["cameras"]
    assert cams["0"] == {"x_m": 50.0, "y_m": -10.0, "dir_deg": 90.0}
    assert cams["2"]["dir_deg"] == 0.0
    d = client.get(scoped(pid, "/analysis/calib")).json()
    assert d["cameras"]["0"]["x_m"] == 50.0
    # missing fields / out of range -> 422
    for bad in ({"cameras": {"1": {"x_m": 10.0, "y_m": 10.0}}},
                {"cameras": {"1": {"x_m": 200.0, "y_m": 10.0,
                                   "dir_deg": 0.0}}},
                {"cameras": {"1": {"x_m": 10.0, "y_m": -50.0,
                                   "dir_deg": 0.0}}},
                {"cameras": {"x": {"x_m": 10.0, "y_m": 10.0,
                                   "dir_deg": 0.0}}},
                {"cameras": "nope"}):
        assert client.put(scoped(pid, "/analysis/calib/cameras"),
                          json=bad).status_code == 422


def test_players_v2_endpoints(client):
    import highlights.app.backend.main as m
    pid = _done_multiangle(client)
    p = m.get_registry().get(pid)
    # no calib -> 409
    r = client.post(scoped(pid, "/analysis/players/v2/run"), json={})
    assert r.status_code == 409
    # seed calib + a v2 tracks.json
    (p.root / "multiangle" / "calib.json").write_text(json.dumps(
        {"angles": {"0": {"pts": [], "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          "rms_m": 0.1}},
         "pitch": {"len_m": 100.0, "wid_m": 64.0}}))
    v2dir = p.root / "analysis" / "players_v2"
    v2dir.mkdir(parents=True)
    (v2dir / "tracks.json").write_text(json.dumps({
        "step": 0.5, "t0": 0.0,
        "tracks": [{"id": 1, "team": "A", "start": 0.0, "end": 5.0,
                    "xy": [[10.0, 20.0], [10.2, 20.0], [None, None],
                           [10.6, 20.0]],
                    "dist_m": 0.6, "sprints": 0,
                    "crops": ["v2_1_0.jpg"]}],
        "summary": {"n_tracks": 1, "median_visible": 1.0,
                    "mean_len_s": 5.0, "visible_hist": [1, 1, 1, 1]},
        "ball": [[1.0, 50.0, 30.0]]}))
    d = client.get(scoped(pid, "/analysis/players/v2/tracks")).json()
    assert d["summary"]["n_tracks"] == 1
    assert d["tracks"][0]["crops"] == [
        f"/api/projects/{pid}/analysis/players/v2/crops/v2_1_0.jpg"]
    # paths in pitch space
    d = client.get(scoped(pid, "/analysis/players/paths")).json()
    assert d["space"] == "pitch"
    assert d["pitch"] == {"len_m": 100.0, "wid_m": 64.0}
    assert d["visible_hist"] == [1, 1, 1, 1]
    assert d["ball"] == [[1.0, 50.0, 30.0]]
    assert d["tracks"][0]["pts"][0] == [0.0, 10.0, 20.0]
    # roster naming accepts v2 ids
    r = client.put(scoped(pid, "/analysis/players/roster"),
                   json={"players": [{"id": "p1", "name": "Nine",
                                      "team": "A", "tracklet_ids": [1]}],
                         "scorers": {}})
    assert r.status_code == 200, r.text


def test_ball_path_rows_format():
    from highlights.app.backend.players_api import _ball_path
    cols = ["t", "ball_conf", "ball_x", "ball_y"]
    feats = {"columns": cols,
             "rows": [[0.0, 0.1, 0.5, 0.5],   # conf too low
                      [1.0, 0.6, 0.4, 0.3],
                      [2.0, 0.7, 0.0, 0.3],   # x=0: skip
                      [3.0, 0.9, 0.8, 0.85]]}
    assert _ball_path(feats, 10.0) == [[11.0, 0.4, 0.3],
                                       [13.0, 0.8, 0.85]]
    assert _ball_path({"columns": ["t"], "rows": []}, 0.0) == []


def test_player_identities_endpoints(client):
    import highlights.app.backend.main as m
    pid = _done_multiangle(client)
    p = _write_teams(client, pid)
    assert client.get(scoped(pid, "/players/identities")).status_code == 404
    v2dir = p.root / "analysis" / "players_v2"
    v2dir.mkdir(parents=True)

    def walk(tid, start, end, x):
        n = round((end - start) / 0.5) + 1
        return {"id": tid, "team": "A", "start": start, "end": end,
                "xy": [[x, 20.0]] * n, "dist_m": 0.0, "sprints": 0,
                "crops": [f"v2_{tid}_0.jpg"]}
    (v2dir / "tracks.json").write_text(json.dumps({
        "step": 0.5, "t0": 0.0,
        "tracks": [walk(1, 0.0, 30.0, 10.0), walk(2, 32.0, 60.0, 10.5)],
        "summary": {"n_tracks": 2, "visible_hist": [1] * 121}}))
    r = client.post(scoped(pid, "/players/identities/rebuild"), json={})
    assert r.status_code == 200, r.text
    d = client.get(scoped(pid, "/players/identities")).json()
    assert [i["id"] for i in d["identities"]] == ["A1"]
    ident = d["identities"][0]
    assert ident["track_ids"] == [1, 2]
    assert all(c.startswith(f"/api/projects/{pid}/analysis/players/v2/crops/")
               for c in ident["crops"])
    assert d["teams"]["A"]["hex"] == "#f08c00"

    r = client.put(scoped(pid, "/players/identities/A1"), json={"name": "Nine"})
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "A1", "name": "Nine"}
    names = json.loads((v2dir / "names.json").read_text())
    assert names["names"]["A1"]["name"] == "Nine"
    assert client.get(scoped(pid, "/players/identities")
                      ).json()["identities"][0]["name"] == "Nine"
    # survives an offline re-link
    client.post(scoped(pid, "/players/identities/rebuild"), json={})
    assert client.get(scoped(pid, "/players/identities")
                      ).json()["identities"][0]["name"] == "Nine"
    # radar paths carry the identity
    paths = client.get(scoped(pid, "/analysis/players/paths")).json()
    assert {t["identity_id"] for t in paths["tracks"]} == {"A1"}

    assert client.put(scoped(pid, "/players/identities/A9"),
                      json={"name": "x"}).status_code == 404
    assert client.put(scoped(pid, "/players/identities/bad-id"),
                      json={"name": "x"}).status_code == 404
    assert client.put(scoped(pid, "/players/identities/A1"),
                      json={"name": "x" * 61}).status_code == 422
    # another user cannot see it
    other = client.get(scoped(pid, "/players/identities"),
                       headers={"X-User": "someone-else"})
    assert other.status_code in (401, 403, 404)
    assert m.get_registry().get(pid) is not None


def test_identity_reel_endpoints(client):
    import subprocess as sp
    pid = _done_multiangle(client)
    p = _write_teams(client, pid)
    v2dir = p.root / "analysis" / "players_v2"
    v2dir.mkdir(parents=True)
    xs = [10.0] * 20 + [10.0 + 3.5 * k for k in range(1, 7)]
    xs += [xs[-1]] * 20
    track = {"id": 1, "team": "A", "start": 5.0,
             "end": 5.0 + 0.5 * (len(xs) - 1),
             "xy": [[x, 20.0] for x in xs], "dist_m": 0.0, "sprints": 0,
             "crops": []}
    (v2dir / "tracks.json").write_text(json.dumps({
        "step": 0.5, "t0": 0.0, "tracks": [track], "ball": [],
        "summary": {"n_tracks": 1, "visible_hist": [1] * 60}}))
    (p.root / "multiangle").mkdir(exist_ok=True)
    (p.root / "multiangle" / "cut_range.json").write_text(
        json.dumps({"lo": 0.0, "hi": 40.0}))
    assert client.post(scoped(pid, "/players/identities/rebuild"),
                       json={}).status_code == 200
    reel = scoped(pid, "/players/identities/A1/reel")
    assert client.get(reel).status_code == 404
    assert client.post(scoped(pid, "/players/identities/B4/reel")
                       ).status_code == 404
    match = p.root / "match.mp4"
    if match.is_symlink() or match.exists():
        match.unlink()
    assert client.post(reel).status_code == 409
    sp.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc=duration=30:size=160x120:rate=10", "-c:v", "libx264",
            "-preset", "ultrafast", str(match)], check=True)
    r = client.post(reel)
    assert r.status_code == 200, r.text
    assert r.json()["status"]["state"] in ("queued", "running", "done")
    deadline = time.time() + 30
    d = {}
    while time.time() < deadline:
        d = client.get(reel).json()
        if d["status"]["state"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert d["status"]["state"] == "done", d
    assert d["url"] == f"/api/projects/{pid}/players/identities/A1/reel.mp4"
    assert d["manifest"]["items"][0]["type"] == "sprint"
    mp4 = client.get(d["url"], params={"user": "tester"})
    assert mp4.status_code == 200
    assert mp4.headers["content-type"] == "video/mp4"
    other = client.get(reel, headers={"X-User": "someone-else"})
    assert other.status_code in (401, 403, 404)
