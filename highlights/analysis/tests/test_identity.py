"""identity tests: whole-match linking respects team / overlap / team-size
constraints, aggregates member-track stats and keeps names across
re-runs."""

import itertools
import json

import numpy as np
import pytest

from highlights.analysis import identity as ident
from highlights.analysis.identity import (
    build_identities,
    link_identities,
    motion_stats,
    reattach_names,
    set_name,
)


def _walk(tid, team, start, end, x0, y0, vx=0.0, vy=0.0, crops=None):
    """Track moving at constant velocity from (x0, y0) over [start, end]."""
    n = round((end - start) / 0.5) + 1
    xy = [[round(x0 + vx * k * 0.5, 3), round(y0 + vy * k * 0.5, 3)]
          for k in range(n)]
    return {"id": tid, "team": team, "start": start, "end": end, "xy": xy,
            "dist_m": 0.0, "sprints": 0, "crops": crops or []}


def test_links_continuation_not_far_jump():
    # player 1 runs right and disappears at t=20 near x=30; track 2
    # reappears 3 s later near x=34 (plausible), track 3 at x=5 (not).
    tracks = [
        _walk(1, "A", 0.0, 20.0, 10.0, 20.0, vx=1.0),
        _walk(2, "A", 23.0, 60.0, 34.0, 20.0, vx=0.2),
        _walk(3, "A", 23.0, 60.0, 5.0, 40.0),
    ]
    doc = link_identities(tracks, team_size=2, merge=False,
                          window=(0.0, 60.0))
    owner = {t: i["id"] for i in doc["identities"] for t in i["track_ids"]}
    assert owner[1] == owner[2]
    assert owner[3] != owner[1]


def test_never_crosses_teams_and_overlap():
    tracks = [
        _walk(1, "A", 0.0, 20.0, 10.0, 20.0),
        _walk(2, "B", 21.0, 40.0, 10.0, 20.0),     # same spot, other team
        _walk(3, "A", 10.0, 30.0, 10.5, 20.0),     # overlaps 1 by 10 s
        _walk(4, None, 0.0, 40.0, 24.0, 10.0),     # referee / unknown
    ]
    doc = link_identities(tracks, team_size=11, merge=False,
                          window=(0.0, 40.0))
    for i in doc["identities"]:
        teams = {t["team"] for t in tracks if t["id"] in i["track_ids"]}
        assert teams == {i["team"]}
        assert not {1, 3} <= set(i["track_ids"])
    assert 4 in doc["unassigned_track_ids"]
    assert doc["quality"]["overlap_violations"] == 0


def test_team_size_cap_holds_at_every_second():
    rng = np.random.default_rng(0)
    tracks = []
    tid = 1
    # 6 simultaneous players per team, each fragmented into 3 pieces
    for team in ("A", "B"):
        for p in range(6):
            y = 5.0 + 6.0 * p
            for k, (s, e) in enumerate([(0, 30), (32, 70), (73, 100)]):
                x = 10.0 + rng.normal(0, 0.2) + k * 0.5
                tracks.append(_walk(tid, team, float(s), float(e), x, y))
                tid += 1
    # extra short noise fragments overlapping everything
    for _ in range(10):
        s = float(rng.integers(0, 90))
        tracks.append(_walk(tid, "A", s, s + 8.0, 40.0, 20.0))
        tid += 1
    doc = link_identities(tracks, team_size=4, merge=False,
                          window=(0.0, 100.0))
    q = doc["quality"]
    assert q["max_concurrent"]["A"] <= 4
    assert q["max_concurrent"]["B"] <= 4
    assert sum(1 for i in doc["identities"] if i["team"] == "A") <= 4
    # chains are overlap-free
    by = {t["id"]: t for t in tracks}
    for i in doc["identities"]:
        ivs = sorted((by[t]["start"], by[t]["end"]) for t in i["track_ids"])
        for (_a0, a1), (b0, _b1) in itertools.pairwise(ivs):
            assert a1 - b0 < 1.0
    # with enough capacity the 6 fragmented players become 6 identities
    doc = link_identities(tracks, team_size=6, merge=False,
                          window=(0.0, 100.0))
    b = [i for i in doc["identities"] if i["team"] == "B"]
    assert len(b) == 6
    assert all(len(i["track_ids"]) == 3 for i in b)


def test_aggregation_sums_member_stats():
    # 10 s at 2 m/s then 10 s gap then 10 s at 7 m/s (sprint)
    t1 = _walk(1, "A", 0.0, 10.0, 0.0, 10.0, vx=2.0)
    t2 = _walk(2, "A", 12.0, 22.0, 22.0, 10.0, vx=7.0)
    doc = link_identities([t1, t2], team_size=1, merge=False,
                          window=(0.0, 22.0))
    (i,) = doc["identities"]
    assert i["track_ids"] == [1, 2]
    assert i["first_s"] == 0.0 and i["last_s"] == 22.0
    assert i["coverage_s"] == pytest.approx(20.0)
    assert i["coverage_pct"] == pytest.approx(100 * 20 / 22, abs=0.1)
    # distance: ~20 m + ~70 m, the unobserved gap contributes nothing
    assert i["distance_m"] == pytest.approx(90.0, rel=0.08)
    assert i["top_speed_ms"] == pytest.approx(7.0, abs=0.2)
    assert i["sprints"] == 1
    assert 12.0 <= i["sprint_events"][0]["t"] <= 22.0


def test_motion_stats_ignores_glitches():
    xy = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [30.0, 0.0], [1.5, 0.0],
          [2.0, 0.0], [None, None], [2.5, 0.0]]
    ms = motion_stats(xy)
    assert ms["top_speed_ms"] < ident.GLITCH_MS
    assert ms["sprints"] == 0


def _write_v2(tmp_path, tracks):
    v2 = tmp_path / "players_v2"
    (v2 / "crops").mkdir(parents=True)
    (v2 / "tracks.json").write_text(json.dumps(
        {"step": 0.5, "t0": 0.0, "tracks": tracks,
         "summary": {"n_tracks": len(tracks),
                     "visible_hist": [1] * 201}}))
    return v2


def _two_players():
    return [
        _walk(1, "A", 0.0, 40.0, 10.0, 10.0),
        _walk(2, "A", 43.0, 100.0, 10.5, 10.0),
        _walk(3, "A", 0.0, 60.0, 30.0, 30.0),
        _walk(4, "A", 62.0, 100.0, 30.0, 30.5),
    ]


def test_build_writes_doc_and_names_survive_rerun(tmp_path):
    v2 = _write_v2(tmp_path, _two_players())
    doc = build_identities(v2, team_size=2, log=lambda _m: None)
    assert (v2 / "identities.json").is_file()
    assert len(doc["identities"]) == 2
    for key in ("id", "team", "track_ids", "first_s", "last_s",
                "coverage_s", "distance_m", "sprints", "top_speed_ms",
                "crops", "name"):
        assert key in doc["identities"][0]
    iid = next(i["id"] for i in doc["identities"] if 3 in i["track_ids"])
    set_name(v2, iid, "Keeper Kim")
    saved = json.loads((v2 / "names.json").read_text())
    assert saved["names"][iid]["name"] == "Keeper Kim"
    on_disk = json.loads((v2 / "identities.json").read_text())
    assert next(i for i in on_disk["identities"]
                if i["id"] == iid)["name"] == "Keeper Kim"

    # re-run with player 2 now longer (gets the lower id): the name
    # follows the tracks, not the id
    tracks = _two_players()
    tracks[0] = _walk(1, "A", 0.0, 42.0, 10.0, 10.0)
    tracks[1] = _walk(2, "A", 45.0, 100.0, 10.5, 10.0)
    tracks[2] = _walk(3, "A", 20.0, 60.0, 30.0, 30.0)
    v2b = _write_v2(tmp_path / "b", tracks)
    (v2b / "names.json").write_text((v2 / "names.json").read_text())
    doc2 = build_identities(v2b, team_size=2, log=lambda _m: None)
    named = [i for i in doc2["identities"] if i["name"]]
    assert len(named) == 1
    assert 3 in named[0]["track_ids"]

    # clearing removes it
    set_name(v2b, named[0]["id"], "")
    assert json.loads((v2b / "names.json").read_text())["names"] == {}


def test_reattach_one_name_per_identity():
    idents = [{"id": "A1", "track_ids": [1, 2, 3]},
              {"id": "A2", "track_ids": [4]}]
    names = {"names": {"A1": {"name": "x", "track_ids": [1, 2]},
                       "A7": {"name": "y", "track_ids": [3, 4]}}}
    out = reattach_names(idents, names)
    assert out == {"A1": "x", "A2": "y"}


def test_cli(tmp_path, capsys):
    v2 = _write_v2(tmp_path, _two_players())
    assert ident.main([str(v2), "--team-size", "2"]) == 0
    assert json.loads((v2 / "identities.json").read_text())["identities"]
    assert ident.main([str(tmp_path / "nope")]) == 1


# ---------- user anchors (shirt numbers) ----------

def test_anchor_different_numbers_never_link():
    # track 1 ends at t=20 near x=30; both continuations are plausible,
    # but only track 2 carries the same anchored number
    tracks = [
        _walk(1, "A", 0.0, 20.0, 10.0, 20.0, vx=1.0),
        _walk(2, "A", 23.0, 60.0, 34.0, 20.0, vx=0.2),
        _walk(3, "A", 23.0, 60.0, 33.0, 20.0, vx=0.2),
    ]
    anchors = {1: ("A", 7), 2: ("A", 7), 3: ("A", 9)}
    doc = link_identities(tracks, team_size=3, merge=False,
                          window=(0.0, 60.0), anchors=anchors)
    owner = {t: i["id"] for i in doc["identities"] for t in i["track_ids"]}
    assert owner[1] == owner[2]
    assert owner[3] != owner[1]
    assert next(i for i in doc["identities"] if 1 in i["track_ids"])["number"] == 7


def test_anchor_same_number_bridges_long_gap():
    # 100 s apart is beyond MAX_GAP_S: no flow edge, but the shared
    # number concatenates the chains post-flow
    tracks = [
        _walk(1, "A", 0.0, 30.0, 10.0, 20.0),
        _walk(2, "A", 140.0, 200.0, 15.0, 25.0),
        _walk(3, "A", 0.0, 200.0, 40.0, 10.0),
    ]
    anchors = {1: ("A", 7), 2: ("A", 7)}
    doc = link_identities(tracks, team_size=3, merge=False,
                          window=(0.0, 200.0), anchors=anchors)
    ident = next(i for i in doc["identities"] if i["number"] == 7)
    assert set(ident["track_ids"]) == {1, 2}
    assert ident["anchored"] is True


def test_anchored_short_track_still_assigned():
    tracks = [
        _walk(1, "A", 0.0, 2.0, 10.0, 20.0),   # < MIN_TRACK_S
        _walk(2, "A", 10.0, 60.0, 12.0, 20.0),
    ]
    doc = link_identities(tracks, team_size=2, merge=False,
                          window=(0.0, 60.0), anchors={1: ("A", 5)})
    assert 1 not in doc["unassigned_track_ids"]
    assert next(i for i in doc["identities"] if 1 in i["track_ids"])["number"] == 5
