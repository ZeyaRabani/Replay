"""Manual merge/split/detach/assign of player identities."""

import json

import pytest

from highlights.analysis.identity import edit_identities, identity_tracks


def _walk(tid, team, start, end, x):
    n = max(1, round((end - start) / 0.5) + 1)
    return {"id": tid, "team": team, "start": start, "end": end,
            "xy": [[x, 20.0]] * n, "dist_m": 10.0, "sprints": 0,
            "crops": [f"v2_{tid}_0.jpg", f"v2_{tid}_1.jpg"]}


def _v2(tmp_path):
    v2 = tmp_path / "players_v2"
    (v2 / "crops").mkdir(parents=True)
    tracks = [_walk(1, "A", 0.0, 30.0, 10.0),
              _walk(2, "A", 32.0, 60.0, 10.5),
              _walk(3, "A", 62.0, 90.0, 11.0),
              _walk(4, "A", 92.0, 120.0, 11.5),
              _walk(5, "B", 0.0, 60.0, 50.0),
              _walk(6, None, 0.0, 10.0, 30.0)]
    (v2 / "tracks.json").write_text(json.dumps(
        {"step": 0.5, "t0": 0.0, "tracks": tracks,
         "summary": {"n_tracks": 6, "visible_hist": [1] * 241}}))
    doc = {"identities": [
        {"id": "A1", "team": "A", "role": "outfield", "name": "Nine",
         "track_ids": [1, 2, 3], "unit_ids": [1, 2, 3],
         "coverage_s": 88.0, "coverage_pct": 73.3},
        {"id": "A2", "team": "A", "role": "outfield", "name": None,
         "track_ids": [4], "unit_ids": [4],
         "coverage_s": 28.0, "coverage_pct": 23.3},
        {"id": "B1", "team": "B", "role": "outfield", "name": None,
         "track_ids": [5], "unit_ids": [5],
         "coverage_s": 60.0, "coverage_pct": 50.0}],
        "unassigned_track_ids": [6],
        "quality": {"team_size": 11, "per_team": {}},
        "window": [0.0, 120.0]}
    (v2 / "identities.json").write_text(json.dumps(doc))
    (v2 / "names.json").write_text(json.dumps(
        {"names": {"A1": {"name": "Nine", "track_ids": [1, 2, 3]}}}))
    return v2


def _doc(v2):
    return json.loads((v2 / "identities.json").read_text())


def _by_id(doc):
    return {i["id"]: i for i in doc["identities"]}


def test_split_creates_lowest_free_id(tmp_path):
    v2 = _v2(tmp_path)
    doc = edit_identities(v2, {"op": "split", "iid": "A1",
                               "track_ids": [3]})
    ids = _by_id(doc)
    assert "A3" in ids  # A1, A2 taken -> A3
    assert ids["A1"]["track_ids"] == [1, 2]
    assert ids["A3"]["track_ids"] == [3] and ids["A3"]["team"] == "A"
    assert ids["A3"]["name"] is None and ids["A3"]["n_links"] is None
    assert ids["A1"]["coverage_s"] > ids["A3"]["coverage_s"] > 0
    assert doc["quality"]["per_team"]["A"]["n_identities"] == 3
    edits = json.loads((v2 / "identity_edits.json").read_text())
    assert edits[-1]["op"] == "split" and "at" in edits[-1]
    # names.json track_ids shrink with the parent
    names = json.loads((v2 / "names.json").read_text())
    assert names["names"]["A1"]["track_ids"] == [1, 2]


def test_split_rejects_bad_subsets(tmp_path):
    v2 = _v2(tmp_path)
    with pytest.raises(ValueError):
        edit_identities(v2, {"op": "split", "iid": "A1",
                             "track_ids": [3, 4]})
    with pytest.raises(ValueError):
        edit_identities(v2, {"op": "split", "iid": "A1",
                             "track_ids": [1, 2, 3]})
    with pytest.raises(ValueError):
        edit_identities(v2, {"op": "split", "iid": "Z9",
                             "track_ids": [1]})


def test_merge_unions_and_keeps_name(tmp_path):
    v2 = _v2(tmp_path)
    doc = edit_identities(v2, {"op": "merge", "into": "A2",
                               "from": "A1"})
    ids = _by_id(doc)
    assert "A1" not in ids
    assert ids["A2"]["track_ids"] == [1, 2, 3, 4]
    assert ids["A2"]["name"] == "Nine"  # into's name empty -> from's
    names = json.loads((v2 / "names.json").read_text())
    assert names["names"]["A2"]["name"] == "Nine"
    assert "A1" not in names["names"]
    with pytest.raises(ValueError):
        edit_identities(v2, {"op": "merge", "into": "A2",
                             "from": "B1"})


def test_detach_moves_to_unassigned_and_deletes_empty(tmp_path):
    v2 = _v2(tmp_path)
    doc = edit_identities(v2, {"op": "detach", "iid": "A1",
                               "track_ids": [1]})
    ids = _by_id(doc)
    assert ids["A1"]["track_ids"] == [2, 3]
    assert doc["unassigned_track_ids"] == [1, 6]
    # detaching everything deletes the card
    doc = edit_identities(v2, {"op": "detach", "iid": "A2",
                               "track_ids": [4]})
    assert "A2" not in _by_id(doc)
    assert doc["unassigned_track_ids"] == [1, 4, 6]


def test_assign_pulls_from_unassigned(tmp_path):
    v2 = _v2(tmp_path)
    doc = edit_identities(v2, {"op": "assign", "iid": "B1",
                               "track_ids": [6]})
    ids = _by_id(doc)
    assert ids["B1"]["track_ids"] == [5, 6]
    assert doc["unassigned_track_ids"] == []
    with pytest.raises(ValueError):
        edit_identities(v2, {"op": "assign", "iid": "B1",
                             "track_ids": [99]})
    with pytest.raises(ValueError):
        edit_identities(v2, {"op": "bogus", "iid": "B1"})


def test_identity_tracks_lists_members_and_unassigned(tmp_path):
    v2 = _v2(tmp_path)
    items = identity_tracks(v2, "A1")
    assert [t["id"] for t in items] == [1, 2, 3]
    assert items[0]["crop"] == "v2_1_1.jpg"  # middle of 2 crops
    assert items[0]["dur_s"] == 30.0 and items[0]["n_crops"] == 2
    assert [t["id"] for t in identity_tracks(v2, "unassigned")] == [6]
    with pytest.raises(KeyError):
        identity_tracks(v2, "Z9")
