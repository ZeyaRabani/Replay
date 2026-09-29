"""groups_v2 tests: fused tracks cluster by crop colour, never across
a >=1 s time overlap."""

import json

import numpy as np

from highlights.analysis.groups_v2 import build_groups_v2, merge_duplicates


def _tr(tid, team, start, end, x, y):
    """Track standing still at (x, y) for [start, end)."""
    n = max(1, round((end - start) / 0.5))
    return {"id": tid, "team": team, "start": start, "end": end,
            "xy": [[x, y]] * n, "dist_m": 0.0, "sprints": 0,
            "crops": []}


def _crop(path, bgr):
    import cv2
    img = np.full((40, 16, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_groups_v2_clusters_by_colour(tmp_path):
    v2 = tmp_path / "players_v2"
    crops = v2 / "crops"
    crops.mkdir(parents=True)
    red, blue = (0, 0, 255), (255, 0, 0)
    tracks = [
        # red team-A pair, disjoint in time -> one group
        {"id": 1, "team": "A", "start": 0.0, "end": 10.0,
         "dist_m": 100.0, "sprints": 1, "crops": ["v2_1_0.jpg"]},
        {"id": 2, "team": "A", "start": 20.0, "end": 30.0,
         "dist_m": 50.0, "sprints": 0, "crops": ["v2_2_0.jpg"]},
        # same colour but overlaps id 1 -> cannot share its group
        {"id": 3, "team": "A", "start": 5.0, "end": 15.0,
         "dist_m": 60.0, "sprints": 2, "crops": ["v2_3_0.jpg"]},
        # blue team-B pair, disjoint -> one group
        {"id": 4, "team": "B", "start": 0.0, "end": 8.0,
         "dist_m": 80.0, "sprints": 0, "crops": ["v2_4_0.jpg"]},
        {"id": 5, "team": "B", "start": 12.0, "end": 25.0,
         "dist_m": 40.0, "sprints": 3, "crops": ["v2_5_0.jpg"]},
        # red but team B -> never with the team-A reds
        {"id": 6, "team": "B", "start": 40.0, "end": 50.0,
         "dist_m": 30.0, "sprints": 0, "crops": ["v2_6_0.jpg"]},
    ]
    _crop(crops / "v2_1_0.jpg", red)
    _crop(crops / "v2_2_0.jpg", red)
    _crop(crops / "v2_3_0.jpg", red)
    _crop(crops / "v2_4_0.jpg", blue)
    _crop(crops / "v2_5_0.jpg", blue)
    _crop(crops / "v2_6_0.jpg", red)
    (v2 / "tracks.json").write_text(
        json.dumps({"step": 0.5, "t0": 0.0, "tracks": tracks,
                    "summary": {"n_tracks": 6}}))

    out = build_groups_v2(v2)
    groups = out["groups"]
    member_of = {tid: g for g in groups for tid in g["track_ids"]}

    # same-colour non-overlapping tracks land together
    assert member_of[1] is member_of[2]
    assert member_of[4] is member_of[5]
    # overlapping tracks never share a group
    assert member_of[1] is not member_of[3]
    # team never crossed
    assert member_of[1] is not member_of[6]
    # output shape
    assert out["n_tracks"] == 6
    assert out["n_grouped"] == 6
    g = member_of[1]
    assert g["id"] and g["team"] == "A"
    assert set(g) >= {"id", "team", "track_ids", "minutes", "dist_m",
                      "sprints", "crops", "start", "end"}
    assert g["minutes"] == round((10.0 + 10.0) / 60.0, 2)
    assert g["start"] == 0.0 and g["end"] == 30.0
    assert (v2 / "groups.json").exists()


def test_merge_duplicates_merges_nearby():
    a = _tr(1, "A", 0.0, 10.0, 10.0, 20.0)
    b = _tr(2, "A", 2.0, 8.0, 11.0, 20.0)      # 1 m away, inside a
    out = merge_duplicates([a, b])
    assert len(out) == 1
    m = out[0]
    assert m["id"] == 1                       # longest member's id
    assert m["member_ids"] == [1, 2]
    assert m["team"] == "A"
    assert m["start"] == 0.0 and m["end"] == 10.0
    # xy averaged over the overlap: (10+11)/2 at t=2..8, 10 outside
    assert m["xy"][0] == [10.0, 20.0]
    mid = round((4.0 - 0.0) / 0.5)
    assert m["xy"][mid] == [10.5, 20.0]


def test_merge_duplicates_constant_offset():
    # same player from two cameras: 6 m constant displacement
    a = _tr(1, "A", 0.0, 10.0, 10.0, 20.0)
    b = _tr(2, "A", 0.0, 10.0, 16.0, 20.0)
    out = merge_duplicates([a, b])
    assert len(out) == 1
    assert out[0]["member_ids"] == [1, 2]


def test_merge_duplicates_varying_offset():
    # 6 m apart but the offset swings >3 m -> different players
    a = _tr(1, "A", 0.0, 10.0, 10.0, 20.0)
    b = _tr(2, "A", 0.0, 10.0, 16.0, 20.0)
    for i, p in enumerate(b["xy"]):
        p[0] = 16.0 + (4.0 if i % 2 else -4.0)   # offset 2..10 m
    out = merge_duplicates([a, b])
    assert len(out) == 2


def test_merge_duplicates_keeps_distant():
    a = _tr(1, "A", 0.0, 10.0, 10.0, 20.0)
    b = _tr(2, "A", 0.0, 10.0, 22.0, 20.0)     # 12 m constant offset
    out = merge_duplicates([a, b])
    assert len(out) == 2


def test_merge_duplicates_never_crosses_teams():
    a = _tr(1, "A", 0.0, 10.0, 10.0, 20.0)
    b = _tr(2, "B", 0.0, 10.0, 10.5, 20.0)     # co-located, other team
    out = merge_duplicates([a, b])
    assert len(out) == 2


def test_group_track_ids_cover_merged_members(tmp_path):
    v2 = tmp_path / "players_v2"
    crops = v2 / "crops"
    crops.mkdir(parents=True)
    red = (0, 0, 255)
    # duplicates of the same player: 1 m apart, overlapping
    tracks = [
        dict(_tr(1, "A", 0.0, 60.0, 10.0, 20.0), crops=["v2_1_0.jpg"]),
        dict(_tr(2, "A", 0.0, 60.0, 11.0, 20.0), crops=["v2_2_0.jpg"]),
    ]
    _crop(crops / "v2_1_0.jpg", red)
    _crop(crops / "v2_2_0.jpg", red)
    (v2 / "tracks.json").write_text(
        json.dumps({"step": 0.5, "t0": 0.0, "tracks": tracks,
                    "summary": {"n_tracks": 2}}))
    out = build_groups_v2(v2)
    assert out["n_tracks"] == 2
    assert out["n_merged"] == 1
    assert len(out["groups"]) == 1
    assert out["groups"][0]["track_ids"] == [1, 2]
    assert out["groups"][0]["n_members"] == 2
