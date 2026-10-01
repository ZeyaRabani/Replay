"""Playlist grouping/parsing unit tests."""

from highlights.app.backend.playlist import group_videos, parse_title

ENTRIES = [
    {"id": "v1", "title": "28/8/26 Green team wins Green end right",
     "duration": 3800},
    {"id": "v2", "title": "28/8/26 Green team wins Green end left",
     "duration": 3700},
    {"id": "v3", "title": "28/8/26 Green team wins Orange end",
     "duration": 3600},
    {"id": "v4", "title": "28/8/26 Green Team wins All Angles",
     "duration": 3800},
    {"id": "v5", "title": "20/10/2023 Afghanistan wins Afghanistan end",
     "duration": 5400},
    {"id": "v6", "title": "10/6/22 Green team wins. Orange end",
     "duration": 3700},
    {"id": "v7", "title": "15/20/21 Weird date Cam A", "duration": 100},
    {"id": "v8", "title": "Random clip no date", "duration": 60},
]


def test_parse_title_dates():
    assert parse_title("28/8/26 Green team wins All Angles")[0] == "2026-08-28"
    assert parse_title("20/10/2023 Afghanistan wins x")[0] == "2023-10-20"
    assert parse_title("10/6/22 Green team wins. y")[0] == "2022-06-10"
    # garbage date -> raw string key
    assert parse_title("15/20/21 Weird date Cam A")[0] == "15/20/21"
    assert parse_title("Random clip no date")[0] is None


def test_group_labels_and_angles():
    groups = {g["date"]: g for g in group_videos(ENTRIES)}
    g28 = groups["2026-08-28"]
    assert g28["label"] == "Green team wins"
    assert g28["n_angles"] == 3                    # v1, v2, v3 (not all-angles)
    assert g28["all_angles_uploaded"] is True
    assert g28["done"] is True
    assert len(g28["videos"]) == 4
    assert groups["2023-10-20"]["label"] == "Afghanistan wins"
    g10 = groups["2022-06-10"]
    assert g10["label"] == "Green team wins"
    assert g10["n_angles"] == 1 and g10["done"] is False
    # garbage groups kept separate
    assert groups["15/20/21"]["n_angles"] == 1
    assert groups[None]["n_angles"] == 1


def test_group_sorting_newest_first():
    ds = [g["date"] for g in group_videos(ENTRIES)]
    assert ds[0] == "2026-08-28"        # newest parsed date first
    assert ds[1] == "2023-10-20"
    assert ds[2] == "2022-06-10"
    assert set(ds[3:]) == {"15/20/21", None}   # unparseable last
