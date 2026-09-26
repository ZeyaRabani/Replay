import json

from highlights.multiangle.run import _load_track_rows, angle_track_window


def test_angle_track_window_offsets_and_pad():
    # angle 0 (reference): pad 30 on both sides
    lo_f, hi_f = angle_track_window(1740.0, 5640.0, offset=0.0, duration=6200.0)
    assert (lo_f, hi_f) == (1710.0, 5670.0)
    # a later-starting angle (offset > 0) maps the window earlier in file time
    lo_f, hi_f = angle_track_window(1740.0, 5640.0, offset=5.0, duration=6200.0)
    assert (lo_f, hi_f) == (1705.0, 5665.0)


def test_angle_track_window_clamps_to_video():
    lo_f, hi_f = angle_track_window(10.0, 5640.0, offset=0.0, duration=6200.0)
    assert lo_f == 0.0
    lo_f, hi_f = angle_track_window(1740.0, 9000.0, offset=0.0, duration=6200.0)
    assert hi_f == 6200.0
    # window fully outside the video -> empty range, not inverted
    lo_f, hi_f = angle_track_window(10.0, 20.0, offset=-8000.0, duration=6200.0)
    assert lo_f == hi_f


def test_load_track_rows_densifies_windowed_rows(tmp_path):
    td = tmp_path / "track"
    td.mkdir()
    rows = [
        [1740.0, 5, "[]"],
        [1741.0, 6, "[[0.5, 0.9]]"],
    ]
    (td / "features_1s.json").write_text(json.dumps({
        "columns": ["t", "n_players", "players_xy"],
        "rows": rows,
    }))
    tr = _load_track_rows(tmp_path)
    assert len(tr["n_players"]) == 1742
    assert tr["n_players"][0] == 0.0       # gap before the window -> zeros
    assert tr["n_players"][1739] == 0.0
    assert tr["n_players"][1740] == 5.0
    assert tr["n_players"][1741] == 6.0
    assert tr["players_xy"][0] == []
    assert tr["players_xy"][1741] == [[0.5, 0.9]]


def test_load_track_rows_missing(tmp_path):
    assert _load_track_rows(tmp_path) == {}
