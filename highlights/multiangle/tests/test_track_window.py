import json
import types

from highlights.multiangle.run import (
    _load_track_rows,
    angle_track_window,
    apply_match_window_src,
)


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


def _ctx(pipe, durs=(0.0, 0.0, 0.0)):
    return types.SimpleNamespace(
        pipe=pipe, log=lambda *a, **k: None,
        angles=[{}, {}, {}], duration=lambda i: durs[i])


def test_apply_match_window_src_converts_with_offsets(tmp_path):
    src = {"angle": 2, "start": 900.0, "end": 2000.0}
    (tmp_path / "match_window_src.json").write_text(json.dumps(src))
    sync = {"offsets": [0.0, 5.0, -748.0]}
    apply_match_window_src(_ctx(tmp_path), sync)
    # shared-T = file_t + offset[2]
    assert json.loads((tmp_path / "cut_range.json").read_text()) \
        == {"lo": 152.0, "hi": 1252.0}
    # idempotent — same content, no error
    apply_match_window_src(_ctx(tmp_path), sync)
    assert json.loads((tmp_path / "cut_range.json").read_text()) \
        == {"lo": 152.0, "hi": 1252.0}


def test_apply_match_window_src_clips_and_missing(tmp_path):
    # no src file -> nothing happens
    apply_match_window_src(_ctx(tmp_path), {"offsets": [0.0]})
    assert not (tmp_path / "cut_range.json").exists()
    # negative shared-T clips to 0
    (tmp_path / "match_window_src.json").write_text(
        json.dumps({"angle": 0, "start": 10.0, "end": 100.0}))
    apply_match_window_src(_ctx(tmp_path), {"offsets": [-50.0]})
    assert json.loads((tmp_path / "cut_range.json").read_text()) \
        == {"lo": 0.0, "hi": 50.0}


def test_apply_match_window_src_resolves_longest_angle(tmp_path):
    # angle null = "measured on the longest video"
    (tmp_path / "match_window_src.json").write_text(
        json.dumps({"angle": None, "start": 900.0, "end": 2000.0}))
    ctx = _ctx(tmp_path, durs=(5000.0, 6200.0, 4000.0))  # a1 longest
    apply_match_window_src(ctx, {"offsets": [0.0, 5.0, -748.0]})
    assert json.loads((tmp_path / "cut_range.json").read_text()) \
        == {"lo": 905.0, "hi": 2005.0}
    # resolved angle persisted back into the src file
    assert json.loads((tmp_path / "match_window_src.json").read_text())["angle"] == 1


def test_apply_match_window_src_waits_for_all_durations(tmp_path):
    # angle null with an angle still unprobed (duration 0) must NOT
    # resolve to a wrong index — no cut_range until all durations known
    (tmp_path / "match_window_src.json").write_text(
        json.dumps({"angle": None, "start": 1080.0, "end": 4200.0}))
    ctx = _ctx(tmp_path, durs=(5048.0, 5158.0, 0.0))  # a2 not downloaded
    apply_match_window_src(ctx, {"offsets": [0.0, 5.0, -748.0]})
    assert not (tmp_path / "cut_range.json").exists()
    assert json.loads((tmp_path / "match_window_src.json").read_text()
                      )["angle"] is None
    # once a2 has a duration, resolution picks the longest
    ctx = _ctx(tmp_path, durs=(5048.0, 5158.0, 6200.0))
    apply_match_window_src(ctx, {"offsets": [0.0, 5.0, -748.0]})
    assert json.loads((tmp_path / "cut_range.json").read_text()) \
        == {"lo": 332.0, "hi": 3452.0}
    assert json.loads((tmp_path / "match_window_src.json").read_text())["angle"] == 2
