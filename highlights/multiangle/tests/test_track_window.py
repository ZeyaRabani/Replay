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


def test_track_window_via_timemap(tmp_path):
    """A piecewise map widens the file range across a jump."""
    segs = [{"file_lo": 0, "file_hi": 50, "offset": 10.0},
            {"file_lo": 53, "file_hi": 100, "offset": 12.0}]
    # shared 30..80 -> covered file 20..68 (before pad)
    lo_f, hi_f = angle_track_window(30.0, 80.0, segs, 100.0, pad=0.0)
    assert (lo_f, hi_f) == (20.0, 68.0)


def _track_ctx(tmp_path):
    import json as _json

    from highlights.multiangle.run import Ctx
    for i in (0, 1):
        adir = tmp_path / "angles" / f"a{i}"
        (adir / "pipeline").mkdir(parents=True)
        (adir / "pipeline" / "probe.json").write_text(
            _json.dumps({"duration_s": 500.0}))
        (adir / "match.mp4").write_bytes(b"v")
    adir = tmp_path / "angles" / "a1"
    (adir / "pipeline" / "candidates.json").write_text("{}")
    pipe = tmp_path / "multiangle"
    pipe.mkdir()
    # shared window 10..490; a1 map is the identity file range 0..500
    # (offset 0), needing file 0..500 after pad
    (pipe / "sync.json").write_text(_json.dumps(
        {"offsets": [0.0, 0.0],
         "timemap": [[{"file_lo": 0, "file_hi": 500, "offset": 0.0}],
                     [{"file_lo": 0, "file_hi": 500, "offset": 0.0}]],
         "coverage": {"union": [0.0, 500.0]}}))
    (pipe / "cut_range.json").write_text(_json.dumps(
        {"lo": 10.0, "hi": 490.0}))

    class _Status:
        def update(self, **kw):
            pass
    return Ctx(project_dir=tmp_path, pipe=pipe, status=_Status(),
               angles=[{"dir": tmp_path / "angles" / "a0",
                        "label": "a0", "url": None},
                       {"dir": adir, "label": "a1", "url": None}]), adir


def test_stage_track_skips_covered_window(tmp_path):
    """An existing features_1s.json whose meta window covers the new
    map's needs is kept, not re-tracked."""
    import json as _json

    from highlights.multiangle.run import stage_track
    ctx, adir = _track_ctx(tmp_path)
    out = adir / "track" / "features_1s.json"
    out.parent.mkdir(parents=True)
    out.write_text(_json.dumps(
        {"meta": {"start_s": 0.0, "end_s": 500.0}, "rows": []}))
    (tmp_path / "angles" / "a0" / "track").mkdir(exist_ok=True)
    (tmp_path / "angles" / "a0" / "track" / "features_1s.json").write_text(
        _json.dumps({"meta": {"start_s": 0.0, "end_s": 500.0}, "rows": []}))
    stage_track(ctx)      # returns without spawning when covered


def test_stage_track_retracks_narrow_window(tmp_path, monkeypatch):
    """A saved track narrower than the needed window is re-tracked."""
    import json as _json
    import subprocess as sp

    from highlights.multiangle.run import stage_track
    ctx, adir = _track_ctx(tmp_path)
    out = adir / "track" / "features_1s.json"
    out.parent.mkdir(parents=True)
    out.write_text(_json.dumps(
        {"meta": {"start_s": 200.0, "end_s": 300.0}, "rows": []}))
    spawned = []
    monkeypatch.setattr(sp, "Popen",
                        lambda cmd, **kw: spawned.append(cmd) or _Done())
    monkeypatch.setattr(
        "highlights.multiangle.proxy.ensure_analysis_proxy",
        lambda *a, **k: type("PI", (), {"path": None, "offset": 0.0,
                                       "src": "original"})())

    class _Done:
        def poll(self): return 0
    monkeypatch.setenv("HL_TRACK_PROXY", "0")
    stage_track(ctx)
    assert spawned and "--end-s" in " ".join(spawned[0])


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
