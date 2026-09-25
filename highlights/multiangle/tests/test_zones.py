"""Manual ball zones in the director."""

import numpy as np

from highlights.multiangle.director import _in_poly, cut_director


def _track(T, cluster=1.0, ball_conf=0.0, ball_xy=(0.0, 0.0)):
    return {
        "ball_conf": np.full(T, ball_conf),
        "ball_size": np.full(T, ball_conf * 0.1),
        "ball_x": np.full(T, ball_xy[0]),
        "ball_y": np.full(T, ball_xy[1]),
        "cluster": np.full(T, cluster),
        "event": np.zeros(T),
    }


def _run(zones, T=300, zone_ok=None):
    avail = np.ones((2, T), dtype=bool)
    tr = [_track(T, cluster=8.0), _track(T, cluster=2.0)]
    # angle 1 sees the ball at (.2,.5) during t=100..200
    tr[1]["ball_conf"][100:201] = 0.8
    tr[1]["ball_size"][100:201] = 0.05
    tr[1]["ball_x"][100:201] = 0.2
    tr[1]["ball_y"][100:201] = 0.5
    motion = [np.ones(T), np.ones(T)]
    return cut_director(tr, avail, motion, zones=zones, zone_ok=zone_ok)


def test_zone_cut_overrides_cluster():
    zones = [[], [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]]]
    d = _run(zones)
    zone_segs = [s for s in d["segments"] if s["rule"] == "zone"]
    assert zone_segs, "expected zone segments"
    for s in zone_segs:
        assert s["angle"] == 1
    t0 = min(s["t_start"] for s in zone_segs)
    t1 = max(s["t_end"] for s in zone_segs)
    assert t0 <= 103            # cuts to the zoned angle quickly
    assert t1 >= 200            # plus linger (~203)
    assert d["zones_used"] is True
    assert d["ratios"].get("zone", 0) > 0
    # while a zone angle is eligible the director never uses the cluster rule
    for s in d["segments"]:
        if s["rule"] == "cluster":
            assert s["t_end"] <= t0 or s["t_start"] >= t1
    # outside the zone window the strong cluster camera holds
    assert d["segments"][0]["angle"] == 0


def test_zones_none_identical():
    a = _run(None)
    b = _run(zones=None)
    assert a["segments"] == b["segments"]
    assert a["zones_used"] is False


def test_zone_ok_suspends():
    zones = [[], [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]]]
    ok = np.ones((2, 300), dtype=bool)
    ok[1, 120:] = False          # camera moved at t=120
    d = _run(zones, zone_ok=ok)
    zone_segs = [s for s in d["segments"] if s["rule"] == "zone"]
    assert zone_segs, "expected zone segments before the camera moved"
    # no new zone cuts once the view check fails at t=120
    assert all(s["t_start"] <= 123 for s in zone_segs)


def test_in_poly():
    poly = [[0, 0], [1, 0], [1, 1], [0, 1]]
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]])
    assert _in_poly(pts, poly).tolist() == [True, False, False]


def test_ctx_duration_probe_fallback(tmp_path):
    """Recut bug regression: ctx.durations empty (no stage_sync) must fall
    back to angles/aN/pipeline/probe.json — else dur=0 pinned every second
    to okarr[0] and suspended 100% of the angle's zones."""
    import json as _json

    from highlights.multiangle.run import Ctx, _map_view_ok

    adir = tmp_path / "angles" / "a1"
    (adir / "pipeline").mkdir(parents=True)
    (adir / "pipeline" / "probe.json").write_text(
        _json.dumps({"duration_s": 5600.0}))
    ctx = Ctx(project_dir=tmp_path, pipe=tmp_path / "ma",
              status=None, angles=[{"dir": tmp_path / "angles" / "a0"},
                                    {"dir": adir}])
    assert ctx.durations == []              # recut: stage_sync never ran
    assert ctx.duration(1) == 5600.0
    assert ctx.durations[1] == 5600.0       # cached

    # and the viewcheck mapping uses it: ok=False only at file t=0
    times = np.arange(0, 5600, 10.0)
    okarr = np.ones(len(times), dtype=bool)
    okarr[0] = False                        # camera still being set up at t=0
    row = _map_view_ok(times, okarr, T=5400, lo=0.0, off=0.0, dur=5600.0)
    assert row.sum() > 5000                 # not all-False
    assert not row[0]
    # dur=0 (the old bug) collapses everything onto the t=0 sample
    row_bug = _map_view_ok(times, okarr, T=5400, lo=0.0, off=0.0, dur=0.0)
    assert not row_bug.any()


def test_ctx_union_cut_range(tmp_path):
    """Ctx.union: no cut_range.json -> coverage union; file -> clipped."""
    from highlights.multiangle.run import Ctx

    ctx = Ctx(project_dir=tmp_path, pipe=tmp_path / "ma",
              status=None, angles=[])
    ctx.pipe.mkdir(parents=True)
    sync = {"coverage": {"union": [100.0, 6100.0]}}
    assert ctx.union(sync) == (100.0, 6100.0)
    import json as _json
    (ctx.pipe / "cut_range.json").write_text(
        _json.dumps({"lo": 1900.0, "hi": 5700.0}))
    assert ctx.union(sync) == (1900.0, 5700.0)
    # out-of-union values are clipped
    (ctx.pipe / "cut_range.json").write_text(
        _json.dumps({"lo": 0.0, "hi": 99999.0}))
    assert ctx.union(sync) == (100.0, 6100.0)


def test_stage_fuse_match_window_cut_range(tmp_path):
    """With cut_range.json, stage_fuse writes match_window=[0, hi-lo] and
    clips halves to the output duration; without it, a0's window is shifted
    and clamped (regression for the re-cut overwrite bug)."""
    import json as _json

    from highlights.multiangle.run import Ctx, stage_fuse

    def _proj():
        proj = tmp_path / "p"
        ma = proj / "multiangle"
        ma.mkdir(parents=True)
        (ma / "sync.json").write_text(_json.dumps({
            "offsets": [0.0, -10.0],
            "coverage": {"union": [100.0, 200.0]}}))
        angles = []
        for i in range(2):
            d = proj / "angles" / f"a{i}" / "pipeline"
            d.mkdir(parents=True)
            (d / "candidates.json").write_text(_json.dumps({"events": []}))
            angles.append({"label": f"a{i}", "dir": d.parent})
        (angles[0]["dir"] / "pipeline" / "match_window.json").write_text(
            _json.dumps({"match_window": [110.0, 195.0],
                         "halves": [{"start": 110.0, "end": 152.0},
                                    {"start": 190.0, "end": 195.0}]}))
        ctx = Ctx(project_dir=proj, pipe=ma, status=None, angles=angles)
        return proj, ctx

    # no cut range: a0 window shifted (offset0 - lo = -100) and clamped
    proj, ctx = _proj()
    stage_fuse(ctx)
    mw = _json.loads((proj / "pipeline" / "match_window.json").read_text())
    assert mw["match_window"] == [10.0, 95.0]
    assert mw["halves"] == [{"start": 10.0, "end": 52.0},
                            {"start": 90.0, "end": 95.0}]

    # with cut range: window = whole rendered video, halves re-clipped
    proj2 = tmp_path / "p2"
    proj2.mkdir()
    import shutil as _sh
    _sh.copytree(proj, proj2, dirs_exist_ok=True)
    ma2 = proj2 / "multiangle"
    (ma2 / "cut_range.json").write_text(_json.dumps({"lo": 110.0, "hi": 150.0}))
    ctx2 = Ctx(project_dir=proj2, pipe=ma2, status=None,
               angles=[{"label": "a0", "dir": proj2 / "angles" / "a0"},
                       {"label": "a1", "dir": proj2 / "angles" / "a1"}])
    stage_fuse(ctx2)
    mw2 = _json.loads((proj2 / "pipeline" / "match_window.json").read_text())
    assert mw2["match_window"] == [0.0, 40.0]
    # halves clipped to [0,40] of the cut (a0 times stay as-is but clamped)
    assert all(0 <= h["start"] < h["end"] <= 40.0 for h in mw2["halves"])
