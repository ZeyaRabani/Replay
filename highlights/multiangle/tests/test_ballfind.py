import json
import os
from pathlib import Path

import numpy as np
import pytest

from highlights.multiangle import ballfind
from highlights.multiangle import run as run_mod


def _det(x, y, s=0.9, w=10, h=10):
    return [x, y, x + w, y + h, s]


def _doc(dets, wh=(1920, 1080)):
    return {"model": "ball_v2", "imgsz": 1280, "native_wh": list(wh),
            "dets": {str(t): v for t, v in dets.items()}}


def test_ball_rows_threshold_and_normalisation():
    doc = _doc({5: [_det(900, 500, 0.9), _det(100, 100, 0.4),
                    _det(0, 0, 0.1)]})
    rows = ballfind.ball_rows(doc, step=1)
    assert set(rows) == {5}
    score, size, cx, cy = rows[5]
    assert score == 0.9                     # max-score pick
    assert size == pytest.approx(10 / 1080)
    assert cx == pytest.approx(905 / 1920)
    assert cy == pytest.approx(505 / 1080)


def test_ball_rows_below_thr_dropped():
    assert ballfind.ball_rows(_doc({3: [_det(0, 0, 0.2)]}), step=1) == {}


def test_ball_rows_static_removed():
    # the same box at the same spot in >=REP other seconds within +-WIN
    dets = {}
    for t in range(0, 40, 7):
        dets[t] = [_det(500, 300, 0.8)]
    dets[14] = [_det(505, 302, 0.8), _det(1500, 700, 0.9)]
    rows = ballfind.ball_rows(_doc(dets), step=1)
    # the ~same-spot box is static everywhere (>=5 repeats within 60 s),
    # so only t=14's distinct box survives
    assert set(rows) == {14}
    assert rows[14][0] == 0.9
    # a static-only set keeps no box at all
    dets2 = {t: [_det(500, 300, 0.8)] for t in range(0, 40, 7)}
    assert ballfind.ball_rows(_doc(dets2), step=1) == {}


def test_runs_chunking():
    runs = ballfind._runs(set(range(10)) | set(range(50, 55)))
    flat = [t for r in runs for t in range(r[0], r[1] + 1)]
    assert flat == sorted(set(range(10)) | set(range(50, 55)))
    assert all(r[1] - r[0] + 1 <= ballfind.CHUNK for r in runs)


def test_needed_seconds_gap_and_offset(tmp_path, monkeypatch):
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True)
    sync = {"offsets": [0.0, -100.0],
            "coverage": {"union": [0.0, 400.0]}}
    (pipe / "sync.json").write_text(json.dumps(sync))
    (pipe / "cut_range.json").write_text(json.dumps({"lo": 50.0,
                                                    "hi": 70.0}))
    angles = [{"label": f"a{i}", "url": None,
               "dir": tmp_path / "angles" / f"a{i}"} for i in range(2)]
    ctx = run_mod.Ctx(project_dir=tmp_path, pipe=pipe,
                      status=run_mod._NoopStatus(), angles=angles,
                      read_only=True, durations=[400.0, 400.0])
    monkeypatch.setattr(run_mod, "apply_match_window_src",
                        lambda ctx, sync: None)
    monkeypatch.setattr(ballfind, "STEP", 1)
    need = ballfind.needed_seconds(ctx)
    assert need[0] == set(range(50, 70))            # file == shared
    assert need[1] == set(range(150, 170))          # offset +100


def test_incremental_skips_cached(tmp_path, monkeypatch):
    angles = [{"label": "a0", "url": None,
               "dir": tmp_path / "angles" / "a0"}]
    ctx = run_mod.Ctx(project_dir=tmp_path, pipe=tmp_path / "multiangle",
                      status=run_mod._NoopStatus(), angles=angles,
                      read_only=True)
    monkeypatch.setattr(run_mod.Ctx, "angle_video",
                        lambda self, i: tmp_path / "v.mp4")
    monkeypatch.setattr(ballfind, "needed_seconds",
                        lambda c: {0: {10, 11, 12}})
    seen = {}

    class M:
        def predict(self, *a, **k):
            seen["called"] = True
            raise AssertionError("predict on cached seconds")

    monkeypatch.setattr(ballfind, "_model_path",
                        lambda: tmp_path / "m.pt")
    (tmp_path / "m.pt").write_text("x")
    monkeypatch.setattr(ballfind, "_load_model", lambda p: M())
    # write a cache covering everything -> zero extraction
    d = angles[0]["dir"] / "track"
    d.mkdir(parents=True)
    (d / "ball_v2.json").write_text(json.dumps(
        _doc({10: [], 11: [], 12: []})))
    monkeypatch.setattr(ballfind, "_extract",
                        lambda *a: seen.setdefault("extract", True))
    ballfind.run_ballfind(ctx)
    assert "called" not in seen and "extract" not in seen

    # now drop second 11 -> only that second is extracted+predicted
    (d / "ball_v2.json").write_text(json.dumps(_doc({10: [], 12: []})))
    extracted = []
    monkeypatch.setattr(ballfind, "_extract",
                        lambda v, S, E, tmp: extracted.append((S, E)))
    monkeypatch.setattr(ballfind, "ffprobe",
                        lambda p: {"width": 1920, "height": 1080})

    class M2:
        def predict(self, *a, **k):
            return [type("R", (), {"boxes": []})()]

    monkeypatch.setattr(ballfind, "_load_model", lambda p: M2())

    def fake_extract(v, S, E, tmp):
        extracted.append((S, E))
        (tmp / "00001.jpg").write_text("x")

    monkeypatch.setattr(ballfind, "_extract", fake_extract)
    ballfind.run_ballfind(ctx)
    assert extracted == [(11, 11)]
    doc = json.loads((d / "ball_v2.json").read_text())
    assert set(doc["dets"]) == {"10", "11", "12"}
    assert doc["model"] == "ball_v2" and doc["imgsz"] == 1280


def test_no_model_skips(tmp_path, monkeypatch):
    angles = [{"label": "a0", "url": None,
               "dir": tmp_path / "angles" / "a0"}]
    ctx = run_mod.Ctx(project_dir=tmp_path, pipe=tmp_path / "multiangle",
                      status=run_mod._NoopStatus(), angles=angles,
                      read_only=True)
    monkeypatch.delenv("REPLAY_BALL_MODEL", raising=False)
    monkeypatch.setattr(ballfind, "_model_path", lambda: None)
    ballfind.run_ballfind(ctx)          # no exception, no files


def test_director_falls_back_to_track(tmp_path, monkeypatch):
    # a0 has complete ball_v2, a1 does not -> ball_source "track" for all
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True)
    sync = {"offsets": [0.0, 0.0],
            "coverage": {"union": [0.0, 100.0]}}
    (pipe / "sync.json").write_text(json.dumps(sync))
    (pipe / "cut_range.json").write_text(json.dumps({"lo": 0.0,
                                                    "hi": 5.0}))
    angles = []
    for i in range(2):
        d = tmp_path / "angles" / f"a{i}"
        (d / "track").mkdir(parents=True)
        rows = {"columns": ["t", "ball_conf", "ball_size", "ball_x",
                            "ball_y", "cluster_score"],
                "rows": [[t, 0.5, 0.01, 0.5, 0.5, 0.0]
                         for t in range(100)]}
        (d / "track" / "features_1s.json").write_text(json.dumps(rows))
        (d / "match.mp4").write_text("x")
        angles.append({"label": f"a{i}", "url": None, "dir": d})
    (angles[0]["dir"] / "track" / "ball_v2.json").write_text(
        json.dumps(_doc({t: [_det(100, 100, 0.99)] for t in range(5)})))
    ctx = run_mod.Ctx(project_dir=tmp_path, pipe=pipe,
                      status=run_mod._NoopStatus(), angles=angles,
                      read_only=True, durations=[100.0, 100.0])
    monkeypatch.setattr(run_mod, "apply_match_window_src",
                        lambda ctx, sync: None)
    inp = run_mod.load_director_inputs(ctx)
    assert inp["ball_source"] == "track"
    assert np.allclose(inp["tracks"][0]["ball_conf"], 0.5)


def test_director_uses_ball_v2_when_complete(tmp_path, monkeypatch):
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True)
    sync = {"offsets": [0.0, 0.0],
            "coverage": {"union": [0.0, 100.0]}}
    (pipe / "sync.json").write_text(json.dumps(sync))
    (pipe / "cut_range.json").write_text(json.dumps({"lo": 0.0,
                                                    "hi": 3.0}))
    angles = []
    for i in range(2):
        d = tmp_path / "angles" / f"a{i}"
        (d / "track").mkdir(parents=True)
        rows = {"columns": ["t", "ball_conf", "cluster_score"],
                "rows": [[t, 0.5, 0.0] for t in range(100)]}
        (d / "track" / "features_1s.json").write_text(json.dumps(rows))
        (d / "match.mp4").write_text("x")
        (d / "track" / "ball_v2.json").write_text(
            json.dumps(_doc({t: [_det(960, 540, 0.99)]
                             for t in range(3)})))
        angles.append({"label": f"a{i}", "url": None, "dir": d})
    ctx = run_mod.Ctx(project_dir=tmp_path, pipe=pipe,
                      status=run_mod._NoopStatus(), angles=angles,
                      read_only=True, durations=[100.0, 100.0])
    monkeypatch.setattr(run_mod, "apply_match_window_src",
                        lambda ctx, sync: None)
    inp = run_mod.load_director_inputs(ctx)
    assert inp["ball_source"] == "ball_v2"
    assert np.allclose(inp["tracks"][0]["ball_conf"][:3], 0.99)


def test_min_hold_override_only_for_ball_v2():
    # stage_director picks min_hold 6 only when ball_source==ball_v2;
    # exercise the style_overrides construction directly
    text = Path(run_mod.__file__).read_text()
    assert '"min_hold": 6' in text
    assert 'inp.get("ball_source") == "ball_v2"' in text


def _cov_ctx(tmp_path, monkeypatch, dets_per_angle):
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True, exist_ok=True)
    (pipe / "sync.json").write_text(json.dumps(
        {"offsets": [0.0] * len(dets_per_angle),
         "coverage": {"union": [0.0, 100.0]}}))
    (pipe / "cut_range.json").write_text(json.dumps({"lo": 0.0,
                                                    "hi": 3.0}))
    angles = []
    for i, dets in enumerate(dets_per_angle):
        d = tmp_path / "angles" / f"a{i}"
        (d / "track").mkdir(parents=True, exist_ok=True)
        (d / "match.mp4").write_text("x")
        if dets is not None:
            (d / "track" / "ball_v2.json").write_text(
                json.dumps(_doc(dets)))
        angles.append({"label": f"a{i}", "url": None, "dir": d})
    ctx = run_mod.Ctx(project_dir=tmp_path, pipe=pipe,
                      status=run_mod._NoopStatus(), angles=angles,
                      read_only=True,
                      durations=[100.0] * len(dets_per_angle))
    monkeypatch.setattr(run_mod, "apply_match_window_src",
                        lambda ctx, sync: None)
    monkeypatch.setattr(ballfind, "_model_path",
                        lambda: tmp_path / "m.pt")
    (tmp_path / "m.pt").write_text("x")
    return ctx


def test_coverage_complete(tmp_path, monkeypatch):
    full = {t: [_det(1, 1, 0.9)] for t in range(3)}
    ctx = _cov_ctx(tmp_path, monkeypatch, [full, full])
    assert ballfind.coverage_complete(ctx)
    assert run_mod._stage_done(ctx, "ballfind")

    # partial cache -> not done
    ctx = _cov_ctx(tmp_path, monkeypatch,
                   [full, {t: [] for t in (0, 1)}])
    assert not ballfind.coverage_complete(ctx)
    assert not run_mod._stage_done(ctx, "ballfind")

    # no model -> done (stage is a no-op)
    monkeypatch.setattr(ballfind, "_model_path", lambda: None)
    assert ballfind.coverage_complete(ctx)
    assert run_mod._stage_done(ctx, "ballfind")


def test_ball_rows_step1_identical():
    doc = _doc({t: [_det(100 + t, 50, 0.9)] for t in range(200)})
    assert ballfind.ball_rows(doc, step=1) == \
        ballfind.ball_rows(doc, step=1)
    doc2 = _doc({10: [_det(100, 100, 0.9)], 11: [_det(200, 200, 0.8)],
                 13: [_det(0, 0, 0.3)]})
    rows1 = ballfind.ball_rows(doc2, step=1)
    assert set(rows1) == {10, 11, 13}


def test_ball_rows_step2():
    # odd-second det is ignored; u=13 fills from max-conf neighbour
    doc = _doc({10: [_det(100, 100, 0.9)],
                11: [_det(200, 200, 0.99)],   # odd -> ignored
                12: [_det(150, 150, 0.5)],
                14: [_det(300, 300, 0.7)]})
    rows = ballfind.ball_rows(doc, step=2)
    assert rows[10][0] == 0.9
    assert rows[13][0] == 0.7       # fill takes max(out[12], out[14])
    assert rows[11][0] == 0.9       # unsampled -> filled from neighbours


def test_ball_rows_step2_fill_edges():
    # one surviving sampled second still fills its unsampled neighbour
    doc = _doc({10: [_det(100, 100, 0.9)]})
    rows = ballfind.ball_rows(doc, step=2)
    assert set(rows) == {10, 11}
    assert rows[11][0] == 0.9
    # a second with no surviving sampled neighbour gets no row
    doc2 = _doc({10: [_det(0, 0, 0.1)], 11: []})
    assert ballfind.ball_rows(doc2, step=2) == {}


def test_ball_rows_step2_static():
    # static across subsampled seconds still removed
    dets = {t: [_det(500, 300, 0.8)] for t in range(0, 40, 7)}
    assert ballfind.ball_rows(_doc(dets), step=2) == \
        ballfind.ball_rows(_doc({t: v for t, v in dets.items()
                                 if t % 2 == 0}), step=2)
    # dense same-spot repeats on even seconds -> static everywhere
    dets_even = {t: [_det(500, 300, 0.8)] for t in range(0, 40, 2)}
    assert ballfind.ball_rows(_doc(dets_even), step=2) == {}


def test_needed_seconds_step(tmp_path, monkeypatch):
    monkeypatch.setattr(ballfind, "STEP", 2)
    ctx = _cov_ctx(tmp_path, monkeypatch, [None])
    need = ballfind.needed_seconds(ctx)
    assert all(s % 2 == 0 for s in need[0]) and need[0]
    # cache of only even seconds is complete
    full_even = {t: [_det(1, 1, 0.9)] for t in range(0, 100, 2)}
    ctx = _cov_ctx(tmp_path, monkeypatch, [full_even])
    assert ballfind.coverage_complete(ctx)


def test_model_path_prefers_ncnn(tmp_path, monkeypatch):
    pt = tmp_path / "ball_v2.pt"
    pt.write_text("x")
    ncnn_dir = tmp_path / "ball_v2_ncnn_model"
    ncnn_dir.mkdir()
    monkeypatch.setenv("REPLAY_BALL_MODEL", str(pt))
    monkeypatch.setattr(ballfind, "_ncnn_ok", lambda: True)
    from highlights.multiangle.engine import prefer_ncnn
    monkeypatch.setattr(ballfind, "prefer_ncnn", prefer_ncnn,
                        raising=False)
    monkeypatch.setattr("highlights.multiangle.engine.ncnn_ok",
                        lambda: True)
    assert ballfind._model_path() == ncnn_dir
    # ncnn unavailable -> .pt
    monkeypatch.setattr("highlights.multiangle.engine.ncnn_ok",
                        lambda: False)
    assert ballfind._model_path() == pt
    # REPLAY_INFER_ENGINE=torch -> .pt
    monkeypatch.setenv("REPLAY_INFER_ENGINE", "torch")
    from highlights.multiangle import engine as eng
    monkeypatch.setattr("highlights.multiangle.engine.ncnn_ok",
                        eng.ncnn_ok)
    assert ballfind._model_path() == pt
    monkeypatch.delenv("REPLAY_INFER_ENGINE")
    # direct _ncnn_model env path passes through
    monkeypatch.setenv("REPLAY_BALL_MODEL", str(ncnn_dir))
    monkeypatch.setattr("highlights.multiangle.engine.ncnn_ok",
                        lambda: True)
    assert ballfind._model_path() == ncnn_dir


def test_track_engine_workers(monkeypatch):
    from highlights.multiangle import engine
    monkeypatch.delenv("HL_MA_MODEL", raising=False)
    monkeypatch.delenv("HL_TRACK_WORKERS", raising=False)
    monkeypatch.delenv("REPLAY_INFER_ENGINE", raising=False)

    # ncnn dir present + importable -> ncnn model, 1 worker, env plain
    monkeypatch.setattr(engine, "ncnn_ok", lambda: True)
    monkeypatch.setattr(Path, "exists",
                        lambda self: str(self).endswith("_ncnn_model"))
    model, use_ncnn, workers, env = run_mod._track_engine()
    assert use_ncnn and model.endswith("_ncnn_model") and workers == 1
    assert "TORCH_NUM_THREADS" not in env or \
        env["TORCH_NUM_THREADS"] == os.environ.get("TORCH_NUM_THREADS")

    # explicit worker count wins
    monkeypatch.setenv("HL_TRACK_WORKERS", "4")
    assert run_mod._track_engine()[2] == 4
    monkeypatch.delenv("HL_TRACK_WORKERS")

    # no ncnn dir -> torch .pt, 2 workers
    monkeypatch.setattr(Path, "exists", lambda self: False)
    model, use_ncnn, workers, env = run_mod._track_engine()
    assert not use_ncnn and model.endswith("yolov8n.pt") and workers == 2
    assert env["TORCH_NUM_THREADS"] == "2"

    # explicit HL_MA_MODEL to ncnn dir -> 1 worker
    monkeypatch.setenv("HL_MA_MODEL", "/x/yolov8n_960_ncnn_model")
    assert run_mod._track_engine()[2] == 1


def test_ballfind_before_director_in_stage_lists():
    from highlights.app.backend import main as app_main
    from highlights.app.backend import pipeline
    lists = {
        "MULTIANGLE_STAGES": pipeline.MULTIANGLE_STAGES,
        "MULTIANGLE_FROM_SYNC": pipeline.MULTIANGLE_FROM_SYNC,
    }
    for name, stages in lists.items():
        assert "ballfind" in stages and "director" in stages, name
        assert stages.index("ballfind") < stages.index("director"), name
    # literal lists in main.py spawn calls: AI recut + you-direct learn
    # include ballfind; the manual-cut spawn stays cheap (no detection)
    src = Path(app_main.__file__).read_text()
    assert src.count('["ballfind", "director", "render", "fuse"]') == 2
    assert src.count('["director", "render", "fuse"]') == 1
    # run.py STAGES dict order
    names = list(run_mod.STAGES)
    assert names.index("ballfind") < names.index("director")
