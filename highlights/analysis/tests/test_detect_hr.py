"""detect_hr tests: tiling, NMS merge, npz write/resume — fake model."""

import numpy as np
import pytest

from highlights.analysis.detect_hr import detect_angle, nms, tiles
from highlights.pipeline.errors import PipelineError


def test_tiles_cover_frame():
    tl = tiles(1920, 1080)
    assert len(tl) == 4
    # corners anchored to frame edges
    assert tl[0][:2] == (0, 0)
    assert tl[-1][2:] == (1920, 1080)
    tw, th = tl[0][2] - tl[0][0], tl[0][3] - tl[0][1]
    assert (tw, th) == (1056, 594)          # half + 10% overlap


def test_nms_merges_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 120, 120]],
                     dtype=float)
    conf = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, conf)
    assert sorted(keep) == [0, 2]           # dup suppressed, best conf kept


def _fake_frames(n=8, w=320, h=180):
    for i in range(n):
        yield i * 0.5, np.zeros((h, w, 3), dtype=np.uint8)


def test_detect_writes_and_resumes(tmp_path):
    out = tmp_path / "det_a0.npz"
    calls = []

    def fake_predict(model, tile, imgsz):
        calls.append(1)
        # one person per tile at its centre -> merges to ~4 dets per
        # frame, deduped by NMS where tiles overlap
        h, w = tile.shape[:2]
        return [(np.array([w * 0.45, h * 0.45, w * 0.55, h * 0.95]),
                 0.7)]

    meta = detect_angle("fake.mp4", out, fps=2.0, start_s=0.0, end_s=4.0,
                        model=object(), predict=fake_predict,
                        frames=_fake_frames(), log=lambda m: None)
    z = np.load(out)
    assert len(z["t"]) == meta["n_dets"] > 0
    assert set(np.unique(z["t"])) == {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
    assert z["team"].tolist() == [""] * len(z["t"])
    assert (tmp_path / "det_a0.meta.json").exists()

    # resume: same call again — frames before last t skipped (still
    # decoded but predict not needed beyond done_s; count doesn't grow)
    n1 = len(z["t"])
    meta2 = detect_angle("fake.mp4", out, fps=2.0, start_s=0.0,
                         end_s=4.0, model=object(),
                         predict=fake_predict, frames=_fake_frames(),
                         log=lambda m: None)
    z2 = np.load(out)
    assert len(z2["t"]) == n1 and meta2["n_dets"] == n1


def test_eta_abort(tmp_path):
    import time
    from unittest.mock import patch

    def slow_predict(model, tile, imgsz):
        time.sleep(0.05)                    # >8 h at 2fps over 100 s
        return []

    frames = _fake_frames(n=25)
    # fake a very long window via end_s
    with pytest.raises(PipelineError, match="aborting"), \
            patch("highlights.analysis.detect_hr.ETA_PROBE", 4):
        detect_angle("fake.mp4", tmp_path / "d.npz", fps=2.0,
                     start_s=0.0, end_s=1e6, model=object(),
                     predict=slow_predict, frames=frames,
                     log=lambda m: None)
