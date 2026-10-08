"""stage_download: angles download in parallel (HL_DOWNLOAD_WORKERS)."""

import time
from pathlib import Path

import pytest

import highlights.pipeline.download as dl_mod
from highlights.multiangle import run
from highlights.pipeline.errors import PipelineError


def _ctx(tmp_path: Path, n: int) -> run.Ctx:
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True)
    angles = [{"label": f"a{i}", "url": f"http://x/{i}",
               "dir": tmp_path / "angles" / f"a{i}"} for i in range(n)]
    return run.Ctx(project_dir=tmp_path, pipe=pipe, status=None,
                   angles=angles)


def test_download_angles_parallel(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 2)
    starts = []

    def fake_download(url, dest_dir, status=None, cookies=None, log=print):
        starts.append((time.monotonic(), url))
        time.sleep(0.2)

    monkeypatch.setattr(dl_mod, "download", fake_download)
    run.stage_download(ctx)
    assert len(starts) == 2
    assert max(t for t, _ in starts) - min(t for t, _ in starts) < 0.1


def test_download_serial_checks(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 2)
    # a0 has a file already, a1 has no url -> sequential checks, no pool
    ctx.angles[0]["dir"].mkdir(parents=True)
    (ctx.angles[0]["dir"] / "match.mp4").write_bytes(b"x")
    ctx.angles[1]["url"] = None
    with pytest.raises(PipelineError, match="no file and no url"):
        run.stage_download(ctx)
