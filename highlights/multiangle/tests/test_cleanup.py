"""cleanup_caches: render caches removed after a successful run, sources kept."""

from pathlib import Path

from highlights.multiangle import run


def _ctx(tmp_path: Path) -> run.Ctx:
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True)
    return run.Ctx(project_dir=tmp_path, pipe=pipe, status=None,
                   angles=[{"label": "a0", "url": None,
                            "dir": tmp_path / "angles" / "a0"}])


def _layout(ctx: run.Ctx) -> dict:
    mezz = ctx.pipe / "mezz"
    segs = ctx.pipe / "segs"
    mezz.mkdir()
    segs.mkdir()
    (mezz / "0_abc.mp4").write_bytes(b"m" * 100)
    (segs / "deadbeef.mp4").write_bytes(b"s" * 50)
    src_dir = ctx.angles[0]["dir"]
    src_dir.mkdir(parents=True)
    src = src_dir / "match.mp4"
    src.write_bytes(b"source")
    cuts = ctx.project_dir / "cuts"
    cuts.mkdir()
    cut = cuts / "cut1.mp4"
    cut.write_bytes(b"cut")
    match = ctx.project_dir / "match.mp4"
    match.write_bytes(b"out")
    return {"mezz": mezz, "segs": segs, "src": src, "cut": cut,
            "match": match}


def test_cleanup_removes_caches_keeps_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_KEEP_CACHES", raising=False)
    ctx = _ctx(tmp_path)
    p = _layout(ctx)
    freed = run.cleanup_caches(ctx)
    assert freed == 150
    assert not p["mezz"].exists()
    assert not p["segs"].exists()
    for kept in (p["src"], p["cut"], p["match"]):
        assert kept.exists()
    # idempotent on a clean tree
    assert run.cleanup_caches(ctx) == 0


def test_cleanup_keep_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_KEEP_CACHES", "1")
    ctx = _ctx(tmp_path)
    p = _layout(ctx)
    assert run.cleanup_caches(ctx) == 0
    assert p["mezz"].exists() and p["segs"].exists()
    assert (p["mezz"] / "0_abc.mp4").exists()
    assert (p["segs"] / "deadbeef.mp4").exists()
