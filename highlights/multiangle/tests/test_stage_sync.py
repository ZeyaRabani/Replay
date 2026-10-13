"""run.stage_sync never stops for low-confidence sync, and run.main
forces re-execution when --offsets is given (the override's outputs
already exist, so skip-on-existing would silently ignore them)."""

import json
from contextlib import nullcontext

from highlights.multiangle import run


def _ctx(tmp_path, n=2):
    angles = []
    for i in range(n):
        d = tmp_path / "angles" / f"a{i}"
        (d / "pipeline" / "audio").mkdir(parents=True)
        (d / "pipeline" / "audio" / "audio.wav").write_bytes(b"")
        (d / "pipeline" / "probe.json").write_text(
            json.dumps({"duration_s": 100.0}))
        angles.append({"dir": d})
    pipe = tmp_path / "multiangle"
    pipe.mkdir(parents=True, exist_ok=True)
    return run.Ctx(project_dir=tmp_path, pipe=pipe, status=None,
                   angles=angles)


def test_stage_sync_low_confidence_continues(tmp_path, monkeypatch):
    """needs_manual non-empty -> return normally (no SystemExit, no
    needs_input status write — status is None, an update would crash)."""
    ctx = _ctx(tmp_path)
    out = {"offsets": [0.0, 5.0], "needs_manual": [1], "method": "xcorr",
           "coverage": {"intersection": [5.0, 100.0], "union": [0.0, 100.0]},
           "confidence_note": "unconfident"}
    import highlights.multiangle.sync as sync_mod
    monkeypatch.setattr(sync_mod, "write_sync", lambda *a, **k: out)
    monkeypatch.setattr(run, "apply_match_window_src", lambda c, s: None)

    res = run.stage_sync(ctx)          # must not raise SystemExit
    assert res is out


def test_main_offsets_sets_force(tmp_path, monkeypatch):
    """--offsets 0,5 --stages sync: ctx.force True (re-runs despite
    existing outputs); without --offsets: False."""
    angles = [{"label": f"a{i}", "url": None,
               "dir": tmp_path / "angles" / f"a{i}"} for i in range(2)]
    monkeypatch.setattr(run, "_load_angles", lambda *a, **k: angles)
    monkeypatch.setattr(run, "job_slot", lambda *a, **k: nullcontext())
    seen = []
    monkeypatch.setattr(run, "run_stages",
                        lambda ctx, names: seen.append((ctx, names)))

    rc = run.main(["--project-dir", str(tmp_path), "--stages", "sync",
                   "--offsets", "0,5"])
    assert rc == 0
    assert seen[-1][0].force is True
    assert seen[-1][1] == ["sync"]

    rc = run.main(["--project-dir", str(tmp_path), "--stages", "sync"])
    assert rc == 0
    assert seen[-1][0].force is False
