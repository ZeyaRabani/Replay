"""run_analysis over the committed sample project (no real video)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from highlights.analysis import run as arun
from highlights.analysis.tests.mock_teams import make_mock_teams
from highlights.io import write_json_atomic

SAMPLE = Path(__file__).resolve().parent / "sample"


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    shutil.copytree(SAMPLE, root)
    fused = json.loads((root / "multiangle" / "fused_candidates.json").read_text())
    cands = [{
        "t": e["t"], "type": e["type"],
        "status": "rejected" if e.get("cross_validation") == "rejected"
        else e.get("status", "pending"),
        "confidence": e.get("confidence", 0.0),
        "signals": e.get("signals") or {},
    } for e in fused.get("events", [])]
    project = {
        "id": "sample", "title": "sample", "pipeline_state": "done",
        "source": {"kind": "multiangle", "angles": [
            {"label": "a0"}, {"label": "a1"}, {"label": "a2"}]},
        "candidates": cands,
    }
    write_json_atomic(root / "project.json", project, indent=1)
    return root


def test_resolve_context_no_video(tmp_path):
    root = _project(tmp_path)
    try:
        arun.resolve_context(root)
        raise AssertionError("expected PipelineError for missing video")
    except Exception as e:
        assert "no video for angle" in str(e)


def test_run_analysis_with_mocked_teams(tmp_path, monkeypatch):
    root = _project(tmp_path)
    # fabricate a video file so the teams stage resolves
    for d in (root / "angles").iterdir():
        (d / "match.mp4").write_bytes(b"fake")

    def fake_teams(video, out_path, **kw):
        d = make_mock_teams(root)
        write_json_atomic(Path(out_path), d, indent=0)
        return d

    monkeypatch.setattr(arun.teams_mod, "run_teams_pass", fake_teams)
    res = arun.run_analysis(root, log=lambda m: None)
    assert (root / "analysis" / "match_stats.json").is_file()
    assert (root / "analysis" / "summary.json").is_file()
    stats = json.loads((root / "analysis" / "match_stats.json").read_text())
    assert stats["caveats"]
    want_ref = max(range(3), key=lambda i: arun._angle_duration(root / "angles" / f"a{i}"))
    assert stats["ref_angle"] == want_ref
    assert len(stats["halves"]) == 2
    for s in stats["shots"]:
        assert {"t_shared", "t_out", "t_file"} <= set(s)
    text = res["summary"]["text"]
    assert len(text) > 50
