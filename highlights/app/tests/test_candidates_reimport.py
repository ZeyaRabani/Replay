"""Self-healing candidates re-import: drop negative-t events and reload
when the pipeline rewrites candidates.json before any user decision."""

import json
import os

from conftest import new_project

import highlights.app.backend.main as m
from highlights.app.backend import pipeline
from highlights.app.backend.schemas import CandidatesFile
from highlights.app.backend.store import make_candidates


def _ev(t, conf=0.7):
    return {"type": "shot", "t": t, "t_start": t - 2.0, "t_end": t + 2.0,
            "confidence": conf}


def _cands_json(*ts):
    return json.dumps({"events": [_ev(t) for t in ts],
                       "video_duration_s": 600.0})


def _done_project(client, video):
    pid = new_project(client, video)
    p = m.get_registry().get(pid)
    pipeline.write_status(p, {
        "state": "done", "stage": "done", "progress": 1.0,
        "stage_progress": 1.0, "message": "done", "error": None,
        "started_at": 1.0, "updated_at": 2.0, "finished_at": 2.0,
        "pid": None, "video_path": str(video),
        "video": {"duration_s": 600.0, "width": 320, "height": 240,
                  "fps": 10.0},
    })
    p.set_pipeline_state("done")
    return p


def _write_cands(p, text, mtime):
    f = p.pipeline_dir / "candidates.json"
    f.write_text(text)
    os.utime(f, (mtime, mtime))
    return f


def test_make_candidates_drops_negative_t():
    cf = CandidatesFile(**{
        "events": [_ev(-1315.0, 0.9), _ev(50.0, 0.7), _ev(80.0, 0.5)],
        "video_duration_s": 600.0})
    cands = make_candidates(cf, 600.0)
    assert [c.t for c in cands] == [50.0, 80.0]
    assert [c.id for c in cands] == ["c001", "c002"]
    assert sorted(c.rank for c in cands) == [1, 2]


def test_reimport_when_file_newer_and_all_pending(
        client, sample_video, monkeypatch):
    p = _done_project(client, sample_video)
    _write_cands(p, _cands_json(50.0), mtime=1000)
    pipeline.refresh(p)
    assert [c.t for c in p.candidates] == [50.0]
    assert p.meta["candidates_src_mtime"] == 1000

    # pipeline rewrote the file (newer mtime, extra event) -> re-import
    _write_cands(p, _cands_json(50.0, 90.0), mtime=2000)
    pipeline.refresh(p)
    assert [c.t for c in p.candidates] == [50.0, 90.0]
    assert p.meta["candidates_src_mtime"] == 2000

    # same mtime -> no re-import even with different content
    f = _write_cands(p, _cands_json(50.0, 90.0, 120.0), mtime=2000)
    pipeline.refresh(p)
    assert [c.t for c in p.candidates] == [50.0, 90.0]
    f.unlink()


def test_reimport_skipped_after_user_decision(
        client, sample_video, monkeypatch):
    p = _done_project(client, sample_video)
    _write_cands(p, _cands_json(50.0), mtime=1000)
    pipeline.refresh(p)
    assert len(p.candidates) == 1

    # user confirms the card -> a newer file must NOT clobber it
    p.candidates[0].status = "confirmed"
    p.save()
    _write_cands(p, _cands_json(50.0, 90.0), mtime=2000)
    pipeline.refresh(p)
    assert [c.t for c in p.candidates] == [50.0]
    assert p.candidates[0].status == "confirmed"
    # src mtime left at the first import
    assert p.meta["candidates_src_mtime"] == 1000
