"""Verdict learning: dynamic windows in the store, render-triggered
training, /api/learning endpoint."""

import json
import time
from pathlib import Path

from conftest import SAMPLE, new_project, scoped

APP = Path(__file__).resolve().parents[1]


def _load(client, video):
    pid = new_project(client, video)
    r = client.post(scoped(pid, "/video"), json={"path": str(video)})
    assert r.status_code == 200, r.text
    r = client.post(scoped(pid, "/candidates/load"),
                    json={"path": str(SAMPLE / "candidates_short.json")})
    assert r.status_code == 200, r.text
    return pid, r.json()


def test_store_uses_event_window_when_sane():
    from highlights.app.backend.schemas import CandidatesFile
    from highlights.app.backend.store import make_candidates
    cf = CandidatesFile(**{
        "events": [
            {"type": "goal", "t": 50.0, "t_start": 42.0, "t_end": 65.0,
             "confidence": 0.9},
            {"type": "shot", "t": 80.0, "t_start": 0.0, "t_end": 1.0,
             "confidence": 0.5},   # insane window -> default 3/3
        ]})
    cands = make_candidates(cf, 200.0)
    assert cands[0].clip_start == 42.0 and cands[0].clip_end == 65.0
    assert cands[1].clip_start == 77.0 and cands[1].clip_end == 83.0


def test_learning_endpoint_empty(client, monkeypatch, tmp_path):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    r = client.get("/api/learning")
    assert r.status_code == 200
    assert r.json() == {"trained": False}


def test_learning_endpoint_meta(client, monkeypatch, tmp_path):
    ld = tmp_path / "learn"
    ld.mkdir()
    (ld / "verdict_lr.json").write_text(json.dumps(
        {"trained": True, "n_examples": 42, "version": "verdict_lr v1"}))
    monkeypatch.setenv("HL_LEARN_DIR", str(ld))
    assert client.get("/api/learning").json()["n_examples"] == 42


def test_render_triggers_training(client, sample_video, monkeypatch, tmp_path):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    pid, cands = _load(client, sample_video)
    r = client.post(scoped(pid, "/render"),
                    json={"ids": [cands[0]["id"]], "overlay": False})
    assert r.status_code == 200, r.text
    # _train_after_render is a daemon thread; wait for the history event
    import highlights.app.backend.history as history
    deadline = time.time() + 15
    kinds = []
    while time.time() < deadline:
        kinds = [e["kind"] for e in history.events(pid)]
        if any(k.startswith("model_train") for k in kinds):
            break
        time.sleep(0.3)
    assert "model_trained" in kinds or "model_train_skipped" in kinds


def test_stats5_uses_verdicts(client, sample_video, tmp_path):
    pid, cands = _load(client, sample_video)
    # no stats.json yet -> 404
    assert client.get(scoped(pid, "/stats")).status_code == 404
    import highlights.app.backend.main as m
    store = m.get_registry().get(pid)
    st = {"goals": 99, "highlights": 0}
    (store.pipeline_dir / "stats.json").write_text(json.dumps(st))
    # no verdicts yet -> pipeline stats returned unchanged
    r = client.get(scoped(pid, "/stats"))
    assert r.json()["goals"] == 99
    # make one goal + one non-goal verdict
    goal = next(c for c in cands if c["type"] == "goal")
    other = next(c for c in cands if c["type"] != "goal")
    client.patch(scoped(pid, f"/candidates/{goal['id']}"), json={"status": "confirmed"})
    client.patch(scoped(pid, f"/candidates/{other['id']}"), json={"status": "confirmed"})
    st = client.get(scoped(pid, "/stats")).json()
    assert st["goals"] == 1 and st["highlights"] == 1
    assert st["basis_events"] == "user verdicts"
