"""Project meta (pitch/camera tags) + feedback export."""

import json

from conftest import scoped


def _wait_done(client, pid, timeout=10.0, states=("done", "failed")):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(scoped(pid, "")).json()
        if d["pipeline_state"] in states:
            return d
        time.sleep(0.05)
    return client.get(scoped(pid, "")).json()


def test_meta_persisted_and_returned(client, sample_video):
    r = client.post("/api/projects", json={
        "path": str(sample_video), "pitch_type": "11", "camera": "ultrawide"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["meta"] == {"pitch_type": "11", "camera": "ultrawide"}
    # survives reload from disk: detail endpoint reads project.json fresh
    pid = d["id"]
    d2 = client.get(scoped(pid, "")).json()
    assert d2["meta"]["pitch_type"] == "11"
    assert d2["meta"]["camera"] == "ultrawide"


def test_meta_absent_is_empty(client, sample_video):
    r = client.post("/api/projects", json={"path": str(sample_video)})
    assert r.status_code == 200, r.text
    assert r.json()["meta"] == {}


def test_meta_invalid_rejected(client, sample_video):
    r = client.post("/api/projects", json={
        "path": str(sample_video), "pitch_type": "12"})
    assert r.status_code == 422
    r = client.post("/api/projects", json={
        "path": str(sample_video), "camera": "drone"})
    assert r.status_code == 422


def test_feedback_export(client, sample_video):
    r = client.post("/api/projects", json={
        "path": str(sample_video), "pitch_type": "7", "camera": "normal"})
    pid = r.json()["id"]
    _wait_done(client, pid)
    cands = client.get(scoped(pid, "/candidates")).json()
    assert len(cands) >= 2
    cid = cands[0]["id"]
    r = client.patch(scoped(pid, f"/candidates/{cid}"),
                     json={"status": "confirmed"})
    assert r.status_code == 200

    r = client.get("/api/feedback/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    # only non-pending: at least the confirmed one, none pending
    assert all(l["status"] != "pending" for l in lines)
    assert any(l["candidate_id"] == cid and l["status"] == "confirmed"
               for l in lines)
    row = next(l for l in lines if l["candidate_id"] == cid)
    assert row["project_id"] == pid
    assert row["meta"] == {"pitch_type": "7", "camera": "normal"}
    assert row["source_kind"] == "path"
    assert row["video"]["duration_s"] > 0
    assert set(row) == {"project_id", "owner", "title", "meta", "video",
                       "candidate_id", "type", "t", "t_start", "t_end",
                       "confidence", "signals", "status", "cross_validation",
                       "source_kind"}

    r = client.get("/api/feedback/export", params={"all": "1"})
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    assert any(l["status"] == "pending" for l in lines)
