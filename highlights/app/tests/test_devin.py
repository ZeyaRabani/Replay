"""Help -> Devin handoff: config gate, tools, ticket routes, ownership.
The Devin HTTP layer (_request) is stubbed — no network."""

import json
import time

from conftest import new_project

import highlights.app.backend.main as m
from highlights.app.backend.assistant import devin, tools


def _cfg(monkeypatch, tmp_path, users=None):
    cfg = {"api_key": "SECRET-KEY", "allowed_users": users or ["tester"],
           "daily_limit": 3}
    cfg_path = tmp_path / "devin.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(devin, "CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(devin, "TICKETS_PATH",
                        str(tmp_path / "devin_tickets.json"))
    monkeypatch.setattr(devin, "_cfg", None)
    monkeypatch.setattr(devin, "_cfg_tried", False)
    monkeypatch.setattr(devin, "_session_cache", {})
    calls = []

    def fake_request(method, path, body=None):
        calls.append((method, path, body))
        if method == "POST" and path == "/v1/sessions":
            return {"session_id": "sid-1", "url": "https://app.devin.ai/s/1"}
        if method == "GET":
            return {"status_enum": "blocked",
                    "messages": [
                        {"type": "devin_message", "message": "looking into it",
                         "timestamp": 1.0},
                        {"type": "user_message", "message": "thanks",
                         "timestamp": 2.0},
                        {"type": "system", "message": "internal",
                         "timestamp": 3.0},
                    ]}
        return {}
    monkeypatch.setattr(devin, "_request", fake_request)
    return calls


def test_unconfigured(client, monkeypatch, tmp_path):
    monkeypatch.setattr(devin, "CONFIG_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(devin, "TICKETS_PATH", str(tmp_path / "t.json"))
    monkeypatch.setattr(devin, "_cfg", None)
    monkeypatch.setattr(devin, "_cfg_tried", False)
    assert not devin.configured()
    r = client.get("/api/assistant/status")
    assert r.json()["devin"] is False
    names = [t["name"] for t in tools.schemas("tester")]
    assert "send_to_devin" not in names
    r = client.post("/api/assistant/devin/sid-1/message",
                    json={"message": "hi"})
    assert r.status_code == 503


def test_send_needs_confirmation(client, monkeypatch, tmp_path):
    calls = _cfg(monkeypatch, tmp_path)
    ok, res, _ = tools.execute("tester", "send_to_devin",
                             {"summary": "cut looks wrong", "confirmed": False})
    assert not ok and "needs confirmation" in res
    assert calls == []        # no API call before the user agrees


def test_send_confirmed_creates_session(client, sample_video, monkeypatch,
                                        tmp_path):
    calls = _cfg(monkeypatch, tmp_path)
    pid = new_project(client, sample_video)
    p = m.get_registry().get(pid)
    p.log_path.write_text("line1\nline2 tracking ok\n")
    ok, res, _ = tools.execute(
        "tester", "send_to_devin",
        {"summary": "the cut skipped the goal", "confirmed": True},
        ctx={"path": f"/projects/{pid}", "project_id": pid,
             "messages": [{"role": "user", "content": "it's wrong"}]})
    assert ok and res["sent"] and res["url"]
    assert len(calls) == 1 and calls[0][1] == "/v1/sessions"
    prompt = calls[0][2]["prompt"]
    assert "the cut skipped the goal" in prompt
    assert f"/projects/{pid}" in prompt
    assert p.title in prompt
    assert "line2 tracking ok" in prompt
    assert "user: it's wrong" in prompt
    assert "SECRET-KEY" not in prompt
    tickets = devin.list_tickets("tester")
    assert len(tickets) == 1 and tickets[0]["session_id"] == "sid-1"
    assert tickets[0]["user"] == "tester" and tickets[0]["project_id"] == pid


def test_send_refused_for_non_allowed_user(client, monkeypatch, tmp_path):
    calls = _cfg(monkeypatch, tmp_path, users=["admin"])
    ok, res, _ = tools.execute("tester", "send_to_devin",
                               {"summary": "x", "confirmed": True})
    assert not ok and "not allowed" in res
    assert calls == []
    assert "send_to_devin" not in [t["name"] for t in tools.schemas("tester")]
    assert "send_to_devin" in [t["name"] for t in tools.schemas("admin")]


def test_daily_limit(client, monkeypatch, tmp_path):
    calls = _cfg(monkeypatch, tmp_path)
    for i in range(3):
        devin.add_ticket({"session_id": f"s{i}", "user": "tester",
                          "title": "t", "project_id": None,
                          "created_at": time.time()})
    ok, res, _ = tools.execute("tester", "send_to_devin",
                               {"summary": "x", "confirmed": True})
    assert not ok and "limit" in res
    assert calls == []


def test_devin_tickets_route_own_only(client, monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path)
    devin.add_ticket({"session_id": "sid-1", "user": "tester", "title": "mine",
                      "project_id": None, "created_at": time.time()})
    devin.add_ticket({"session_id": "sid-2", "user": "other", "title": "theirs",
                      "project_id": None, "created_at": time.time()})
    r = client.get("/api/assistant/devin")
    assert r.status_code == 200
    rows = r.json()["tickets"]
    assert len(rows) == 1 and rows[0]["session_id"] == "sid-1"
    assert rows[0]["status"] == "needs_reply"      # blocked mapped
    msgs = rows[0]["messages"]
    assert [x["source"] for x in msgs] == ["devin", "user"]  # system dropped
    assert msgs[0]["text"] == "looking into it"


def test_reply_route_ownership(client, monkeypatch, tmp_path):
    calls = _cfg(monkeypatch, tmp_path)
    devin.add_ticket({"session_id": "sid-1", "user": "tester", "title": "mine",
                      "project_id": None, "created_at": time.time()})
    devin.add_ticket({"session_id": "sid-2", "user": "other", "title": "theirs",
                      "project_id": None, "created_at": time.time()})
    r = client.post("/api/assistant/devin/sid-2/message",
                    json={"message": "hi"})
    assert r.status_code == 404
    r = client.post("/api/assistant/devin/sid-1/message",
                    json={"message": "please hurry"})
    assert r.status_code == 200 and r.json()["ok"]
    assert calls == [("POST", "/v1/sessions/sid-1/message",
                      {"message": "please hurry"})]


def test_check_and_reply_tools(client, monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path)
    devin.add_ticket({"session_id": "sid-1", "user": "tester", "title": "mine",
                      "project_id": None, "created_at": time.time()})
    ok, res, _ = tools.execute("tester", "check_devin_fix", {})
    assert ok and res["status"] == "needs_reply"
    assert res["devin_messages"] == ["looking into it"]
    ok, res, _ = tools.execute("tester", "reply_to_devin",
                               {"message": "try the next file"})
    assert ok and res["sent"]
    ok, res, _ = tools.execute("other", "check_devin_fix",
                               {"session_id": "sid-1"})
    assert not ok and res == "ticket not found"
