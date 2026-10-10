"""Replay assistant: status gate, tool-call loop, destructive guards,
cross-user denial, and llm.py provider request/response shapes."""

import json
import urllib.request

from conftest import new_project

import highlights.app.backend.main as m
from highlights.app.backend.assistant import llm


def _env(monkeypatch, provider="openai"):
    monkeypatch.setenv("HL_ASSISTANT_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    monkeypatch.delenv("HL_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("HL_ASSISTANT_BASE_URL", raising=False)


def test_status_503_when_unconfigured(client, monkeypatch):
    monkeypatch.delenv("HL_ASSISTANT_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.get("/api/assistant/status")
    assert r.status_code == 200 and r.json() == {
        "configured": False, "model": None}
    r = client.post("/api/assistant/chat", json={
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]


def test_chat_tool_loop(client, sample_video, monkeypatch):
    _env(monkeypatch)
    pid = new_project(client, sample_video)
    calls = []

    def fake_chat(system, messages, tools_):
        calls.append(messages)
        if len(calls) == 1:
            return {"text": None, "tool_calls": [
                {"id": "c1", "name": "list_projects", "arguments": {}}],
                "_raw_message": {"role": "assistant", "content": None,
                                 "tool_calls": []}}
        # second round: the tool result is the last message; it must
        # contain our project title
        last = json.dumps(messages[-1])
        assert "sample" in last or pid in last
        return {"text": "Your match is queued.", "tool_calls": []}

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = client.post("/api/assistant/chat", json={
        "messages": [{"role": "user", "content": "what is running?"}],
        "page": {"path": "/projects"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "Your match is queued."
    assert body["model"] == "gpt-4o-mini"
    assert body["actions"][0]["tool"] == "list_projects"
    assert body["actions"][0]["ok"] is True


def test_delete_requires_confirmation(client, sample_video, monkeypatch):
    _env(monkeypatch)
    pid = new_project(client, sample_video)

    n = [0]

    def fake_chat(system, messages, tools_):
        n[0] += 1
        if n[0] == 1:
            return {"text": None, "tool_calls": [
                {"id": "c1", "name": "delete_project",
                 "arguments": {"project_id": pid, "confirmed": False}}]}
        return {"text": "refused", "tool_calls": []}

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = client.post("/api/assistant/chat", json={
        "messages": [{"role": "user", "content": "delete it"}]})
    assert r.status_code == 200, r.text
    [act] = r.json()["actions"]
    assert act["tool"] == "delete_project" and act["ok"] is False
    assert "confirmation" in act["summary"]
    assert m.get_registry().get(pid) is not None


def test_other_users_project_denied(client, sample_video, monkeypatch):
    _env(monkeypatch)
    other_pid = new_project(client, sample_video)
    p = m.get_registry().get(other_pid)
    p.owner = "someone_else"
    p.save()

    n = [0]

    def fake_chat(system, messages, tools_):
        n[0] += 1
        if n[0] == 1:
            return {"text": None, "tool_calls": [
                {"id": "c1", "name": "get_project_status",
                 "arguments": {"project_id": other_pid}}]}
        return {"text": "cannot see it", "tool_calls": []}

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = client.post("/api/assistant/chat", json={
        "messages": [{"role": "user", "content": "status?"}]})
    assert r.status_code == 200, r.text
    [act] = r.json()["actions"]
    assert act["ok"] is False and "not found" in act["summary"]


# ---------- llm.py request builders / response parsing (no network) ----------


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _capture(monkeypatch, payload):
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["headers"] = {k.lower(): v for k, v in req.header_items()}
        sent["body"] = json.loads(req.data.decode())
        return _Resp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


TOOLS = [{"name": "list_projects", "description": "List matches.",
          "parameters": {"type": "object", "properties": {},
                         "required": []}}]


def test_openai_request_and_parse(monkeypatch):
    _env(monkeypatch)
    sent = _capture(monkeypatch, {
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "list_projects", "arguments": "{}"}}]}}]})
    out = llm.chat("sys", [{"role": "user", "content": "hi"}], TOOLS)
    assert sent["url"].endswith("/chat/completions")
    assert sent["headers"]["authorization"] == "Bearer x"
    tool = sent["body"]["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "list_projects"
    assert tool["function"]["parameters"]["type"] == "object"
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert out["tool_calls"] == [
        {"id": "c1", "name": "list_projects", "arguments": {}}]
    tr = llm.tool_result_message("c1", "list_projects", "[]")
    assert tr["role"] == "tool" and tr["tool_call_id"] == "c1"


def test_anthropic_request_and_parse(monkeypatch):
    _env(monkeypatch, provider="anthropic")
    sent = _capture(monkeypatch, {
        "content": [
            {"type": "text", "text": "checking…"},
            {"type": "tool_use", "id": "u1", "name": "list_projects",
             "input": {}}]})
    out = llm.chat("sys", [{"role": "user", "content": "hi"}], TOOLS)
    assert sent["url"].endswith("/v1/messages")
    assert sent["headers"]["x-api-key"] == "y"
    assert sent["headers"]["anthropic-version"] == "2023-06-01"
    tool = sent["body"]["tools"][0]
    assert tool["name"] == "list_projects"
    assert tool["input_schema"]["type"] == "object"
    assert sent["body"]["system"] == "sys"
    assert sent["body"]["model"] == "claude-3-5-haiku-latest"
    assert out["text"] == "checking…"
    assert out["tool_calls"] == [
        {"id": "u1", "name": "list_projects", "arguments": {}}]
    tr = llm.tool_result_message("u1", "list_projects", "[]")
    assert tr["role"] == "user"
    assert tr["content"][0]["type"] == "tool_result"
    assert tr["content"][0]["tool_use_id"] == "u1"
