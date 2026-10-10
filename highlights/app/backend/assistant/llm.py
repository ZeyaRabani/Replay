"""Provider client for the Replay assistant — stdlib urllib only.

Config (env):
  HL_ASSISTANT_PROVIDER  openai|anthropic; default: openai if OPENAI_API_KEY,
                         else anthropic if ANTHROPIC_API_KEY, else unconfigured
  HL_ASSISTANT_MODEL     default gpt-4o-mini / claude-3-5-haiku-latest
  HL_ASSISTANT_BASE_URL  optional override (default the provider's API root)
"""

from __future__ import annotations

import json
import os
import urllib.request

_TIMEOUT = 60.0

# ToolSpec = {"name": str, "description": str, "parameters": JSON-schema dict}
# LLMReply = {"text": str|None, "tool_calls": [{"id","name","arguments"}]}


def provider() -> str | None:
    p = (os.environ.get("HL_ASSISTANT_PROVIDER") or "").strip().lower()
    if p in ("openai", "anthropic"):
        return p
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def configured() -> bool:
    return provider() is not None


def model() -> str | None:
    p = provider()
    if p is None:
        return None
    m = (os.environ.get("HL_ASSISTANT_MODEL") or "").strip()
    if m:
        return m
    return "gpt-4o-mini" if p == "openai" else "claude-3-5-haiku-latest"


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


def _openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in tools]


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]} for t in tools]


def chat(system: str, messages: list[dict], tools: list[dict]) -> dict:
    """One provider round-trip. Returns LLMReply."""
    p = provider()
    if p == "openai":
        return _openai_chat(system, messages, tools)
    if p == "anthropic":
        return _anthropic_chat(system, messages, tools)
    raise RuntimeError("assistant not configured")


def _openai_chat(system: str, messages: list[dict], tools: list[dict]) -> dict:
    base = (os.environ.get("HL_ASSISTANT_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": model(),
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": _openai_tools(tools),
    }
    resp = _post(f"{base}/chat/completions",
                 {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                 body)
    msg = resp["choices"][0]["message"]
    calls = [{
        "id": tc["id"],
        "name": tc["function"]["name"],
        "arguments": (json.loads(tc["function"].get("arguments") or "{}")
                      if isinstance(tc["function"].get("arguments"), str)
                      else tc["function"].get("arguments") or {}),
    } for tc in (msg.get("tool_calls") or [])]
    return {"text": msg.get("content"), "tool_calls": calls,
            "_raw_message": msg}


def _anthropic_chat(system: str, messages: list[dict],
                    tools: list[dict]) -> dict:
    base = (os.environ.get("HL_ASSISTANT_BASE_URL")
            or "https://api.anthropic.com").rstrip("/")
    body = {
        "model": model(),
        "max_tokens": 2048,
        "system": system,
        "messages": messages,
        "tools": _anthropic_tools(tools),
    }
    resp = _post(f"{base}/v1/messages",
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                  "anthropic-version": "2023-06-01"}, body)
    text = None
    calls = []
    for block in resp.get("content") or []:
        if block.get("type") == "text":
            text = (text or "") + block.get("text", "")
        elif block.get("type") == "tool_use":
            calls.append({"id": block["id"], "name": block["name"],
                          "arguments": block.get("input") or {}})
    return {"text": text, "tool_calls": calls,
            "_raw_message": {"role": "assistant",
                             "content": resp.get("content") or []}}


def tool_result_message(call_id: str, name: str, content_str: str,
                        assistant_msg: dict | None = None) -> dict:
    """The provider-shaped message that carries one tool result back."""
    p = provider()
    if p == "openai":
        return {"role": "tool", "tool_call_id": call_id,
                "name": name, "content": content_str}
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id,
                         "content": content_str}]}
