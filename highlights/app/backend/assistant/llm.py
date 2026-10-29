"""Provider client for the Replay assistant — stdlib urllib only.

Config (env):
  HL_ASSISTANT_PROVIDER  openai|anthropic; default: openai if OPENAI_API_KEY,
                         else anthropic if ANTHROPIC_API_KEY, else unconfigured
  HL_ASSISTANT_MODEL     default gpt-4o-mini / claude-3-5-haiku-latest
  HL_ASSISTANT_BASE_URL  optional override (default the provider's API root)

Optional file config: REPLAY_LLM_CONFIG (default /data/shared/llm.json,
mode 600):
  {"provider": "openai", "api_key": "...", "base_url": "...",
   "model": "..."}
Every field overrides the matching env var (provider ->
HL_ASSISTANT_PROVIDER, api_key -> OPENAI_API_KEY/ANTHROPIC_API_KEY by
provider, base_url -> HL_ASSISTANT_BASE_URL, model -> HL_ASSISTANT_MODEL).
Read on each call through a small mtime cache, so writing the file takes
effect WITHOUT a restart. Missing file/fields -> env behaviour unchanged.
The api_key is never logged.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

_TIMEOUT = 60.0
CONFIG_PATH = os.environ.get("REPLAY_LLM_CONFIG", "/data/shared/llm.json")

# ToolSpec = {"name": str, "description": str, "parameters": JSON-schema dict}
# LLMReply = {"text": str|None, "tool_calls": [{"id","name","arguments"}]}

_cfg: dict = {}
_cfg_mtime = -1.0


def _file_cfg() -> dict:
    """llm.json, re-read only when its mtime changed."""
    global _cfg, _cfg_mtime
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        mtime = -1.0
    if mtime == _cfg_mtime:
        return _cfg
    _cfg_mtime = mtime
    try:
        v = json.loads(Path(CONFIG_PATH).read_text())
        _cfg = v if isinstance(v, dict) else {}
    except Exception:
        _cfg = {}
    return _cfg


def _merged() -> dict:
    """Effective config: file fields win over env, per-field."""
    f = _file_cfg()
    prov = (str(f.get("provider") or "").strip().lower()
            or (os.environ.get("HL_ASSISTANT_PROVIDER") or "").strip().lower())
    if prov not in ("openai", "anthropic"):
        prov = None
    if prov is None and not f.get("provider"):
        if os.environ.get("OPENAI_API_KEY"):
            prov = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            prov = "anthropic"
    key_env = ("OPENAI_API_KEY" if prov == "openai"
               else "ANTHROPIC_API_KEY" if prov == "anthropic" else "")
    return {
        "provider": prov,
        "api_key": (str(f.get("api_key") or "").strip()
                    or os.environ.get(key_env, "")),
        "base_url": (str(f.get("base_url") or "").strip()
                     or (os.environ.get("HL_ASSISTANT_BASE_URL")
                         or "").strip()),
        "model": (str(f.get("model") or "").strip()
                  or (os.environ.get("HL_ASSISTANT_MODEL") or "").strip()),
    }


def provider() -> str | None:
    return _merged()["provider"]


def configured() -> bool:
    return provider() is not None


def model() -> str | None:
    cfg = _merged()
    if cfg["provider"] is None:
        return None
    if cfg["model"]:
        return cfg["model"]
    return ("gpt-4o-mini" if cfg["provider"] == "openai"
            else "claude-3-5-haiku-latest")


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
    cfg = _merged()
    base = (cfg["base_url"] or "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": model(),
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": _openai_tools(tools),
    }
    resp = _post(f"{base}/chat/completions",
                 {"Authorization": f"Bearer {_merged()['api_key']}"},
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
    base = (_merged()["base_url"]
            or "https://api.anthropic.com").rstrip("/")
    body = {
        "model": model(),
        "max_tokens": 2048,
        "system": system,
        "messages": messages,
        "tools": _anthropic_tools(tools),
    }
    resp = _post(f"{base}/v1/messages",
                 {"x-api-key": _merged()["api_key"],
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
