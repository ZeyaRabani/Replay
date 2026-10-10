"""Replay assistant HTTP routes: POST /api/assistant/chat +
GET /api/assistant/status. Owns the chat loop: LLM <-> tool calls."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import llm, tools

MAX_HISTORY = 30
MAX_MSG_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 12000
MAX_ITERATIONS = 8


def _knowledge() -> str:
    return (Path(__file__).parent / "KNOWLEDGE.md").read_text()


class ChatMessage(BaseModel):
    role: str
    content: str


class PageCtx(BaseModel):
    path: str = ""
    project_id: str | None = None


class ChatBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    page: PageCtx = Field(default_factory=PageCtx)


def make_router(current_user):
    """current_user: main's auth dependency (X-User header -> profile name)."""
    router = APIRouter()

    @router.get("/api/assistant/status")
    def status(user: str = Depends(current_user)) -> dict:
        return {"configured": llm.configured(), "model": llm.model()}

    @router.post("/api/assistant/chat")
    def chat(body: ChatBody, user: str = Depends(current_user)) -> dict:
        if not llm.configured():
            raise HTTPException(503, "assistant not configured")
        t0 = time.time()
        from highlights.app.backend import main as m

        msgs = [{"role": mm.role, "content": mm.content[:MAX_MSG_CHARS]}
                for mm in body.messages[-MAX_HISTORY:]
                if mm.role in ("user", "assistant") and mm.content]
        if not msgs or msgs[-1]["role"] != "user":
            raise HTTPException(422, "last message must be from the user")

        admin = "yes" if m._is_admin(user) else "no"
        system = (
            _knowledge()
            + f"\n\n## Live context\nUser profile: {user} (admin: {admin}). "
            f"Current page: {body.page.path or 'unknown'}. "
            f"Project open: {body.page.project_id or 'none'}. "
            f"Server time (UTC): "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}.")

        convo = list(msgs)
        actions: list[dict] = []
        reply: str | None = None
        for _ in range(MAX_ITERATIONS):
            resp = llm.chat(system, convo, tools.schemas())
            calls = resp.get("tool_calls") or []
            if not calls:
                reply = resp.get("text")
                break
            # record the assistant turn that requested the tools so the
            # provider sees calls and results paired correctly
            convo.append(resp.get("_raw_message")
                         or {"role": "assistant", "content": resp.get("text")})
            for call in calls:
                ok, result, summary = tools.execute(
                    user, call["name"], call.get("arguments") or {})
                payload = (result if isinstance(result, str)
                           else json.dumps(result, default=str))
                payload = payload[:MAX_TOOL_RESULT_CHARS]
                actions.append({"tool": call["name"], "ok": ok,
                                "summary": summary})
                convo.append(llm.tool_result_message(
                    call["id"], call["name"], payload))
        else:
            reply = None
        if not reply:
            reply = ("I ran out of steps — please ask again more "
                     "specifically.")
        print(f"assistant: user={user} n_messages={len(msgs)} "
              f"tools={[a['tool'] for a in actions]} "
              f"{time.time() - t0:.1f}s")
        return {"reply": reply, "actions": actions, "model": llm.model()}

    return router
