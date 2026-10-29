"""Replay assistant HTTP routes: POST /api/assistant/chat +
GET /api/assistant/status. Owns the chat loop: LLM <-> tool calls."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import devin, llm, tools

MAX_HISTORY = 30
MAX_MSG_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 12000
MAX_ITERATIONS = 8


def _err_body(e: Exception) -> str:
    """The error body — read ONCE (HTTPError streams are single-use)."""
    with contextlib.suppress(Exception):
        return (e.read().decode() if hasattr(e, "read") else str(e))[:400]
    return ""


def _err_desc(e: Exception, body: str) -> str:
    """Status + short body for logs; never headers/keys."""
    code = getattr(e, "code", None) or type(e).__name__
    return f"{code} {body[:160]}"


def _provider_error_text(e: Exception, body: str) -> str:
    code = getattr(e, "code", None)
    low = (body or "").lower()
    if code == 402 or "resource_exhausted" in low or "credit" in low:
        return ("Help's AI (Google Gemini) is out of credits, so I can't "
                "answer right now. The Devin tab still works.")
    if code == 429:
        return ("Help's AI is busy right now. Please try again in a "
                "minute.")
    return (f"Help's AI is unavailable right now (error {code or 'unknown'}). "
            "Please try again later.")


def _knowledge() -> str:
    return (Path(__file__).parent / "KNOWLEDGE.md").read_text()


class ChatMessage(BaseModel):
    role: str
    content: str


class PageCtx(BaseModel):
    path: str = ""
    project_id: str | None = None


class DevinMessageBody(BaseModel):
    message: str = ""


class ChatBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    page: PageCtx = Field(default_factory=PageCtx)


def make_router(current_user):
    """current_user: main's auth dependency (X-User header -> profile name)."""
    router = APIRouter()

    @router.get("/api/assistant/status")
    def status(user: str = Depends(current_user)) -> dict:
        return {"configured": llm.configured(), "model": llm.model(),
                "devin": devin.configured() and devin.allowed(user)}

    @router.get("/api/assistant/devin")
    def devin_tickets(user: str = Depends(current_user)) -> dict:
        if not (devin.configured() and devin.allowed(user)):
            return {"tickets": []}
        mine = sorted(devin.list_tickets(user),
                      key=lambda t: t.get("created_at", 0), reverse=True)
        out = []
        for i, t in enumerate(mine):
            row = {"session_id": t["session_id"], "url": t.get("url"),
                   "title": t.get("title"), "project_id": t.get("project_id"),
                   "created_at": t.get("created_at"), "messages": []}
            if i < 3:
                try:
                    sess = devin.get_session(t["session_id"])
                    row["status"] = devin.map_status(sess)
                    row["messages"] = devin.session_messages(sess)
                except devin.DevinError:
                    row["status"] = "unknown"
            else:
                row["status"] = "finished"
            out.append(row)
        return {"tickets": out}

    @router.post("/api/assistant/devin/{session_id}/message")
    def devin_reply(session_id: str, body: DevinMessageBody,
                    user: str = Depends(current_user)) -> dict:
        if not devin.configured():
            raise HTTPException(503, "devin not configured")
        if devin.get_ticket(user, session_id) is None:
            raise HTTPException(404, "ticket not found")
        msg = (body.message or "").strip()
        if not msg:
            raise HTTPException(422, "message is required")
        try:
            devin.send_message(session_id, msg)
        except devin.DevinError as e:
            raise HTTPException(502, str(e)) from e
        return {"ok": True}

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
            try:
                resp = llm.chat(system, convo, tools.schemas(user))
            except Exception as e:
                # provider outage/billing: answer in-app, never 500
                ebody = _err_body(e)
                reply = _provider_error_text(e, ebody)
                print(f"assistant: llm.chat failed: {_err_desc(e, ebody)}")
                break
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
                    user, call["name"], call.get("arguments") or {},
                    ctx={"path": body.page.path,
                         "project_id": body.page.project_id,
                         "messages": msgs})
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
