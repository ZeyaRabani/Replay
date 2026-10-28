"""Devin sessions from the Help panel — stdlib urllib only.

Config: /data/shared/devin.json
  {"api_key": "apk_...", "allowed_users": ["zeya", "admin"],
   "daily_limit": 3}
Missing file or key -> configured() False and everything no-ops.
The api_key is never logged or included in errors.

Tickets (one per session) live in /data/shared/devin_tickets.json:
  [{"session_id","url","user","title","project_id","created_at"}]

Devin API v1 (https://api.devin.ai): POST /v1/sessions,
GET /v1/sessions/{sid}, POST /v1/sessions/{sid}/message.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_PATH = os.environ.get("REPLAY_DEVIN_CONFIG", "/data/shared/devin.json")
TICKETS_PATH = os.environ.get("REPLAY_DEVIN_TICKETS",
                              "/data/shared/devin_tickets.json")
BASE = "https://api.devin.ai"
_TIMEOUT = 30.0
DAY_S = 24 * 3600
_SESSION_CACHE_S = 10.0

_cfg: dict | None = None
_cfg_tried = False
_lock = threading.Lock()
_session_cache: dict[str, tuple[float, dict]] = {}

PROMPT_TEMPLATE = """\
You are fixing a problem reported from the Help button of Replay (https://replay-highlights.vercel.app), Zeya's football-highlight app. Zeya will not open the Devin app: every message you send them is shown inside Replay's Help panel, and their replies come back to you the same way. Write to them in plain, non-technical language, and keep it short.

## The report
From profile: {user}
Page: {page}
Match open: {project_line}
Problem, as described in the Help chat:
{summary}

Recent Help chat (oldest first):
{chat}

## Live details captured when the report was sent (UTC {now})
{context}

## Where things are
- Code: GitHub ZeyaRabani/Replay. `main` is exactly what is live. The app is in highlights/ (backend: highlights/app/backend, FastAPI; frontend: highlights/app/frontend, Vite + React).
- Backend: Oracle VM 79.72.77.103 (user ubuntu, SSH private key in secret REPLAY_ORACLE_SSH_KEY), Docker container deploy-replay-1, data under /data inside the container.
- Frontend: Vercel project replay-highlights (token in secret REPLAY_VERCEL_TOKEN), deployed with the Vercel CLI, not from git.
- GitHub pushes: use secret SPLICE_GITHUB_TOKEN (it can push to ZeyaRabani/Replay) via https://x-access-token:$SPLICE_GITHUB_TOKEN@github.com/ZeyaRabani/Replay.git. It cannot open pull requests.

## Rules
1. Find the cause first. If it isn't a code problem (for example expired YouTube cookies, a private video, or a feature used the wrong way), tell Zeya how to fix it instead of changing code.
2. Keep changes minimal. Run `pytest highlights/app/tests -q` and `ruff check highlights`; for frontend changes also `npx tsc --noEmit` and `npx vite build` in highlights/app/frontend.
3. Commit on a branch named devin/<unix-time>-help-<short-slug>, push it, then fast-forward `main` to it (Zeya has approved merging fixes into main). Never force-push.
4. Restarting the backend kills running matches and downloads. Only restart when no match is processing and nobody is downloading; if something is running, wait and check again about every 10 minutes. Back up every file you replace in the container first. After restarting, check that GET /api/users returns 200, and roll back from the backup if it doesn't.
5. Never delete or overwrite matches, cuts, source videos or the archive, and never upload to or delete from the Cloudflare R2 bucket by hand. Never print or commit secrets.
6. Don't ask Zeya technical questions. Ask only when you truly need a decision from them, as one short plain-language question.
7. When you're done, send one short final message: what was wrong, what you changed, and whether it's live. Then stop.
"""


class DevinError(Exception):
    pass


def _load_cfg() -> dict | None:
    global _cfg, _cfg_tried
    if _cfg_tried:
        return _cfg
    _cfg_tried = True
    try:
        cfg = json.loads(Path(CONFIG_PATH).read_text())
        if cfg.get("api_key"):
            _cfg = cfg
    except Exception:
        pass
    return _cfg


def configured() -> bool:
    return _load_cfg() is not None


def allowed(user: str) -> bool:
    cfg = _load_cfg()
    if cfg is None:
        return False
    users = cfg.get("allowed_users")
    return not isinstance(users, list) or user in users


def can_create(user: str) -> bool:
    """Fewer than daily_limit tickets by this user in the last 24 h."""
    if not configured():
        return False
    cfg = _load_cfg() or {}
    limit = int(cfg.get("daily_limit") or 3)
    cutoff = time.time() - DAY_S
    n = sum(1 for t in _tickets()
            if t.get("user") == user and t.get("created_at", 0) > cutoff)
    return n < limit


# ---- ticket store -------------------------------------------------------

def _tickets() -> list[dict]:
    try:
        v = json.loads(Path(TICKETS_PATH).read_text())
        return v if isinstance(v, list) else []
    except Exception:
        return []


def list_tickets(user: str) -> list[dict]:
    return [t for t in _tickets() if t.get("user") == user]


def get_ticket(user: str, sid: str) -> dict | None:
    for t in _tickets():
        if t.get("session_id") == sid:
            return t if t.get("user") == user else None
    return None


def add_ticket(t: dict) -> None:
    with _lock:
        rows = _tickets()
        rows.append(t)
        path = Path(TICKETS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", dir=path.parent, delete=False) as tf:
            json.dump(rows, tf)
            tmp = tf.name
        os.replace(tmp, path)


# ---- API ----------------------------------------------------------------

def _request(method: str, path: str, body: dict | None = None) -> dict:
    key = (_load_cfg() or {}).get("api_key")
    if not key:
        raise DevinError("devin not configured")
    req = urllib.request.Request(
        BASE + path, method=method,
        data=(json.dumps(body).encode() if body is not None else None),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        snippet = ""
        with contextlib.suppress(Exception):
            snippet = e.read().decode()[:200]
        raise DevinError(f"devin API {e.code}: {snippet}") from e
    except Exception as e:
        raise DevinError(f"devin API: {e}") from e


def create_session(prompt: str, title: str) -> dict:
    return _request("POST", "/v1/sessions",
                    {"prompt": prompt, "title": title,
                     "tags": ["replay-help"], "idempotent": False})


def get_session(sid: str) -> dict:
    with _lock:
        hit = _session_cache.get(sid)
        if hit and time.time() - hit[0] < _SESSION_CACHE_S:
            return hit[1]
    out = _request("GET", f"/v1/sessions/{sid}")
    with _lock:
        _session_cache[sid] = (time.time(), out)
    return out


def send_message(sid: str, text: str) -> None:
    _request("POST", f"/v1/sessions/{sid}/message", {"message": text})
    with _lock:
        _session_cache.pop(sid, None)


# ---- prompt building ----------------------------------------------------

def build_prompt(user: str, page: str, project_id: str | None,
                 summary: str, chat: list[dict]) -> str:
    """Fill PROMPT_TEMPLATE. `chat` = [{role, content}] from the Help panel."""
    from . import tools as _t

    project_line = "none"
    context: dict = {}
    if project_id:
        m = _t._m()
        p = m.get_registry().get(project_id)
        if p and p.owner == user:
            project_line = f"{p.title} (id {p.id})"
            with contextlib.suppress(Exception):
                context["project_status"] = _t._t_get_project_status(
                    user, {"project_id": project_id})
            with contextlib.suppress(Exception):
                if p.log_path.is_file():
                    lines = p.log_path.read_text(
                        encoding="utf-8", errors="replace").splitlines()
                    context["pipeline_log_tail"] = lines[-60:]
    with contextlib.suppress(Exception):
        context["server_health"] = _t._t_get_server_health(user, {})

    chat_lines = []
    for msg in chat[-10:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        chat_lines.append(f"{role}: {(msg.get('content') or '')[:500]}")

    return (PROMPT_TEMPLATE
            .replace("{user}", user)
            .replace("{page}", page or "unknown")
            .replace("{project_line}", project_line)
            .replace("{summary}", summary)
            .replace("{chat}", "\n".join(chat_lines) or "none")
            .replace("{now}", time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.gmtime()))
            .replace("{context}",
                     json.dumps(context, default=str, indent=1)[:12000]))


# ---- session mapping ----------------------------------------------------

def map_status(sess: dict) -> str:
    """Devin session status -> UI bucket."""
    s = str(sess.get("status_enum") or sess.get("status") or "").lower()
    if s.startswith("blocked"):
        return "needs_reply"
    if s.startswith(("working", "resumed", "resume_requested")):
        return "working"
    if s.startswith(("finished", "expired", "suspend_requested")):
        return "finished"
    return "working"


def session_messages(sess: dict) -> list[dict]:
    """[{source: devin|user, text, ts}] from a GET /v1/sessions reply."""
    out = []
    for ev in sess.get("messages") or []:
        typ = ev.get("type")
        if typ == "devin_message":
            src = "devin"
        elif typ == "user_message":
            src = "user"
        else:
            continue
        out.append({"source": src, "text": ev.get("message") or "",
                    "ts": ev.get("timestamp")})
    return out
