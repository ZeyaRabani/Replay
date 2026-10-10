"""Assistant tool definitions + executors. Every executor takes the
authenticated user and refuses (404-style error string) to touch a project
owned by someone else — same rule as the HTTP endpoints."""

from __future__ import annotations

import json
import shutil
import time

COOKIE_STEPS = """\
To refresh YouTube cookies:
1. In Chrome: extensions -> "Get cookies.txt LOCALLY" -> Details -> enable
   "Allow in Incognito".
2. Open an incognito window, sign in to youtube.com (a throwaway account is
   recommended), play any video for a few seconds.
3. Click the extension -> Export (Netscape format) -> cookies.txt downloads.
4. Close the incognito window right away; don't use that account normally.
5. In Replay -> Projects page -> "YouTube access" -> Remove old -> paste the
   file contents -> Save (as admin, tick "share" so everyone benefits).
Waiting matches resume automatically — no restart needed."""


def _schemas() -> list[dict]:
    return [
        {"name": "list_projects",
         "description": "List the user's matches with live pipeline state.",
         "parameters": {"type": "object", "properties": {},
                        "required": []}},
        {"name": "get_project_status",
         "description": "Get one match's pipeline state plus the last ~40 log lines and, for multi-angle, the sync result.",
         "parameters": {"type": "object", "properties": {
             "project_id": {"type": "string"}},
             "required": ["project_id"]}},
        {"name": "get_server_health",
         "description": "Disk usage, biggest projects, learning-model meta, cookie files status, server time and running/queued counts.",
         "parameters": {"type": "object", "properties": {},
                        "required": []}},
        {"name": "retry_project",
         "description": "Restart the pipeline for a match (same as 'Retry now').",
         "parameters": {"type": "object", "properties": {
             "project_id": {"type": "string"}},
             "required": ["project_id"]}},
        {"name": "cancel_project",
         "description": "Cancel a running pipeline; requires confirmed=true after the user agreed.",
         "parameters": {"type": "object", "properties": {
             "project_id": {"type": "string"},
             "confirmed": {"type": "boolean"}},
             "required": ["project_id", "confirmed"]}},
        {"name": "set_sync_offsets",
         "description": "Set manual sync offsets (seconds each angle leads/lags angle 1) and resume a multi-angle match from sync.",
         "parameters": {"type": "object", "properties": {
             "project_id": {"type": "string"},
             "offsets": {"type": "array", "items": {"type": "number"}}},
             "required": ["project_id", "offsets"]}},
        {"name": "recut_project",
         "description": "Re-run the cut for a done multi-angle match with a different cut style (fast or normal).",
         "parameters": {"type": "object", "properties": {
             "project_id": {"type": "string"},
             "cut_style": {"type": "string", "enum": ["fast", "normal"]}},
             "required": ["project_id", "cut_style"]}},
        {"name": "delete_project",
         "description": "Delete a match (videos+caches removed, record kept in Archive, decisions kept for learning); requires confirmed=true after the user agreed.",
         "parameters": {"type": "object", "properties": {
             "project_id": {"type": "string"},
             "confirmed": {"type": "boolean"}},
             "required": ["project_id", "confirmed"]}},
        {"name": "list_archive",
         "description": "List the user's archived (deleted) matches.",
         "parameters": {"type": "object", "properties": {},
                        "required": []}},
        {"name": "restart_from_archive",
         "description": "Create a new match from an archived record with the same settings; requires confirmed=true after the user agreed.",
         "parameters": {"type": "object", "properties": {
             "match_id": {"type": "string"},
             "confirmed": {"type": "boolean"}},
             "required": ["match_id", "confirmed"]}},
        {"name": "get_youtube_cookie_steps",
         "description": "Get the exact step list for exporting and saving YouTube cookies.",
         "parameters": {"type": "object", "properties": {},
                        "required": []}},
    ]


def schemas() -> list[dict]:
    return _schemas()


def _m():
    from highlights.app.backend import main
    return main


def _project_or_err(user: str, project_id: str):
    """(project, err). Refuses projects owned by someone else."""
    m = _m()
    p = m.get_registry().get(project_id or "")
    if p is None or p.owner != user:
        return None, "project not found"
    return p, None


def _t_list_projects(user: str, _a: dict):
    m = _m()
    out = []
    for p in m.get_registry().list_projects():
        if p.owner != user:
            continue
        s = m.summary(p)
        out.append({
            "id": s["id"], "title": s["title"], "created_at": s["created_at"],
            "source_kind": (s["source"] or {}).get("kind"),
            "n_angles": s["n_angles"], "mode": s["mode"],
            "cut_style": (s["meta"] or {}).get("cut_style"),
            "pipeline_state": s["pipeline_state"], "stage": s["stage"],
            "progress": s["progress"], "message": s["message"],
            "auto_retry": s["auto_retry"],
            "n_candidates": s["n_candidates"],
            "n_confirmed": s["n_confirmed"],
        })
    return out


def _t_get_project_status(user: str, a: dict):
    from highlights.app.backend import pipeline
    p, err = _project_or_err(user, a.get("project_id"))
    if err:
        return err
    st = pipeline.status_with_log(p)
    st["log"] = (st.get("log") or [])[-40:]
    sync_f = p.multiangle_dir / "sync.json" if p.is_multiangle else None
    if sync_f and sync_f.is_file():
        sync = json.loads(sync_f.read_text())
        st["sync"] = {
            "method": sync.get("method"),
            "offsets": sync.get("offsets"),
            "needs_manual": sync.get("needs_manual"),
            "confidence_note": sync.get("confidence_note"),
        }
    return st


def _t_get_server_health(user: str, _a: dict):
    m = _m()
    reg = m.get_registry()
    wd = m.workdir()
    usage = shutil.disk_usage(wd)
    seen: set = set()
    per = []
    for p in reg.list_projects():
        try:
            b = m._du(p.root, seen)
        except Exception:
            b = 0
        per.append({"id": p.id, "title": p.title, "owner": p.owner,
                    "bytes": b})
    per.sort(key=lambda r: r["bytes"], reverse=True)

    learning: dict = {"trained": False}
    try:
        from highlights.pipeline.learn import learn_dir
        meta_path = learn_dir() / "verdict_lr.json"
        if meta_path.is_file():
            learning = json.loads(meta_path.read_text())
    except Exception as e:
        learning = {"error": str(e)}

    uck = m._user_cookies_path(user)
    sck = m._shared_cookies_path()
    states = {"running": 0, "queued": 0}
    for p in reg.list_projects():
        try:
            st = m.pipeline.refresh(p)
        except Exception:
            st = None
        s = (st or {}).get("state") or p.pipeline_state
        if s == "running":
            states["running"] += 1
        elif s == "queued":
            states["queued"] += 1

    return {
        "disk": {"total_bytes": usage.total, "used_bytes": usage.used,
                 "free_bytes": usage.free},
        "largest_projects": per[:5],
        "learning": learning,
        "cookies": {
            "user_saved": uck.is_file(),
            "user_updated_at": uck.stat().st_mtime if uck.is_file() else None,
            "shared_available": sck.is_file(),
            "shared_mtime": sck.stat().st_mtime if sck.is_file() else None,
        },
        "jobs": states,
        "server_time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }


def _t_retry_project(user: str, a: dict):
    m = _m()
    from highlights.app.backend import pipeline
    p, err = _project_or_err(user, a.get("project_id"))
    if err:
        return err
    try:
        out = m._respawn_project(p, user)
    except pipeline.PipelineBusy as e:
        return f"cannot retry: {e}"
    return {"started": True, "pipeline": out}


def _t_cancel_project(user: str, a: dict):
    from highlights.app.backend import pipeline
    p, err = _project_or_err(user, a.get("project_id"))
    if err:
        return err
    if not a.get("confirmed"):
        return "needs confirmation: ask the user first, then call again with confirmed=true"
    return pipeline.cancel(p)


def _t_set_sync_offsets(user: str, a: dict):
    m = _m()
    from highlights.app.backend import pipeline
    p, err = _project_or_err(user, a.get("project_id"))
    if err:
        return err
    try:
        out = m.put_multiangle_offsets(
            m.OffsetsPut(offsets=[float(x) for x in a.get("offsets") or []]),
            p)
    except m.HTTPException as e:
        return f"error {e.status_code}: {e.detail}"
    except pipeline.PipelineBusy as e:
        return f"cannot set offsets: {e}"
    return {"started": True, "pipeline": out}


def _t_recut_project(user: str, a: dict):
    m = _m()
    from highlights.app.backend import pipeline
    p, err = _project_or_err(user, a.get("project_id"))
    if err:
        return err
    try:
        out = m.recut_multiangle(
            m.RecutPut(style=a.get("cut_style") or "fast"), p, user)
    except m.HTTPException as e:
        return f"error {e.status_code}: {e.detail}"
    except pipeline.PipelineBusy as e:
        return f"cannot re-cut: {e}"
    return {"recut_started": True, "pipeline": out}


def _t_delete_project(user: str, a: dict):
    m = _m()
    p, err = _project_or_err(user, a.get("project_id"))
    if err:
        return err
    if not a.get("confirmed"):
        return "needs confirmation: ask the user first, then call again with confirmed=true"
    m.delete_project(p)
    return {"deleted": True, "id": p.id,
            "note": "record kept in the Archive section"}


def _t_list_archive(user: str, _a: dict):
    m = _m()
    out = []
    for r in m.history.list_matches(user, include_deleted=True):
        if not r.get("deleted"):
            continue
        rec = r.get("record") or {}
        src = rec.get("sources") or {}
        out.append({
            "id": r["id"], "title": r["title"],
            "deleted_at": r.get("deleted_at") or rec.get("deleted_at"),
            "n_sources": len(src.get("angles") or []) or (1 if src else 0),
            "n_reels": len(rec.get("artefacts") or []),
        })
    return out


def _t_restart_from_archive(user: str, a: dict):
    m = _m()
    if not a.get("confirmed"):
        return "needs confirmation: ask the user first, then call again with confirmed=true"
    try:
        out = m.restart_match(a.get("match_id") or "", user)
    except m.HTTPException as e:
        return f"error {e.status_code}: {e.detail}"
    return {"restarted": True,
            "new_project": {"id": out.get("id"), "title": out.get("title")}}


def _t_cookie_steps(user: str, _a: dict):
    return COOKIE_STEPS


_EXECUTORS = {
    "list_projects": _t_list_projects,
    "get_project_status": _t_get_project_status,
    "get_server_health": _t_get_server_health,
    "retry_project": _t_retry_project,
    "cancel_project": _t_cancel_project,
    "set_sync_offsets": _t_set_sync_offsets,
    "recut_project": _t_recut_project,
    "delete_project": _t_delete_project,
    "list_archive": _t_list_archive,
    "restart_from_archive": _t_restart_from_archive,
    "get_youtube_cookie_steps": _t_cookie_steps,
}

_SUMMARISE = {
    "list_projects": lambda r: f"listed {len(r)} match(es)",
    "get_project_status": lambda r: (
        f"status {r.get('state')}/{r.get('stage')}" if isinstance(r, dict)
        else str(r)),
    "get_server_health": lambda r: "checked server health",
    "retry_project": lambda r: (
        "retried the match" if isinstance(r, dict) and r.get("started")
        else str(r)),
    "cancel_project": lambda r: (
        "cancelled the run" if isinstance(r, dict)
        and r.get("error") == "cancelled" else str(r)),
    "set_sync_offsets": lambda r: (
        "applied sync offsets" if isinstance(r, dict) and r.get("started")
        else str(r)),
    "recut_project": lambda r: (
        "queued the re-cut" if isinstance(r, dict) and r.get("recut_started")
        else str(r)),
    "delete_project": lambda r: (
        "deleted the match" if isinstance(r, dict) and r.get("deleted")
        else str(r)),
    "list_archive": lambda r: f"listed {len(r)} archived match(es)",
    "restart_from_archive": lambda r: (
        f"restarted as {r['new_project']['title']}"
        if isinstance(r, dict) and r.get("restarted") else str(r)),
    "get_youtube_cookie_steps": lambda r: "returned the cookie steps",
}


def execute(user: str, name: str, arguments: dict) -> tuple[bool, object, str]:
    """Run one tool. Returns (ok, result_or_error_string, summary_line)."""
    fn = _EXECUTORS.get(name)
    if fn is None:
        return False, f"unknown tool {name}", f"unknown tool {name}"
    try:
        res = fn(user, arguments or {})
        ok = not (isinstance(res, str))
        try:
            summary = _SUMMARISE.get(name, lambda r: "done")(res)
        except Exception:
            summary = "done"
        if not ok:
            summary = f"{name}: {res}"
        return ok, res, summary
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", f"{name} failed: {e}"
