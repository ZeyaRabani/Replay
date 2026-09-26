"""Match history + archive (SQLite, deletion-surviving records).

Two tables at ``{workdir}/history.sqlite``:

- ``matches`` — one row per project, kept after the project is deleted.
  ``record`` is a JSON snapshot of everything needed to describe or
  restart the match: sources, window, sync, zones, style, review
  decisions and the list of artefacts moved to ``{workdir}/archive``.
- ``events`` — an append-only timeline (created, stage transitions,
  zones_saved, recut_queued, candidate_confirmed, …).

Every public function opens its own short-lived connection, so calls
are safe from any thread (WAL mode).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .store import ProjectStore, workdir

DB_NAME = "history.sqlite"
ARCHIVE_DIR = "archive"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches(
    id TEXT PRIMARY KEY,
    owner TEXT,
    title TEXT,
    mode TEXT,
    created_at REAL,
    deleted_at REAL,
    record JSON
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    ts REAL,
    kind TEXT,
    stage TEXT,
    status TEXT,
    detail JSON
);
CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id, id);
"""


def _db_path() -> Path:
    return workdir() / DB_NAME


def archive_root() -> Path:
    return workdir() / ARCHIVE_DIR


@contextmanager
def _conn():
    db = sqlite3.connect(_db_path(), check_same_thread=False)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        yield db
        db.commit()
    finally:
        db.close()


def _init(db: sqlite3.Connection) -> None:
    db.executescript(_SCHEMA)


def init_db() -> None:
    with _conn() as db:
        _init(db)


def log(match_id: str, kind: str, stage: str | None = None,
        status: str | None = None, **fields) -> None:
    """Append an event to a match's timeline."""
    with _conn() as db:
        _init(db)
        db.execute(
            "INSERT INTO events(match_id, ts, kind, stage, status, detail)"
            " VALUES(?,?,?,?,?,?)",
            (match_id, time.time(), kind, stage, status,
             json.dumps(fields, default=str)))


def events(match_id: str) -> list[dict]:
    """Timeline for one match, newest first."""
    with _conn() as db:
        _init(db)
        rows = db.execute(
            "SELECT id, ts, kind, stage, status, detail FROM events"
            " WHERE match_id=? ORDER BY id DESC", (match_id,)).fetchall()
    return [{"id": r["id"], "ts": r["ts"], "kind": r["kind"],
             "stage": r["stage"], "status": r["status"],
             "detail": json.loads(r["detail"] or "{}")}
            for r in rows]


def _row_summary(row: sqlite3.Row, full: bool = False) -> dict:
    record = json.loads(row["record"] or "{}")
    out = {
        "id": row["id"],
        "owner": row["owner"],
        "title": row["title"],
        "mode": row["mode"],
        "created_at": row["created_at"],
        "deleted_at": row["deleted_at"],
        "n_angles": len((record.get("sources") or {}).get("angles") or []),
        "style": record.get("style"),
        "artefacts": record.get("artefacts") or [],
        "deleted": row["deleted_at"] is not None,
    }
    if full:
        out["record"] = record
    return out


def list_matches(owner: str | None = None,
                 include_deleted: bool = False) -> list[dict]:
    with _conn() as db:
        _init(db)
        q = "SELECT * FROM matches"
        cond, args = [], []
        if owner is not None:
            cond.append("owner=?")
            args.append(owner)
        if not include_deleted:
            cond.append("deleted_at IS NULL")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        rows = db.execute(q + " ORDER BY created_at DESC", args).fetchall()
    return [_row_summary(r) for r in rows]


def get_match(match_id: str) -> dict | None:
    with _conn() as db:
        _init(db)
        row = db.execute(
            "SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
    return _row_summary(row, full=True) if row else None


def mark_deleted(match_id: str) -> None:
    with _conn() as db:
        _init(db)
        db.execute("UPDATE matches SET deleted_at=? WHERE id=?",
                   (time.time(), match_id))


def remove_match(match_id: str) -> None:
    with _conn() as db:
        _init(db)
        db.execute("DELETE FROM matches WHERE id=?", (match_id,))
        db.execute("DELETE FROM events WHERE match_id=?", (match_id,))
    shutil.rmtree(archive_root() / match_id, ignore_errors=True)


def _write_record(match_id: str, record: dict) -> None:
    with _conn() as db:
        _init(db)
        db.execute("UPDATE matches SET record=? WHERE id=?",
                   (json.dumps(record, default=str), match_id))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _angle_durations(p: ProjectStore) -> list[float]:
    out = []
    for i, _a in enumerate(p.source_info.get("angles") or []):
        st = _read_json(p.angle_dir(i) / "pipeline" / "status.json") or {}
        pr = _read_json(p.angle_dir(i) / "pipeline" / "probe.json") or {}
        out.append(float((st.get("video") or pr).get("duration_s") or 0.0))
    return out


def _candidate_windows(p: ProjectStore, status: str) -> list[dict]:
    return [
        {"id": c.id, "type": c.type, "t_start": c.t_start,
         "t_end": c.t_end, "t": c.t}
        for c in p.candidates if c.status == status
    ]


def build_record(p: ProjectStore, artefacts: list[str] | None = None) -> dict:
    """The deletion-surviving snapshot of a project."""
    ma = p.multiangle_dir if p.is_multiangle else None
    durs = _angle_durations(p) if p.is_multiangle else []
    timing_ref = (int(max(range(len(durs)), key=lambda i: durs[i]))
                  if durs and any(durs) else None)
    record = {
        "sources": p.source_info,
        "mode": "multiangle" if p.is_multiangle else "single",
        "meta": p.meta,
        "window": {
            "cut_range": _read_json(ma / "cut_range.json") if ma else None,
            "match_window": _read_json(p.pipeline_dir / "match_window.json"),
            "match_window_src": _read_json(ma / "match_window_src.json") if ma else None,
        },
        "timing_reference": timing_ref,
        "durations": durs,
        "sync": _read_json(ma / "sync.json") if ma else None,
        "zones": _read_json(ma / "zones.json") if ma else None,
        "style": p.meta.get("cut_style"),
        "candidates": {
            "confirmed": _candidate_windows(p, "confirmed"),
            "rejected": _candidate_windows(p, "rejected"),
        },
        "artefacts": artefacts or [],
        "snapshotted_at": time.time(),
    }
    return record


def upsert_match(p: ProjectStore) -> None:
    """Insert/update the match row for a project; keeps deleted_at."""
    with _conn() as db:
        _init(db)
        prev = db.execute("SELECT record FROM matches WHERE id=?",
                          (p.id,)).fetchone()
        old_art = []
        if prev and prev["record"]:
            try:
                old_art = (json.loads(prev["record"]) or {}).get(
                    "artefacts") or []
            except Exception:
                old_art = []
        record = build_record(p, artefacts=old_art)
        db.execute(
            "INSERT INTO matches(id, owner, title, mode, created_at,"
            " deleted_at, record) VALUES(?,?,?,?,?,"
            " (SELECT deleted_at FROM matches WHERE id=?), ?)"
            " ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,"
            " title=excluded.title, mode=excluded.mode,"
            " record=excluded.record",
            (p.id, p.owner, p.title, record["mode"], p.created_at,
             p.id, json.dumps(record, default=str)))


# ---------- status-transition tracking ----------


_seen: dict[str, tuple[str | None, str | None]] = {}


def track_status(p: ProjectStore, status: dict) -> None:
    """Log stage/state transitions when ``status`` differs from the last
    status seen for this project (called from pipeline.refresh — the
    single choke point all status changes surface through)."""
    key = p.id
    prev = _seen.get(key)
    cur = (status.get("state"), status.get("stage"))
    _seen[key] = cur
    if prev is None:
        return
    pstate, pstage = prev
    state, stage = cur
    if state == "running" and stage and stage != pstage:
        if pstage:
            log(key, "stage_done", stage=pstage, status="done")
        log(key, "stage_start", stage=stage, status="running")
        upsert_match(p)
    elif state != pstate:
        if state == "done":
            log(key, "stage_done", stage=pstage, status="done")
            upsert_match(p)
        elif state == "failed":
            log(key, "stage_failed", stage=pstage,
                status="failed", message=status.get("message"))
            upsert_match(p)
        elif state in ("queued", "running", "paused"):
            log(key, "resumed" if pstate == "paused" else "stage_start",
                stage=stage, status=state)


# ---------- archive on delete ----------


_ARCHIVE_FILES = [
    ("match.mp4", "match.mp4"),                       # active director cut
    ("project.json", "project.json"),
    ("multiangle/zones.json", "zones.json"),
    ("multiangle/sync.json", "sync.json"),
    ("multiangle/director.json", "director.json"),
    ("multiangle/fused_candidates.json", "fused_candidates.json"),
    ("pipeline/stats.json", "stats.json"),
]


def archive_project(p: ProjectStore) -> list[str]:
    """Move kept artefacts to {workdir}/archive/{id}/ before the project
    dir is removed; stores their names in the match record. Returns the
    artefact names."""
    dest = archive_root() / p.id
    dest.mkdir(parents=True, exist_ok=True)
    # build the record BEFORE moving files — it reads zones/sync/etc.
    record = build_record(p, artefacts=[])
    kept: list[str] = []
    for rel, name in _ARCHIVE_FILES:
        src = p.root / rel
        if src.is_file():
            shutil.move(str(src), str(dest / name))
            kept.append(name)
    reels = p.root / "renders"
    if reels.is_dir():
        for f in sorted(reels.glob("*.mp4")):
            shutil.move(str(f), str(dest / f.name))
            kept.append(f.name)
    record["artefacts"] = kept
    _write_record(p.id, record)
    log(p.id, "deleted", artefacts=kept)
    mark_deleted(p.id)
    return kept


def artefact_path(match_id: str, name: str) -> Path | None:
    if "/" in name or ".." in name or "\\" in name:
        return None
    f = archive_root() / match_id / name
    return f if f.is_file() else None


def backfill(reg) -> None:
    """On startup: upsert every existing project + seed a `created`
    event when a match has no events yet."""
    for p in reg.list_projects():
        try:
            upsert_match(p)
            if not events(p.id):
                log(p.id, "created", title=p.title,
                    owner=p.owner, backfill=True)
        except Exception as e:
            print(f"warning: history backfill failed for {p.id}: {e}")
