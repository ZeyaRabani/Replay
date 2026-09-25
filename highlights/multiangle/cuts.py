"""Director-cut snapshots: every completed cut is kept under
<project>/multiangle/cuts/<id>/ so it can be re-activated or downloaded
later. match.mp4 is hardlinked (never rewritten in place — the renderer
writes to a tmp file then os.replace()s)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from highlights.io import write_json_atomic

SNAP_FILES = [
    "multiangle/director.json",
    "multiangle/fused_candidates.json",
    "pipeline/candidates.json",
    "pipeline/probe.json",
    "pipeline/stats.json",
]


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def cut_label(style: str, zones_used: bool) -> str:
    return f"{'Fast' if style == 'fast' else 'Normal'} " \
           f"{'+ zones' if zones_used else '(AI)'}"


def snapshot_cut(project_dir: Path, style: str | None = None) -> dict | None:
    """Snapshot the current cut into multiangle/cuts/<id>/.

    Returns the cut meta dict, or None if there is nothing to snapshot.
    """
    ma = project_dir / "multiangle"
    match = project_dir / "match.mp4"
    director = _read(ma / "director.json")
    if not match.is_file() or not director:
        return None
    cuts_dir = ma / "cuts"
    cid = time.strftime("%Y%m%d-%H%M%S")
    cdir = cuts_dir / cid
    n = 1
    while cdir.exists():
        n += 1
        cdir = cuts_dir / f"{cid}-{n}"
    cid = cdir.name
    cdir.mkdir(parents=True)
    try:
        os.link(match, cdir / "match.mp4")
    except OSError:
        shutil.copy2(match, cdir / "match.mp4")
    for rel in SNAP_FILES:
        src = project_dir / rel
        if src.is_file():
            shutil.copy2(src, cdir / src.name)
    zones_used = bool(director.get("zones_used"))
    st = style or director.get("style") or "normal"
    meta = {
        "id": cid,
        "label": cut_label(st, zones_used),
        "style": st,
        "zones_used": zones_used,
        "n_cuts": director.get("n_cuts"),
        "created_at": time.time(),
    }
    write_json_atomic(cdir / "meta.json", meta, indent=1)
    write_json_atomic(cuts_dir / "active.json", {"id": cid}, indent=1)
    return meta


def list_cuts(project_dir: Path) -> dict:
    cuts_dir = project_dir / "multiangle" / "cuts"
    active = (_read(cuts_dir / "active.json") or {}).get("id")
    metas = []
    if cuts_dir.is_dir():
        for d in cuts_dir.iterdir():
            if not d.is_dir():
                continue
            m = _read(d / "meta.json")
            if m:
                m.setdefault("id", d.name)
                metas.append(m)
    metas.sort(key=lambda m: m.get("created_at", 0))
    return {"active": active, "cuts": metas}


def activate_cut(project_dir: Path, cut_id: str) -> dict | None:
    """Make cuts/<cut_id>/ the live cut. Returns its meta or None."""
    cdir = project_dir / "multiangle" / "cuts" / cut_id
    if not cdir.is_dir() or "/" in cut_id or ".." in cut_id:
        return None
    live_match = project_dir / "match.mp4"
    src = cdir / "match.mp4"
    if src.is_file():
        tmp = project_dir / f".match.{cut_id}.mp4"
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, live_match)
    for rel in SNAP_FILES:
        src_f = cdir / Path(rel).name
        dst = project_dir / rel
        if src_f.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(src_f, tmp)
            os.replace(tmp, dst)
    write_json_atomic(cdir.parent / "active.json", {"id": cut_id}, indent=1)
    return _read(cdir / "meta.json")
