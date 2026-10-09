"""FastAPI backend for the Replay Highlights review/render shell.

Multi-user (X-User header), multi-project, with a detached pipeline
subprocess per project. Run from repo root:

    uvicorn highlights.app.backend.main:app --port 8000
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from highlights.io import write_json_atomic
from highlights.multiangle.cuts import (
    activate_cut,
    cut_info,
    list_cuts,
    snapshot_cut,
)

from . import ffmpeg as fx
from . import history, pipeline, playlist, stats, watchdog
from .schemas import (
    CandidatePatch,
    CandidatesFile,
    CandidatesLoadRequest,
    ClipResult,
    RenderJob,
    RenderRequest,
    VideoInfo,
    VideoRegisterRequest,
)
from .store import ProjectStore, Registry, workdir

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DIST = APP_DIR / "frontend" / "dist"

NO_DOWNLOAD_STAGES = ["probe", "audio", "motion", "features", "score", "candidates", "stats"]

_jobs: dict[str, RenderJob] = {}
_job_owner: dict[str, str] = {}  # job_id -> project_id
_proxy_jobs: dict[str, fx.ProxyJob] = {}  # project_id -> job
_angle_proxy_jobs: dict[tuple[str, int], fx.ProxyJob] = {}  # (project_id, angle) -> job
_trim_jobs: dict[str, fx.TrimJob] = {}    # "pid:start:end" -> job
_jobs_lock = threading.Lock()

_registry: Registry | None = None


PITCH_TYPES = ("11", "9", "7", "5", "other")
CAMERA_TYPES = ("normal", "ultrawide", "zoom", "other")
CUT_STYLES = ("normal", "fast")


def _meta_or_422(pitch_type: str | None, camera: str | None,
                 cut_style: str | None = None) -> dict:
    meta = {}
    if pitch_type is not None:
        if pitch_type not in PITCH_TYPES:
            raise HTTPException(422, f"pitch_type must be one of {PITCH_TYPES}")
        meta["pitch_type"] = pitch_type
    if camera is not None:
        if camera not in CAMERA_TYPES:
            raise HTTPException(422, f"camera must be one of {CAMERA_TYPES}")
        meta["camera"] = camera
    if cut_style is not None:
        if cut_style not in CUT_STYLES:
            raise HTTPException(422, f"cut_style must be one of {CUT_STYLES}")
        meta["cut_style"] = cut_style
    return meta


class ProjectCreate(BaseModel):
    title: str | None = None
    youtube_url: str | None = None
    path: str | None = None
    run_pipeline: bool = True
    cookies_text: str | None = None
    pitch_type: str | None = None
    camera: str | None = None
    cut_style: str | None = None


class UserCreate(BaseModel):
    name: str


class AngleSpec(BaseModel):
    url: str | None = None
    filename: str | None = None
    label: str = ""
    camera: str | None = None


class MultiangleCreate(BaseModel):
    title: str | None = None
    angles: list[AngleSpec]
    cookies_text: str | None = None
    pitch_type: str | None = None
    camera: str | None = None
    cut_style: str | None = None
    # file seconds on match_window_angle (null = the longest angle,
    # resolved at conversion); written to match_window_src.json
    match_window: list[float] | None = None
    match_window_angle: int | None = None


class OffsetsPut(BaseModel):
    offsets: list[float]


class RecutPut(BaseModel):
    style: str
    # output-time seconds of the CURRENT video [start, end]; None = full
    window: list[float] | None = None
    # preview=True + window: re-cut only that stretch, remembering the
    # previous cut_range in cut_range_base.json; the next non-preview
    # recut restores it first
    preview: bool = False
    goal_aware: bool | None = None


class MaSettingsPut(BaseModel):
    goal_aware: bool


class MatchWindowPut(BaseModel):
    start_s: float
    end_s: float


class MaMatchWindowPut(BaseModel):
    # file seconds of `angle`; null/absent = measured on the longest
    # angle, resolved at conversion time. both start/end null -> clear
    angle: int | None = None
    start: float | None = None
    end: float | None = None


class ScorePut(BaseModel):
    home_label: str
    away_label: str
    home_hex: str | None = None
    away_hex: str | None = None


_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _norm_hex(v: str | None, field: str) -> str | None:
    if v is None or v == "":
        return None
    if not _HEX_RE.match(v.strip()):
        raise HTTPException(422, f"{field} must be #rrggbb")
    return "#" + v.strip().lstrip("#").lower()


class ZoneKeyframe(BaseModel):
    t: float = 0.0
    zones: list[list[list[float]]] = []


class ZonesPut(BaseModel):
    # v2: per-angle list of keyframes; legacy: per-angle list of polys
    angles: list[list[ZoneKeyframe | list[list[float]]]]
    ref_t: list[float | None] | None = None


def _zones_to_v2(body: ZonesPut, n: int,
                 durations: list[float]) -> list[list[dict]]:
    """Normalise a PUT body into v2 keyframes per angle."""
    out: list[list[dict]] = []
    for ai, entry in enumerate(body.angles):
        kfs: list[dict] = []
        if entry and all(isinstance(k, ZoneKeyframe) for k in entry):
            for kf in entry:
                if kf.t < 0:
                    raise HTTPException(422, f"angle {ai}: keyframe t < 0")
                kfs.append({"t": float(kf.t), "zones": kf.zones})
        elif entry:
            # legacy flat polygon list -> single keyframe
            rt = (body.ref_t[ai] if body.ref_t and ai < len(body.ref_t)
                  and body.ref_t[ai] is not None else None)
            dur = durations[ai] if ai < len(durations) else 0.0
            t = float(rt) if rt is not None else (dur * 0.3 if dur else 0.0)
            kfs.append({"t": t,
                        "zones": [p for p in entry if isinstance(p, list)]})
        kfs.sort(key=lambda k: k["t"])
        out.append(kfs)
    return out


class TrimRequest(BaseModel):
    start_s: float
    end_s: float


def get_registry() -> Registry:
    global _registry
    wd = workdir()
    if _registry is None or _registry.root != wd:
        reg = Registry(wd)
        ensure_demo(reg)
        for p in reg.list_projects():
            pipeline.reconcile(p)
        try:
            history.backfill(reg)
        except Exception as e:
            print(f"warning: history backfill failed: {e}")
        _registry = reg
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
    _jobs.clear()
    _job_owner.clear()
    _proxy_jobs.clear()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    get_registry()
    watchdog.start()
    yield


app = FastAPI(title="Replay Highlights", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_demo(reg: Registry) -> None:
    """Seed a demo user + demo project from committed fusion outputs."""
    reg.add_user("admin")
    if reg.list_projects():
        return
    cands_path = REPO_ROOT / "highlights" / "fusion" / "outputs" / "candidates.json"
    if not cands_path.is_file():
        return
    demo_video = os.environ.get("HL_DEMO_VIDEO") or (
        "/home/ubuntu/match/match_2400.mp4"
        if Path("/home/ubuntu/match/match_2400.mp4").is_file()
        else "/home/ubuntu/match/match.mp4"
    )
    p = reg.create_project(
        owner="admin",
        title="Demo match (5qj_nsQSzvQ)",
        source={"kind": "path", "url": "https://youtu.be/5qj_nsQSzvQ", "filename": "match.mp4"},
    )
    video_dict = None
    vp = Path(demo_video)
    if vp.is_file():
        try:
            info = fx.probe(vp)
            vi = VideoInfo(path=str(vp.resolve()), registered_at=time.time(), **info)
            p.set_video(vi)
            video_dict = vi.model_dump()
        except Exception as e:
            print(f"warning: could not probe demo video {vp}: {e}")
    try:
        p.load_candidates(CandidatesFile(**json.loads(cands_path.read_text())))
    except Exception as e:
        print(f"warning: could not load demo candidates: {e}")
    now = time.time()
    pipeline.write_status(p, {
        "state": "done",
        "stage": "done",
        "progress": 1.0,
        "stage_progress": 1.0,
        "message": "demo project: precomputed fusion outputs",
        "error": None,
        "started_at": now,
        "updated_at": now,
        "finished_at": now,
        "pid": None,
        "video_path": str(vp.resolve()) if vp.is_file() else None,
        "video": video_dict,
        "download": None,
    })
    p.set_pipeline_state("done")
    try:
        (p.pipeline_dir / "stats.json").write_text(json.dumps(stats.demo_stats(), indent=2))
    except Exception as e:
        print(f"warning: could not compute demo stats: {e}")


# ---------- deps ----------


def current_user(x_user: str | None = Header(default=None, alias="X-User")) -> str:
    if not x_user:
        raise HTTPException(401, "missing X-User header")
    x_user = unquote(x_user)
    if get_registry().get_user(x_user) is None:
        raise HTTPException(401, "unknown user")
    return x_user


def project_dep(project_id: str, user: Annotated[str, Depends(current_user)]) -> ProjectStore:
    p = get_registry().get(project_id)
    if p is None or p.owner != user:
        raise HTTPException(404, "project not found")
    return p


def project_public(project_id: str) -> ProjectStore:
    """No auth — for media GETs fetched via <video>/<img> src attributes."""
    p = get_registry().get(project_id)
    if p is None:
        raise HTTPException(404, "project not found")
    return p


def legacy_project() -> ProjectStore:
    """Legacy unscoped routes map to user demo's newest project."""
    p = get_registry().newest_for("admin")
    if p is None:
        raise HTTPException(
            404,
            "no project: create one via POST /api/projects "
            "(legacy unscoped routes map to user demo's newest project)",
        )
    return p


ScopedP = Annotated[ProjectStore, Depends(project_dep)]
PublicP = Annotated[ProjectStore, Depends(project_public)]
LegacyP = Annotated[ProjectStore, Depends(legacy_project)]
UserDep = Annotated[str, Depends(current_user)]


def _hist(p: ProjectStore, kind: str, **kw) -> None:
    """Log a history event + refresh the match record; never fatal."""
    try:
        history.log(p.id, kind, **kw)
        history.upsert_match(p)
    except Exception:
        pass


# ---------- shared per-project logic ----------


def _video_path(p: ProjectStore) -> Path:
    if p.video is None:
        raise HTTPException(404, "no video registered")
    return Path(p.video.path)


def _proxy_ready(p: ProjectStore) -> bool:
    """True only when a completed proxy exists for the current video."""
    dst = p.root / "proxy.mp4"
    if p.proxy_complete and p.video and p.proxy_source == p.video.path and dst.exists():
        return True
    job = _proxy_jobs.get(p.id)
    running = job is not None and not job.done and job.error is None
    if not running:
        dst.unlink(missing_ok=True)
    return False


def _stats(p: ProjectStore) -> dict:
    """Legacy stats shape (used for render stats.json and GET /api/stats)."""
    cands = p.candidates
    counts: dict[str, dict] = {}
    for c in cands:
        e = counts.setdefault(c.type, {"total": 0, "confirmed": 0, "rejected": 0, "pending": 0})
        e["total"] += 1
        e[c.status] += 1
    xv: dict[str, int] = {}
    for c in cands:
        xv[c.cross_validation] = xv.get(c.cross_validation, 0) + 1
    return {
        "source": p.source,
        "video_duration_s": p.video.duration_s if p.video else 0.0,
        "n_candidates": len(cands),
        "counts_by_type": counts,
        "counts_by_cross_validation": xv,
        "timeline": [
            {
                "id": c.id, "type": c.type, "t": c.t,
                "clip_start": c.clip_start, "clip_end": c.clip_end,
                "confidence": c.confidence, "status": c.status,
                "cross_validation": c.cross_validation,
            }
            for c in sorted(cands, key=lambda c: c.t)
        ],
    }


def _register_video(p: ProjectStore, req: VideoRegisterRequest) -> VideoInfo:
    fp = Path(req.path)
    if not fp.is_file():
        raise HTTPException(404, f"file not found: {fp}")
    resolved = str(fp.resolve())
    if p.video is None or p.video.path != resolved:
        p.invalidate_video()
    info = fx.probe(fp)
    vi = VideoInfo(path=resolved, registered_at=time.time(), **info)
    p.set_video(vi)
    vi.proxy_ready = _proxy_ready(p)
    return vi


def _get_video(p: ProjectStore) -> VideoInfo:
    if p.video is None:
        raise HTTPException(404, "no video registered")
    p.video.proxy_ready = _proxy_ready(p)
    return p.video


def _serve(path: Path) -> FileResponse:
    # Starlette FileResponse handles Range requests (status 206).
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type="video/mp4")


def _build_proxy(p: ProjectStore) -> dict:
    src = _video_path(p)
    if _proxy_ready(p):
        return {"status": "ready"}
    job = _proxy_jobs.get(p.id)
    if job is not None and not job.done and job.error is None:
        return {"status": "started"}
    src_str = str(src)
    job = fx.ProxyJob(
        src, p.root / "proxy.mp4", p.video.duration_s,
        on_success=lambda: p.set_proxy_complete(src_str),
    )
    _proxy_jobs[p.id] = job
    job.start()
    return {"status": "started"}


def _proxy_status(p: ProjectStore) -> dict:
    if _proxy_ready(p):
        return {"ready": True, "progress": 1.0}
    job = _proxy_jobs.get(p.id)
    if job is None:
        return {"ready": False, "progress": 0.0}
    return {"ready": job.done, "progress": job.progress, "error": job.error}


def _proxy_file(p: ProjectStore) -> FileResponse:
    return _serve(p.root / "proxy.mp4")


# ---------- match window + trimmed video ----------

def _match_window_path(p: ProjectStore) -> Path:
    return p.pipeline_dir / "match_window.json"


def _get_match_window(p: ProjectStore) -> dict:
    mw = _read_json(_match_window_path(p)) or {}
    dur = p.video.duration_s if p.video else 0.0
    win = mw.get("match_window")
    if not (isinstance(win, list) and len(win) == 2 and win[1] > win[0]):
        win = [0.0, dur]
    return {"match_window": [float(win[0]), float(win[1])],
            "halves": mw.get("halves"),
            "warning": mw.get("warning"),
            "duration": dur}


def _put_match_window(p: ProjectStore, body: MatchWindowPut) -> dict:
    dur = p.video.duration_s if p.video else 0.0
    if not (0.0 <= body.start_s < body.end_s and body.end_s <= dur):
        raise HTTPException(
            422, f"need 0 <= start < end <= duration ({dur:.1f} s)")
    s, e = float(body.start_s), float(body.end_s)
    mw_path = _match_window_path(p)
    mw = _read_json(mw_path) or {}
    # clip existing halves to the new window; drop halves that go empty
    halves = []
    for h in mw.get("halves") or []:
        hs = max(float(h.get("start", 0.0)), s)
        he = min(float(h.get("end", 0.0)), e)
        if he > hs:
            halves.append({**h, "start": hs, "end": he})
    mw["match_window"] = [s, e]
    mw["halves"] = halves
    mw_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(mw_path, mw, indent=1)
    _hist(p, "window_set", start=s, end=e)
    # recompute stats with the new window; preserve the multiangle block
    try:
        from highlights.pipeline.run import recompute_stats
        stats = recompute_stats(p.pipeline_dir, dur)
        old = _read_json(p.pipeline_dir / "stats.json") or {}
        if "multiangle" in old:
            stats["multiangle"] = old["multiangle"]
        write_json_atomic(p.pipeline_dir / "stats.json", stats, indent=1)
    except Exception as e:
        print(f"warning: stats recompute failed for {p.id}: {e}")
    return _get_match_window(p)


def _trim_out(p: ProjectStore, start: float, end: float) -> Path:
    return p.root / "renders" / f"trimmed_{start:g}_{end:g}.mp4"


def _trim_key(p: ProjectStore, start: float, end: float) -> str:
    return f"{p.id}:{start:g}:{end:g}"


def _window_or_422(p: ProjectStore, start: float, end: float) -> None:
    dur = p.video.duration_s if p.video else 0.0
    if not (0.0 <= start < end and end <= dur):
        raise HTTPException(
            422, f"need 0 <= start < end <= duration ({dur:.1f} s)")


def _start_trim(p: ProjectStore, start: float, end: float) -> dict:
    src = _video_path(p)
    _window_or_422(p, start, end)
    out = _trim_out(p, start, end)
    if out.is_file():
        return {"ready": True, "progress": 1.0}
    key = _trim_key(p, start, end)
    job = _trim_jobs.get(key)
    if job is not None and not job.done and job.error is None:
        return {"ready": False, "progress": job.progress}
    out.parent.mkdir(parents=True, exist_ok=True)
    job = fx.TrimJob(src, out, start, end)
    _trim_jobs[key] = job
    job.start()
    _hist(p, "trimmed", start_s=start, end_s=end)
    return {"ready": False, "progress": job.progress}


def _trim_status(p: ProjectStore, start: float, end: float) -> dict:
    if _trim_out(p, start, end).is_file():
        return {"ready": True, "progress": 1.0}
    job = _trim_jobs.get(_trim_key(p, start, end))
    if job is None:
        return {"ready": False, "progress": 0.0}
    return {"ready": job.done, "progress": job.progress, "error": job.error}


def _source_file(p: ProjectStore) -> FileResponse:
    return _serve(_video_path(p))


async def _load_candidates(p: ProjectStore, request: Request) -> list:
    """Accepts multipart file upload OR JSON body {"path": "/abs/candidates.json"}."""
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" in ct:
        form = await request.form()
        file = form.get("file")
        if file is None or not hasattr(file, "read"):
            raise HTTPException(400, "missing 'file' field")
        cf = CandidatesFile(**json.loads(await file.read()))
    elif "application/json" in ct:
        body = CandidatesLoadRequest(**(await request.json()))
        fp = Path(body.path)
        if not fp.is_file():
            raise HTTPException(404, f"file not found: {fp}")
        cf = CandidatesFile(**json.loads(fp.read_text()))
    else:
        raise HTTPException(400, "expected multipart file or JSON body with 'path'")
    return [c.model_dump() for c in p.load_candidates(cf)]


def _list_candidates(p: ProjectStore, sort: str = "confidence") -> list:
    cands = p.candidates
    if p.video and p.video.duration_s > 0:
        cands = [c for c in cands if c.t <= p.video.duration_s + 1]
    cands = sorted(cands, key=lambda c: c.t) if sort == "time" else sorted(cands, key=lambda c: -c.confidence)
    return [c.model_dump() for c in cands]


def _patch_candidate(p: ProjectStore, cand_id: str, patch: CandidatePatch) -> dict:
    c = p.get(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    data = patch.model_dump(exclude_none=True)
    team = data.pop("team", None)
    old_type = c.type
    updated = c.model_copy(update=data)
    duration = p.video.duration_s if p.video else 0.0
    err = fx.validate_clip_window(updated.clip_start, updated.clip_end, duration)
    if err:
        raise HTTPException(422, err)
    for k, v in data.items():
        setattr(c, k, v)
    if team is not None:
        c.signals = {**c.signals, "team": team}
    if "type" in data and "ai_type" not in c.signals:
        c.signals = {**c.signals, "ai_type": old_type}
    p.update(c)
    if data.get("status") in ("confirmed", "rejected"):
        _hist(p, f"candidate_{data['status']}", cand_id=cand_id,
              type=c.type, t_start=c.t_start, t_end=c.t_end)
    elif "type" in data:
        _hist(p, "candidate_type_changed", cand_id=cand_id,
              type=c.type, ai_type=c.signals.get("ai_type"),
              t_start=c.t_start, t_end=c.t_end)
    return c.model_dump()


def _reset_candidate(p: ProjectStore, cand_id: str) -> dict:
    c = p.reset_candidate(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    return c.model_dump()


def _thumb(p: ProjectStore, cand_id: str, t: float | None = None) -> FileResponse:
    c = p.get(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    duration = p.video.duration_s if p.video else 0.0
    tval = c.t if t is None else t
    if duration > 0:
        tval = min(max(0.0, tval), max(0.0, duration - 0.1))
    out = p.thumb_dir() / f"{cand_id}_{tval:.1f}.jpg"
    if not out.exists():
        try:
            fx.thumbnail(_video_path(p), tval, out)
        except RuntimeError as e:
            raise HTTPException(404, f"thumbnail failed: {e}") from e
    if not out.is_file() or out.stat().st_size == 0:
        raise HTTPException(404, "thumbnail produced no frame")
    return FileResponse(out, media_type="image/jpeg")


def _project_old(p: ProjectStore) -> dict:
    return {
        "video": p.video.model_dump() if p.video else None,
        "candidates_version": p.candidates_version,
        "proxy_ready": _proxy_ready(p),
        "mode": "multiangle" if p.is_multiangle else "single",
    }


def _run_render(p: ProjectStore, job_id: str, req: RenderRequest, items: list[dict]) -> None:
    job = _jobs[job_id]
    job.state = "running"
    out_dir = p.root / "renders" / job_id
    try:
        def cb(frac: float, msg: str) -> None:
            job.progress, job.message = frac, msg

        res = fx.render_reel(_video_path(p), items, out_dir, overlay=req.overlay, reencode=req.reencode, progress_cb=cb)
        _hist(p, "rendered_reel", job_id=job_id, n_items=len(items))
        base = f"/api/projects/{p.id}/files/{job_id}"
        job.clips = [
            ClipResult(id=r["id"], path=r["path"], url=f"{base}/clips/{Path(r['path']).name}", duration=r["duration"])
            for r in res["clips"]
        ]
        st = _stats(p)
        st["rendered"] = {
            "n_clips": len(res["clips"]),
            "total_clip_s": sum(r["duration"] for r in res["clips"]),
            "reel_s": res["reel_s"],
        }
        (out_dir / "stats.json").write_text(json.dumps(st, indent=2))
        manifest = [
            {
                "id": it["id"], "t": it["t"], "clip_start": it["clip_start"], "clip_end": it["clip_end"],
                "path": r["path"], "duration": r["duration"],
            }
            for it, r in zip(items, res["clips"], strict=True)
        ]
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        job.reel_url = f"{base}/reel.mp4"
        job.stats_url = f"{base}/stats.json"
        job.state = "done"
        job.progress = 1.0
        job.message = "done"
    except Exception as e:
        job.state = "error"
        job.error = str(e)
        job.message = "error"


def _start_render(p: ProjectStore, req: RenderRequest) -> dict:
    _video_path(p)
    if any(j.state in ("queued", "running") and _job_owner.get(jid) == p.id
           for jid, j in _jobs.items()):
        raise HTTPException(409, "render already running")
    if req.ids:
        cands = [c for c in (p.get(i) for i in req.ids) if c is not None]
    else:
        cands = [c for c in p.candidates if c.status == "confirmed"]
        if not cands:
            cands = [c for c in p.candidates if c.status != "rejected"]
    if not cands:
        raise HTTPException(400, "no candidates selected for render")
    duration = p.video.duration_s if p.video else 0.0
    items = []
    for c in cands:
        err = fx.validate_clip_window(c.clip_start, c.clip_end, duration)
        if err:
            raise HTTPException(422, f"candidate {c.id}: {err}")
        items.append({
            "id": c.id, "t": c.t, "type": c.type,
            "clip_start": c.clip_start, "clip_end": c.clip_end,
            "name": f"{c.rank:02d}_{c.type}_{c.t:07.1f}.mp4",
        })
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = RenderJob(job_id=job_id)
    _job_owner[job_id] = p.id
    threading.Thread(target=_run_render, args=(p, job_id, req, items), daemon=True).start()
    threading.Thread(target=_train_after_render, args=(p,), daemon=True).start()
    return {"job_id": job_id}


def _train_after_render(p: ProjectStore) -> None:
    """Fold this project's verdicts into the global model; never raises."""
    try:
        from highlights.pipeline.learn import train_from_registry
        meta = train_from_registry(get_registry(), p)
        if meta.get("trained"):
            _hist(p, "model_trained", **{k: meta.get(k) for k in (
                "n_examples", "n_pos", "n_neg", "auroc_lopo", "version")})
        else:
            _hist(p, "model_train_skipped", reason=meta.get("reason"))
    except Exception as e:
        _hist(p, "model_train_skipped", reason=f"{type(e).__name__}: {e}")


def _render_status(p: ProjectStore, job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None or _job_owner.get(job_id) != p.id:
        raise HTTPException(404, "job not found")
    return job.model_dump()


def _render_file(p: ProjectStore, job_id: str, name: str) -> FileResponse:
    fp = (p.root / "renders" / job_id / name).resolve()
    root = (p.root / "renders" / job_id).resolve()
    if root not in fp.parents and fp != root:
        raise HTTPException(403, "forbidden")
    if not fp.is_file():
        raise HTTPException(404, "file not found")
    media = "video/mp4" if fp.suffix == ".mp4" else "application/json"
    return FileResponse(fp, media_type=media, filename=fp.name, content_disposition_type="attachment")


def _project_thumb(p: ProjectStore) -> FileResponse:
    if p.video is None:
        raise HTTPException(404, "no video registered")
    tval = max(p.candidates, key=lambda c: c.confidence).t if p.candidates else p.video.duration_s / 2
    duration = p.video.duration_s
    if duration > 0:
        tval = min(max(0.0, tval), max(0.0, duration - 0.1))
    out = p.thumb_dir() / "project.jpg"
    if not out.exists():
        try:
            fx.thumbnail(_video_path(p), tval, out)
        except RuntimeError as e:
            raise HTTPException(404, f"thumbnail failed: {e}") from e
    if not out.is_file():
        raise HTTPException(404, "thumbnail produced no frame")
    return FileResponse(out, media_type="image/jpeg")


def _stats5(p: ProjectStore) -> dict:
    """Contract 5 stats: pipeline/stats.json or computed demo stats."""
    sp = p.pipeline_dir / "stats.json"
    if sp.is_file():
        st = json.loads(sp.read_text())
        if p.is_multiangle:
            st.setdefault("multiangle", {})["score"] = _multiangle_score(p)
        _verdict_stats(p, st)
        return st
    if p.owner == "demo" and p.title.startswith("Demo match"):
        st = stats.demo_stats()
        with contextlib.suppress(OSError):
            sp.write_text(json.dumps(st, indent=2))
        _verdict_stats(p, st)
        return st
    raise HTTPException(404, "stats not available yet (pipeline has not produced stats.json)")


def _verdict_stats(p: ProjectStore, st: dict) -> None:
    """When the user has made verdicts, headline counts come from them."""
    conf = [c for c in p.candidates if c.status == "confirmed"]
    if not conf:
        return
    st["goals"] = sum(1 for c in conf if c.type == "goal")
    st["highlights"] = len(conf) - st["goals"]
    st["basis_events"] = "user verdicts"


def summary(p: ProjectStore) -> dict:
    """ProjectSummary for list/detail responses."""
    status = pipeline.refresh(p)
    return {
        "id": p.id,
        "title": p.title,
        "created_at": p.created_at,
        "source": p.source_info,
        "meta": p.meta,
        "pipeline_state": p.pipeline_state,
        "auto_retry": status.get("auto_retry") if status else None,
        "progress": status["progress"] if status else 0.0,
        "stage": status["stage"] if status else None,
        "message": (
            (status.get("error") or f"failed during {status.get('stage') or 'pipeline'}")
            if status and status.get("state") == "failed"
            else status["message"] if status else ""
        ),
        "video": (
            {
                "duration_s": p.video.duration_s,
                "width": p.video.width,
                "height": p.video.height,
                "fps": p.video.fps,
            }
            if p.video
            else None
        ),
        "n_candidates": len(p.candidates),
        "n_confirmed": sum(1 for c in p.candidates if c.status == "confirmed"),
        "thumb_url": f"/api/projects/{p.id}/thumb.jpg" if p.video else None,
        "mode": "multiangle" if p.is_multiangle else "single",
        "n_angles": len(p.source_info.get("angles") or []) if p.is_multiangle else 1,
        "cut_info": cut_info(p.root) if p.is_multiangle else None,
    }


# ---------- users ----------


@app.post("/api/users")
def create_user(body: UserCreate) -> dict:
    name = body.name.strip()
    if not (1 <= len(name) <= 40):
        raise HTTPException(422, "name must be 1..40 chars")
    return get_registry().add_user(name)


@app.get("/api/users")
def list_users() -> list:
    reg = get_registry()
    counts: dict[str, int] = {}
    for p in reg.list_projects():
        counts[p.owner] = counts.get(p.owner, 0) + 1
    return [
        {"name": u["name"], "created_at": u["created_at"], "n_projects": counts.get(u["name"], 0)}
        for u in reg.list_users()
    ]


# ---------- projects ----------


def _safe_filename(name: str) -> str:
    base = re.sub(r"[/\\]", "", Path(name).name).strip().lstrip(".")
    return base or "upload.mp4"


def _save_cookies(p, text: str | None) -> str | None:
    """Write YouTube cookies to <project>/source/cookies.txt (0600).
    Returns the path, or None. Content is never logged or echoed."""
    if not text or not text.strip():
        return None
    path = p.source_dir / "cookies.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, 0o600)
    return str(path)


def _project_cookies(p) -> str | None:
    """os.pathsep-joined source/cookies*.txt for reruns, else None.
    Ordered cookies.txt, cookies.1.txt, … (newest source first)."""
    def _idx(f: Path) -> int:
        if f.name == "cookies.txt":
            return 0
        try:
            return int(f.name.split(".")[1])
        except (IndexError, ValueError):
            return 1_000_000
    paths = sorted(p.source_dir.glob("cookies*.txt"), key=_idx)
    return os.pathsep.join(str(f) for f in paths) if paths else None


# ---------- per-user saved YouTube cookies ----------

def _is_admin(user: str) -> bool:
    admins = {a.strip() for a in os.environ.get("REPLAY_ADMINS", "john").split(",")
              if a.strip()}
    return user in admins


def _shared_cookies_path() -> Path:
    return workdir() / "shared" / "youtube_cookies.txt"


def _user_cookies_path(user: str) -> Path:
    return workdir() / "users" / user / "youtube_cookies.txt"


def _save_user_cookies(user: str, text: str) -> Path:
    """Persist the user's YouTube cookies (0600). Never logged or echoed."""
    path = _user_cookies_path(user)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, 0o600)
    return path


def cookies_file_hash(f: Path) -> str:
    """Content hash for cookie dedup / watchdog tried-set bookkeeping."""
    return hashlib.sha256(f.read_bytes()).hexdigest()


def _cookies_sources(user: str) -> list[Path]:
    """Every distinct saved cookies file usable for `user` — the user's
    own plus the admin-shared ones — de-duplicated by content, newest
    first. Empty when none exist."""
    cands = [f for f in (_user_cookies_path(user), _shared_cookies_path())
             if f.is_file()]
    cands.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    seen: set[str] = set()
    out = []
    for f in cands:
        h = cookies_file_hash(f)
        if h not in seen:
            seen.add(h)
            out.append(f)
    return out


def _cookies_source(user: str) -> Path | None:
    """The newest usable cookies file for `user`, or None."""
    srcs = _cookies_sources(user)
    return srcs[0] if srcs else None


def _user_default_cookies(p, user: str) -> str | None:
    """Copy every distinct saved cookies file into the project as
    cookies.txt, cookies.1.txt, … and return them os.pathsep-joined
    (single file -> plain path)."""
    dsts = []
    for k, src in enumerate(_cookies_sources(user)):
        dst = p.source_dir / ("cookies.txt" if k == 0 else f"cookies.{k}.txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        os.chmod(dst, 0o600)
        dsts.append(str(dst))
    return os.pathsep.join(dsts) if dsts else None


class CookiesPut(BaseModel):
    cookies_text: str
    share: bool = False


@app.get("/api/me/youtube-cookies")
def get_youtube_cookies(user: UserDep) -> dict:
    path = _user_cookies_path(user)
    return {"saved": path.is_file(),
            "updated_at": path.stat().st_mtime if path.is_file() else None,
            "shared_available": _shared_cookies_path().is_file(),
            "is_admin": _is_admin(user)}


@app.put("/api/me/youtube-cookies")
def put_youtube_cookies(body: CookiesPut, user: UserDep) -> dict:
    text = body.cookies_text or ""
    if not text.strip():
        raise HTTPException(422, "cookies_text is empty")
    if "youtube.com" not in text:
        raise HTTPException(422, "does not look like Netscape cookies for youtube.com")
    if body.share and not _is_admin(user):
        raise HTTPException(403, "only admins can share cookies server-wide")
    if body.share or _is_admin(user):
        # admin cookies are server-wide by design
        sp = _shared_cookies_path()
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(text)
        os.chmod(sp, 0o600)
    path = _save_user_cookies(user, text)
    watchdog.kick()  # cookie-failed projects can resume immediately
    return {"saved": True, "updated_at": path.stat().st_mtime}


@app.delete("/api/admin/youtube-cookies")
def delete_shared_youtube_cookies(user: UserDep) -> Response:
    if not _is_admin(user):
        raise HTTPException(403, "admin only")
    _shared_cookies_path().unlink(missing_ok=True)
    return Response(status_code=204)


@app.delete("/api/me/youtube-cookies")
def delete_youtube_cookies(user: UserDep) -> Response:
    _user_cookies_path(user).unlink(missing_ok=True)
    return Response(status_code=204)


# ---------- server-wide YouTube proxy ----------

PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}


def _shared_proxy_path() -> Path:
    return workdir() / "shared" / "youtube_proxy.txt"


def _yt_proxy() -> str | None:
    return pipeline._shared_proxy()


def _proxy_host(proxy: str) -> str | None:
    """scheme://host:port, credentials stripped — safe to echo/log."""
    u = urlparse(proxy)
    if not u.hostname:
        return None
    port = f":{u.port}" if u.port else ""
    return f"{u.scheme}://{u.hostname}{port}"


class ProxyPut(BaseModel):
    proxy_url: str


@app.get("/api/admin/youtube-proxy")
def get_youtube_proxy(user: UserDep) -> dict:
    if not _is_admin(user):
        raise HTTPException(403, "admin only")
    proxy = _yt_proxy()
    return {"set": proxy is not None,
            "host": _proxy_host(proxy) if proxy else None}


@app.put("/api/admin/youtube-proxy")
def put_youtube_proxy(body: ProxyPut, user: UserDep) -> dict:
    if not _is_admin(user):
        raise HTTPException(403, "admin only")
    proxy = (body.proxy_url or "").strip()
    u = urlparse(proxy)
    if u.scheme not in PROXY_SCHEMES or not u.hostname:
        raise HTTPException(
            422, "proxy_url must be http(s):// or socks5(h):// with a host")
    path = _shared_proxy_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proxy)
    os.chmod(path, 0o600)
    return {"set": True, "host": _proxy_host(proxy)}


@app.delete("/api/admin/youtube-proxy")
def delete_youtube_proxy(user: UserDep) -> Response:
    if not _is_admin(user):
        raise HTTPException(403, "admin only")
    _shared_proxy_path().unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/config")
def get_config() -> dict:
    return {"upload_origin": os.environ.get("HL_PUBLIC_URL") or None}


@app.get("/api/learning")
def get_learning(user: UserDep) -> dict:
    from highlights.pipeline.learn import learn_dir
    ld = learn_dir()
    meta_path = ld / "verdict_lr.json"
    if not meta_path.is_file():
        return {"trained": False}
    return json.loads(meta_path.read_text())


# ---------- playlist ----------

class PlaylistPut(BaseModel):
    url: str


@app.get("/api/playlist")
def get_playlist(user: UserDep) -> dict:
    return playlist.get_state(user, get_registry())


@app.put("/api/playlist")
def put_playlist(body: PlaylistPut, user: UserDep) -> dict:
    return playlist.set_url(user, body.url.strip(), get_registry())


@app.post("/api/playlist/refresh")
def refresh_playlist(user: UserDep) -> dict:
    playlist.refresh_async(user, get_registry())
    return playlist.get_state(user, get_registry())


# ---------- storage ----------

_STORAGE_TTL = 60.0
_storage_cache: tuple[float, dict] | None = None


def _du(root: Path, seen: set) -> int:
    """Bytes under root; hardlinked files counted once via the shared
    (st_dev, st_ino) set."""
    total = 0
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            try:
                st = os.lstat(os.path.join(dp, fn))
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    return total


def _project_sources_bytes(p: ProjectStore, seen: set) -> int:
    total = 0
    if p.is_multiangle:
        for i in range(len(p.source_info.get("angles") or [])):
            total += _du(p.angle_dir(i), seen)
    else:
        total += _du(p.source_dir, seen)
    return total


@app.get("/api/storage")
def get_storage(user: UserDep) -> dict:
    global _storage_cache
    now = time.time()
    if _storage_cache and now - _storage_cache[0] < _STORAGE_TTL:
        return _storage_cache[1]
    wd = workdir()
    usage = shutil.disk_usage(wd)
    seen: set = set()
    per = []
    projects_bytes = 0
    for p in get_registry().list_projects():
        b = _du(p.root, seen)
        sb = _project_sources_bytes(p, seen)
        projects_bytes += b
        per.append({"id": p.id, "title": p.title, "owner": p.owner,
                    "bytes": b, "sources_bytes": sb})
    out = {"total_bytes": usage.total, "used_bytes": usage.used,
           "free_bytes": usage.free, "projects_bytes": projects_bytes,
           "per_project": per}
    _storage_cache = (now, out)
    return out


@app.post("/api/projects")
async def create_project(request: Request, user: UserDep) -> dict:
    reg = get_registry()
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" in ct:
        form = await request.form()
        file = form.get("file")
        if file is None or not hasattr(file, "read"):
            raise HTTPException(400, "missing 'file' field")
        title = (form.get("title") or "").strip() if isinstance(form.get("title"), str) else ""
        cookies_text = form.get("cookies_text") if isinstance(form.get("cookies_text"), str) else None
        name = _safe_filename(file.filename or "upload.mp4")
        meta = _meta_or_422(
            form.get("pitch_type") or None,
            form.get("camera") or None,
            form.get("cut_style") or None)
        # create the project first so we have a source dir
        p = reg.create_project(
            owner=user,
            title=title or name,
            source={"kind": "upload", "url": None, "filename": name},
            meta=meta,
        )
        dst = p.source_dir / name
        with open(dst, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        _save_cookies(p, cookies_text)
        _hist(p, "created", title=p.title)
        _hist(p, "source_added", src="upload", filename=name)
        try:
            pipeline.spawn(p, video=str(dst), stages=NO_DOWNLOAD_STAGES)
        except pipeline.PipelineBusy as e:
            raise HTTPException(409, str(e)) from e
        return summary(p)

    # JSON body
    body = ProjectCreate(**(await request.json()))
    if body.youtube_url:
        url = body.youtube_url.strip()
        parsed = urlparse(url)
        if ("\n" in body.youtube_url or len(url) > 2048
                or parsed.scheme not in ("http", "https") or not parsed.netloc):
            raise HTTPException(
                422, "Enter a valid video URL "
                     "(e.g. https://www.youtube.com/watch?v=...)")
        body.youtube_url = url
        p = reg.create_project(
            owner=user,
            title=body.title or body.youtube_url,
            source={"kind": "youtube", "url": body.youtube_url, "filename": None},
            meta=_meta_or_422(body.pitch_type, body.camera, body.cut_style),
        )
        if body.cookies_text:
            cookies = _save_cookies(p, body.cookies_text)
            _save_user_cookies(user, body.cookies_text)
        else:
            cookies = _user_default_cookies(p, user)
        _hist(p, "created", title=p.title)
        _hist(p, "source_added", src="youtube", url=body.youtube_url)
        try:
            pipeline.spawn(p, youtube_url=body.youtube_url, cookies=cookies)
        except pipeline.PipelineBusy as e:
            raise HTTPException(409, str(e)) from e
        return summary(p)
    if body.path:
        fp = Path(body.path)
        if not fp.is_file():
            raise HTTPException(404, f"file not found: {fp}")
        resolved = str(fp.resolve())
        p = reg.create_project(
            owner=user,
            title=body.title or fp.name,
            source={"kind": "path", "url": resolved, "filename": fp.name},
            meta=_meta_or_422(body.pitch_type, body.camera, body.cut_style),
        )
        _hist(p, "created", title=p.title)
        _hist(p, "source_added", src="path", path=resolved)
        if body.run_pipeline:
            try:
                pipeline.spawn(p, video=resolved, stages=NO_DOWNLOAD_STAGES)
            except pipeline.PipelineBusy as e:
                raise HTTPException(409, str(e)) from e
        else:
            try:
                info = fx.probe(resolved)
            except RuntimeError as e:
                raise HTTPException(422, f"ffprobe failed: {e}") from e
            p.set_video(VideoInfo(path=resolved, registered_at=time.time(), **info))
        return summary(p)
    raise HTTPException(422, "provide file, youtube_url or path")


@app.get("/api/projects")
def list_projects(user: UserDep) -> list:
    return [summary(p) for p in get_registry().list_projects(user)]


@app.get("/api/feedback/export")
def feedback_export(user: UserDep, all: bool = False) -> Response:
    """JSONL of review decisions across all projects — training data export."""
    lines = []
    for p in get_registry().list_projects():
        vid = {
            "width": p.video.width, "height": p.video.height,
            "fps": p.video.fps, "duration_s": p.video.duration_s,
        } if p.video else None
        for c in p.candidates:
            if c.status == "pending" and not all:
                continue
            lines.append(json.dumps({
                "project_id": p.id,
                "owner": p.owner,
                "title": p.title,
                "meta": p.meta,
                "video": vid,
                "candidate_id": c.id,
                "type": c.type,
                "t": c.t,
                "t_start": c.t_start,
                "t_end": c.t_end,
                "confidence": c.confidence,
                "signals": c.signals,
                "status": c.status,
                "cross_validation": c.cross_validation,
                "source_kind": p.source_info.get("kind"),
            }))
    return Response("\n".join(lines) + ("\n" if lines else ""),
                    media_type="application/x-ndjson")


# ---------- multi-angle (Option 2: director cut) ----------


def _norm_angles(angles: list[AngleSpec]) -> list[dict]:
    if not (2 <= len(angles) <= 4):
        raise HTTPException(422, "multi-angle projects need 2..4 angles")
    out = []
    for i, a in enumerate(angles):
        cam = a.camera.strip() if a.camera else ""
        if cam and cam not in CAMERA_TYPES:
            raise HTTPException(
                422, f"angle {i} camera must be one of {CAMERA_TYPES}")
        out.append({
            "url": a.url.strip() if a.url else None,
            "filename": a.filename.strip() if a.filename else None,
            "label": a.label.strip() or f"Angle {i + 1}",
            "camera": cam or None,
        })
    return out


@app.post("/api/projects/multiangle")
def create_multiangle(body: MultiangleCreate, user: UserDep) -> dict:
    angles = _norm_angles(body.angles)
    for i, a in enumerate(angles):
        if not a["url"]:
            raise HTTPException(422, f"angle {i} has no url")
    if body.match_window is not None and not (
            len(body.match_window) == 2 and
            0 <= body.match_window[0] < body.match_window[1]):
        raise HTTPException(422, "match_window must be [start, end] with 0 <= start < end")
    if body.match_window is not None and body.match_window_angle is not None and not (
            0 <= body.match_window_angle < len(angles)):
        raise HTTPException(422, f"match_window_angle must be 0..{len(angles) - 1}")
    p = get_registry().create_project(
        owner=user,
        title=(body.title or "").strip() or angles[0]["url"] or "multi-angle",
        source={"kind": "multiangle", "url": None, "filename": None, "angles": angles},
        meta=_meta_or_422(body.pitch_type, None, body.cut_style),
    )
    if body.match_window:
        p.multiangle_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(p.multiangle_dir / "match_window_src.json",
                          {"angle": body.match_window_angle,
                           "start": float(body.match_window[0]),
                           "end": float(body.match_window[1])}, indent=1)
    if body.cookies_text:
        cookies = _save_cookies(p, body.cookies_text)
        _save_user_cookies(user, body.cookies_text)
    else:
        cookies = _user_default_cookies(p, user)
    _hist(p, "created", title=p.title)
    _hist(p, "source_added", src="multiangle",
          n_angles=len(angles))
    try:
        pipeline.spawn_multiangle(p, cookies=cookies)
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e
    return summary(p)


@app.post("/api/projects/multiangle/upload")
async def create_multiangle_upload(request: Request, user: UserDep) -> dict:
    form = await request.form()
    files = [f for f in form.getlist("files") if hasattr(f, "read")]
    cameras = [x for x in form.getlist("cameras") if isinstance(x, str)]
    title = form.get("title") if isinstance(form.get("title"), str) else ""
    cookies_text = form.get("cookies_text") if isinstance(form.get("cookies_text"), str) else None
    if not (2 <= len(files) <= 4):
        raise HTTPException(422, "multi-angle uploads need 2..4 files")
    for i, c in enumerate(cameras):
        if c and c not in CAMERA_TYPES:
            raise HTTPException(
                422, f"angle {i} camera must be one of {CAMERA_TYPES}")
    meta = _meta_or_422(
        form.get("pitch_type") or None,
        None,
        form.get("cut_style") or None)
    names = [_safe_filename(f.filename or f"angle{i}.mp4") for i, f in enumerate(files)]
    angles = [
        {
            "url": None,
            "filename": name,
            "label": Path(name).stem or f"Angle {i + 1}",
            "camera": (cameras[i].strip() if i < len(cameras) else "") or None,
        }
        for i, name in enumerate(names)
    ]
    p = get_registry().create_project(
        owner=user,
        title=title.strip() or names[0],
        source={"kind": "multiangle", "url": None, "filename": None, "angles": angles},
        meta=meta,
    )
    for i, (f, name) in enumerate(zip(files, names, strict=True)):
        suffix = Path(name).suffix or ".mp4"
        dst = p.angle_dir(i) / f"match{suffix}"
        with open(dst, "wb") as fh:
            while True:
                chunk = await f.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    _save_cookies(p, cookies_text)
    _hist(p, "created", title=p.title)
    _hist(p, "source_added", src="multiangle",
          n_angles=len(angles))
    try:
        pipeline.spawn_multiangle(p)
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e
    return summary(p)


def _require_multiangle(p: ProjectStore) -> None:
    if not p.is_multiangle:
        raise HTTPException(404, "not a multi-angle project")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _multiangle_score(p: ProjectStore) -> dict:
    st = _read_json(p.pipeline_dir / "stats.json") or {}
    score = (st.get("multiangle") or {}).get("score") or {}
    # user-set labels/colours win over the stats block
    sd = _read_json(p.multiangle_dir / "score.json") or {}
    for side in ("home", "away"):
        for k, v in (sd.get(side) or {}).items():
            if v is not None:
                score.setdefault(side, {})[k] = v
    home = score.get("home") or {}
    away = score.get("away") or {}
    n_home = n_away = n_un = 0
    for c in p.candidates:
        if c.status != "confirmed" or c.type != "goal":
            continue
        team = (c.signals or {}).get("team")
        if team == "home":
            n_home += 1
        elif team == "away":
            n_away += 1
        else:
            n_un += 1
    return {
        **score,
        "home": {"label": home.get("label") or "Home", "goals": n_home,
                 "hex": home.get("hex")},
        "away": {"label": away.get("label") or "Away", "goals": n_away,
                 "hex": away.get("hex")},
        "unassigned": n_un,
        "basis": "confirmed goals with team set",
    }


def _angle_probe_duration(p: ProjectStore, i: int) -> float | None:
    """Duration fallback for angles whose status.json lacks a probe block
    (imported projects): per-angle pipeline/probe.json, else ffprobe the
    angle video and cache the result there."""
    probe_path = p.angle_dir(i) / "pipeline" / "probe.json"
    pr = _read_json(probe_path)
    if pr and pr.get("duration_s"):
        return pr["duration_s"]
    video = p.angle_video(i)
    if video is None:
        return None
    try:
        pr = fx.probe(video)
    except Exception:
        return None
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(probe_path, pr, indent=1)
    return pr.get("duration_s")


def _angles_info(p: ProjectStore) -> list[dict]:
    out = []
    for i, a in enumerate(p.source_info.get("angles") or []):
        ast = _read_json(p.angle_dir(i) / "pipeline" / "status.json") or {}
        duration = (ast.get("video") or {}).get("duration_s")
        if duration is None:
            duration = _angle_probe_duration(p, i)
        out.append({
            "index": i,
            "label": a.get("label") or f"Angle {i + 1}",
            "url": a.get("url"),
            "filename": a.get("filename"),
            "duration": duration,
            "status": ast.get("state"),
            "has_file": p.angle_video(i) is not None,
        })
    return out


scoped = APIRouter(prefix="/api/projects/{project_id}")
legacy = APIRouter(prefix="/api")


class TitlePatch(BaseModel):
    title: str


@scoped.patch("")
def patch_project(body: TitlePatch, p: ScopedP) -> dict:
    t = body.title.strip()
    if not (1 <= len(t) <= 120):
        raise HTTPException(422, "title must be 1-120 characters")
    p.title = t
    p.save()
    _hist(p, "renamed", title=t)
    return {"id": p.id, "title": p.title}


@scoped.get("")
def get_project(p: ScopedP) -> dict:
    return {**summary(p), "pipeline": pipeline.refresh(p)}


@scoped.delete("")
def delete_project(p: ScopedP) -> Response:
    status = pipeline.read_status(p)
    if status and pipeline.pid_alive(status.get("pid")):
        pipeline.cancel(p)
    for jid in [j for j, pid in _job_owner.items() if pid == p.id]:
        _jobs.pop(jid, None)
        _job_owner.pop(jid, None)
    _proxy_jobs.pop(p.id, None)
    try:
        # persist training examples now — archive/delete removes the
        # pipeline dir that collect_project_examples needs
        from highlights.pipeline.learn import collect_project_examples, save_examples
        save_examples(collect_project_examples(p))
    except Exception as e:
        print(f"warning: could not save learning examples for {p.id}: {e}")
    try:
        history.archive_project(p)
    except Exception as e:
        print(f"warning: archive failed for {p.id}: {e}")
    get_registry().delete(p.id)
    threading.Thread(target=_train_after_render, args=(p,), daemon=True).start()
    return Response(status_code=204)


def _respawn_project(p: ProjectStore, owner: str,
                     *, stages: list[str] | None = None,
                     force: bool = False) -> dict:
    """Spawn the pipeline exactly like POST /pipeline/run — shared with the
    watchdog so auto-resume behaviour can't diverge from a manual re-run."""
    if p.is_multiangle:
        ck = _user_default_cookies(p, owner) or _project_cookies(p)
        return pipeline.spawn_multiangle(p, stages=stages, force=force,
                                       cookies=ck)
    kind = p.source_info.get("kind")
    if kind == "youtube":
        ck = _user_default_cookies(p, owner) or _project_cookies(p)
        return pipeline.spawn(p, youtube_url=p.source_info.get("url"),
                              stages=stages, force=force,
                              cookies=ck)
    video = p.video.path if p.video else p.source_info.get("url")
    if not video:
        files = [f for f in p.source_dir.iterdir() if f.is_file()]
        if files:
            video = str(files[0])
    if not video:
        raise HTTPException(422, "no video or upload to run the pipeline on")
    return pipeline.spawn(
        p, video=video,
        stages=stages if stages is not None else NO_DOWNLOAD_STAGES,
        force=force,
    )


@scoped.post("/pipeline/run")
def run_pipeline(p: ScopedP, user: UserDep,
                 body: Annotated[dict | None, Body()] = None) -> dict:
    body = body or {}
    stages = body.get("stages")
    force = bool(body.get("force", False))
    try:
        return _respawn_project(p, user, stages=stages, force=force)
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e


@scoped.post("/pipeline/cancel")
def cancel_pipeline(p: ScopedP) -> dict:
    # pause is logged by history.track_status (refresh diff) — one event
    # no matter which path cancelled/paused the run
    return pipeline.cancel(p)


@scoped.get("/pipeline")
def get_pipeline(p: ScopedP) -> dict:
    return pipeline.status_with_log(p)


@scoped.get("/stats")
def get_stats(p: ScopedP) -> dict:
    return _stats5(p)


@scoped.get("/multiangle")
def get_multiangle(p: ScopedP) -> dict:
    _require_multiangle(p)
    director = _read_json(p.multiangle_dir / "director.json")
    if director:
        director = {
            k: v for k, v in director.items()
            if k not in {"segments", "segments_out"}
        }
    cr = _read_json(p.multiangle_dir / "cut_range.json")
    match_window = ([float(cr["lo"]), float(cr["hi"])]
                    if cr and cr.get("hi", 0) > cr.get("lo", 0) else None)
    return {
        "sync": _read_json(p.multiangle_dir / "sync.json"),
        "director": director,
        "angles": _angles_info(p),
        "score": _multiangle_score(p),
        "status": pipeline.read_status(p),
        "cut_style": p.meta.get("cut_style", "normal"),
        "goal_aware": bool(p.meta.get("goal_aware", True)),
        "sources_purged": bool(p.meta.get("sources_purged")),
        "match_window": match_window,
        "match_window_src": _read_json(p.multiangle_dir / "match_window_src.json"),
    }


@scoped.get("/multiangle/settings")
def get_multiangle_settings(p: ScopedP) -> dict:
    _require_multiangle(p)
    return {"goal_aware": bool(p.meta.get("goal_aware", True))}


@scoped.put("/multiangle/settings")
def put_multiangle_settings(body: MaSettingsPut, p: ScopedP) -> dict:
    _require_multiangle(p)
    p.meta["goal_aware"] = body.goal_aware
    p.save()
    return {"goal_aware": body.goal_aware}


@scoped.put("/multiangle/score")
def put_multiangle_score(body: ScorePut, p: ScopedP) -> dict:
    """Persist team labels/colours for the scoreboard + score card."""
    _require_multiangle(p)
    for name, v in (("home_label", body.home_label),
                    ("away_label", body.away_label)):
        if not (1 <= len(v.strip()) <= 20):
            raise HTTPException(422, f"{name} must be 1-20 characters")
    doc = {
        "home": {"label": body.home_label.strip(),
                 "hex": _norm_hex(body.home_hex, "home_hex")},
        "away": {"label": body.away_label.strip(),
                 "hex": _norm_hex(body.away_hex, "away_hex")},
    }
    p.multiangle_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p.multiangle_dir / "score.json", doc, indent=1)
    _hist(p, "score_labels_set", home=doc["home"]["label"],
          away=doc["away"]["label"])
    return _multiangle_score(p)


@scoped.post("/multiangle/scoreboard")
def post_multiangle_scoreboard(p: ScopedP) -> dict:
    """Queue a job that burns scoreboard + clock into the active cut.

    Goals come from confirmed goal candidates with signals.team set
    (times are OUTPUT time of the active cut); the clock starts at the
    detected kickoff (pipeline/match_window.json match_window[0])."""
    _require_multiangle(p)
    status = pipeline.read_status(p)
    if status and status.get("state") in ("queued", "running") \
            and pipeline.pid_alive(status.get("pid")):
        raise HTTPException(409, "pipeline is running")
    if not (p.root / "match.mp4").is_file():
        raise HTTPException(409, "no rendered cut yet — run the pipeline first")
    active_id = (_read_json(p.multiangle_dir / "cuts" / "active.json") or {}).get("id")
    if active_id:
        meta = _read_json(p.multiangle_dir / "cuts" / str(active_id) / "meta.json") or {}
        if meta.get("scoreboard"):
            raise HTTPException(409, "active cut already has a scoreboard — "
                                     "activate the plain cut first")
    goals, n_un = [], 0
    for c in p.candidates:
        if c.status != "confirmed" or c.type != "goal":
            continue
        team = (c.signals or {}).get("team")
        if team in ("home", "away"):
            goals.append({"t": float(c.t), "team": team})
        else:
            n_un += 1
    kickoff = 0.0
    mw = _read_json(p.pipeline_dir / "match_window.json") or {}
    win = mw.get("match_window")
    if isinstance(win, list) and win:
        with contextlib.suppress(Exception):
            kickoff = float(win[0])
    labels = _read_json(p.multiangle_dir / "score.json") or {}
    write_json_atomic(p.multiangle_dir / "scoreboard.json", {
        "home": {"label": (labels.get("home") or {}).get("label") or "Home",
                 "hex": (labels.get("home") or {}).get("hex")},
        "away": {"label": (labels.get("away") or {}).get("label") or "Away",
                 "hex": (labels.get("away") or {}).get("hex")},
        "goals": goals,
        "kickoff": kickoff,
    }, indent=1)
    _ensure_cut_snapshot(p)   # keep the pre-scoreboard cut selectable
    try:
        job = pipeline.spawn_multiangle(
            p, stages=["scoreboard"], force=True,
            style=p.meta.get("cut_style"))
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e
    _hist(p, "scoreboard_requested", n_goals=len(goals))
    return {"job": job, "goals": len(goals), "unassigned": n_un}


@scoped.put("/multiangle/match-window")
def put_multiangle_match_window(body: MaMatchWindowPut, p: ScopedP) -> dict:
    """Set/clear the match window measured in file seconds of `body.angle`.
    Always writes match_window_src.json; when sync.json exists it is also
    converted to shared-T cut_range.json (shared-T = file_t + offset).
    The runner re-derives cut_range on the next sync/track/director."""
    _require_multiangle(p)
    src_path = p.multiangle_dir / "match_window_src.json"
    cr_path = p.multiangle_dir / "cut_range.json"

    def _invalidate_downstream() -> None:
        # director segments are in OUTPUT time (0 = union_lo); a changed
        # union remaps every segment — director/fuse/render must re-run
        for f in ("director.json", "concat.txt", "fused_candidates.json"):
            (p.multiangle_dir / f).unlink(missing_ok=True)
        (p.pipeline_dir / "candidates.json").unlink(missing_ok=True)
        (p.pipeline_dir / "stats.json").unlink(missing_ok=True)
        if p.video and Path(p.video.path).exists():
            Path(p.video.path).unlink()

    if body.start is None and body.end is None:
        had = src_path.exists() or cr_path.exists()
        src_path.unlink(missing_ok=True)
        cr_path.unlink(missing_ok=True)
        if had:
            _invalidate_downstream()
        _hist(p, "window_set", cleared=True)
        return {"match_window": None, "match_window_src": None}
    if body.start is None or body.end is None:
        raise HTTPException(422, "provide both start and end, or neither")
    if not (0.0 <= body.start < body.end):
        raise HTTPException(422, "need 0 <= start < end")
    n_angles = len(p.source_info.get("angles") or [])
    if body.angle is not None and not (0 <= body.angle < max(1, n_angles)):
        raise HTTPException(422, f"angle must be 0..{n_angles - 1}")
    # resolve "measured on the longest angle" only once EVERY angle has a
    # known duration — resolving early picks the wrong angle when a later
    # angle hasn't been downloaded/probed yet
    resolved = body.angle
    if resolved is None:
        durs = [float(a.get("duration") or 0.0) for a in _angles_info(p)]
        if durs and len(durs) == n_angles and all(d > 0 for d in durs):
            resolved = int(max(range(len(durs)), key=lambda i: durs[i]))
    p.multiangle_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(src_path, {"angle": resolved,
                               "start": float(body.start),
                               "end": float(body.end)}, indent=1)
    # convert now if sync already exists; the runner re-applies otherwise
    sync = _read_json(p.multiangle_dir / "sync.json")
    mw = None
    if sync and resolved is not None:
        try:
            from highlights.multiangle.syncmap import file_to_shared, timemap_from_sync
            tm = timemap_from_sync(sync)
            mw = [max(0.0, file_to_shared(tm, resolved, float(body.start))),
                  file_to_shared(tm, resolved, float(body.end))]
            old = _read_json(cr_path)
            if not old or [old.get("lo"), old.get("hi")] != mw:
                write_json_atomic(cr_path, {"lo": mw[0], "hi": mw[1]},
                                  indent=1)
                _invalidate_downstream()
        except Exception:
            mw = None
    p.save()
    _hist(p, "window_set", angle=resolved, start=float(body.start),
          end=float(body.end))
    return {"match_window": mw, "match_window_src":
            {"angle": resolved, "start": float(body.start),
             "end": float(body.end)}}


class MaAnglePut(BaseModel):
    url: str


@scoped.put("/multiangle/angles/{index}")
def put_multiangle_angle(index: int, body: MaAnglePut, p: ScopedP) -> dict:
    """Replace one angle's source URL. Deletes that angle's downloaded
    files + track outputs and everything derived downstream (sync,
    director, fused candidates, rendered video, this angle's cached
    segments) so the next run redownloads/resyncs/rerenders it."""
    _require_multiangle(p)
    angles = p.source_info.get("angles") or []
    if not (0 <= index < len(angles)):
        raise HTTPException(404, f"no angle {index}")
    st = (pipeline.read_status(p) or {}).get("state")
    if st in {"queued", "running"}:
        raise HTTPException(409, f"cannot change angles while a job is {st}")
    url = body.url.strip()
    if not url:
        raise HTTPException(422, "url required")
    angles[index]["url"] = url
    angles[index].pop("filename", None)
    adir = p.angle_dir(index)
    if adir.is_dir():
        shutil.rmtree(adir)
    ma = p.multiangle_dir
    # cached segments + this angle's mezzanine are stale (the seg_key is
    # keyed by angle index + file time, so same key = different video).
    # seg_key now also mixes in the mezzanine key, so computing the stale
    # names is impractical — clear the whole seg cache instead (other
    # angles' segs re-extract in seconds) plus this angle's mezz file.
    shutil.rmtree(ma / "segs", ignore_errors=True)
    for f in (ma / "mezz").glob(f"{index}_*.mp4"):
        f.unlink(missing_ok=True)
    for name in ("sync.json", "director.json", "concat.txt",
                 "fused_candidates.json"):
        (ma / name).unlink(missing_ok=True)
    (p.pipeline_dir / "candidates.json").unlink(missing_ok=True)
    (p.pipeline_dir / "stats.json").unlink(missing_ok=True)
    if p.video and Path(p.video.path).exists():
        Path(p.video.path).unlink()
    # persist the edited source block to disk first — save() prefers the
    # on-disk source (pipeline writes labels there out-of-band)
    try:
        pj = p.root / "project.json"
        data = json.loads(pj.read_text())
        data["source"] = p.source_info
        write_json_atomic(pj, data, indent=2)
    except Exception:
        pass
    p.save()
    _hist(p, "source_added", src="angle", index=index, url=url)
    return {"index": index, "url": url, "angles": angles}


@scoped.post("/purge-sources")
def purge_sources(p: ScopedP) -> dict:
    """Delete the original angle videos + track intermediates; keep the cut."""
    _require_multiangle(p)
    if p.pipeline_state != "done":
        raise HTTPException(409, "pipeline is not done")
    if p.meta.get("sources_purged"):
        return {"freed_bytes": 0, "sources_purged": True}
    freed = 0
    video_exts = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
    for i in range(len(p.source_info.get("angles") or [])):
        v = p.angle_video(i)
        if v is not None:
            freed += v.stat().st_size
            v.unlink()
        td = p.angle_dir(i) / "track"
        if td.is_dir():
            for f in td.iterdir():
                if f.is_file() and f.suffix.lower() in video_exts | {".wav"}:
                    freed += f.stat().st_size
                    f.unlink()
        # 480p analysis proxy (rebuilt on the next track run)
        for f in p.angle_dir(i).glob("analysis_480p.*"):
            if f.is_file():
                freed += f.stat().st_size
                f.unlink()
    # render intermediates (mezzanines are re-built on the next re-cut)
    for dname in ("mezz", "segs"):
        sd = p.multiangle_dir / dname
        if sd.is_dir():
            for f in sd.iterdir():
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
    p.meta["sources_purged"] = True
    p.save()
    _hist(p, "sources_purged", freed_bytes=freed)
    return {"freed_bytes": freed, "sources_purged": True}


@scoped.post("/multiangle/recut")
def recut_multiangle(body: RecutPut, p: ScopedP, user: UserDep) -> dict:
    """Re-run director+render+fuse with a different cut style."""
    _require_multiangle(p)
    if body.style not in CUT_STYLES:
        raise HTTPException(422, f"style must be one of {CUT_STYLES}")
    if p.meta.get("sources_purged"):
        raise HTTPException(409, "angle sources were purged — cannot re-cut")
    _ensure_cut_snapshot(p)   # keep the current cut selectable afterwards
    p.meta["cut_style"] = body.style
    if body.goal_aware is not None:
        p.meta["goal_aware"] = body.goal_aware
    window = body.window
    if window:
        from highlights.multiangle.timemap import from_output_time_with_replays
        current_director = _read_json(
            p.multiangle_dir / "director.json") or {}
        replays = current_director.get("replays") or []
        window = [
            from_output_time_with_replays(float(t), replays)
            for t in window
        ]
    # optional cut range: window is in the current video's output time,
    # stored as absolute shared-T seconds for the runner's ctx.union()
    cr_path = p.multiangle_dir / "cut_range.json"
    cr_base = p.multiangle_dir / "cut_range_base.json"
    if not body.preview and cr_base.exists():
        # a preview range was active: restore the pre-preview cut_range
        # (or delete it when there was none) before the full re-cut
        base = _read_json(cr_base)
        if base and base.get("lo") is not None:
            write_json_atomic(cr_path, base, indent=1)
        else:
            cr_path.unlink(missing_ok=True)
        cr_base.unlink()
    # dur = the CURRENT rendered video's length: an active cut_range's
    # span, else the coverage union span, else the source video
    cur = _read_json(cr_path)
    if cur:
        dur = float(cur["hi"]) - float(cur["lo"])
    else:
        sync = _read_json(p.multiangle_dir / "sync.json") or {}
        try:
            ulo, uhi = sync["coverage"]["union"]
            dur = float(uhi) - float(ulo)
        except Exception:
            dur = p.video.duration_s if p.video else 0.0
    full = (window is None or not window or
            (window[0] <= 0.5 and window[1] >= dur - 0.5))
    if full:
        # "the whole current video": keep an existing cut_range (the
        # current video is itself windowed — deleting it would expand
        # the re-cut to the full coverage union)
        if not cur:
            cr_path.unlink(missing_ok=True)
    else:
        if len(window) != 2:
            raise HTTPException(422, "window must be [start, end]")
        s, e = float(window[0]), float(window[1])
        if not (0 <= s < e <= dur):
            raise HTTPException(
                422, f"need 0 <= start < end <= duration ({dur:.1f} s)")
        if body.preview and not cr_base.exists():
            # remember the pre-preview cut_range so the next full re-cut
            # returns to it; {"none": true} = there wasn't one
            write_json_atomic(
                cr_base,
                dict(cur) if cur else {"none": True}, indent=1)
        cur_lo = float(cur["lo"]) if cur else None
        if cur_lo is None:
            sync = _read_json(p.multiangle_dir / "sync.json") or {}
            try:
                cur_lo = float(sync["coverage"]["union"][0])
            except Exception:
                cur_lo = 0.0
        write_json_atomic(cr_path, {"lo": cur_lo + s, "hi": cur_lo + e},
                          indent=1)
    p.save()
    p.invalidate_video()   # drops proxy.mp4 + thumb cache for the old cut
    try:
        ck = _user_default_cookies(p, user) or _project_cookies(p)
        out = pipeline.spawn_multiangle(
            p, stages=["director", "render", "fuse"], force=True,
            style=body.style, cookies=ck)
        _hist(p, "recut_queued", style=body.style, preview=body.preview,
              window=body.window)
        return out
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e


@scoped.get("/multiangle/director")
def get_multiangle_director(p: ScopedP) -> dict:
    _require_multiangle(p)
    d = _read_json(p.multiangle_dir / "director.json")
    if d is None:
        raise HTTPException(404, "director.json not available yet")
    if "segments_out" in d:
        d = {**d, "segments": d["segments_out"],
             "segments_live": d.get("segments", [])}
        d.pop("segments_out", None)
    return d


@scoped.get("/multiangle/angle/{angle_idx}/video")
def get_angle_video(angle_idx: int, p: PublicP) -> FileResponse:
    _require_multiangle(p)
    _check_angle_idx(p, angle_idx)
    v = p.angle_video(angle_idx)
    if v is None:
        raise HTTPException(404, "angle video not found")
    return _serve(v)


def _check_angle_idx(p: ProjectStore, angle_idx: int) -> None:
    angles = p.source_info.get("angles") or []
    if not (0 <= angle_idx < len(angles)):
        raise HTTPException(404, "angle out of range")


def _angle_proxy_path(p: ProjectStore, angle_idx: int) -> Path:
    return p.angle_dir(angle_idx) / "proxy.mp4"


def _angle_proxy_state(p: ProjectStore, angle_idx: int) -> str:
    if _angle_proxy_path(p, angle_idx).is_file():
        return "ready"
    job = _angle_proxy_jobs.get((p.id, angle_idx))
    if job is not None and not job.done and job.error is None:
        return "started"
    return "missing"


@scoped.get("/multiangle/angle/{angle_idx}/proxy")
def get_angle_proxy(angle_idx: int, p: PublicP) -> FileResponse:
    _require_multiangle(p)
    _check_angle_idx(p, angle_idx)
    return _serve(_angle_proxy_path(p, angle_idx))


@scoped.get("/multiangle/angle/{angle_idx}/proxy/status")
def get_angle_proxy_status(angle_idx: int, p: PublicP) -> dict:
    _require_multiangle(p)
    _check_angle_idx(p, angle_idx)
    return {"status": _angle_proxy_state(p, angle_idx)}


@scoped.post("/multiangle/angle/{angle_idx}/proxy")
def start_angle_proxy(angle_idx: int, p: ScopedP) -> dict:
    _require_multiangle(p)
    _check_angle_idx(p, angle_idx)
    status = _angle_proxy_state(p, angle_idx)
    if status != "missing":
        return {"status": status}
    src = p.angle_video(angle_idx)
    if src is None:
        raise HTTPException(404, "angle video not found")
    duration = (_angles_info(p)[angle_idx].get("duration") or 0.0)
    # ProxyJob writes <name>.part.mp4 and renames on success, so a
    # half-written proxy.mp4 is never served.
    job = fx.ProxyJob(src, _angle_proxy_path(p, angle_idx), float(duration))
    _angle_proxy_jobs[(p.id, angle_idx)] = job
    job.start()
    return {"status": "started"}


@scoped.get("/multiangle/zones")
def get_multiangle_zones(p: PublicP) -> dict:
    _require_multiangle(p)
    n = len(p.source_info.get("angles") or [])
    z = _read_json(p.multiangle_dir / "zones.json")
    if z and isinstance(z.get("angles"), list):
        if z.get("version") == 2:
            return z
        # legacy file -> convert on read (t = ref_t or 0.3*duration)
        from highlights.multiangle.zones import normalize_zones
        durs = [float(a.get("duration") or 0.0) for a in _angles_info(p)]
        return {"version": 2,
                "angles": normalize_zones(z, durs)}
    return {"version": 2, "angles": [[] for _ in range(n)]}


@scoped.get("/multiangle/zones/learned")
def get_learned_zones(p: PublicP) -> dict:
    """Zones learned from you-direct sessions (zones_learned.json)."""
    _require_multiangle(p)
    z = _read_json(p.multiangle_dir / "zones_learned.json")
    if z is None:
        raise HTTPException(404, "no learned zones yet")
    return z


@scoped.put("/multiangle/zones")
def put_multiangle_zones(body: ZonesPut, p: ScopedP) -> dict:
    _require_multiangle(p)
    n = len(p.source_info.get("angles") or [])
    if len(body.angles) != n:
        raise HTTPException(422, f"expected {n} angle entries, got {len(body.angles)}")
    if body.ref_t is not None and len(body.ref_t) != n:
        raise HTTPException(422, f"ref_t must have {n} entries")
    durs = [float(a.get("duration") or 0.0) for a in _angles_info(p)]
    kfs = _zones_to_v2(body, n, durs)
    for ai, entry in enumerate(kfs):
        for kf in entry:
            for poly in kf["zones"]:
                if len(poly) < 3:
                    raise HTTPException(422, f"angle {ai}: polygon needs >= 3 points")
                for pt in poly:
                    if len(pt) != 2 or not all(0.0 <= float(v) <= 1.0 for v in pt):
                        raise HTTPException(422, f"angle {ai}: coords must be [x,y] in 0..1")
    out = {"version": 2, "angles": kfs}
    p.multiangle_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p.multiangle_dir / "zones.json", out, indent=1)
    _hist(p, "zones_saved")
    return out


def _active_cut_meta(p: ProjectStore) -> dict:
    """Active cut's meta.json (range/label); empty dict when absent."""
    try:
        cid = (_read_json(p.multiangle_dir / "cuts" / "active.json")
               or {}).get("id")
        if cid:
            return _read_json(
                p.multiangle_dir / "cuts" / str(cid) / "meta.json") or {}
    except Exception:
        pass
    return {}


def _direct_sessions(p: ProjectStore) -> list[dict]:
    """Saved you-direct session docs, newest first."""
    sdir = p.multiangle_dir / "manual_direct"
    sessions = []
    if sdir.is_dir():
        for f in sorted(sdir.glob("*.json"), reverse=True):
            doc = _read_json(f)
            if doc:
                sessions.append(doc)
    return sessions


def _direct_stretch(p: ProjectStore, t_start: float | None,
                    t_end: float | None) -> dict:
    """Build the /direct/suggest payload for a stretch (or the busiest
    candidate when no explicit stretch is given)."""
    from highlights.multiangle.manual_direct import director_rows, pick_candidate, suggest_stretch
    from highlights.multiangle.timemap import to_output_time_with_replays
    ma = p.multiangle_dir
    director = _read_json(ma / "director.json")
    if not director:
        raise HTTPException(404, "no director output yet")
    sync = _read_json(ma / "sync.json") or {}
    offsets = [float(o) for o in sync.get("offsets") or []]
    n_angles = len(p.source_info.get("angles") or [])
    if len(offsets) < n_angles:
        offsets += [0.0] * (n_angles - len(offsets))
    from highlights.multiangle.syncmap import shared_to_file, timemap_from_sync
    _tm = timemap_from_sync(sync or {"offsets": offsets})
    segs = director.get("segments") or []
    replays = director.get("replays") or []
    meta = _active_cut_meta(p)
    rng = meta.get("range")
    if rng:
        lo, hi = float(rng[0]), float(rng[1])
    else:
        cr = _read_json(ma / "cut_range.json")
        un = ((sync.get("coverage") or {}).get("union")
              or [0.0, 0.0])
        lo = float(cr["lo"]) if cr else float(un[0])
        hi = float(cr["hi"]) if cr else float(un[1])
        if hi <= lo and segs:
            # nothing recorded: the cut covers the director's own span
            hi = lo + max(s["t_end"] for s in segs)
    if t_start is not None and t_end is not None:
        s, e = float(t_start), float(t_end)
        if not (lo <= s < e <= hi):
            raise HTTPException(
                422, f"stretch must lie inside [{round(lo,1)}, {round(hi,1)}]")
        cand = None
    else:
        fused = (_read_json(ma / "fused_candidates.json")
                 or _read_json(p.pipeline_dir / "candidates.json") or {})
        # suggest the next busiest stretch: candidates inside a saved
        # session's span only count as fallback
        covered = _direct_sessions(p)
        def inside(c):
            t = float(c.get("t", 0))
            return any(float(s.get("t_start", -1)) <= t
                       <= float(s.get("t_end", -1)) for s in covered)
        evs = (fused.get("candidates") or fused.get("events") or [])
        fresh = [c for c in evs if not inside(c)]
        cand = (pick_candidate({"candidates": fresh}, (lo, hi))
                or pick_candidate(fused, (lo, hi)))
        if cand is None:
            raise HTTPException(404, "no candidates inside the match window")
        s, e = suggest_stretch(float(cand["t"]), (lo, hi))
    rows = director_rows(segs, lo, s, e)
    return {
        "t_start": round(s, 2), "t_end": round(e, 2),
        "t_start_out": round(
            to_output_time_with_replays(s - lo, replays), 2),
        "offsets": [round(shared_to_file(_tm, i, s, clamp=True), 2)
                    for i in range(len(_tm))],
        "match_window": [round(lo, 2), round(hi, 2)],
        "n_angles": n_angles,
        "n_sessions_saved": len(_direct_sessions(p)),
        "director": rows,
        "candidate": ({"t": round(float(cand["t"]), 2),
                       "type": cand.get("type"),
                       "confidence": cand.get("confidence")}
                      if cand else None),
    }


@scoped.get("/multiangle/direct/suggest")
def get_direct_suggest(p: ScopedP, t_start: float | None = None,
                       t_end: float | None = None) -> dict:
    """Suggest the busiest stretch to hand-direct, or build the payload
    for an arbitrary ?t_start&t_end."""
    _require_multiangle(p)
    if (t_start is None) != (t_end is None):
        raise HTTPException(422, "t_start and t_end must be given together")
    return _direct_stretch(p, t_start, t_end)


@scoped.get("/multiangle/direct/suggest3")
def get_direct_suggest3(p: ScopedP) -> dict:
    """Six busiest 60 s stretches — two per third of the match window.
    Busiest = candidates + 0.5*director switches inside the window;
    windows overlapping a saved session score 0 unless nothing else is
    left."""
    _require_multiangle(p)
    ma = p.multiangle_dir
    director = _read_json(ma / "director.json")
    if not director:
        raise HTTPException(404, "no director output yet")
    segs = director.get("segments") or []
    meta = _active_cut_meta(p)
    rng = meta.get("range")
    if rng:
        lo, hi = float(rng[0]), float(rng[1])
    else:
        sync = _read_json(ma / "sync.json") or {}
        cr = _read_json(ma / "cut_range.json")
        un = ((sync.get("coverage") or {}).get("union") or [0.0, 0.0])
        lo = float(cr["lo"]) if cr else float(un[0])
        hi = float(cr["hi"]) if cr else float(un[1])
        if hi <= lo and segs:
            hi = lo + max(s["t_end"] for s in segs)
    fused = (_read_json(ma / "fused_candidates.json")
             or _read_json(p.pipeline_dir / "candidates.json") or {})
    cand_ts = [float(c.get("t", 0)) for c in
               (fused.get("candidates") or fused.get("events") or [])]
    # director switches = segment boundaries in shared seconds
    switch_ts = [lo + float(s["t_start"]) for s in segs[1:]]
    covered = _direct_sessions(p)

    def overlaps_session(s: float, e: float) -> bool:
        return any(float(x.get("t_start", -1)) < e
                   and float(x.get("t_end", -1)) > s for x in covered)

    W, STEP, PER_THIRD = 60.0, 5.0, 2
    third = (hi - lo) / 3.0
    stretches = []
    for k in range(3):
        t0, t1 = lo + k * third, lo + (k + 1) * third
        scored = []
        s = t0
        while s + W <= t1 + 1e-6:
            e = s + W
            n_c = sum(1 for t in cand_ts if s <= t < e)
            n_sw = sum(1 for t in switch_ts if s <= t < e)
            scored.append((n_c + 0.5 * n_sw, not overlaps_session(s, e),
                           s, e))
            s += STEP
        picked: list[tuple[float, float]] = []
        for _ in range(PER_THIRD):
            free = [x for x in scored if not any(
                x[2] < pe and x[3] > ps for ps, pe in picked)]
            fresh = [x for x in free if x[1]]
            pool = fresh or free
            if not pool:
                break
            _, _, bs, be = max(pool, key=lambda x: (x[0], x[2]))
            picked.append((bs, be))
        for bs, be in sorted(picked):
            stretches.append(_direct_stretch(p, bs, be))
    if not stretches:
        raise HTTPException(404, "no stretches inside the match window")
    return {"stretches": stretches}


class DirectChoice(BaseModel):
    t: float
    angle: int


class DirectSessionPut(BaseModel):
    t_start: float
    t_end: float
    choices: list[DirectChoice]


@scoped.post("/multiangle/direct/sessions")
def post_direct_session(body: DirectSessionPut, p: ScopedP) -> dict:
    """Save a you-direct session and return the director comparison."""
    _require_multiangle(p)
    from highlights.multiangle.manual_direct import compare
    ma = p.multiangle_dir
    director = _read_json(ma / "director.json")
    if not director:
        raise HTTPException(404, "no director output yet")
    n_angles = len(p.source_info.get("angles") or [])
    if not (0 <= body.t_start < body.t_end):
        raise HTTPException(422, "t_start must be < t_end")
    if not body.choices:
        raise HTTPException(422, "choices must be non-empty")
    meta = _active_cut_meta(p)
    rng = meta.get("range")
    lo = float(rng[0]) if rng else 0.0
    choices: dict[int, int] = {}
    for c in body.choices:
        if not (0 <= c.angle < n_angles):
            raise HTTPException(422, f"angle must be 0..{n_angles - 1}")
        if not (body.t_start <= c.t <= body.t_end):
            raise HTTPException(422, "choice t outside [t_start, t_end]")
        choices[int(c.t)] = int(c.angle)
    from highlights.multiangle.manual_direct import director_rows
    rows = director_rows(director.get("segments") or [], lo,
                         body.t_start, body.t_end)
    if not rows:
        raise HTTPException(404, "stretch outside the active cut")
    cmp = compare(choices, rows)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sdir = ma / "manual_direct"
    sdir.mkdir(parents=True, exist_ok=True)
    doc = {"id": stamp, "t_start": body.t_start, "t_end": body.t_end,
           "choices": [{"t": c.t, "angle": c.angle} for c in body.choices],
           "comparison": cmp, "created_at": time.time()}
    write_json_atomic(sdir / f"{stamp}.json", doc, indent=1)
    _hist(p, "manual_direct_saved", session=stamp,
          agreement=cmp["agreement_pct"])
    return {"id": stamp, **doc, "comparison": cmp}


@scoped.get("/multiangle/direct/sessions")
def get_direct_sessions(p: ScopedP) -> dict:
    """List saved you-direct sessions with their comparisons."""
    _require_multiangle(p)
    return {"sessions": _direct_sessions(p)}


class _StatusStub:
    """StatusWriter duck-type for load_director_inputs outside a job."""
    def update(self, *a, **k):
        return None


def _direct_ctx(p: ProjectStore):
    """Minimal run.Ctx over the project's real files (learn replay)."""
    from highlights.multiangle.run import Ctx
    adir = p.root / "angles"
    dirs = sorted(d for d in adir.iterdir() if d.is_dir()) \
        if adir.is_dir() else []
    return Ctx(project_dir=p.root, pipe=p.multiangle_dir,
               status=_StatusStub(),
               angles=[{"dir": d} for d in dirs],
               style=str(p.meta.get("cut_style") or "fast"))


def _learn_agree(sessions: list[dict], segments: list[dict],
                 range_lo: float) -> float | None:
    """Agreement % of director segments vs saved sessions (same scorer
    as learn_prefs's baseline)."""
    from highlights.multiangle.manual_direct import compare, director_rows
    agree = total = 0
    for s in sessions:
        rows = director_rows(segments, range_lo,
                             s["t_start"], s["t_end"])
        choices = {int(c["t"]): int(c["angle"]) for c in s["choices"]}
        cmp = compare(choices, rows)
        for r in cmp["rows"]:
            if r["user"] is not None:
                total += 1
                agree += 1 if r["agree"] else 0
    return round(agree / total * 100, 1) if total else None


def _run_direct_learn(p: ProjectStore, sessions: list[dict]) -> dict:
    """Replay the director on this project's real inputs and
    grid-search the style overrides + prefs that best match the user."""
    from highlights.multiangle.director import cut_director, per_second, resolve_style
    from highlights.multiangle.learned_zones import learn_zones
    from highlights.multiangle.manual_direct import learn_prefs
    from highlights.multiangle.run import _load_zone_inputs, load_director_inputs
    ctx = _direct_ctx(p)
    inp = load_director_inputs(ctx)

    # zone variants: learned (from sessions) / drawn / none — pick the
    # one with the best baseline agreement, ties -> learned>drawn>none
    lz = learn_zones(sessions, inp["tracks"], inp["avail"], inp["lo"],
                     inp["timemap"], inp["durations"])
    variants: dict[str, tuple] = {"none": (None, None, None)}
    if inp["zones"]:
        variants["drawn"] = (inp["zones"], inp["zone_ok"], inp["zone_kf"])
    if lz is not None:
        lz_in = _load_zone_inputs(ctx, lz, inp["avail"], inp["T"],
                                  inp["lo"], inp["timemap"])
        if lz_in[0]:
            variants["learned"] = lz_in[:3]

    def baseline_agree(zk):
        z, zok, zkf = zk
        segs = cut_director(
            inp["tracks"], inp["avail"], inp["motion"], ctx.style,
            zones=z, zone_ok=zok, zone_kf=zkf)["segments"]
        return _learn_agree(sessions, segs, inp["lo"])

    zone_agree = {k: baseline_agree(v) for k, v in variants.items()}
    order = {"learned": 0, "drawn": 1, "none": 2}
    chosen = max(variants, key=lambda k: (
        zone_agree[k] if zone_agree[k] is not None else -1, -order[k]))
    zones, zone_ok, zone_kf = variants[chosen]
    zone_agreement = {"drawn": zone_agree.get("drawn"),
                      "learned": zone_agree.get("learned"),
                      "none": zone_agree.get("none")}

    # persist learned zones for diagnostics; active only when chosen
    if lz is not None:
        lz = {**lz, "chosen": chosen == "learned",
              "active": chosen == "learned"}
        write_json_atomic(p.multiangle_dir / "zones_learned.json",
                          lz, indent=1)

    def replay(overrides, prefs=None):
        return cut_director(
            inp["tracks"], inp["avail"], inp["motion"], ctx.style,
            zones=zones, zone_ok=zone_ok, zone_kf=zone_kf,
            style_overrides=overrides or None, prefs=prefs)["segments"]

    def cand(overrides, prefs):
        sty = resolve_style(ctx.style, overrides or None)
        a, _s, r, _S, _b, _zs, _zb = per_second(
            inp["tracks"], inp["avail"],
            zones=zones, zone_ok=zone_ok, zone_kf=zone_kf,
            linger=sty.zone_linger, strong=sty.zone_ball_strong,
            prefs=prefs)
        return a, r

    res = learn_prefs(sessions, replay, cand, range_lo=inp["lo"],
                      n_angles=len(inp["tracks"]))
    res["zone_source"] = chosen
    res["zone_agreement"] = zone_agreement
    res["learned_cells"] = lz["cells"] if lz is not None else None
    return res


class DirectLearnPut(BaseModel):
    recut: bool = False


@scoped.post("/multiangle/direct/learn")
def post_direct_learn(p: ScopedP, body: DirectLearnPut | None = None,
                      user: UserDep = "") -> dict:
    """Learn style overrides from the saved you-direct sessions and
    optionally enqueue a re-cut that uses them."""
    _require_multiangle(p)
    sessions = _direct_sessions(p)
    if not sessions:
        raise HTTPException(409, "save a you-direct session first")
    res = _run_direct_learn(p, sessions)
    ma = p.multiangle_dir
    write_json_atomic(ma / "direct_learn.json",
                      {**res, "created_at": time.time()}, indent=1)
    write_json_atomic(ma / "director_params.json",
                      {"style_overrides": res["best"],
                       "prefs": res.get("prefs")}, indent=1)
    _hist(p, "manual_direct_learned",
          agreement_before=res["agreement_pct_before"],
          agreement_after=res["agreement_pct_after"],
          overrides=res["best"], n_sessions=res["n_sessions"])
    job = None
    if body and body.recut:
        _ensure_cut_snapshot(p)
        ck = _user_default_cookies(p, user) or _project_cookies(p)
        try:
            job = pipeline.spawn_multiangle(
                p, stages=["director", "render", "fuse"], force=True,
                style=str(p.meta.get("cut_style") or "fast"), cookies=ck)
        except pipeline.PipelineBusy as e:
            raise HTTPException(409, str(e)) from e
    return {"learn": res, "job": job}


@scoped.get("/multiangle/direct/learn")
def get_direct_learn(p: ScopedP) -> dict:
    """Last saved learn result."""
    _require_multiangle(p)
    d = _read_json(p.multiangle_dir / "direct_learn.json")
    if d is None:
        raise HTTPException(404, "no learn result yet")
    return d


def _ensure_cut_snapshot(p: ProjectStore) -> None:
    """Legacy projects predate cuts/: snapshot the live cut once so it
    appears in the cuts list and stays selectable after a re-cut."""
    cuts_dir = p.multiangle_dir / "cuts"
    has_cuts = cuts_dir.is_dir() and any(d.is_dir() for d in cuts_dir.iterdir())
    if has_cuts:
        return
    with contextlib.suppress(Exception):
        snapshot_cut(p.root, p.meta.get("cut_style", "normal"))


@scoped.get("/multiangle/cuts")
def get_cuts(p: ScopedP) -> dict:
    _require_multiangle(p)
    _ensure_cut_snapshot(p)
    return list_cuts(p.root)


@scoped.post("/multiangle/cuts/{cut_id}/activate")
def activate_cut_route(cut_id: str, p: ScopedP) -> dict:
    _require_multiangle(p)
    status = pipeline.read_status(p)
    if status and status.get("state") in ("queued", "running") \
            and pipeline.pid_alive(status.get("pid")):
        raise HTTPException(409, "pipeline is running")
    meta = activate_cut(p.root, cut_id)
    if meta is None:
        raise HTTPException(404, "unknown cut")
    with contextlib.suppress(Exception):
        pipeline.sync_candidate_timeline(p)
    p.meta["cut_style"] = meta.get("style") or p.meta.get("cut_style", "normal")
    p.save()
    p.invalidate_video()   # drop proxy/thumbs for the previous cut
    # refresh the registered video + status.json so the UI sees this cut
    info = _read_json(p.pipeline_dir / "probe.json") or \
        _read_json(p.multiangle_dir / "cuts" / cut_id / "probe.json")
    try:
        info = info or fx.probe(p.root / "match.mp4")
    except Exception:
        info = info or {}
    resolved = str((p.root / "match.mp4").resolve())
    p.set_video(VideoInfo(path=resolved, registered_at=time.time(), **info))
    if status:
        status["video"] = info
        status["video_path"] = resolved
        pipeline.write_status(p, status)
    _hist(p, "cut_activated", cut_id=cut_id)
    return list_cuts(p.root)


@scoped.delete("/multiangle/cuts/{cut_id}")
def delete_cut_route(cut_id: str, p: ScopedP) -> dict:
    _require_multiangle(p)
    status = pipeline.read_status(p)
    if status and status.get("state") in ("queued", "running") \
            and pipeline.pid_alive(status.get("pid")):
        raise HTTPException(409, "pipeline is running")
    cur = list_cuts(p.root)
    if cur["active"] == cut_id:
        raise HTTPException(409, "cannot delete the active cut")
    cdir = p.multiangle_dir / "cuts" / cut_id
    if not cdir.is_dir() or "/" in cut_id or ".." in cut_id:
        raise HTTPException(404, "unknown cut")
    shutil.rmtree(cdir)
    _hist(p, "cut_deleted", cut_id=cut_id)
    return list_cuts(p.root)


@scoped.get("/multiangle/cuts/{cut_id}/match.mp4")
def get_cut_file(cut_id: str, p: PublicP) -> FileResponse:
    _require_multiangle(p)
    cdir = p.multiangle_dir / "cuts" / cut_id
    f = cdir / "match.mp4"
    if "/" in cut_id or ".." in cut_id or not f.is_file():
        raise HTTPException(404, "cut video not found")
    meta = _read_json(cdir / "meta.json") or {}
    label = meta.get("label") or cut_id
    fname = _safe_filename(f"{p.title} - {label}.mp4")
    return FileResponse(f, media_type="video/mp4", filename=fname)


@scoped.get("/multiangle/angle/{angle_idx}/frame.jpg")
def get_angle_frame(angle_idx: int, p: PublicP, t: float | None = None,
                    h: int = 180) -> FileResponse:
    _require_multiangle(p)
    angles = p.source_info.get("angles") or []
    if not (0 <= angle_idx < len(angles)):
        raise HTTPException(404, "angle out of range")
    v = p.angle_video(angle_idx)
    if v is None:
        raise HTTPException(404, "angle video not found (purged?)")
    if t is None:
        probe = _read_json(p.angle_dir(angle_idx) / "pipeline" / "probe.json") or {}
        t = float(probe.get("duration_s") or 60.0) * 0.3
    h = max(120, min(1080, int(h)))
    out = p.thumb_dir() / (f"angle{angle_idx}_{t:.0f}.jpg" if h == 180
                           else f"angle{angle_idx}_{t:.0f}_h{h}.jpg")
    if not out.is_file():
        try:
            fx.thumbnail(v, t, out, height=h)
        except Exception as e:
            raise HTTPException(500, f"frame extract failed: {e}") from e
    return FileResponse(out, media_type="image/jpeg")


@scoped.put("/multiangle/offsets")
def put_multiangle_offsets(body: OffsetsPut, p: ScopedP) -> dict:
    _require_multiangle(p)
    n = len(p.source_info.get("angles") or [])
    if len(body.offsets) != n:
        raise HTTPException(422, f"expected {n} offsets, got {len(body.offsets)}")
    if body.offsets[0] != 0:
        raise HTTPException(422, "reference angle offset must be 0")
    try:
        return pipeline.spawn_multiangle(
            p, stages=pipeline.MULTIANGLE_FROM_SYNC,
            offsets=body.offsets, cookies=_project_cookies(p))
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e


@scoped.get("/thumb.jpg")
def project_thumb(p: PublicP) -> FileResponse:
    return _project_thumb(p)


# ---------- video ----------


@scoped.post("/video")
def s_register_video(req: VideoRegisterRequest, p: ScopedP) -> VideoInfo:
    return _register_video(p, req)


@legacy.post("/video")
def l_register_video(req: VideoRegisterRequest, p: LegacyP) -> VideoInfo:
    return _register_video(p, req)


@scoped.get("/video")
def s_get_video(p: ScopedP) -> VideoInfo:
    return _get_video(p)


@legacy.get("/video")
def l_get_video(p: LegacyP) -> VideoInfo:
    return _get_video(p)


@scoped.post("/video/proxy")
def s_build_proxy(p: ScopedP) -> dict:
    return _build_proxy(p)


@legacy.post("/video/proxy")
def l_build_proxy(p: LegacyP) -> dict:
    return _build_proxy(p)


@scoped.get("/video/proxy/status")
def s_proxy_status(p: ScopedP) -> dict:
    return _proxy_status(p)


@legacy.get("/video/proxy/status")
def l_proxy_status(p: LegacyP) -> dict:
    return _proxy_status(p)


@scoped.get("/video/proxy.mp4")
def s_proxy_file(p: PublicP) -> FileResponse:
    return _proxy_file(p)


@legacy.get("/video/proxy.mp4")
def l_proxy_file(p: LegacyP) -> FileResponse:
    return _proxy_file(p)


@scoped.get("/video/source.mp4")
def s_source_file(p: PublicP) -> FileResponse:
    return _source_file(p)


@scoped.get("/match-window")
def s_get_match_window(p: ScopedP) -> dict:
    return _get_match_window(p)


@legacy.get("/match-window")
def l_get_match_window(p: LegacyP) -> dict:
    return _get_match_window(p)


@scoped.put("/match-window")
def s_put_match_window(body: MatchWindowPut, p: ScopedP) -> dict:
    return _put_match_window(p, body)


@legacy.put("/match-window")
def l_put_match_window(body: MatchWindowPut, p: LegacyP) -> dict:
    return _put_match_window(p, body)


@scoped.post("/video/trim")
def s_start_trim(body: TrimRequest, p: ScopedP) -> dict:
    return _start_trim(p, body.start_s, body.end_s)


@legacy.post("/video/trim")
def l_start_trim(body: TrimRequest, p: LegacyP) -> dict:
    return _start_trim(p, body.start_s, body.end_s)


@scoped.get("/video/trim/status")
def s_trim_status(p: ScopedP, start: float = 0.0, end: float = 0.0) -> dict:
    return _trim_status(p, start, end)


@legacy.get("/video/trim/status")
def l_trim_status(p: LegacyP, start: float = 0.0, end: float = 0.0) -> dict:
    return _trim_status(p, start, end)


@scoped.get("/video/trimmed.mp4")
def s_trimmed_file(p: PublicP, start: float = 0.0, end: float = 0.0) -> Response:
    """First request kicks off the stream-copy; 202 + progress until ready."""
    out = _trim_out(p, start, end)
    if not out.is_file():
        st = _start_trim(p, start, end)
        if not st["ready"]:
            return JSONResponse(st, status_code=202)
    if not out.is_file():
        raise HTTPException(404, "trimmed file not ready")
    return FileResponse(out, media_type="video/mp4",
                        filename="match_trimmed.mp4")


@legacy.get("/video/source.mp4")
def l_source_file(p: LegacyP) -> FileResponse:
    return _source_file(p)


# ---------- candidates ----------


@scoped.post("/candidates/load")
async def s_load_candidates(request: Request, p: ScopedP) -> list:
    return await _load_candidates(p, request)


@legacy.post("/candidates/load")
async def l_load_candidates(request: Request, p: LegacyP) -> list:
    return await _load_candidates(p, request)


@scoped.get("/candidates")
def s_list_candidates(p: ScopedP, sort: str = "confidence") -> list:
    return _list_candidates(p, sort)


@legacy.get("/candidates")
def l_list_candidates(p: LegacyP, sort: str = "confidence") -> list:
    return _list_candidates(p, sort)


@scoped.patch("/candidates/{cand_id}")
def s_patch_candidate(cand_id: str, patch: CandidatePatch, p: ScopedP) -> dict:
    return _patch_candidate(p, cand_id, patch)


@legacy.patch("/candidates/{cand_id}")
def l_patch_candidate(cand_id: str, patch: CandidatePatch, p: LegacyP) -> dict:
    return _patch_candidate(p, cand_id, patch)


@scoped.post("/candidates/{cand_id}/reset")
def s_reset_candidate(cand_id: str, p: ScopedP) -> dict:
    return _reset_candidate(p, cand_id)


@legacy.post("/candidates/{cand_id}/reset")
def l_reset_candidate(cand_id: str, p: LegacyP) -> dict:
    return _reset_candidate(p, cand_id)


@scoped.get("/candidates/{cand_id}/thumb.jpg")
def s_thumb(cand_id: str, p: PublicP, t: float | None = None) -> FileResponse:
    return _thumb(p, cand_id, t)


@legacy.get("/candidates/{cand_id}/thumb.jpg")
def l_thumb(cand_id: str, p: LegacyP, t: float | None = None) -> FileResponse:
    return _thumb(p, cand_id, t)


# ---------- project / render / stats ----------


@scoped.get("/project")
def s_project(p: ScopedP) -> dict:
    return _project_old(p)


@legacy.get("/project")
def l_project(p: LegacyP) -> dict:
    return _project_old(p)


@scoped.post("/render")
def s_start_render(req: RenderRequest, p: ScopedP) -> dict:
    return _start_render(p, req)


@legacy.post("/render")
def l_start_render(req: RenderRequest, p: LegacyP) -> dict:
    return _start_render(p, req)


@scoped.get("/render/{job_id}")
def s_render_status(job_id: str, p: ScopedP) -> dict:
    return _render_status(p, job_id)


@legacy.get("/render/{job_id}")
def l_render_status(job_id: str, p: LegacyP) -> dict:
    return _render_status(p, job_id)


@scoped.get("/renders")
def s_list_renders(p: ScopedP) -> list[dict]:
    """Newest-first list of previous render jobs with downloadable files."""
    base_dir = p.root / "renders"
    out = []
    if base_dir.is_dir():
        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            files = [
                {"name": f.relative_to(d).as_posix(),
                 "size": f.stat().st_size,
                 "url": f"/api/projects/{p.id}/files/{d.name}/"
                        f"{f.relative_to(d).as_posix()}"}
                for f in sorted(d.rglob("*")) if f.is_file()
            ]
            out.append({"job_id": d.name,
                        "created_at": d.stat().st_mtime,
                        "files": files})
    out.sort(key=lambda e: -e["created_at"])
    return out


@scoped.get("/files/{job_id}/{name:path}")
def s_render_file(job_id: str, name: str, p: PublicP) -> FileResponse:
    return _render_file(p, job_id, name)


@legacy.get("/files/{job_id}/{name:path}")
def l_render_file(job_id: str, name: str, p: LegacyP) -> FileResponse:
    return _render_file(p, job_id, name)


@legacy.get("/stats")
def l_stats(p: LegacyP) -> dict:
    return _stats(p)


# ---------- history / archive ----------


def _match_or_404(match_id: str, user: str) -> dict:
    m = history.get_match(match_id)
    if m is None or (m["owner"] != user and not _is_admin(user)):
        raise HTTPException(404, "match not found")
    return m


@app.get("/api/history")
def get_history(user: UserDep, include_deleted: bool = False) -> list:
    return history.list_matches(user, include_deleted=include_deleted)


@app.get("/api/history/downloads")
def get_downloads(user: UserDep) -> dict:
    return history.download_counts(None if _is_admin(user) else user)


@app.get("/api/history/{match_id}/events")
def get_history_events(match_id: str, user: UserDep) -> list:
    _match_or_404(match_id, user)
    return history.events(match_id)


@app.get("/api/history/{match_id}/artefacts/{name:path}")
def get_history_artefact(match_id: str, name: str) -> FileResponse:
    """Public like other media GETs — artefact names live under an
    unguessable match id, same pattern as cut/thumb downloads."""
    f = history.artefact_path(match_id, name)
    if f is None:
        raise HTTPException(404, "artefact not found")
    return FileResponse(f, filename=_safe_filename(name))


@app.post("/api/history/{match_id}/restart")
def restart_match(match_id: str, user: UserDep) -> dict:
    """Create a NEW project for the same owner from the archived record:
    same sources, title + \" (restart)\", window/zones/style as hints."""
    m = _match_or_404(match_id, user)
    rec = m.get("record") or {}
    src = rec.get("sources") or {}
    meta = dict(rec.get("meta") or {})
    if rec.get("style"):
        meta["cut_style"] = rec["style"]
    title = f"{m['title']} (restart)"[:120]
    reg = get_registry()
    if rec.get("mode") == "multiangle":
        angles = [
            {"url": a.get("url"), "filename": a.get("filename"),
             "label": a.get("label") or f"Angle {i + 1}"}
            for i, a in enumerate(src.get("angles") or [])
        ]
        if not (2 <= len(angles) <= 4):
            raise HTTPException(422, "record has no usable angle list")
        p = reg.create_project(
            owner=m["owner"], title=title,
            source={"kind": "multiangle", "url": None, "filename": None,
                    "angles": angles}, meta=meta)
        win = (rec.get("window") or {})
        p.multiangle_dir.mkdir(parents=True, exist_ok=True)
        if win.get("match_window_src"):
            write_json_atomic(
                p.multiangle_dir / "match_window_src.json",
                win["match_window_src"], indent=1)
        if win.get("cut_range"):
            write_json_atomic(
                p.multiangle_dir / "cut_range.json",
                win["cut_range"], indent=1)
        if rec.get("zones") and isinstance(rec["zones"], dict):
            write_json_atomic(
                p.multiangle_dir / "zones.json", rec["zones"], indent=1)
        try:
            out = pipeline.spawn_multiangle(
                p, cookies=_user_default_cookies(p, user))
        except pipeline.PipelineBusy as e:
            raise HTTPException(409, str(e)) from e
    else:
        kind = src.get("kind") or "path"
        if not src.get("url") and not src.get("filename"):
            raise HTTPException(422, "record has no usable source")
        p = reg.create_project(
            owner=m["owner"], title=title,
            source={"kind": kind, "url": src.get("url"),
                    "filename": src.get("filename")}, meta=meta)
        mw = (rec.get("window") or {}).get("match_window")
        if mw:
            write_json_atomic(
                _match_window_path(p), {"match_window": mw}, indent=1)
        ck = _user_default_cookies(p, user)
        try:
            if kind == "youtube":
                out = pipeline.spawn(p, youtube_url=src["url"], cookies=ck)
            else:
                video = src.get("url")
                if not video or not Path(video).is_file():
                    files = [f for f in p.source_dir.iterdir()
                             if f.is_file()]
                    video = str(files[0]) if files else None
                if not video:
                    raise HTTPException(
                        422, "source file no longer available to restart")
                out = pipeline.spawn(p, video=video,
                                     stages=NO_DOWNLOAD_STAGES)
        except pipeline.PipelineBusy as e:
            raise HTTPException(409, str(e)) from e
    history.log(match_id, "restarted", new_project=p.id)
    _hist(p, "created", restarted_from=match_id)
    _hist(p, "source_added", src=(rec.get("mode") or src.get("kind")),
          n_angles=len(src.get("angles") or []))
    return {**summary(p), "pipeline": out}


@app.delete("/api/history/{match_id}")
def delete_history(match_id: str, user: UserDep) -> Response:
    """Really delete: drop the archived artefacts + match row + events."""
    _match_or_404(match_id, user)
    history.remove_match(match_id)
    return Response(status_code=204)


app.include_router(scoped)
app.include_router(legacy)

from .analysis_api import make_router as _analysis_router  # noqa: E402
from .players_api import make_router as _players_router  # noqa: E402

app.include_router(_analysis_router(ScopedP))
app.include_router(_players_router(ScopedP, PublicP))


# ---------- frontend ----------

# static assets get a real mount; the catch-all below is the SPA fallback so
# client-side routes like /projects/<id> serve index.html
if DIST.is_dir() and (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(404, "not found")
    if not DIST.is_dir():
        raise HTTPException(404, "not found")
    fp = (DIST / full_path).resolve()
    if full_path and fp.is_file() and DIST.resolve() in fp.parents:
        return FileResponse(fp)
    index = DIST / "index.html"
    if not index.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(index)
