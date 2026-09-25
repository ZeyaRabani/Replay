"""FastAPI backend for the Replay Highlights review/render shell.

Multi-user (X-User header), multi-project, with a detached pipeline
subprocess per project. Run from repo root:

    uvicorn highlights.app.backend.main:app --port 8000
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ffmpeg as fx
from . import pipeline, stats
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
_jobs_lock = threading.Lock()

_registry: Registry | None = None


PITCH_TYPES = ("11", "9", "7", "5", "other")
CAMERA_TYPES = ("normal", "ultrawide", "zoom", "other")


def _meta_or_422(pitch_type: str | None, camera: str | None) -> dict:
    meta = {}
    if pitch_type is not None:
        if pitch_type not in PITCH_TYPES:
            raise HTTPException(422, f"pitch_type must be one of {PITCH_TYPES}")
        meta["pitch_type"] = pitch_type
    if camera is not None:
        if camera not in CAMERA_TYPES:
            raise HTTPException(422, f"camera must be one of {CAMERA_TYPES}")
        meta["camera"] = camera
    return meta


class ProjectCreate(BaseModel):
    title: str | None = None
    youtube_url: str | None = None
    path: str | None = None
    run_pipeline: bool = True
    cookies_text: str | None = None
    pitch_type: str | None = None
    camera: str | None = None


class UserCreate(BaseModel):
    name: str


class AngleSpec(BaseModel):
    url: str | None = None
    filename: str | None = None
    label: str = ""


class MultiangleCreate(BaseModel):
    title: str | None = None
    angles: list[AngleSpec]
    cookies_text: str | None = None
    pitch_type: str | None = None
    camera: str | None = None


class OffsetsPut(BaseModel):
    offsets: list[float]


def get_registry() -> Registry:
    global _registry
    wd = workdir()
    if _registry is None or _registry.root != wd:
        reg = Registry(wd)
        ensure_demo(reg)
        for p in reg.list_projects():
            pipeline.reconcile(p)
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
    reg.add_user("demo")
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
        owner="demo",
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
    p = get_registry().newest_for("demo")
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
    cands = sorted(cands, key=lambda c: c.t) if sort == "time" else sorted(cands, key=lambda c: -c.confidence)
    return [c.model_dump() for c in cands]


def _patch_candidate(p: ProjectStore, cand_id: str, patch: CandidatePatch) -> dict:
    c = p.get(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    data = patch.model_dump(exclude_none=True)
    team = data.pop("team", None)
    updated = c.model_copy(update=data)
    duration = p.video.duration_s if p.video else 0.0
    err = fx.validate_clip_window(updated.clip_start, updated.clip_end, duration)
    if err:
        raise HTTPException(422, err)
    for k, v in data.items():
        setattr(c, k, v)
    if team is not None:
        c.signals = {**c.signals, "team": team}
    p.update(c)
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
    return {"job_id": job_id}


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
        return st
    if p.owner == "demo" and p.title.startswith("Demo match"):
        st = stats.demo_stats()
        with contextlib.suppress(OSError):
            sp.write_text(json.dumps(st, indent=2))
        return st
    raise HTTPException(404, "stats not available yet (pipeline has not produced stats.json)")


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
    """Path to an existing source/cookies.txt for reruns, else None."""
    path = p.source_dir / "cookies.txt"
    return str(path) if path.is_file() else None


# ---------- per-user saved YouTube cookies ----------

def _user_cookies_path(user: str) -> Path:
    return workdir() / "users" / user / "youtube_cookies.txt"


def _save_user_cookies(user: str, text: str) -> Path:
    """Persist the user's YouTube cookies (0600). Never logged or echoed."""
    path = _user_cookies_path(user)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, 0o600)
    return path


def _user_default_cookies(p, user: str) -> str | None:
    """Copy the user's saved cookies into the project, returning the path."""
    src = _user_cookies_path(user)
    if not src.is_file():
        return None
    dst = p.source_dir / "cookies.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    os.chmod(dst, 0o600)
    return str(dst)


class CookiesPut(BaseModel):
    cookies_text: str


@app.get("/api/me/youtube-cookies")
def get_youtube_cookies(user: UserDep) -> dict:
    path = _user_cookies_path(user)
    return {"saved": path.is_file(),
            "updated_at": path.stat().st_mtime if path.is_file() else None}


@app.put("/api/me/youtube-cookies")
def put_youtube_cookies(body: CookiesPut, user: UserDep) -> dict:
    text = body.cookies_text or ""
    if not text.strip():
        raise HTTPException(422, "cookies_text is empty")
    if "youtube.com" not in text:
        raise HTTPException(422, "does not look like Netscape cookies for youtube.com")
    path = _save_user_cookies(user, text)
    return {"saved": True, "updated_at": path.stat().st_mtime}


@app.delete("/api/me/youtube-cookies")
def delete_youtube_cookies(user: UserDep) -> Response:
    _user_cookies_path(user).unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/config")
def get_config() -> dict:
    return {"upload_origin": os.environ.get("HL_PUBLIC_URL") or None}


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
            form.get("camera") or None)
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
            meta=_meta_or_422(body.pitch_type, body.camera),
        )
        if body.cookies_text:
            cookies = _save_cookies(p, body.cookies_text)
            _save_user_cookies(user, body.cookies_text)
        else:
            cookies = _user_default_cookies(p, user)
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
            meta=_meta_or_422(body.pitch_type, body.camera),
        )
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
        out.append({
            "url": a.url.strip() if a.url else None,
            "filename": a.filename.strip() if a.filename else None,
            "label": a.label.strip() or f"Angle {i + 1}",
        })
    return out


@app.post("/api/projects/multiangle")
def create_multiangle(body: MultiangleCreate, user: UserDep) -> dict:
    angles = _norm_angles(body.angles)
    for i, a in enumerate(angles):
        if not a["url"]:
            raise HTTPException(422, f"angle {i} has no url")
    p = get_registry().create_project(
        owner=user,
        title=(body.title or "").strip() or angles[0]["url"] or "multi-angle",
        source={"kind": "multiangle", "url": None, "filename": None, "angles": angles},
        meta=_meta_or_422(body.pitch_type, body.camera),
    )
    if body.cookies_text:
        cookies = _save_cookies(p, body.cookies_text)
        _save_user_cookies(user, body.cookies_text)
    else:
        cookies = _user_default_cookies(p, user)
    try:
        pipeline.spawn_multiangle(p, cookies=cookies)
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e
    return summary(p)


@app.post("/api/projects/multiangle/upload")
async def create_multiangle_upload(request: Request, user: UserDep) -> dict:
    form = await request.form()
    files = [f for f in form.getlist("files") if hasattr(f, "read")]
    labels = [x for x in form.getlist("labels") if isinstance(x, str)]
    title = form.get("title") if isinstance(form.get("title"), str) else ""
    cookies_text = form.get("cookies_text") if isinstance(form.get("cookies_text"), str) else None
    if not (2 <= len(files) <= 4):
        raise HTTPException(422, "multi-angle uploads need 2..4 files")
    meta = _meta_or_422(
        form.get("pitch_type") or None,
        form.get("camera") or None)
    names = [_safe_filename(f.filename or f"angle{i}.mp4") for i, f in enumerate(files)]
    angles = [
        {
            "url": None,
            "filename": name,
            "label": (labels[i].strip() if i < len(labels) else "") or f"Angle {i + 1}",
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
    score: dict = {}
    st = _read_json(p.pipeline_dir / "stats.json") or {}
    score = (st.get("multiangle") or {}).get("score") or {}
    if not score:
        score = _read_json(p.multiangle_dir / "score.json") or {}
    home_label = (score.get("home") or {}).get("label") or "Home"
    away_label = (score.get("away") or {}).get("label") or "Away"
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
        "home": {"label": home_label, "goals": n_home},
        "away": {"label": away_label, "goals": n_away},
        "unassigned": n_un,
        "basis": "confirmed goals with team set",
    }


def _angles_info(p: ProjectStore) -> list[dict]:
    out = []
    for i, a in enumerate(p.source_info.get("angles") or []):
        ast = _read_json(p.angle_dir(i) / "pipeline" / "status.json") or {}
        duration = (ast.get("video") or {}).get("duration_s")
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
    get_registry().delete(p.id)
    return Response(status_code=204)


@scoped.post("/pipeline/run")
def run_pipeline(p: ScopedP, user: UserDep,
                 body: Annotated[dict | None, Body()] = None) -> dict:
    body = body or {}
    stages = body.get("stages")
    force = bool(body.get("force", False))
    try:
        if p.is_multiangle:
            ck = _user_default_cookies(p, user) or _project_cookies(p)
            return pipeline.spawn_multiangle(p, stages=stages, force=force,
                                           cookies=ck)
        kind = p.source_info.get("kind")
        if kind == "youtube":
            ck = _user_default_cookies(p, user) or _project_cookies(p)
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
    except pipeline.PipelineBusy as e:
        raise HTTPException(409, str(e)) from e


@scoped.post("/pipeline/cancel")
def cancel_pipeline(p: ScopedP) -> dict:
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
        director = {k: v for k, v in director.items() if k != "segments"}
    return {
        "sync": _read_json(p.multiangle_dir / "sync.json"),
        "director": director,
        "angles": _angles_info(p),
        "score": _multiangle_score(p),
        "status": pipeline.read_status(p),
    }


@scoped.get("/multiangle/director")
def get_multiangle_director(p: ScopedP) -> dict:
    _require_multiangle(p)
    d = _read_json(p.multiangle_dir / "director.json")
    if d is None:
        raise HTTPException(404, "director.json not available yet")
    return d


@scoped.get("/multiangle/angle/{angle_idx}/video")
def get_angle_video(angle_idx: int, p: PublicP) -> FileResponse:
    _require_multiangle(p)
    angles = p.source_info.get("angles") or []
    if not (0 <= angle_idx < len(angles)):
        raise HTTPException(404, "angle out of range")
    v = p.angle_video(angle_idx)
    if v is None:
        raise HTTPException(404, "angle video not found")
    return _serve(v)


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


@scoped.get("/files/{job_id}/{name:path}")
def s_render_file(job_id: str, name: str, p: PublicP) -> FileResponse:
    return _render_file(p, job_id, name)


@legacy.get("/files/{job_id}/{name:path}")
def l_render_file(job_id: str, name: str, p: LegacyP) -> FileResponse:
    return _render_file(p, job_id, name)


@legacy.get("/stats")
def l_stats(p: LegacyP) -> dict:
    return _stats(p)


app.include_router(scoped)
app.include_router(legacy)


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
