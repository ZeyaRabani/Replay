"""FastAPI backend for the Replay Highlights review/render shell.

Run from repo root:  uvicorn highlights.app.backend.main:app --port 8000
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import ffmpeg as fx
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
from .store import STORE

APP_DIR = Path(__file__).resolve().parents[1]
DIST = APP_DIR / "frontend" / "dist"

app = FastAPI(title="Replay Highlights")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, RenderJob] = {}
_proxy_job: fx.ProxyJob | None = None
_jobs_lock = threading.Lock()


def _wd() -> Path:
    return STORE.root


def _video_path() -> Path:
    if STORE.video is None:
        raise HTTPException(404, "no video registered")
    return Path(STORE.video.path)


def _stats() -> dict:
    cands = STORE.candidates
    counts: dict[str, dict] = {}
    for c in cands:
        e = counts.setdefault(c.type, {"total": 0, "confirmed": 0, "rejected": 0, "pending": 0})
        e["total"] += 1
        e[c.status] += 1
    xv: dict[str, int] = {}
    for c in cands:
        xv[c.cross_validation] = xv.get(c.cross_validation, 0) + 1
    return {
        "source": STORE.source,
        "video_duration_s": STORE.video.duration_s if STORE.video else 0.0,
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


# ---------- video ----------


@app.post("/api/video")
def register_video(req: VideoRegisterRequest) -> VideoInfo:
    p = Path(req.path)
    if not p.is_file():
        raise HTTPException(404, f"file not found: {p}")
    info = fx.probe(p)
    vi = VideoInfo(path=str(p.resolve()), proxy_ready=(_wd() / "proxy.mp4").exists(), **info)
    STORE.set_video(vi)
    return vi


@app.get("/api/video")
def get_video() -> VideoInfo:
    if STORE.video is None:
        raise HTTPException(404, "no video registered")
    STORE.video.proxy_ready = (_wd() / "proxy.mp4").exists()
    return STORE.video


def _serve(path: Path) -> FileResponse:
    # Starlette FileResponse handles Range requests (status 206).
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/video/proxy")
def build_proxy() -> dict:
    global _proxy_job
    src = _video_path()
    dst = _wd() / "proxy.mp4"
    if dst.exists():
        return {"status": "ready"}
    _proxy_job = fx.ProxyJob(src, dst, STORE.video.duration_s)
    _proxy_job.start()
    return {"status": "started"}


@app.get("/api/video/proxy/status")
def proxy_status() -> dict:
    if (_wd() / "proxy.mp4").exists():
        return {"ready": True, "progress": 1.0}
    if _proxy_job is None:
        return {"ready": False, "progress": 0.0}
    return {"ready": _proxy_job.done, "progress": _proxy_job.progress, "error": _proxy_job.error}


@app.get("/api/video/proxy.mp4")
def proxy_file() -> FileResponse:
    return _serve(_wd() / "proxy.mp4")


@app.get("/api/video/source.mp4")
def source_file() -> FileResponse:
    return _serve(_video_path())


# ---------- candidates ----------


@app.post("/api/candidates/load")
async def load_candidates(request: Request) -> list:
    """Accepts multipart file upload OR JSON body {"path": "/abs/candidates.json"}."""
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" in ct:
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(400, "missing 'file' field")
        cf = CandidatesFile(**json.loads(await file.read()))
    elif "application/json" in ct:
        body = CandidatesLoadRequest(**(await request.json()))
        p = Path(body.path)
        if not p.is_file():
            raise HTTPException(404, f"file not found: {p}")
        cf = CandidatesFile(**json.loads(p.read_text()))
    else:
        raise HTTPException(400, "expected multipart file or JSON body with 'path'")
    return [c.model_dump() for c in STORE.load_candidates(cf)]


@app.get("/api/candidates")
def list_candidates(sort: str = "confidence") -> list:
    cands = STORE.candidates
    cands = sorted(cands, key=lambda c: c.t) if sort == "time" else sorted(cands, key=lambda c: -c.confidence)
    return [c.model_dump() for c in cands]


@app.patch("/api/candidates/{cand_id}")
def patch_candidate(cand_id: str, patch: CandidatePatch) -> dict:
    c = STORE.get(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    data = patch.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(c, k, v)
    duration = STORE.video.duration_s if STORE.video else 0.0
    if c.clip_start >= c.clip_end:
        raise HTTPException(422, "clip_start must be < clip_end")
    if c.clip_start < 0 or (duration and c.clip_end > duration):
        raise HTTPException(422, "clip window outside video duration")
    STORE.update(c)
    return c.model_dump()


@app.post("/api/candidates/{cand_id}/reset")
def reset_candidate(cand_id: str) -> dict:
    c = STORE.reset_candidate(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    return c.model_dump()


@app.get("/api/candidates/{cand_id}/thumb.jpg")
def thumb(cand_id: str, t: float | None = None) -> FileResponse:
    c = STORE.get(cand_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    tval = c.t if t is None else t
    out = _wd() / "thumbs" / f"{cand_id}_{tval:.1f}.jpg"
    if not out.exists():
        try:
            fx.thumbnail(_video_path(), tval, out)
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e
    return FileResponse(out, media_type="image/jpeg")


# ---------- render ----------


def _run_render(job_id: str, req: RenderRequest, items: list[dict]) -> None:
    job = _jobs[job_id]
    job.state = "running"
    out_dir = _wd() / "renders" / job_id
    try:
        def cb(frac: float, msg: str) -> None:
            job.progress, job.message = frac, msg

        res = fx.render_reel(_video_path(), items, out_dir, overlay=req.overlay, reencode=req.reencode, progress_cb=cb)
        job.clips = [
            ClipResult(id=r["id"], path=r["path"], url=f"/api/files/{job_id}/clips/{Path(r['path']).name}", duration=r["duration"])
            for r in res["clips"]
        ]
        stats = _stats()
        stats["rendered"] = {
            "n_clips": len(res["clips"]),
            "total_clip_s": sum(r["duration"] for r in res["clips"]),
            "reel_s": res["reel_s"],
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        manifest = [
            {
                "id": it["id"], "t": it["t"], "clip_start": it["clip_start"], "clip_end": it["clip_end"],
                "path": r["path"], "duration": r["duration"],
            }
            for it, r in zip(items, res["clips"], strict=True)
        ]
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        job.reel_url = f"/api/files/{job_id}/reel.mp4"
        job.stats_url = f"/api/files/{job_id}/stats.json"
        job.state = "done"
        job.progress = 1.0
        job.message = "done"
    except Exception as e:
        job.state = "error"
        job.error = str(e)
        job.message = "error"


@app.post("/api/render")
def start_render(req: RenderRequest) -> dict:
    _video_path()
    if req.ids:
        cands = [c for c in (STORE.get(i) for i in req.ids) if c is not None]
    else:
        cands = [c for c in STORE.candidates if c.status == "confirmed"]
        if not cands:
            cands = [c for c in STORE.candidates if c.status != "rejected"]
    if not cands:
        raise HTTPException(400, "no candidates selected for render")
    items = [
        {
            "id": c.id, "t": c.t, "type": c.type,
            "clip_start": c.clip_start, "clip_end": c.clip_end,
            "name": f"{c.rank:02d}_{c.type}_{c.t:07.1f}.mp4",
        }
        for c in cands
    ]
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = RenderJob(job_id=job_id)
    threading.Thread(target=_run_render, args=(job_id, req, items), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/render/{job_id}")
def render_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.model_dump()


@app.get("/api/files/{job_id}/{name:path}")
def render_file(job_id: str, name: str) -> FileResponse:
    p = (_wd() / "renders" / job_id / name).resolve()
    root = (_wd() / "renders" / job_id).resolve()
    if root not in p.parents and p != root:
        raise HTTPException(403, "forbidden")
    if not p.is_file():
        raise HTTPException(404, "file not found")
    media = "video/mp4" if p.suffix == ".mp4" else "application/json"
    return FileResponse(p, media_type=media, filename=p.name, content_disposition_type="attachment")


@app.get("/api/stats")
def stats() -> dict:
    return _stats()


# ---------- frontend ----------

if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
