"""Cloudflare R2 fast downloads.

match.mp4 objects are pushed to the `replay-videos` bucket under
``{pid}/match-{size}-{mtime}.mp4`` (the scheme r2up.py uses) and served
via 6 h presigned GETs. The bucket is hard-capped at ``cap_bytes``:
auto uploads may only evict objects older than 24 h and stay deferred
when nothing may go; the deferred queue is retried hourly. A single
worker thread uploads one object at a time.

Everything is a no-op when ``REPLAY_R2_CONFIG`` (default
/data/shared/r2.json) is missing or invalid, or boto3 isn't installed.
The file's contents are never logged.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

CONFIG_PATH = os.environ.get("REPLAY_R2_CONFIG", "/data/shared/r2.json")
DEFER_RETRY_S = 3600
AUTO_MIN_AGE_S = 24 * 3600
PRESIGN_TTL_S = 6 * 3600
HEAD_CACHE_S = 60
MPU_GRACE_S = 3600
_CHUNK = 64 * 1024 * 1024

_cfg: dict | None = None
_cfg_tried = False
_client = None
_q: list[dict] = []          # pending jobs: {pid, path, auto, defer_until}
_q_cv = threading.Condition()
_state: dict[str, dict] = {} # pid -> last known status dict
_head_cache: dict[str, tuple[float, dict]] = {}  # key -> (ts, meta)
_worker_started = False
_lock = threading.Lock()


def _load_cfg() -> dict | None:
    global _cfg, _cfg_tried
    if _cfg_tried:
        return _cfg
    _cfg_tried = True
    try:
        cfg = json.loads(Path(CONFIG_PATH).read_text())
        need = ("endpoint", "access_key_id", "secret_access_key",
                "bucket", "cap_bytes")
        if all(cfg.get(k) for k in need):
            _cfg = cfg
    except Exception:
        pass
    return _cfg


def enabled() -> bool:
    """True when the config exists and boto3 is importable."""
    if _load_cfg() is None:
        return False
    try:
        import boto3  # noqa: F401
        return True
    except Exception:
        return False


def _mk_client():
    """Build the boto3 S3 client. Separated so tests can stub it."""
    import boto3
    cfg = _load_cfg()
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
    )


def _s3():
    global _client
    if _client is None:
        _client = _mk_client()
    return _client


def _bucket() -> str:
    return _load_cfg()["bucket"]


def _cap() -> int:
    return int(_load_cfg()["cap_bytes"])


def object_key(pid: str, path: Path) -> str:
    st = Path(path).stat()
    return f"{pid}/match-{st.st_size}-{int(st.st_mtime)}.mp4"


# ---------- status ----------

def _head(key: str) -> dict | None:
    now = time.time()
    hit = _head_cache.get(key)
    if hit and now - hit[0] < HEAD_CACHE_S:
        return hit[1]
    try:
        meta = _s3().head_object(Bucket=_bucket(), Key=key)
    except Exception:
        meta = None
    _head_cache[key] = (now, meta)
    return meta


def status(pid: str, path: Path) -> dict:
    """{state, progress, detail?}: off|none|queued|uploading|ready|
    too_big|error."""
    if not enabled():
        return {"state": "off", "progress": 0.0}
    cur = _state.get(pid)
    if cur and cur.get("state") in ("queued", "uploading", "too_big"):
        return cur
    try:
        key = object_key(pid, path)
    except OSError:
        return {"state": "none", "progress": 0.0}
    if _head(key):
        return {"state": "ready", "progress": 1.0}
    if cur:
        return cur
    return {"state": "none", "progress": 0.0}


def presign(key: str, filename: str) -> str:
    """GET URL, 6 h, Content-Disposition attachment (RFC 5987)."""
    asc = "".join(c if 32 <= ord(c) < 127 and c not in '"\\'
                  else "_" for c in filename) or "match.mp4"
    cd = (f"attachment; filename=\"{asc}\"; "
          f"filename*=UTF-8''{quote(filename)}")
    return _s3().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": _bucket(),
            "Key": key,
            "ResponseContentDisposition": cd,
            "ResponseContentType": "video/mp4",
        },
        ExpiresIn=PRESIGN_TTL_S,
    )


# ---------- bucket helpers ----------

def _list_objects() -> list[dict]:
    out: list[dict] = []
    tok = None
    while True:
        kw: dict[str, Any] = {"Bucket": _bucket()}
        if tok:
            kw["ContinuationToken"] = tok
        r = _s3().list_objects_v2(**kw)
        out += r.get("Contents", [])
        if not r.get("IsTruncated"):
            return out
        tok = r.get("NextContinuationToken")


def _abort_stale_mpus() -> None:
    """Abort every multipart upload older than MPU_GRACE_S — incomplete
    parts still count toward storage."""
    now = time.time()
    try:
        tok = None
        while True:
            kw: dict[str, Any] = {"Bucket": _bucket()}
            if tok:
                kw["UploadIdMarker"] = tok[0]
                kw["KeyMarker"] = tok[1]
            r = _s3().list_multipart_uploads(**kw)
            for u in r.get("Uploads", []):
                initiated = u["Initiated"].timestamp()
                if now - initiated > MPU_GRACE_S:
                    with contextlib.suppress(Exception):
                        _s3().abort_multipart_upload(
                            Bucket=_bucket(), Key=u["Key"],
                            UploadId=u["UploadId"])
            if not r.get("IsTruncated"):
                return
            tok = (r.get("NextUploadIdMarker"), r.get("NextKeyMarker"))
    except Exception:
        pass


def _evict_for(size: int, keep_key: str, auto: bool) -> bool:
    """Free space for `size` bytes. Returns False when the object
    cannot fit under the cap given the eviction rules."""
    if size > _cap():
        return False
    objs = _list_objects()
    used = sum(o["Size"] for o in objs)
    if used + size <= _cap():
        return True
    # evict oldest first; never touch keep_key; auto may not evict
    # objects younger than 24 h
    now = time.time()
    cands = sorted(
        (o for o in objs if o["Key"] != keep_key
         and (not auto or now - o["LastModified"].timestamp() > AUTO_MIN_AGE_S)),
        key=lambda o: o["LastModified"])
    for o in cands:
        if used + size <= _cap():
            break
        try:
            _s3().delete_object(Bucket=_bucket(), Key=o["Key"])
            used -= o["Size"]
        except Exception:
            pass
    return used + size <= _cap()


# ---------- worker ----------

def _run_upload(job: dict) -> None:
    pid, path, auto = job["pid"], Path(job["path"]), job["auto"]
    key = object_key(pid, path)
    try:
        _abort_stale_mpus()
        # remove other stale keys under {pid}/ so one video lives per pid
        for o in _list_objects():
            if o["Key"].startswith(f"{pid}/") and o["Key"] != key:
                with contextlib.suppress(Exception):
                    _s3().delete_object(Bucket=_bucket(), Key=o["Key"])
        if not _evict_for(path.stat().st_size, key, auto):
            _state[pid] = {"state": "too_big", "progress": 0.0,
                           "key": key, "auto": auto}
            if auto:
                # stays deferred: retry hourly
                job["defer_until"] = time.time() + DEFER_RETRY_S
                with _q_cv:
                    _q.append(job)
                    _q_cv.notify()
            return
        from boto3.s3.transfer import TransferConfig
        st_size = path.stat().st_size
        _state[pid] = {"state": "uploading", "progress": 0.0, "key": key}

        def cb(n: int, s=st_size) -> None:
            _state[pid]["_sent"] = _state[pid].get("_sent", 0) + n
            _state[pid]["progress"] = min(1.0, _state[pid]["_sent"] / s)

        _s3().upload_file(
            str(path), _bucket(), key, Callback=cb,
            Config=TransferConfig(multipart_chunksize=_CHUNK,
                                  max_concurrency=4))
        _head_cache.pop(key, None)
        _state[pid] = {"state": "ready", "progress": 1.0, "key": key}
    except Exception as e:
        _state[pid] = {"state": "error", "progress": 0.0,
                       "key": key, "detail": str(e)[:300]}


def _worker() -> None:
    while True:
        with _q_cv:
            while True:
                now = time.time()
                idx = next((i for i, j in enumerate(_q)
                            if j.get("defer_until", 0) <= now), None)
                if idx is not None:
                    job = _q.pop(idx)
                    break
                waits = [j["defer_until"] - now for j in _q]
                _q_cv.wait(timeout=min(waits) if waits else 3600)
        _run_upload(job)


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
        # abort stale multipart uploads once at start too
        with contextlib.suppress(Exception):
            _abort_stale_mpus()
        threading.Thread(target=_worker, daemon=True,
                         name="r2-upload").start()


def enqueue(pid: str, path: Path, auto: bool = False) -> dict:
    """Queue pid's video for upload. Returns the current status."""
    if not enabled():
        return {"state": "off", "progress": 0.0}
    cur = status(pid, path)
    if cur.get("state") in ("ready", "uploading"):
        return cur
    with _q_cv:
        if not any(j["pid"] == pid for j in _q):
            _q.append({"pid": pid, "path": str(path), "auto": auto,
                       "defer_until": 0})
            _state[pid] = {"state": "queued", "progress": 0.0}
            _q_cv.notify()
    _ensure_worker()
    return _state.get(pid, {"state": "queued", "progress": 0.0})


# ---------- test seam ----------

def _reset() -> None:
    """Test-only: drop all state."""
    global _cfg, _cfg_tried, _client, _worker_started
    _cfg = None
    _cfg_tried = False
    _client = None
    _worker_started = False
    with _q_cv:
        _q.clear()
    _state.clear()
    _head_cache.clear()
