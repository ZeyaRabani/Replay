"""YouTube download stage via the yt-dlp Python API."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yt_dlp

from .errors import PipelineError

BOT_CHECK_MARKERS = ("Sign in to confirm", "not a bot")

BOT_CHECK_MSG = (
    "YouTube blocked the automated download (sign-in / bot check). "
    "Upload the video file instead, or provide a cookies file "
    "(--cookies / HL_YT_COOKIES)."
)


def _progress_hook(status, d: dict) -> None:
    if status is None:
        return
    st = d.get("status")
    if st == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        done = d.get("downloaded_bytes") or 0
        frac = min(1.0, done / total) if total else 0.0
        height = (d.get("info_dict") or {}).get("height")
        res = f"{height}p" if height else "?"
        msg = f"Downloading {res} ({frac:.0%})"
        status.update(stage_progress=frac, message=msg)
    elif st == "finished":
        status.update(stage_progress=1.0, message="download finished, merging")


def download(url: str, dest_dir: str | Path, status=None,
             cookies: str | None = None, log=print) -> Path:
    """Download `url` as dest_dir/match.mp4 (or .mkv if mp4 merge impossible).

    Returns the resolved output path. Raises PipelineError on failure; the
    YouTube bot-check case gets an actionable message.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cookiefile = cookies or os.environ.get("HL_YT_COOKIES")

    opts: dict = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "outtmpl": str(dest_dir / "match.%(ext)s"),
        "noplaylist": True,
        "progress_hooks": [lambda d: _progress_hook(status, d)],
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 20,
        "concurrent_fragment_downloads": 4,
    }
    js = {}
    if shutil.which("deno"):
        js["deno"] = {}
    if shutil.which("node"):
        js["node"] = {}
    if js:
        opts["js_runtimes"] = js
        log(f"js runtimes: {','.join(js)}")
    else:
        log("warning: no JS runtime on PATH; YouTube JS challenges may fail")
    if cookiefile:
        opts["cookiefile"] = cookiefile

    # Real-world finding: YouTube DASH formats (271/251) can 403 mid-fetch
    # while the HLS variants download fine. Retry once on 403 with HLS.
    formats = [
        opts["format"],
        "bv*[protocol^=m3u8]+ba[protocol^=m3u8]/"
        "bv*[protocol^=m3u8]+ba/bv*+ba/b",
    ]
    info = None
    for attempt, fmt in enumerate(formats):
        opts["format"] = fmt
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if any(m in msg for m in BOT_CHECK_MARKERS):
                raise PipelineError(BOT_CHECK_MSG) from e
            if attempt == 0 and ("403" in msg or "Forbidden" in msg):
                log("DASH download got 403; retrying with HLS streams")
                continue
            raise PipelineError(msg) from e

    # resolve the actual output file
    out: Path | None = None
    try:
        req = (info or {}).get("requested_downloads") or []
        if req and req[0].get("filepath"):
            p = Path(req[0]["filepath"])
            if p.exists():
                out = p
    except (AttributeError, IndexError, KeyError):
        pass
    if out is None:
        cands = sorted(dest_dir.glob("match.*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        cands = [p for p in cands if p.suffix != ".part"]
        if cands:
            out = cands[0]
    if out is None or not out.exists():
        raise PipelineError(f"download finished but no match.* file in {dest_dir}")
    if out.suffix == ".mkv":
        log("warning: could not merge to mp4; keeping match.mkv")

    if status is not None:
        w, h = (info or {}).get("width"), (info or {}).get("height")
        status.update(
            download={
                "format": (info or {}).get("format"),
                "resolution": f"{w}x{h}" if w and h else None,
                "filesize": out.stat().st_size,
            },
            video_path=str(out),
            force=True,
        )
    log(f"downloaded {url} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
