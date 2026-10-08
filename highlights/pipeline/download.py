"""YouTube download stage via the yt-dlp Python API."""

from __future__ import annotations

import os
import re
import shutil
import time
import urllib.parse
from pathlib import Path

import yt_dlp

from .errors import PipelineError

BOT_CHECK_MARKERS = ("Sign in to confirm", "not a bot")

BOT_CHECK_MSG = (
    "YouTube blocked the automated download (sign-in / bot check). "
    "Check the bgutil-provider service is running (HL_POT_PROVIDER_URL), "
    "or upload the video file instead / provide a cookies file "
    "(--cookies / HL_YT_COOKIES), or set a residential proxy "
    "(HL_YT_PROXY / the YouTube access panel)."
)

COOKIES_REJECTED_MSG = (
    "YouTube rejected the saved cookies (expired/rotated). Re-export them "
    "from an incognito window (log in to YouTube there, export with the "
    "Get cookies.txt extension, close the window without browsing) and "
    "paste into the YouTube access panel, then Restart."
)

# transient YouTube/network failures worth an outer (long) retry
TRANSIENT_RE = re.compile(
    r"5\d\d|Service Unavailable|timed out|Connection reset|"
    r"Temporary failure|Name or service not known|Network is unreachable|"
    r"Remote end closed|EOF occurred|incomplete read",
    re.IGNORECASE,
)
TRANSIENT_BACKOFF_S = [30, 60, 120, 300, 600, 900, 1200, 1800]
TRANSIENT_MAX_ATTEMPTS = len(TRANSIENT_BACKOFF_S)

# injectable for tests
_sleep = time.sleep


def _is_bot_check(msg: str) -> bool:
    return any(m in msg for m in BOT_CHECK_MARKERS)


def is_cookie_failure(msg: str) -> bool:
    """True for saved-cookies-rejected or bot-check failures — retried only
    when the cookies change, never on a timer."""
    return (msg.startswith(COOKIES_REJECTED_MSG[:40])
            or msg.startswith(BOT_CHECK_MSG[:40]))


def is_transient_failure(msg: str) -> bool:
    """True for YouTube 5xx / network errors that clear on their own."""
    return bool(TRANSIENT_RE.search(msg))


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


def _extract(opts: dict, formats: list[str], url: str, log) -> dict:
    """One download attempt over the DASH->HLS format fallback list.

    Returns the yt-dlp info dict. Raises DownloadError unchanged; the
    caller decides whether the failure is bot-check / transient / fatal.
    """
    for attempt, fmt in enumerate(formats):
        opts["format"] = fmt
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if (attempt == 0 and not _is_bot_check(msg)
                    and ("403" in msg or "Forbidden" in msg)):
                log("DASH download got 403; retrying with HLS streams")
                continue
            raise
    return {}


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
        "retry_sleep_functions": {
            "http": lambda n: min(60, 2 ** n),
            "fragment": lambda n: min(30, 2 ** n),
            "extractor": lambda n: min(60, 5 * n),
        },
        "fragment_retries": 20,
        "extractor_retries": 5,
        "continuedl": True,
        "socket_timeout": 30,
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
    pot_url = os.environ.get("HL_POT_PROVIDER_URL")
    if pot_url:
        opts["extractor_args"] = {"youtubepot-bgutilhttp": {"base_url": [pot_url]}}
        log(f"pot provider: {pot_url}")
    proxy = os.environ.get("HL_YT_PROXY")
    if proxy:
        opts["proxy"] = proxy
        u = urllib.parse.urlparse(proxy)
        log(f"yt proxy: {u.scheme}://{u.hostname}:{u.port}")

    # Real-world finding: YouTube DASH formats (271/251) can 403 mid-fetch
    # while the HLS variants download fine. Retry once on 403 with HLS.
    formats = [
        opts["format"],
        "bv*[protocol^=m3u8]+ba[protocol^=m3u8]/"
        "bv*[protocol^=m3u8]+ba/bv*+ba/b",
    ]
    info = None
    cookies_dropped = False
    attempt = 0
    while True:
        try:
            info = _extract(opts, formats, url, log)
            break
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if _is_bot_check(msg):
                if cookiefile and not cookies_dropped:
                    # saved cookies rejected: retry once via the pot provider
                    cookies_dropped = True
                    opts.pop("cookiefile", None)
                    log("saved cookies rejected; retrying once without them")
                    continue
                raise PipelineError(
                    COOKIES_REJECTED_MSG if cookiefile else BOT_CHECK_MSG) from e
            if not is_transient_failure(msg):
                raise PipelineError(msg) from e
            attempt += 1
            if attempt >= TRANSIENT_MAX_ATTEMPTS:
                raise PipelineError(
                    f"{msg} (after {TRANSIENT_MAX_ATTEMPTS} attempts)") from e
            wait = TRANSIENT_BACKOFF_S[attempt - 1]
            short = msg.splitlines()[0][:80]
            note = (f"YouTube unavailable ({short}); retrying in {wait}s "
                    f"(attempt {attempt}/{TRANSIENT_MAX_ATTEMPTS})")
            log(note)
            if status is not None:
                status.update(message=note)
            _sleep(wait)

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
