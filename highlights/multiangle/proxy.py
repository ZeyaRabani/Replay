"""480p analysis proxies: low-res copies of each angle used ONLY for
tracking (the render still encodes from the original).

Resolution-bound decode means tracking a 480p proxy is ~3x faster than
the 1080p source, with identical ball/player detection at the 960-wide
YOLO input. Order of attempts:
  1. existing analysis_480p.mp4 that probes ok -> reuse;
  2. YouTube url -> yt-dlp <=480p video-only (same yt-dlp Python API,
     cookies and retry behaviour as the main downloader);
  3. ffmpeg transcode of the original (window-limited to the track
     range +/-5 s when a window exists — the file-time shift is stored
     as "offset" in analysis_480p.json so the reader can map back).

The proxy must have identical frame timing to the original: its duration
is asserted within 1 s of the original's (windowed transcode compares
against the window span), else we fall back down the chain.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from highlights.io import write_json_atomic

NAME = "analysis_480p"


@dataclass
class ProxyInfo:
    path: Path          # may be the original when every path failed
    offset: float       # file seconds subtracted -> proxy-local time
    src: str            # "youtube-480p" | "transcode-480p" | "original"


def _probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _ytdlp_480(url: str, dest: Path, cookies: str | None, log) -> Path:
    """yt-dlp <=480p video-only into dest (reusing the main downloader's
    cookie + retry behaviour). Returns dest or raises."""
    import yt_dlp

    ck = cookies or os.environ.get("HL_YT_COOKIES")
    opts: dict = {
        "format": ("bestvideo[height<=480][ext=mp4]/"
                   "bestvideo[height<=480]/worst"),
        "outtmpl": str(dest) + ".part.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 20,
        "concurrent_fragment_downloads": 4,
    }
    if ck:
        opts["cookiefile"] = ck
    import shutil
    js = {}
    if shutil.which("deno"):
        js["deno"] = {}
    if shutil.which("node"):
        js["node"] = {}
    if js:
        opts["js_runtimes"] = js
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    parts = sorted(dest.parent.glob(dest.name + ".part.*"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not parts:
        raise RuntimeError("yt-dlp finished but left no .part output")
    os.replace(parts[0], dest)
    return dest


def _transcode_480(video: Path, dest: Path, lo: float, hi: float,
                   log) -> Path:
    tmp = dest.with_name(dest.stem + ".part.mp4")
    tmp.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-ss", f"{lo:.3f}", "-i", str(video), "-t", f"{hi - lo:.3f}",
           "-vf", "scale=-2:480",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-g", "30", "-threads", "2", "-an", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"proxy transcode failed: {proc.stderr[-500:]}")
    os.replace(tmp, dest)
    return dest


def ensure_analysis_proxy(angle_dir: Path, source_video: Path,
                          url: str | None = None,
                          cookies: str | None = None,
                          window: tuple[float, float] | None = None,
                          log=print) -> ProxyInfo:
    """Return a ProxyInfo for angle_dir — proxy file + the file-time
    offset the reader must subtract (0 for full-length proxies)."""
    angle_dir = Path(angle_dir)
    dest = angle_dir / f"{NAME}.mp4"
    side = angle_dir / f"{NAME}.json"
    src_dur = _probe_duration(source_video)
    # window for windowed transcodes: track range padded 5 s each side
    if window is not None:
        lo_w = max(0.0, window[0] - 5.0)
        hi_w = min(src_dur if src_dur > 0 else window[1],
                   window[1] + 5.0)
        want_dur = hi_w - lo_w
    else:
        lo_w, want_dur = 0.0, src_dur

    if dest.exists() and side.exists():
        try:
            meta = json.loads(side.read_text())
            off = float(meta.get("offset") or 0.0)
            if _probe_duration(dest) > 0:
                return ProxyInfo(dest, off,
                                 str(meta.get("src") or "youtube-480p"))
        except Exception:
            pass
    dest.unlink(missing_ok=True)
    side.unlink(missing_ok=True)

    t0 = time.time()
    # (b) YouTube 480p stream, full length (offset 0)
    if url:
        try:
            _ytdlp_480(url, dest, cookies, log)
            d = _probe_duration(dest)
            if src_dur > 0 and abs(d - src_dur) > 1.0:
                raise RuntimeError(
                    f"proxy duration {d:.1f} != source {src_dur:.1f}")
            meta = {"src": "youtube-480p", "offset": 0.0}
            write_json_atomic(side, meta)
            log(f"proxy: a youtube 480p download took "
                f"{time.time() - t0:.0f} s")
            return ProxyInfo(dest, 0.0, "youtube-480p")
        except Exception as e:
            log(f"proxy: youtube 480p failed ({e}); transcoding")
            dest.unlink(missing_ok=True)
    # (c) transcode the original (windowed when a track window exists)
    t0 = time.time()
    try:
        if window is not None and want_dur < src_dur - 1.0:
            _transcode_480(source_video, dest, lo_w,
                           lo_w + want_dur, log)
            off = lo_w
        else:
            _transcode_480(source_video, dest, 0.0,
                           src_dur if src_dur > 0 else 1e9, log)
            off = 0.0
        d = _probe_duration(dest)
        if d <= 0 or (want_dur > 0 and abs(d - want_dur) > 1.0):
            raise RuntimeError(f"transcoded proxy duration {d:.1f} s")
        write_json_atomic(side, {"src": "transcode-480p", "offset": off})
        log(f"proxy: transcode-480p took {time.time() - t0:.0f} s "
            f"(offset {off:.0f} s)")
        return ProxyInfo(dest, off, "transcode-480p")
    except Exception as e:
        log(f"proxy: transcode failed ({e}); using the original")
        dest.unlink(missing_ok=True)
        side.unlink(missing_ok=True)
        return ProxyInfo(Path(source_video), 0.0, "original")
