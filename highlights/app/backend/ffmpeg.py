"""FFmpeg helpers: probe, thumbnails, proxy, clip cut, overlay, concat reel."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(path: str | Path) -> dict[str, Any]:
    """Return {duration_s, width, height, fps} via ffprobe."""
    r = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate,codec_type",
            "-of", "json", str(path),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()}")
    data = json.loads(r.stdout or "{}")
    vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = 0.0
    rate = vstream.get("r_frame_rate", "0/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "duration_s": float(data.get("format", {}).get("duration", 0.0) or 0.0),
        "width": int(vstream.get("width", 0) or 0),
        "height": int(vstream.get("height", 0) or 0),
        "fps": fps,
    }


def drawtext_supported() -> bool:
    r = _run(["ffmpeg", "-hide_banner", "-filters"])
    return "drawtext" in (r.stdout or "")


def find_font() -> str | None:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def mmss(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def thumbnail(src: str | Path, t: float, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = _run(
        [
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", "scale=-2:180", "-q:v", "4", str(out),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg thumbnail failed: {r.stderr.strip()[-400:]}")
    return out


def _overlay_filter(label: str) -> str | None:
    """drawtext filter for '{TYPE}  {mm:ss}' bottom-left, white on semi-transparent box."""
    if not drawtext_supported():
        return None
    font = find_font()
    if font is None:
        return None
    text = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
    return (
        f"drawtext=fontfile={font}:text='{text}':fontsize=28:fontcolor=white:"
        "x=24:y=h-th-24:box=1:boxcolor=black@0.5:boxborderw=10"
    )


def cut_clip(
    src: str | Path,
    start: float,
    dur: float,
    out: str | Path,
    overlay_label: str | None = None,
    reencode: bool = False,
) -> Path:
    """Cut [start, start+dur] to out.mp4. Stream copy unless overlay/reencode."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = _overlay_filter(overlay_label) if overlay_label else None
    if overlay_label and vf is None:
        print("warning: drawtext/font unavailable, skipping overlay")
    if vf or reencode:
        filters = f"scale=-2:720,{vf}" if vf else "scale=-2:720"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
            "-vf", filters,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", str(out),
        ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {r.stderr.strip()[-400:]}")
    return out


def concat_reel(clips: list[str | Path], out: str | Path) -> Path:
    """Concat clips in order to out.mp4; stream-copy, fall back to re-encode."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    listfile = out.with_suffix(".list.txt")
    listfile.write_text("".join(f"file '{Path(c).resolve()}'\n" for c in clips))
    r = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])
    if r.returncode != 0:
        r = _run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
            ]
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr.strip()[-400:]}")
    return out


class ProxyJob:
    """Background low-res proxy build with progress parsing."""

    def __init__(self, src: str | Path, dst: str | Path, duration_s: float) -> None:
        self.src, self.dst, self.duration = Path(src), Path(dst), duration_s
        self.progress = 0.0
        self.done = dst.exists() if hasattr(dst, "exists") else False
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.done = Path(self.dst).exists()
        if self.done:
            self.progress = 1.0
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        cmd = [
            "ffmpeg", "-y", "-i", str(self.src),
            "-vf", "scale=-2:360", "-r", "15",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
            "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(self.dst),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms=") and self.duration > 0:
                try:
                    ms = float(line.split("=", 1)[1])
                    self.progress = min(1.0, ms / 1e6 / self.duration)
                except ValueError:
                    pass
        proc.wait()
        if proc.returncode == 0 and Path(self.dst).exists():
            self.progress = 1.0
            self.done = True
        else:
            self.error = f"proxy ffmpeg exited {proc.returncode}"


def render_reel(
    src: str | Path,
    items: list[dict[str, Any]],
    out_dir: str | Path,
    overlay: bool = True,
    reencode: bool = False,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Cut clips + concat reel.

    items: list of {"id", "t", "type", "clip_start", "clip_end", "name" (optional)}.
    Returns {"clips": [{id, path, duration}], "reel": path, "reel_s": float}.
    """
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(items, key=lambda c: c["clip_start"])
    n = len(items)
    results = []
    for i, c in enumerate(items):

        def report(frac: float, msg: str) -> None:
            if progress_cb:
                progress_cb(frac, msg)

        name = c.get("name") or f"{i + 1:02d}_{c['type']}_{c['t']:07.1f}.mp4"
        path = clips_dir / name
        label = f"{str(c['type']).upper()}  {mmss(c['t'])}" if overlay else None
        dur = max(0.1, c["clip_end"] - c["clip_start"])
        cut_clip(src, c["clip_start"], dur, path, overlay_label=label, reencode=reencode)
        results.append({"id": c["id"], "path": str(path), "duration": dur})
        report((i + 1) / (n + 1), f"clip {i + 1}/{n}")
    reel = out_dir / "reel.mp4"
    concat_reel([r["path"] for r in results], reel)
    if progress_cb:
        progress_cb(1.0, "done")
    reel_s = probe(reel)["duration_s"]
    return {"clips": results, "reel": str(reel), "reel_s": reel_s}
