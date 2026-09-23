"""ffmpeg helpers: stream-copy chunking and clip extraction."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    idx: int
    path: Path
    t0: float        # actual start time on the source clock (first-frame pts)
    duration: float


def _first_pts(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pts_time,best_effort_timestamp_time,pkt_pts_time", "-read_intervals", "%+#1",
         "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    frames = json.loads(out).get("frames", [])
    if not frames:
        return 0.0
    for key in ("pts_time", "best_effort_timestamp_time", "pkt_pts_time"):
        if key in frames[0]:
            return float(frames[0][key])
    return 0.0


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def split(video: Path, out_dir: Path, chunk_s: float) -> list[Chunk]:
    """Keyframe-aligned stream-copy split; t0 is the real pts of each chunk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = _duration(video)
    chunks: list[Chunk] = []
    t = 0.0
    idx = 0
    while t < total - 0.05:
        dst = out_dir / f"chunk_{idx:03d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                        "-t", f"{chunk_s:.3f}", "-c", "copy", str(dst)], check=True)
        if not dst.exists() or dst.stat().st_size == 0:
            break
        chunks.append(Chunk(idx, dst, t + _first_pts(dst), _duration(dst)))
        t += chunk_s
        idx += 1
    return chunks


def extract_clip(video: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(video),
                    "-t", f"{end - start:.3f}", "-vf", "scale=-2:720", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", str(dst)], check=True)
