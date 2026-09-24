"""ffmpeg helper: clip extraction (720p re-encode)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_clip(video: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(video),
                    "-t", f"{end - start:.3f}", "-vf", "scale=-2:720", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", str(dst)], check=True)
