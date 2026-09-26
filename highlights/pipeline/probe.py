"""ffprobe wrapper: duration/size/fps of a video file."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe(path: str | Path) -> dict[str, Any]:
    """Return {duration_s, width, height, fps} via ffprobe."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate,codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True,
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
