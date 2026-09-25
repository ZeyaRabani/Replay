"""Render the director cut: per-segment re-encode + concat.

Each segment is cut from its angle's file at file-time T - offset_i,
normalised to a 1920x1080 canvas (aspect preserved, letterboxed — never
cropped) with that angle's own audio (normalised to 48 kHz stereo AAC so
the concat demuxer can stream-copy), then concat-demuxed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CANVAS = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
CRF = "18"
PRESET = "veryfast"
AUDIO_BITRATE = "192k"


def segment_cmd(video: str, t_file: float, dur: float, out: str) -> list[str]:
    """ffmpeg cmd for one segment at canvas resolution with its own audio."""
    return ["ffmpeg", "-y", "-v", "error",
            "-ss", f"{t_file:.3f}", "-i", video, "-t", f"{dur:.3f}",
            "-vf", CANVAS + ",fps=30",
            "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            out]


def concat_file(segs: list[str], path: str | Path) -> Path:
    p = Path(path)
    p.write_text("".join(f"file '{s}'\n" for s in segs))
    return p


def concat_cmd(list_file: str | Path, out: str) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", out]


def run(cmd: list[str], log=print) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def render(videos: list[str], offsets: list[float], segments: list[dict],
           union_lo: float, union_hi: float, workdir: Path, out_path: Path,
           ref_video: str, log=print) -> Path:
    """Render segments to out_path. Videos[i] is angle i's file."""
    seg_dir = workdir / "segs"
    seg_dir.mkdir(parents=True, exist_ok=True)
    files = []
    total = max(1.0, union_hi - union_lo)
    for k, s in enumerate(segments):
        a = s["angle"]
        t0 = max(union_lo, s["t_start"])
        t1 = min(union_hi, s["t_end"])
        if t1 <= t0:
            continue
        t_file = t0 - offsets[a]
        dur = t1 - t0
        out = seg_dir / f"{k:04d}.mp4"
        run(segment_cmd(videos[a], t_file, dur, str(out)), log)
        files.append(f"segs/{out.name}")  # concat resolves relative to list dir
        if k % 10 == 0:
            log(f"render: seg {k}/{len(segments)} ({100*(t0-union_lo)/total:.0f}%)")
    lst = concat_file(files, workdir / "concat.txt")
    run(concat_cmd(lst, str(out_path)), log)
    return out_path
