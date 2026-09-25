"""Render the director cut: per-segment re-encode + concat.

Segments are in OUTPUT time (rendered-video seconds; 0 = coverage-union
start), as written by stage_director. A segment [t0, t1) shown on angle a
is cut from that angle's file at file-time t0 + union_lo - offsets[a],
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


def plan_segments(segments: list[dict], offsets: list[float],
                  union_lo: float, union_hi: float,
                  durations: list[float]) -> list[dict]:
    """Map output-time segments to per-angle file time.

    director.json segments are in output time (0 = union_lo, so the
    renderable range is [0, hi-lo]); angle a's file time for output
    second t is t + union_lo - offsets[a]. Segments whose start lands
    beyond that angle's file duration come back with "skip": True.
    """
    dur_out = union_hi - union_lo
    plans = []
    for k, s in enumerate(segments):
        a = s["angle"]
        t0 = max(0.0, float(s["t_start"]))
        t1 = min(dur_out, float(s["t_end"]))
        if t1 <= t0:
            continue
        t_file = t0 + union_lo - offsets[a]
        if a < len(durations) and durations[a] and t_file >= durations[a]:
            plans.append({"seg_index": k, "angle": a, "t_file": t_file,
                          "skip": True})
            continue
        plans.append({"seg_index": k, "angle": a, "t0": t0, "t1": t1,
                      "t_file": t_file, "dur": t1 - t0})
    return plans


def render(videos: list[str], offsets: list[float], segments: list[dict],
           union_lo: float, union_hi: float, workdir: Path, out_path: Path,
           ref_video: str, durations: list[float] | None = None,
           log=print) -> Path:
    """Render segments to out_path. Videos[i] is angle i's file."""
    seg_dir = workdir / "segs"
    seg_dir.mkdir(parents=True, exist_ok=True)
    files = []
    total = max(1.0, union_hi - union_lo)
    plans = plan_segments(segments, offsets, union_lo, union_hi,
                          durations or [])
    for p in plans:
        if p.get("skip"):
            log(f"render: seg {p['seg_index']} (angle {p['angle']}) starts at "
                f"file t={p['t_file']:.1f} s, beyond that angle's video — skipped")
            continue
        a = p["angle"]
        out = seg_dir / f"{p['seg_index']:04d}.mp4"
        run(segment_cmd(videos[a], p["t_file"], p["dur"], str(out)), log)
        files.append(f"segs/{out.name}")  # concat resolves relative to list dir
        if p["seg_index"] % 10 == 0:
            log(f"render: seg {p['seg_index']}/{len(segments)} "
                f"({100*p['t0']/total:.0f}%)")
    lst = concat_file(files, workdir / "concat.txt")
    # unlink first: out_path may be a hardlink to another project's file
    # (e.g. a copied demo project) and ffmpeg would clobber the shared inode
    Path(out_path).unlink(missing_ok=True)
    run(concat_cmd(lst, str(out_path)), log)
    return out_path
