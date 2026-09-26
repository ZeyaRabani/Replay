"""Render the director cut: per-segment re-encode + concat.

Segments are in OUTPUT time (rendered-video seconds; 0 = coverage-union
start), as written by stage_director. A segment [t0, t1) shown on angle a
is cut from that angle's file at file-time t0 + union_lo - offsets[a],
normalised to a 1920x1080 canvas (aspect preserved, letterboxed — never
cropped) with that angle's own audio (normalised to 48 kHz stereo AAC so
the concat demuxer can stream-copy), then concat-demuxed.
"""

from __future__ import annotations

import hashlib
import os
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
            "-threads", "2",
            out]


def seg_key(angle: int, t_file: float, dur: float) -> str:
    """Content key for a cached segment: same inputs -> same file."""
    raw = f"{angle}|{t_file:.3f}|{dur:.3f}|{CANVAS}|{CRF}|{PRESET}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def concat_file(segs: list[str], path: str | Path) -> Path:
    from highlights.io import write_text_atomic
    p = Path(path)
    write_text_atomic(p, "".join(f"file '{s}'\n" for s in segs))
    return p


def concat_cmd(list_file: str | Path, out: str) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", out]


def run(cmd: list[str], log=print) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _seg_ok(path: Path) -> bool:
    """A written segment is usable iff it exists, is non-empty and
    actually carries at least one stream (ffmpeg can exit 0 yet leave a
    truncated header-only file — e.g. killed mid-mux)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return proc.returncode == 0 and bool(proc.stdout.strip())


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
    used: set[str] = set()
    reused = 0
    to_encode = []        # (plan, out_path) — cache misses
    total = max(1.0, union_hi - union_lo)
    plans = plan_segments(segments, offsets, union_lo, union_hi,
                          durations or [])
    for p in plans:
        if p.get("skip"):
            log(f"render: seg {p['seg_index']} (angle {p['angle']}) starts at "
                f"file t={p['t_file']:.1f} s, beyond that angle's video — skipped")
            continue
        a = p["angle"]
        out = seg_dir / f"{seg_key(a, p['t_file'], p['dur'])}.mp4"
        used.add(out.name)
        if out.exists() and out.stat().st_size > 0:
            reused += 1
        else:
            to_encode.append((p, out))
        files.append(f"segs/{out.name}")  # concat resolves relative to list dir
    # encode missing segments HL_RENDER_WORKERS at a time (threads are
    # fine: the work is in ffmpeg subprocesses); cache hits keep order
    from concurrent.futures import ThreadPoolExecutor
    workers = int(os.environ.get("HL_RENDER_WORKERS", "2"))

    def _enc(p: dict, out: Path) -> None:
        tmp = out.with_name(out.stem + ".part.mp4")
        tmp.unlink(missing_ok=True)
        run(segment_cmd(videos[p["angle"]], p["t_file"],
                        p["dur"], str(tmp)), log)
        os.replace(tmp, out)

    def _encode_all(jobs: list[tuple[dict, Path]], log_progress: bool) -> None:
        done_n = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [(p, ex.submit(_enc, p, out)) for p, out in jobs]
            for p, fut in futs:
                fut.result()
                done_n += 1
                if log_progress and (
                        p["seg_index"] % 10 == 0 or done_n == len(futs)):
                    log(f"render: seg {p['seg_index']}/{len(segments)} "
                        f"({100*p['t0']/total:.0f}%)")

    _encode_all(to_encode, log_progress=True)
    log(f"render: reused {reused}/{reused + len(to_encode)} segments")
    # verify every referenced segment before concat: a failed or
    # truncated encode must never reach the concat demuxer
    plan_by_name = {
        seg_key(p["angle"], p["t_file"], p["dur"]) + ".mp4":
        (p, seg_dir / (seg_key(p["angle"], p["t_file"], p["dur"]) + ".mp4"))
        for p in plans if not p.get("skip")}
    bad = [n for n in plan_by_name if not _seg_ok(seg_dir / n)]
    if bad:
        log(f"render: {len(bad)} invalid segment(s) "
            f"({', '.join(bad[:5])}{'...' if len(bad) > 5 else ''}) - re-encoding")
        _encode_all([plan_by_name[n] for n in bad], log_progress=False)
        still = [n for n in bad if not _seg_ok(seg_dir / n)]
        if still:
            raise RuntimeError(
                f"render: {len(still)} segment(s) still invalid after "
                f"re-encode: {', '.join(still[:5])}")
    # prune cache entries not referenced by this render
    for f in seg_dir.glob("*.mp4"):
        if f.name not in used and f.suffix == ".mp4":
            f.unlink()
    lst = concat_file(files, workdir / "concat.txt")
    # write to a sibling tmp then os.replace: out_path may be hardlinked
    # into cuts/ snapshots or a copied project — never clobber in place
    tmp = Path(out_path).with_name(Path(out_path).name + ".part.mp4")
    tmp.unlink(missing_ok=True)
    run(concat_cmd(lst, str(tmp)), log)
    os.replace(tmp, out_path)
    return out_path
