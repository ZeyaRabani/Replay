"""Render the director cut via per-angle mezzanines + segment extraction.

Segments are in OUTPUT time (rendered-video seconds; 0 = coverage-union
start), as written by stage_director. A segment [t0, t1) shown on angle a
is cut from that angle's file at file-time t0 + union_lo - offsets[a].

Pipeline: once per angle we encode a "mezzanine" covering the renderable
file range — canvas-normalised 1920x1080 30 fps (aspect preserved,
letterboxed — never cropped) with the same CRF/PRESET/audio as a direct
segment encode, plus a fixed 1-second GOP so integer-second cuts land on
keyframes. Segments are then extracted with input-side -ss and -c:v copy
(audio re-encoded at the cut so boundaries stay clean) and concat-demuxed
with -c copy — a re-cut is near-instant once the mezzanines exist. The
mezzanine key (angle + file range + encode settings + gop30) is part of
each segment's seg_key, so a rebuilt mezzanine invalidates old segments.
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
    """ffmpeg cmd for one segment at canvas resolution with its own audio.
    (Fallback path — mezzanine extraction is the normal one.)"""
    return ["ffmpeg", "-y", "-v", "error",
            "-ss", f"{t_file:.3f}", "-i", video, "-t", f"{dur:.3f}",
            "-vf", CANVAS + ",fps=30",
            "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            "-threads", "2",
            out]


def extract_cmd(mezz: str, t_rel: float, dur: float, out: str) -> list[str]:
    """Cut [t_rel, t_rel+dur) out of a mezzanine: video stream-copied on the
    forced 1 s keyframe grid, audio re-encoded so cut points stay clean."""
    return ["ffmpeg", "-y", "-v", "error",
            "-ss", f"{t_rel:.3f}", "-i", mezz, "-t", f"{dur:.3f}",
            "-frames:v", str(round(dur * 30)),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            out]


def mezz_key(angle: int, t0: float, t1: float) -> str:
    raw = (f"{angle}|{t0:.3f}|{t1:.3f}|{CANVAS}|{CRF}|{PRESET}|gop30")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def mezz_cmd(video: str, t0: float, t1: float, out: str) -> list[str]:
    """Encode the mezzanine for one angle: same canvas/codec settings as a
    direct segment encode, plus a fixed 1 s GOP (-progress pipe:1 for the
    caller's % logging)."""
    return ["ffmpeg", "-y", "-v", "error",
            "-nostats", "-progress", "pipe:1",
            "-ss", f"{t0:.3f}", "-i", video, "-t", f"{t1 - t0:.3f}",
            "-vf", CANVAS + ",fps=30",
            "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
            "-pix_fmt", "yuv420p",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-force_key_frames", "expr:gte(t,n_forced*1)",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            "-threads", "2",
            out]


def seg_key(angle: int, t_file: float, dur: float, mezz: str = "") -> str:
    """Content key for a cached segment: same inputs -> same file. The
    mezzanine key is mixed in so rebuilding a mezzanine invalidates the
    segments extracted from it."""
    raw = f"{angle}|{t_file:.3f}|{dur:.3f}|{CANVAS}|{CRF}|{PRESET}|{mezz}"
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


def _run_progress(cmd: list[str], dur_s: float, tag: str,
                  log=print) -> None:
    """Run an ffmpeg cmd that reports `-progress` on stdout; log every
    ~10% of dur_s worth of out_time."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    next_mark = 10
    for line in proc.stdout:
        if not line.startswith("out_time_ms="):
            continue
        try:
            done_us = float(line.split("=", 1)[1].strip() or 0)
        except ValueError:
            continue
        pct = 100 * done_us / 1e6 / max(dur_s, 1e-9)
        while pct >= next_mark:
            log(f"{tag} {next_mark}%")
            next_mark += 10
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{err[-2000:]}")


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


def mezz_range(i: int, offsets: list[float], union_lo: float,
               union_hi: float, durations: list[float]
               ) -> tuple[float, float]:
    """File-time range of angle i covered by the render, clipped to
    [0, duration] when the duration is known."""
    m0 = max(0.0, union_lo - offsets[i])
    m1 = max(m0, union_hi - offsets[i])
    if i < len(durations) and durations[i]:
        m1 = min(m1, float(durations[i]))
    return m0, m1


def render(videos: list[str], offsets: list[float], segments: list[dict],
           union_lo: float, union_hi: float, workdir: Path, out_path: Path,
           ref_video: str, durations: list[float] | None = None,
           log=print) -> Path:
    """Render segments to out_path. Videos[i] is angle i's file."""
    durations = durations or []
    seg_dir = workdir / "segs"
    mezz_dir = workdir / "mezz"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mezz_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    workers = int(os.environ.get("HL_RENDER_WORKERS", "2"))

    # per-angle mezzanines covering the renderable file range
    mezzs: dict[int, tuple[Path, float, float, str]] = {}
    for i, v in enumerate(videos):
        m0, m1 = mezz_range(i, offsets, union_lo, union_hi, durations)
        if m1 <= m0:
            continue
        key = mezz_key(i, m0, m1)
        mezzs[i] = (mezz_dir / f"{i}_{key}.mp4", m0, m1, key)

    def _build_mezz(i: int) -> None:
        mp, m0, m1, _ = mezzs[i]
        if _seg_ok(mp):
            log(f"render: mezzanine cached (angle {i}, {m1 - m0:.0f} s)")
            return
        tmp = mp.with_name(mp.stem + ".part.mp4")
        tmp.unlink(missing_ok=True)
        _run_progress(mezz_cmd(videos[i], m0, m1, str(tmp)), m1 - m0,
                      f"render: mezzanine angle {i}", log)
        os.replace(tmp, mp)
        if not _seg_ok(mp):
            raise RuntimeError(f"render: mezzanine angle {i} invalid")
        log(f"render: mezzanine built (angle {i}, {m1 - m0:.0f} s)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_build_mezz, mezzs))
    # drop stale mezzanines (different key / range)
    live = {mp.name for mp, *_ in mezzs.values()}
    for f in mezz_dir.glob("*.mp4"):
        if f.name not in live:
            f.unlink()

    files = []
    used: set[str] = set()
    reused = 0
    to_encode = []        # (plan, out_path) — cache misses
    total = max(1.0, union_hi - union_lo)
    plans = plan_segments(segments, offsets, union_lo, union_hi, durations)
    for p in plans:
        if p.get("skip"):
            log(f"render: seg {p['seg_index']} (angle {p['angle']}) starts at "
                f"file t={p['t_file']:.1f} s, beyond that angle's video — skipped")
            continue
        a = p["angle"]
        mk = mezzs[a][3] if a in mezzs else ""
        out = seg_dir / f"{seg_key(a, p['t_file'], p['dur'], mk)}.mp4"
        used.add(out.name)
        if out.exists() and out.stat().st_size > 0:
            reused += 1
        else:
            to_encode.append((p, out))
        files.append(f"segs/{out.name}")  # concat resolves relative to list dir
    # extract missing segments HL_RENDER_WORKERS at a time (threads are
    # fine: the work is in ffmpeg subprocesses); cache hits keep order

    def _enc(p: dict, out: Path) -> None:
        tmp = out.with_name(out.stem + ".part.mp4")
        tmp.unlink(missing_ok=True)
        a = p["angle"]
        if a in mezzs:
            mp, m0, _m1, _k = mezzs[a]
            run(extract_cmd(str(mp), p["t_file"] - m0, p["dur"],
                            str(tmp)), log)
        else:
            run(segment_cmd(videos[a], p["t_file"], p["dur"], str(tmp)), log)
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
    plan_by_name = {}
    for p in plans:
        if p.get("skip"):
            continue
        a = p["angle"]
        mk = mezzs[a][3] if a in mezzs else ""
        n = seg_key(a, p["t_file"], p["dur"], mk) + ".mp4"
        plan_by_name[n] = (p, seg_dir / n)
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
