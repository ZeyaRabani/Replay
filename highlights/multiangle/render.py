"""Render director segments on an exact shared output-time timeline.

Each angle gets one output-time mezzanine spanning the coverage union. Its
video is 30 fps on the common frame grid and its sibling WAV is 48 kHz PCM;
real pieces map shared time through the angle timemap and uncovered regions
are black/silence. Director segment boundaries are rounded once to common
integer seconds. Video segments are frame-exact stream copies, while audio is
assembled as one sample-exact WAV with a short crossfade at camera changes.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np

MEZZ_SIZE = "1920x1080"
CANVAS = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
CRF = "18"
PRESET = "veryfast"
AUDIO_BITRATE = "192k"
FPS = 30
SAMPLES_PER_FRAME = 1600
SAMPLE_RATE = 48000
CROSSFADE_SAMPLES = 480
DRAW_TEXT_FONTS = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _duration_frames(t0: float, t1: float) -> tuple[int, int]:
    return round(t0 * FPS), round(t1 * FPS)


def mezz_pieces(i: int, tm, union_lo: float, union_hi: float,
                durations: list[float]) -> list[dict]:
    """Output-time pieces for an angle; later timemap segments win overlaps."""
    from highlights.multiangle.syncmap import as_tm

    maps = as_tm(tm, durations)
    if i >= len(maps):
        return []
    segs = maps[i]
    edges = {float(union_lo), float(union_hi)}
    for seg in segs:
        off = float(seg["offset"])
        flo, fhi = float(seg["file_lo"]), float(seg["file_hi"])
        for shared in (flo + off, fhi + off):
            if union_lo < shared < union_hi:
                edges.add(shared)
        if flo < 0 < fhi:
            edges.add(off)
        if i < len(durations) and durations[i]:
            d = float(durations[i])
            if flo < d < fhi:
                edges.add(d + off)
    points = sorted(edges)
    pieces = []
    for slo, shi in itertools.pairwise(points):
        n0, n1 = _duration_frames(slo - union_lo, shi - union_lo)
        if n1 <= n0:
            continue
        mid = (slo + shi) / 2
        selected = next((s for s in reversed(segs)
                         if s["file_lo"] + s["offset"] <= mid
                         <= s["file_hi"] + s["offset"]), None)
        # Audio gets sample-exact bounds: the piece covers samples
        # round(τ0*48000)..round(τ1*48000) even when the video bounds
        # snap to the frame grid.
        a0 = round((slo - union_lo) * SAMPLE_RATE)
        a1 = round((shi - union_lo) * SAMPLE_RATE)
        if selected:
            offset = float(selected["offset"])
            file_mid = mid - offset
            dur = float(durations[i]) if i < len(durations) and durations[i] else None
            if file_mid >= 0 and (dur is None or file_mid < dur):
                tau0 = n0 / FPS
                file_start = max(float(selected["file_lo"]),
                                 union_lo + tau0 - offset)
                if dur is not None:
                    file_start = min(file_start, dur)
                a_start = min(max(slo - offset,
                                  float(selected["file_lo"])),
                              float(selected["file_hi"]))
                pieces.append({"n0": n0, "n1": n1, "frames": n1 - n0,
                               "a0": a0, "a1": a1, "a_start": a_start,
                               "file_start": file_start, "real": True,
                               "offset": offset})
                continue
        pieces.append({"n0": n0, "n1": n1, "frames": n1 - n0,
                       "a0": a0, "a1": a1, "a_start": None,
                       "file_start": None, "real": False, "offset": None})
    return pieces


def _serial(pieces: list[dict]) -> list[dict]:
    return [{k: p.get(k) for k in
             ("n0", "n1", "a0", "a1", "a_start", "file_start", "real")}
            for p in pieces]


def mezz_key(angle: int, pieces: list[dict]) -> str:
    """Content key including output-time mapping and otime4 cache salt."""
    raw = json.dumps(["otime4", angle, CANVAS, CRF, PRESET, _serial(pieces)],
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def chunk_pieces(pieces: list[dict], c: int, ch: int) -> list[dict]:
    """Slice full-union pieces to output frames [c*ch*FPS, (c+1)*ch*FPS)."""
    total = max((p["n1"] for p in pieces), default=0)
    c0, c1 = c * ch * FPS, min((c + 1) * ch * FPS, total)
    out = []
    for p in pieces:
        n0, n1 = max(p["n0"], c0), min(p["n1"], c1)
        if n1 <= n0:
            continue
        q = {"n0": n0 - c0, "n1": n1 - c0, "frames": n1 - n0,
             "a0": (n0 - c0) * SAMPLES_PER_FRAME,
             "a1": (n1 - c0) * SAMPLES_PER_FRAME,
             "real": p["real"], "offset": p.get("offset")}
        if p["real"]:
            shift = (n0 - p["n0"]) / FPS
            q["file_start"] = p["file_start"] + shift
            q["a_start"] = (p.get("a_start") or 0.0) + shift
        else:
            q["file_start"] = None
            q["a_start"] = None
        out.append(q)
    return out


def chunk_key(angle: int, sliced: list[dict]) -> str:
    """Content key for one chunk's piece mapping (position-independent)."""
    raw = json.dumps(["chunk1", angle, CANVAS, CRF, PRESET, _serial(sliced)],
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _has_audio(video: str) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=False)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def mezz_cmd(video: str, pieces: list[dict], out: str | None,
             wav_out: str | None, threads: int = 2, video_out: bool = True,
             audio_out: bool = True) -> list[str]:
    """Build the output-time mezz video and/or PCM audio WAV in one run."""
    if not video_out and not audio_out:
        raise ValueError("mezz_cmd: nothing to produce")
    if (video_out and out is None) or (audio_out and wav_out is None):
        raise ValueError("mezz_cmd: missing output path")
    args = ["ffmpeg", "-y", "-v", "error", "-nostats", "-progress", "pipe:1"]
    filters = []
    n = len(pieces)
    if not n:
        raise ValueError("mezzanine needs at least one output piece")
    has_audio = _has_audio(video) if audio_out else False
    input_idx = 0
    for j, piece in enumerate(pieces):
        frames = piece["frames"]
        samples = piece["a1"] - piece["a0"]
        if piece["real"]:
            span = frames / FPS + 1
            if video_out:
                vi = input_idx
                args += ["-ss", f"{piece['file_start']:.6f}", "-t",
                         f"{span:.6f}", "-i", video]
                input_idx += 1
                filters.append(
                    f"[{vi}:v:0]{CANVAS},setsar=1,fps=30,"
                    f"tpad=stop_mode=clone:stop_duration=2,"
                    f"trim=end_frame={frames},setpts=PTS-STARTPTS[v{j}]")
            if audio_out:
                if has_audio:
                    # Input seeking lands on a packet boundary; -copyts keeps
                    # original pts so atrim can slice by absolute file time
                    # sample-exactly.
                    ai = input_idx
                    a_start = float(piece["a_start"])
                    a_len = samples / SAMPLE_RATE
                    a_seek = max(0.0, a_start - 2.0)
                    args += ["-copyts", "-ss", f"{a_seek:.6f}", "-t",
                             f"{a_start + a_len - a_seek + 2:.6f}",
                             "-i", video]
                    input_idx += 1
                    filters.append(
                        f"[{ai}:a:0]aresample=48000,aformat=sample_fmts=s16:"
                        f"channel_layouts=stereo,"
                        f"atrim=start={a_start:.6f}:end={a_start + a_len:.6f},"
                        f"asetpts=PTS-STARTPTS,apad,atrim=end_sample={samples},"
                        f"asetpts=PTS-STARTPTS[a{j}]")
                else:
                    args += ["-f", "lavfi", "-t", f"{span:.6f}", "-i",
                             "anullsrc=r=48000:cl=stereo"]
                    ai = input_idx
                    input_idx += 1
                    filters.append(
                        f"[{ai}:a:0]aresample=48000,aformat=sample_fmts=s16:"
                        f"channel_layouts=stereo,apad,"
                        f"atrim=end_sample={samples},"
                        f"asetpts=PTS-STARTPTS[a{j}]")
        else:
            if video_out:
                filters.append(
                    f"color=black:s={MEZZ_SIZE}:r=30,setsar=1,"
                    f"trim=end_frame={frames},setpts=PTS-STARTPTS[v{j}]")
            if audio_out:
                filters.append(
                    f"anullsrc=r=48000:cl=stereo,atrim=end_sample={samples},"
                    f"asetpts=PTS-STARTPTS[a{j}]")
    # concat pads each segment's shorter stream to the longer one on
    # ffmpeg >=5, which would offset the sample-exact audio bounds — keep
    # the video and audio chains independent.
    if video_out:
        vpads = "".join(f"[v{j}]" for j in range(n))
        filters.append(f"{vpads}concat=n={n}:v=1:a=0[vout]")
    if audio_out:
        apads = "".join(f"[a{j}]" for j in range(n))
        filters.append(f"{apads}concat=n={n}:v=0:a=1[aout]")
    args += ["-filter_complex", ";".join(filters)]
    if video_out:
        args += ["-map", "[vout]", "-r", "30",
                 "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
                 "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "30",
                 "-sc_threshold", "0", "-force_key_frames",
                 "expr:gte(t,n_forced*1)", "-threads", str(threads),
                 "-video_track_timescale", "15360", "-an", out]
    if audio_out:
        args += ["-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "48000",
                 "-ac", "2", wav_out]
    return args


def plan_segments(segments: list[dict], tm=None, union_lo: float = 0.0,
                  union_hi: float = 0.0,
                  durations: list[float] | None = None) -> list[dict]:
    """Snap output-time director segments to the common integer-second grid."""
    plans = []
    for idx, seg in enumerate(segments):
        k0 = round(float(seg["t_start"]))
        k1 = round(float(seg["t_end"]))
        if plans:
            prev = plans[-1]
            if k0 > prev["k1"]:
                prev["k1"] = k0
                prev["frames"] = (prev["k1"] - prev["k0"]) * FPS
                prev["samples"] = (prev["k1"] - prev["k0"]) * SAMPLE_RATE
            k0 = max(k0, prev["k1"])
        if k1 <= k0:
            continue
        plans.append({"seg_index": idx, "angle": int(seg["angle"]),
                      "k0": k0, "k1": k1, "frames": (k1 - k0) * FPS,
                      "samples": (k1 - k0) * SAMPLE_RATE,
                      "speed": float(seg.get("speed", 1.0)),
                      "overlay": seg.get("overlay"),
                      "source_t0": float(seg.get("t_src_start", seg["t_start"])),
                      "source_t1": float(seg.get("t_src_end", seg["t_end"]))})
    if plans:
        if any(plans[i]["k0"] != plans[i - 1]["k1"]
               for i in range(1, len(plans))):
            raise ValueError("plan_segments: non-contiguous plans")
        total = sum(p["frames"] for p in plans)
        if total != (plans[-1]["k1"] - plans[0]["k0"]) * FPS:
            raise ValueError("plan_segments: frames do not cover timeline")
    return plans


def extract_cmd(mezz: str, k0: int, req: int, out: str) -> list[str]:
    """Extract exactly req seconds of video-only 30 fps mezz frames."""
    return ["ffmpeg", "-y", "-v", "error", "-ss", f"{k0:.3f}", "-i", mezz,
            "-t", f"{req:.3f}", "-frames:v", str(req * FPS),
            "-c:v", "copy", "-video_track_timescale", "15360", "-an", out]


def seg_key(angle: int, k0: int, req: int, mezz: str = "",
            speed: float = 1.0, overlay: str | None = None) -> str:
    raw = f"otime2|{angle}|{k0}|{req}|{mezz}"
    if speed != 1.0 or overlay:
        raw += f"|{speed:.6f}|{overlay or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def _drawtext_available() -> bool:
    if not _drawtext_font():
        return False
    try:
        output = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                capture_output=True, text=True,
                                check=False).stdout
    except OSError:
        return False
    return "drawtext" in output


def _drawtext_font() -> Path | None:
    return next((font for font in DRAW_TEXT_FONTS if font.is_file()), None)


def replay_cmd(src: str, t_rel: float, dur_src: float, speed: float,
               out: str, overlay: str | None = None,
               from_mezz: bool = True) -> list[str]:
    filters = ([] if from_mezz else [CANVAS])
    filters.extend([f"setpts=PTS/{speed}", "fps=30", "setsar=1"])
    if overlay and _drawtext_available():
        font = _drawtext_font()
        filters.append(
            f"drawtext=fontfile={font}:text={overlay}:x=48:y=40:"
            "fontsize=44:fontcolor=white:box=1:boxcolor=black@0.55:"
            "boxborderw=14")
    return ["ffmpeg", "-y", "-v", "error", "-ss", f"{t_rel:.3f}",
            "-t", f"{dur_src:.3f}", "-i", src,
            "-t", f"{dur_src / speed:.3f}", "-vf", ",".join(filters),
            "-frames:v", str(round(dur_src / speed * FPS)), "-an",
            "-c:v", "libx264", "-preset", PRESET, "-crf", CRF,
            "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "30",
            "-sc_threshold", "0", "-threads", "2",
            "-video_track_timescale", "15360", out]


def concat_file(segs: list[str], path: str | Path) -> Path:
    from highlights.io import write_text_atomic
    p = Path(path)
    write_text_atomic(p, "".join(f"file '{s}'\n" for s in segs))
    return p


def concat_cmd(list_file: str | Path, audio: str, out: str) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-i", audio, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-ar", "48000", "-ac", "2", "-movflags", "+faststart", out]


def run(cmd: list[str], log=print) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _run_progress(cmd: list[str], dur_s: float, tag: str,
                  log=print) -> None:
    """Run ffmpeg and log progress in ~10% increments."""
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
    if not path.exists() or path.stat().st_size == 0:
        return False
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _wav_ok(path: Path, samples: int) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return (wf.getnchannels() == 2 and wf.getsampwidth() == 2
                    and wf.getframerate() == SAMPLE_RATE
                    and wf.getnframes() == samples)
    except (OSError, wave.Error):
        return False


def _build_audio(plans: list[dict], wavs: dict[int, Path], out: Path,
                 log=print) -> int:
    """Stream-copy output-order PCM, crossfading 10 ms at angle switches."""
    total_samples = sum(p["samples"] for p in plans)
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(2)
        dst.setsampwidth(2)
        dst.setframerate(SAMPLE_RATE)
        last_angle = None
        for plan in plans:
            n_samples = plan["samples"]
            start = plan["k0"] * SAMPLE_RATE
            if plan["angle"] not in wavs:
                audio = np.zeros((n_samples, 2), dtype=np.int16)
            elif plan["speed"] != 1.0:
                audio = _replay_audio(wavs[plan["angle"]], plan,
                                      n_samples)
            else:
                audio = _read_pcm(wavs[plan["angle"]], start, n_samples)
            cross = (min(CROSSFADE_SAMPLES, n_samples)
                     if last_angle is not None
                     and last_angle != plan["angle"] else 0)
            if cross:
                previous = (_read_pcm(wavs[last_angle], start, cross)
                            if last_angle in wavs
                            else np.zeros((cross, 2), dtype=np.int16))
                alpha = (np.arange(cross, dtype=np.float32) + 1) / cross
                mixed = (previous.astype(np.float32) * (1 - alpha[:, None])
                         + audio[:cross].astype(np.float32) * alpha[:, None])
                audio[:cross] = np.clip(np.rint(mixed), -32768, 32767).astype(np.int16)
            dst.writeframesraw(audio.astype("<i2", copy=False).tobytes())
            last_angle = plan["angle"]
    if total_samples != sum(p["frames"] for p in plans) * SAMPLES_PER_FRAME:
        raise AssertionError("audio sample/frame invariant violated")
    if not _wav_ok(out, total_samples):
        raise RuntimeError("assembled audio.wav has incorrect sample count")
    return total_samples


def _read_pcm(path: Path, start: int, count: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        total = wf.getnframes()
        start = max(0, min(start, total))
        wf.setpos(start)
        raw = wf.readframes(count)
    arr = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
    if len(arr) < count:
        arr = np.pad(arr, ((0, count - len(arr)), (0, 0)))
    return arr.copy()


def _replay_audio(wav: Path, plan: dict, samples: int) -> np.ndarray:
    src_dur = plan["source_t1"] - plan["source_t0"]
    speed = plan["speed"]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{plan['source_t0']:.6f}",
         "-t", f"{src_dur:.6f}", "-i", str(wav), "-af",
         f"atempo={speed},aresample=48000,apad,atrim=end_sample={samples}",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
         "pipe:1"], capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
    if len(arr) < samples:
        arr = np.pad(arr, ((0, samples - len(arr)), (0, 0)))
    return arr[:samples].copy()


def _mezz_audio_at(wavs: dict[int, Path], angle: int, k0: int,
                   n_samples: int) -> np.ndarray:
    return _read_pcm(wavs[angle], k0 * SAMPLE_RATE, n_samples)


def render(videos: list[str], tm, segments: list[dict],
           union_lo: float, union_hi: float, workdir: Path, out_path: Path,
           ref_video: str, durations: list[float] | None = None,
           log=print) -> Path:
    """Render with every angle mapped to shared output time."""
    durations = durations or []
    seg_dir, mezz_dir = workdir / "segs", workdir / "mezz"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mezz_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = mezz_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    import math
    from concurrent.futures import ThreadPoolExecutor

    chunk_mode = os.environ.get("HL_RENDER_MODE", "chunk") != "mezz"
    ch = max(1, int(os.environ.get("HL_RENDER_CHUNK", "10")))
    total_s = union_hi - union_lo
    n_chunks = max(1, math.ceil(total_s / ch))

    mezzs: dict[int, tuple[Path, Path, str, list[dict]]] = {}
    for i, video in enumerate(videos):
        pieces = mezz_pieces(i, tm, union_lo, union_hi, durations)
        if not pieces:
            continue
        key = mezz_key(i, pieces)
        mezzs[i] = (mezz_dir / f"{i}_{key}.mp4",
                    mezz_dir / f"{i}_{key}.wav", key, pieces)
    n_mezz = len(mezzs)
    mezz_threads = max(1, -(-(os.cpu_count() or 2) // max(1, n_mezz)))

    plans = plan_segments(segments)

    def _plan_chunk_range(plan: dict) -> tuple[int, int]:
        """[cc0, cc1] chunk indices the plan needs on its angle."""
        if plan["speed"] != 1.0 or plan["overlay"]:
            s0 = max(0.0, math.floor(plan["source_t0"]) - 1.0)
            s1 = min(total_s, math.ceil(plan["source_t1"]) + 1.0)
            return int(s0) // ch, max(0, math.ceil(s1) - 1) // ch
        return plan["k0"] // ch, (plan["k1"] - 1) // ch

    # angle -> {chunk index -> (path, key)} for the chunks this cut needs;
    # angles with a valid full mezz mp4 use the mezz fast path instead
    need_chunks: dict[int, dict[int, tuple[Path, str]]] = {}
    if chunk_mode:
        for i, (mp, _wp, _key, pieces) in mezzs.items():
            if _seg_ok(mp):
                continue
            idxs: set[int] = set()
            for plan in plans:
                if plan["angle"] == i:
                    cc0, cc1 = _plan_chunk_range(plan)
                    idxs.update(range(cc0, cc1 + 1))
            need_chunks[i] = {}
            for c in sorted(idxs):
                if c < 0 or c >= n_chunks:
                    continue
                sliced = chunk_pieces(pieces, c, ch)
                if not sliced:
                    continue
                k = chunk_key(i, sliced)
                need_chunks[i][c] = (chunk_dir / f"{i}_{k}.mp4", k)

    def _seg_source_key(i: int, plan: dict) -> str:
        """Cache key fragment covering the video sources a segment uses."""
        if not chunk_mode or i not in need_chunks or _seg_ok(mezzs[i][0]):
            return mezzs[i][2] if i in mezzs else ""
        cc0, cc1 = _plan_chunk_range(plan)
        keys = [need_chunks[i][c][1] for c in range(cc0, cc1 + 1)
                if c in need_chunks[i]]
        return hashlib.sha1(
            ("chunk|" + "|".join(keys)).encode()).hexdigest()[:16]

    def _build_mezz(i: int) -> None:
        mp, wp, _key, pieces = mezzs[i]
        total_samples = sum(x["a1"] - x["a0"] for x in pieces)
        if _seg_ok(mp) and _wav_ok(wp, total_samples):
            log(f"render: mezzanine cached (angle {i})")
            return
        mp.unlink(missing_ok=True)
        wp.unlink(missing_ok=True)
        tmp_mp = mp.with_name(mp.stem + ".part.mp4")
        tmp_wp = wp.with_name(wp.stem + ".part.wav")
        tmp_mp.unlink(missing_ok=True)
        tmp_wp.unlink(missing_ok=True)
        cmd = mezz_cmd(videos[i], pieces, str(tmp_mp), str(tmp_wp),
                       threads=mezz_threads)
        _run_progress(cmd, total_samples / SAMPLE_RATE,
                      f"render: mezzanine angle {i}", log)
        if not _seg_ok(tmp_mp) or not _wav_ok(tmp_wp, total_samples):
            raise RuntimeError(f"render: mezzanine angle {i} invalid")
        os.replace(tmp_mp, mp)
        os.replace(tmp_wp, wp)
        log(f"render: mezzanine built (angle {i})")

    def _build_wav(i: int) -> None:
        _mp, wp, _key, pieces = mezzs[i]
        total_samples = sum(x["a1"] - x["a0"] for x in pieces)
        if _wav_ok(wp, total_samples):
            log(f"render: audio cached (angle {i})")
            return
        wp.unlink(missing_ok=True)
        tmp_wp = wp.with_name(wp.stem + ".part.wav")
        tmp_wp.unlink(missing_ok=True)
        cmd = mezz_cmd(videos[i], pieces, None, str(tmp_wp),
                       threads=mezz_threads, video_out=False)
        _run_progress(cmd, total_samples / SAMPLE_RATE,
                      f"render: audio angle {i}", log)
        if not _wav_ok(tmp_wp, total_samples):
            raise RuntimeError(f"render: audio angle {i} invalid")
        os.replace(tmp_wp, wp)

    def _build_chunk(i: int, c: int) -> None:
        mp, _k = need_chunks[i][c]
        if _seg_ok(mp):
            return
        tmp = mp.with_name(mp.stem + ".part.mp4")
        tmp.unlink(missing_ok=True)
        cmd = mezz_cmd(videos[i], chunk_pieces(mezzs[i][3], c, ch),
                       str(tmp), None, threads=1, audio_out=False)
        run(cmd, log)
        if not _seg_ok(tmp):
            raise RuntimeError(f"render: chunk {c} angle {i} invalid")
        os.replace(tmp, mp)

    if chunk_mode:
        with ThreadPoolExecutor(max_workers=max(1, n_mezz)) as ex:
            list(ex.map(_build_wav, mezzs))
        jobs = [(i, c) for i, cs in need_chunks.items() for c in cs
                if not _seg_ok(cs[c][0])]
        built = 0
        with ThreadPoolExecutor(max_workers=max(1, int(
                os.environ.get("HL_RENDER_CHUNK_WORKERS", "4")))) as ex:
            futs = [ex.submit(_build_chunk, i, c) for i, c in jobs]
            for fut in futs:
                fut.result()
                built += 1
                if built % 20 == 0 or built == len(jobs):
                    log(f"render: chunks built {built}/{len(jobs)} "
                        + " ".join(
                            f"(angle {i}: {len(cs)} used of {n_chunks})"
                            for i, cs in sorted(need_chunks.items())))
        if not jobs:
            log("render: chunks built 0/0")
    else:
        with ThreadPoolExecutor(max_workers=max(1, n_mezz)) as ex:
            list(ex.map(_build_mezz, mezzs))
    live = {f.name for mp, wp, *_ in mezzs.values() for f in (mp, wp)}
    for f in mezz_dir.iterdir():
        if f.is_file() and f.name not in live and (
                f.suffix in (".mp4", ".wav") or ".part." in f.name):
            f.unlink()
    # keep every chunk still matching the current pieces of any angle,
    # needed by this cut or not: a later recut moving back to an earlier
    # camera choice reuses them
    live_chunks = set()
    for i, (_mp, _wp, _key, pieces) in mezzs.items():
        for c in range(n_chunks):
            sliced = chunk_pieces(pieces, c, ch)
            if sliced:
                live_chunks.add(f"{i}_{chunk_key(i, sliced)}.mp4")
    for f in chunk_dir.iterdir():
        if f.is_file() and (".part." in f.name or (
                f.suffix == ".mp4" and f.name not in live_chunks)):
            f.unlink()
    wavs = {i: val[1] for i, val in mezzs.items()}

    files, used, to_encode = [], set(), []
    reused = 0
    for plan in plans:
        angle = plan["angle"]
        req = plan["k1"] - plan["k0"]
        key = _seg_source_key(angle, plan)
        out = seg_dir / f"{seg_key(angle, plan['k0'], req, key, plan['speed'], plan['overlay'])}.mp4"
        used.add(out.name)
        if out.exists() and out.stat().st_size:
            reused += 1
        else:
            to_encode.append((plan, out))
        files.append(f"segs/{out.name}")

    def _seg_src(plan: dict, tag: str) -> tuple[Path, float]:
        """(source mp4, time offset of k0/source_t0 inside it)."""
        angle = plan["angle"]
        if not chunk_mode or _seg_ok(mezzs[angle][0]):
            return mezzs[angle][0], 0.0
        cc0, cc1 = _plan_chunk_range(plan)
        srcs = [str(need_chunks[angle][c][0])
                for c in range(cc0, cc1 + 1) if c in need_chunks[angle]]
        lst = seg_dir / f"srccat_{tag}.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in srcs))
        tmp = seg_dir / f"srctmp_{tag}.mp4"
        tmp.unlink(missing_ok=True)
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy",
             "-video_track_timescale", "15360", str(tmp)], log)
        lst.unlink(missing_ok=True)
        return tmp, cc0 * ch

    def _enc(plan: dict, out: Path) -> None:
        tmp = out.with_name(out.stem + ".part.mp4")
        tmp.unlink(missing_ok=True)
        angle = plan["angle"]
        req = plan["k1"] - plan["k0"]
        src, base = _seg_src(plan, out.stem)
        try:
            if plan["speed"] != 1.0 or plan["overlay"]:
                cmd = replay_cmd(str(src), plan["source_t0"] - base,
                                 plan["source_t1"] - plan["source_t0"],
                                 plan["speed"], str(tmp), plan["overlay"])
                run(cmd, log)
                # Exact output frame count enforced after replay video encode.
                fix = ["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
                       "-vf", f"fps=30,trim=end_frame={plan['frames']},setpts=PTS-STARTPTS",
                       "-frames:v", str(plan["frames"]), "-an", "-c:v", "libx264",
                       "-preset", PRESET, "-crf", CRF, "-pix_fmt", "yuv420p",
                       "-video_track_timescale", "15360", str(out)]
                run(fix, log)
                tmp.unlink(missing_ok=True)
            else:
                run(extract_cmd(str(src), plan["k0"] - base, req,
                                str(tmp)), log)
                os.replace(tmp, out)
        finally:
            if src != mezzs.get(angle, (None,))[0]:
                src.unlink(missing_ok=True)

    def _encode_all(jobs: list[tuple[dict, Path]], progress: bool) -> None:
        done = 0
        with ThreadPoolExecutor(max_workers=int(os.environ.get("HL_RENDER_WORKERS", "2"))) as ex:
            futures = [(p, ex.submit(_enc, p, o)) for p, o in jobs]
            for plan, fut in futures:
                fut.result()
                done += 1
                if progress and (plan["seg_index"] % 10 == 0 or done == len(jobs)):
                    log(f"render: seg {plan['seg_index']}/{len(segments)}")

    _encode_all(to_encode, True)
    log(f"render: reused {reused}/{reused + len(to_encode)} segments")
    bad = [seg_dir / name for name in used if not _seg_ok(seg_dir / name)]
    if bad:
        byname = {}
        for plan in plans:
            angle = plan["angle"]
            key = _seg_source_key(angle, plan)
            req = plan["k1"] - plan["k0"]
            name = f"{seg_key(angle, plan['k0'], req, key, plan['speed'], plan['overlay'])}.mp4"
            byname[name] = (plan, seg_dir / name)
        repairs = [byname[x.name] for x in bad if x.name in byname]
        if repairs:
            _encode_all(repairs, False)
        if any(not _seg_ok(x) for x in bad):
            raise RuntimeError("render: invalid video segment after re-encode")
    for f in seg_dir.glob("*.mp4"):
        if f.name not in used:
            f.unlink()

    audio_tmp = workdir / "audio.wav"
    _build_audio(plans, wavs, audio_tmp, log)
    lst = concat_file(files, workdir / "concat.txt")
    tmp_out = Path(out_path).with_name(Path(out_path).name + ".part.mp4")
    tmp_out.unlink(missing_ok=True)
    run(concat_cmd(lst, str(audio_tmp), str(tmp_out)), log)
    os.replace(tmp_out, out_path)
    audio_tmp.unlink(missing_ok=True)
    return out_path
