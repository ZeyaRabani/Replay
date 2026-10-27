"""Chunked-mezz rendering: piece slicing, mode split, mode parity."""

import json
import subprocess
import wave
from pathlib import Path

import pytest

from highlights.multiangle import render

FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
FPS = render.FPS


def _mkvideo(path: Path, seconds: float = 8.0, size: str = "320x180",
             freq: int = 440) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size={size}:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)], check=True)
    return path


def _probe(path: Path) -> dict:
    return json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,nb_frames,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True,
         check=True).stdout)


def _frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def _pcm(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", "pipe:1"],
        capture_output=True, check=True).stdout


def _psnr(a: Path, b: Path) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(a), "-i", str(b),
         "-lavfi", "psnr", "-f", "null", "-"],
        capture_output=True, text=True)
    line = [l for l in proc.stderr.splitlines() if "average:" in l][-1]
    return float(line.split("average:", 1)[1].split()[0])


def test_chunk_pieces_rebuilds_full_mapping():
    pieces = [
        {"n0": 0, "n1": 250, "frames": 250, "a0": 0, "a1": 250 * 1600,
         "a_start": 1.0, "file_start": 1.0, "real": True, "offset": 1.0},
        {"n0": 250, "n1": 350, "frames": 100, "a0": 250 * 1600,
         "a1": 350 * 1600, "a_start": None, "file_start": None,
         "real": False, "offset": None},
        {"n0": 350, "n1": 700, "frames": 350, "a0": 350 * 1600,
         "a1": 700 * 1600, "a_start": 4.0, "file_start": 5.5,
         "real": True, "offset": 0.0},
    ]
    ch = 4  # 4 s chunks: 0-120, 120-240, ..., short last 480-700
    rebuilt: dict[int, tuple[bool, float | None]] = {}
    for c in range(-(-700 // (ch * FPS))):
        for p in render.chunk_pieces(pieces, c, ch):
            for n in range(c * ch * FPS + p["n0"],
                           c * ch * FPS + p["n1"]):
                fs = (None if not p["real"]
                      else p["file_start"]
                      + (n - c * ch * FPS - p["n0"]) / FPS)
                assert n not in rebuilt
                rebuilt[n] = (p["real"], fs)
    assert len(rebuilt) == 700
    for n in range(700):
        src = next(p for p in pieces if p["n0"] <= n < p["n1"])
        real, fs = rebuilt[n]
        assert real == src["real"]
        if real:
            exp = src["file_start"] + (n - src["n0"]) / FPS
            assert fs == pytest.approx(exp)


def test_chunk_key_is_position_independent():
    # black (non-real) content is identical wherever it sits -> shared file
    pieces = [{"n0": 0, "n1": 600, "frames": 600, "a0": 0,
               "a1": 600 * 1600, "a_start": None, "file_start": None,
               "real": False, "offset": None}]
    a = render.chunk_key(0, render.chunk_pieces(pieces, 0, 10))
    b = render.chunk_key(0, render.chunk_pieces(pieces, 1, 10))
    assert a == b
    c = render.chunk_key(0, render.chunk_pieces(pieces, 0, 5))
    assert a != c
    # real content maps to different file times -> different key
    real = [dict(pieces[0], real=True, file_start=0.0, a_start=0.0,
                 offset=0.0)]
    assert render.chunk_key(0, render.chunk_pieces(real, 0, 10)) != \
        render.chunk_key(0, render.chunk_pieces(real, 1, 10))


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_mezz_cmd_mode_split(tmp_path):
    src = _mkvideo(tmp_path / "src.mp4", 6)
    pieces = [{"n0": 0, "n1": 180, "frames": 180, "a0": 0,
               "a1": 180 * 1600, "a_start": 0.0, "file_start": 0.0,
               "real": True, "offset": 0.0}]

    wav = tmp_path / "a.wav"
    cmd = render.mezz_cmd(str(src), pieces, None, str(wav), threads=1,
                          video_out=False)
    assert "m.mp4" not in cmd and "[vout]" not in " ".join(cmd)
    assert not any(x.endswith(".mp4") and x != str(src) for x in cmd)
    render.run(cmd, lambda *_: None)
    with wave.open(str(wav), "rb") as wf:
        assert wf.getnframes() == 180 * 1600

    mp4 = tmp_path / "v.mp4"
    cmd = render.mezz_cmd(str(src), pieces, str(mp4), None, threads=1,
                          audio_out=False)
    assert not any("a:0" in x or "pcm" in x or x.endswith(".wav")
                   for x in cmd)
    render.run(cmd, lambda *_: None)
    info = _probe(mp4)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 180
    assert len(info["streams"]) == 1  # video only


SEGS = [{"t_start": 0, "t_end": 7.2, "angle": 0},
        {"t_start": 7.2, "t_end": 16.8, "angle": 1},
        {"t_start": 16.8, "t_end": 22.3, "angle": 0},
        {"t_start": 22.3, "t_end": 30.6, "angle": 1},
        {"t_start": 30.6, "t_end": 40.0, "angle": 0}]


def _render_env(mode: str, monkeypatch):
    monkeypatch.setenv("HL_RENDER_MODE", mode)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_chunk_mode_matches_mezz_mode(tmp_path, monkeypatch):
    # two 60 s sources, a1 offset -12 s -> 12 s black lead-in for a1
    a0 = _mkvideo(tmp_path / "a0.mp4", 60, freq=330)
    a1 = _mkvideo(tmp_path / "a1.mp4", 60, size="640x360", freq=550)
    kw = dict(videos=[str(a0), str(a1)], tm=[0.0, -12.0],
              union_lo=0, union_hi=40, durations=[60, 60],
              ref_video=str(a0), log=lambda *_: None)

    _render_env("mezz", monkeypatch)
    out_m = render.render(segments=SEGS, workdir=tmp_path / "wm",
                          out_path=tmp_path / "om.mp4", **kw)
    _render_env("chunk", monkeypatch)
    monkeypatch.setenv("HL_RENDER_CHUNK", "10")
    out_c = render.render(segments=SEGS, workdir=tmp_path / "wc",
                          out_path=tmp_path / "oc.mp4", **kw)

    assert _frames(out_m) == _frames(out_c) == 40 * FPS
    dm = float(_probe(out_m)["format"]["duration"])
    dc = float(_probe(out_c)["format"]["duration"])
    assert dm == pytest.approx(dc, abs=0.05)
    assert _pcm(out_m) == _pcm(out_c)          # audio bit-identical
    assert _psnr(out_m, out_c) > 40


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_chunk_mode_encodes_only_used_chunks(tmp_path, monkeypatch):
    a0 = _mkvideo(tmp_path / "a0.mp4", 60, freq=330)
    a1 = _mkvideo(tmp_path / "a1.mp4", 60, freq=550)
    # angle 1 used for 10 s of a 40 s cut -> <= 2 chunks built for it
    segs = [{"t_start": 0, "t_end": 33, "angle": 0},
            {"t_start": 33, "t_end": 40, "angle": 1}]
    monkeypatch.setenv("HL_RENDER_MODE", "chunk")
    logs = []
    render.render(videos=[str(a0), str(a1)], tm=[0.0, 0.0],
                  segments=segs, union_lo=0, union_hi=40,
                  durations=[60, 60], workdir=tmp_path / "w",
                  out_path=tmp_path / "o.mp4", ref_video=str(a0),
                  log=logs.append)
    chunks = sorted((tmp_path / "w" / "mezz" / "chunks").glob("*.mp4"))
    a1_chunks = [c for c in chunks if c.name.startswith("1_")]
    assert len(a1_chunks) == 1        # seconds 30-40 -> chunk 3 only
    a0_chunks = [c for c in chunks if c.name.startswith("0_")]
    assert len(a0_chunks) == 4        # seconds 0-40 -> chunks 0..3


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_chunk_recut_reuses_untouched_chunks(tmp_path, monkeypatch):
    a0 = _mkvideo(tmp_path / "a0.mp4", 60, freq=330)
    a1 = _mkvideo(tmp_path / "a1.mp4", 60, freq=550)
    monkeypatch.setenv("HL_RENDER_MODE", "chunk")
    kw = dict(videos=[str(a0), str(a1)], tm=[0.0, 0.0],
              union_lo=0, union_hi=40, durations=[60, 60],
              ref_video=str(a0))
    work = tmp_path / "w"
    render.render(segments=SEGS, workdir=work,
                  out_path=tmp_path / "o1.mp4", log=lambda *_: None, **kw)
    before = {p.name: p.stat().st_mtime_ns
              for p in (work / "mezz" / "chunks").glob("*.mp4")}
    # change only the last segment's angle: it must not rebuild old chunks
    segs2 = SEGS[:-1] + [{"t_start": 30.6, "t_end": 40.0, "angle": 1}]
    logs = []
    render.render(segments=segs2, workdir=work,
                  out_path=tmp_path / "o2.mp4", log=logs.append, **kw)
    after = {p.name: p.stat().st_mtime_ns
             for p in (work / "mezz" / "chunks").glob("*.mp4")}
    new_or_rebuilt = {n for n in after
                      if before.get(n) != after[n]}
    # segment 30.6-40 on angle1 touches chunks 30-40 (a1 chunk 3) only;
    # no other chunk file may be replaced
    assert all(n.startswith("1_") for n in new_or_rebuilt)
    # earlier identical chunk content is reused, not re-encoded
    kept = [m for m in logs if "chunks built" in m]
    assert kept and int(kept[-1].split("built ")[1].split("/")[0]) <= 1
