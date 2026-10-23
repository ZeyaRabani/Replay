"""End-to-end exact sync across fractional and piecewise camera offsets."""

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from highlights.multiangle import render
from highlights.multiangle.syncmap import shared_to_file

FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
FFPROBE = subprocess.run(["which", "ffprobe"], capture_output=True).returncode == 0
W, H, FPS, DURATION = 160, 90, 30, 24
UNION_LO, UNION_HI = 0.0, 20.0
TM = [
    [{"file_lo": 0.0, "file_hi": DURATION, "offset": 0.0}],
    [{"file_lo": 0.0, "file_hi": 10.0, "offset": 3.37},
     {"file_lo": 10.0, "file_hi": DURATION, "offset": 2.77}],
]


def _source(path: Path):
    raw = bytearray()
    for frame in range(DURATION * FPS):
        image = np.full((H, W, 3), 48, dtype=np.uint8)
        for bit in range(10):
            value = 235 if frame & (1 << bit) else 18
            x = 3 + bit * 13
            image[8:22, x:x + 10] = value
        raw.extend(image.tobytes())
    raw_path = path.with_suffix(".rgb")
    raw_path.write_bytes(raw)
    samples = np.zeros((DURATION * 48000, 2), dtype=np.int16)
    phase = 0.25
    for sec in range(DURATION):
        start = round((sec + phase) * 48000)
        t = np.arange(480, dtype=np.float64) / 48000
        click = (24000 * np.sin(2 * np.pi * 2000 * t)).astype(np.int16)
        samples[start:start + len(click), :] = click[:, None]
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(samples.astype("<i2").tobytes())
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format",
         "rgb24", "-video_size", f"{W}x{H}", "-framerate", str(FPS),
         "-i", str(raw_path), "-i", str(wav), "-map", "0:v", "-map", "1:a",
         "-c:v", "ffv1", "-level", "3", "-c:a", "pcm_s16le", str(path)],
        check=True)
    raw_path.unlink()
    wav.unlink()
    return path


def _decode_video(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, H, W, 3)


def _barcode(frame: np.ndarray) -> int:
    value = 0
    for bit in range(10):
        x = 3 + bit * 13
        if float(frame[8:22, x:x + 10].mean()) > 120:
            value |= 1 << bit
    return value


def _decode_audio(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
         "-ac", "2", "-ar", "48000", "pipe:1"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2)


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_render_sync_piecewise_fractional_timemap(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "MEZZ_SIZE", "160x90")
    monkeypatch.setattr(
        render, "CANVAS",
        "scale=160:90:force_original_aspect_ratio=decrease,"
        "pad=160:90:(ow-iw)/2:(oh-ih)/2")
    videos = [_source(tmp_path / "a0.mkv"), _source(tmp_path / "a1.mkv")]
    segments = [
        {"t_start": 0.0, "t_end": 1.4, "angle": 0},
        {"t_start": 1.4, "t_end": 4.4, "angle": 1},
        {"t_start": 4.4, "t_end": 7.0, "angle": 0},
        {"t_start": 7.0, "t_end": 10.4, "angle": 1},
        {"t_start": 10.4, "t_end": 11.4, "angle": 0},
        {"t_start": 11.4, "t_end": 14.0, "angle": 1},
        {"t_start": 14.0, "t_end": 20.0, "angle": 0},
    ]
    plans = render.plan_segments(segments)
    out = render.render([str(v) for v in videos], TM, segments,
                        UNION_LO, UNION_HI, tmp_path / "work",
                        tmp_path / "out.mp4", str(videos[0]),
                        durations=[DURATION, DURATION], log=lambda *_: None)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,nb_frames", "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video_stream["nb_frames"]) == 30 * 20
    assert abs(float(info["format"]["duration"]) - 20) <= 1024 / 48000

    frames = _decode_video(out)
    elapsed = 0
    checked = []
    for plan in plans:
        # Check at least one content frame per director segment. Skip the
        # first mapped angle-1 portion before that camera enters coverage.
        candidates = [plan["k0"] + 0.5]
        if plan["angle"] == 1 and plan["k0"] == 11:
            candidates.append(13.5)  # after the piecewise timemap jump
        if plan["angle"] == 1 and plan["k0"] == 1:
            candidates = [3.5]
        for tau in candidates:
            file_t = shared_to_file(TM, plan["angle"], UNION_LO + tau)
            if file_t is None:
                continue
            idx = round(tau * FPS)
            got = _barcode(frames[idx])
            want = round(file_t * FPS)
            assert abs(got - want) <= 1, (tau, got, want)
            checked.append(tau)
        elapsed += plan["frames"]
    assert len(checked) >= len(plans)

    # Each source has a 10 ms 2 kHz click at file time n+0.25. Convert
    # those events through each active output-time map, then check the
    # final AAC decode around every expected event across the whole cut.
    pcm = _decode_audio(out)
    expected = []
    for plan in plans:
        for piece in render.mezz_pieces(
                plan["angle"], TM, UNION_LO, UNION_HI,
                [DURATION, DURATION]):
            if not piece["real"]:
                continue
            lo = max(plan["k0"], piece["n0"] / FPS)
            hi = min(plan["k1"], piece["n1"] / FPS)
            for n in range(DURATION):
                tau = n + 0.25 + piece["offset"] - UNION_LO
                if lo <= tau < hi and plan["k0"] <= tau < plan["k1"]:
                    expected.append(tau)
    assert expected
    # Localize each click by cross-correlating with the known 10 ms burst —
    # robust to AAC attack/decay smearing at codec block edges.
    template = np.sin(
        2 * np.pi * 2000 * np.arange(480) / 48000).astype(np.float64)
    for tau in expected:
        center = round(tau * 48000)
        left, right = max(0, center - 960), min(len(pcm), center + 960)
        mono = pcm[left:right, 0].astype(np.float64)
        assert np.abs(mono).max() > 5000, f"missing click at {tau:.6f}s"
        windows = np.lib.stride_tricks.sliding_window_view(mono, 480)
        start = left + int(np.argmax(np.abs(windows @ template)))
        assert abs(start / 48000 - tau) <= 0.003, (tau, start / 48000)
