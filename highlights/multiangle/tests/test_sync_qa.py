"""sync_qa: render/map checks against the camera files."""

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from highlights.multiangle import render, sync_qa

FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
FFPROBE = subprocess.run(["which", "ffprobe"], capture_output=True).returncode == 0
W, H, FPS, DURATION = 160, 90, 30, 60
CLICKS = sorted(np.random.default_rng(7).uniform(1.0, 58.0, 80).tolist())
TM = [
    [{"file_lo": 0.0, "file_hi": DURATION, "offset": 0.0}],
    [{"file_lo": 0.0, "file_hi": DURATION, "offset": 2.5}],
]


def _source(path: Path, audio_shift: float = 0.0):
    raw = bytearray()
    for frame in range(DURATION * FPS):
        image = np.full((H, W, 3), 48, dtype=np.uint8)
        for bit in range(11):
            value = 235 if frame & (1 << bit) else 18
            x = 2 + bit * 12
            image[8:22, x:x + 10] = value
        raw.extend(image.tobytes())
    raw_path = path.with_suffix(".rgb")
    raw_path.write_bytes(raw)
    samples = np.zeros((DURATION * 48000, 2), dtype=np.int16)
    for t0 in CLICKS:
        t0 += audio_shift
        if t0 < 0:
            continue
        start = round(t0 * 48000)
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


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("qa_src")
    # angle 1's file clock runs 2.5 s behind: the same real-world click
    # sits 2.5 s earlier in its file (true timemap offset +2.5).
    return [str(_source(d / "a0.mkv")),
            str(_source(d / "a1.mkv", audio_shift=-2.5))]


@pytest.fixture(scope="module")
def cut(tmp_path_factory, sources):
    d = tmp_path_factory.mktemp("qa_render")
    segments = [
        {"t_start": float(i * 5), "t_end": float(i * 5 + 5),
         "angle": i % 2}
        for i in range(10)
    ]
    out = render.render(sources, TM, segments, 0.0, 50.0, d / "work",
                        d / "out.mp4", sources[0],
                        durations=[DURATION, DURATION], log=lambda *_: None)
    return str(out), segments


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_check_render_passes_exact_cut(cut, sources):
    out, segments = cut
    envs = [sync_qa.envelope(v) for v in sources]
    res = sync_qa.check_render(out, sources, TM, segments, 0.0, envs=envs)
    assert res["ok"] is True
    assert res["verified"] is True, res


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_check_render_catches_audio_delay(cut, sources, tmp_path):
    out, segments = cut
    bad = tmp_path / "bad_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", out, "-c:v", "copy",
         "-af", "adelay=500|500", "-c:a", "aac", str(bad)], check=True)
    envs = [sync_qa.envelope(v) for v in sources]
    res = sync_qa.check_render(str(bad), sources, TM, segments, 0.0,
                               envs=envs)
    assert res["ok"] is False
    assert res["audio"]["ok"] is False


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_check_render_catches_video_shift(cut, sources, tmp_path):
    out, segments = cut
    bad = tmp_path / "bad_video.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", out,
         "-vf", "tpad=start_duration=1:start_mode=clone",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
         "-pix_fmt", "yuv420p", "-c:a", "copy", str(bad)], check=True)
    envs = [sync_qa.envelope(v) for v in sources]
    res = sync_qa.check_render(str(bad), sources, TM, segments, 0.0,
                               envs=envs)
    assert res["ok"] is False
    assert res["video"]["ok"] is False


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_check_map_flags_wrong_offset(sources):
    good = sync_qa.check_map(sources, TM, 0.0, 60.0, step_s=6.0, win_s=4.0)
    assert 1 not in good["bad_angles"]
    wrong = [TM[0],
             [{"file_lo": 0.0, "file_hi": DURATION, "offset": 3.5}]]
    res = sync_qa.check_map(sources, wrong, 0.0, 60.0,
                            step_s=6.0, win_s=4.0)
    assert res["bad_angles"] == [1]


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_check_timeline(tmp_path):
    good = tmp_path / "good.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=160x90:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(good)], check=True)
    res = sync_qa.check_timeline(str(good), 60)
    assert res["ok"] is True
    assert res["frames"] == 60
    # frame count mismatch -> False
    assert sync_qa.check_timeline(str(good), 61)["ok"] is False
    # deliberate pts corruption: re-timestamped at 15 fps spacing
    gap = tmp_path / "gap.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(good),
         "-vf", "fps=15", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "copy", str(gap)], check=True)
    res = sync_qa.check_timeline(str(gap), 60)
    assert res["ok"] is False
    assert res["bad_gaps"] >= 1


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_check_render_catches_wrong_segment_content(cut, sources, tmp_path):
    """Video seq check: a cut whose one segment shows the wrong file time
    (shifted >=1 s) must fail even though timestamps are perfect."""
    out, segments = cut
    bad = tmp_path / "bad_content.mp4"
    # shift the whole cut's picture by +1 s: every sampled segment then
    # shows content 1 s earlier than the map expects
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", out, "-ss", "1.0",
         "-i", out, "-filter_complex",
         "[0:v][1:v]overlay=0:0:eof_action=pass[v]",
         "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset",
         "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "copy",
         str(bad)], check=True)
    envs = [sync_qa.envelope(v) for v in sources]
    res = sync_qa.check_render(str(bad), sources, TM, segments, 0.0,
                               envs=envs)
    assert res["ok"] is False


def test_gray_frames_tonemaps_hdr_canvas_only(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # enough gray pixels for reshape (frames may be empty)
        return type("P", (),
                    {"stdout": np.zeros(3 * sync_qa.W * sync_qa.H,
                                        np.uint8).tobytes(),
                     "returncode": 0})()

    from highlights.multiangle import color
    monkeypatch.setattr(color, "is_hdr", lambda p: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    sync_qa._gray_frames("v.mp4", 0.0, 0.1, canvas=True)
    vf = calls[-1][calls[-1].index("-vf") + 1]
    assert "zscale=t=linear:npl=203" in vf and "tonemap=mobius" in vf
    sync_qa._gray_frames("v.mp4", 0.0, 0.1, canvas=False)
    vf = calls[-1][calls[-1].index("-vf") + 1]
    assert "zscale" not in vf
    monkeypatch.setattr(color, "is_hdr", lambda p: False)
    sync_qa._gray_frames("v.mp4", 0.0, 0.1, canvas=True)
    vf = calls[-1][calls[-1].index("-vf") + 1]
    assert "zscale" not in vf


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")
def test_gray_frames_matches_tonemapped_mezz(tmp_path):
    """PQ source through mezz_cmd, then _gray_frames(canvas=True) on the
    raw PQ source must return ~the same luma as the tone-mapped mezz —
    the comparison picture checks do."""
    src = tmp_path / "hdr.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=duration=2:size=320x180:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-pix_fmt", "yuv420p10le", "-color_trc", "smpte2084",
         "-color_primaries", "bt2020", "-colorspace", "bt2020nc",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", str(src)], check=True)
    from highlights.multiangle.color import is_hdr
    is_hdr.cache_clear()
    pieces = [{"n0": 0, "n1": 60, "frames": 60,
               "a0": 0, "a1": 96000, "a_start": 0.0,
               "file_start": 0.0, "real": True}]
    mezz = tmp_path / "mezz.mp4"
    r = subprocess.run(render.mezz_cmd(str(src), pieces, str(mezz),
                                       str(tmp_path / "mezz.wav")),
                     capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    src_g = sync_qa._gray_frames(str(src), 0.0, 0.5, canvas=True)
    mezz_g = sync_qa._gray_frames(str(mezz), 0.0, 0.5, canvas=False)
    n = min(len(src_g), len(mezz_g))
    assert n > 0
    assert np.abs(src_g[:n] - mezz_g[:n]).mean() < 8.0
