"""Render command, cache and output timing smoke tests."""

import json
import subprocess
from pathlib import Path

import pytest

from highlights.multiangle import render

FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


def _mkvideo(path: Path, seconds: float = 8.0, size: str = "320x180") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size={size}:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)], check=True)
    return path


def _probe(path: Path) -> dict:
    return json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,nb_frames,width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True,
        check=True).stdout)


def test_extract_and_mezz_command_shapes():
    cmd = render.extract_cmd("/v/a.mp4", 12, 8, "/tmp/seg.mp4")
    assert cmd[cmd.index("-ss") + 1] == "12.000"
    assert cmd[cmd.index("-frames:v") + 1] == "240"
    assert "-an" in cmd and "-c:v" in cmd
    pieces = [{"n0": 0, "n1": 240, "frames": 240,
               "a0": 0, "a1": 384000, "a_start": 1.25,
               "file_start": 1.25, "real": True}]
    cmd = render.mezz_cmd("/v/a.mp4", pieces, "m.mp4", "m.wav", threads=1)
    assert cmd[cmd.index("-threads") + 1] == "1"
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "fps=30" in graph and "trim=end_frame=240" in graph
    # /v/a.mp4 has no audio stream -> anullsrc piece audio input
    assert any("anullsrc" in a for a in cmd)
    assert "end_sample=384000" in graph
    assert cmd[cmd.index("-an") + 1] == "m.mp4"
    assert cmd[-1] == "m.wav"


def test_output_time_plan_snaps_shared_boundaries():
    plans = render.plan_segments([
        {"t_start": 0.49, "t_end": 1.51, "angle": 0},
        {"t_start": 1.51, "t_end": 3.49, "angle": 1},
        {"t_start": 3.49, "t_end": 4.51, "angle": 0},
    ])
    assert [(p["k0"], p["k1"], p["frames"]) for p in plans] == [
        (0, 2, 60), (2, 3, 30), (3, 5, 60)]
    assert sum(p["frames"] for p in plans) == (round(4.51) - round(0.49)) * 30
    assert render.plan_segments([
        {"t_start": 1.1, "t_end": 1.4, "angle": 0}]) == []


def test_seg_key_includes_otime_mezz_and_exact_range():
    k = render.seg_key(1, 5, 2, "abc")
    assert k != render.seg_key(1, 5, 2, "def")
    assert k != render.seg_key(1, 6, 2, "abc")
    assert k != render.seg_key(1, 5, 2, "abc", 0.5, "REPLAY")


def test_replay_cmd_is_video_only():
    cmd = render.replay_cmd("source.mp4", 1.0, 2.0, 0.5, "replay.mp4")
    assert "-an" in cmd
    assert "setpts=PTS/0.5" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-frames:v") + 1] == "120"


def test_concat_file_and_command(tmp_path):
    lst = render.concat_file(["a.mp4", "b.mp4"], tmp_path / "c.txt")
    assert lst.read_text() == "file 'a.mp4'\nfile 'b.mp4'\n"
    cmd = render.concat_cmd(lst, "audio.wav", "out.mp4")
    assert cmd[cmd.index("-i") + 1] == str(lst)
    assert cmd[cmd.index("-map") + 1] == "0:v"
    assert cmd[cmd.index("-map") + 3] == "1:a"


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_three_segment_render_is_frame_exact(tmp_path):
    v0 = _mkvideo(tmp_path / "a0.mp4", 10)
    v1 = _mkvideo(tmp_path / "a1.mp4", 10)
    segs = [{"t_start": 0, "t_end": 3, "angle": 0},
            {"t_start": 3, "t_end": 5, "angle": 1},
            {"t_start": 5, "t_end": 8, "angle": 0}]
    out = render.render([str(v0), str(v1)], [0.0, 0.0], segs, 0, 8,
                        tmp_path / "work", tmp_path / "out.mp4", str(v0),
                        log=lambda *_: None)
    info = _probe(out)
    assert abs(float(info["format"]["duration"]) - 8) < 0.1
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 8 * 30
    assert (video["width"], video["height"]) == (1920, 1080)
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert float(audio.get("duration", 8)) == pytest.approx(8, abs=0.03)
    assert not (tmp_path / "work" / "audio.wav").exists()


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_segment_cache_reuse_and_prune(tmp_path):
    v0, v1 = _mkvideo(tmp_path / "a0.mp4", 8), _mkvideo(tmp_path / "a1.mp4", 8)
    segs = [{"t_start": 0, "t_end": 3, "angle": 0},
            {"t_start": 3, "t_end": 5, "angle": 1}]
    logs = []
    kw = dict(videos=[str(v0), str(v1)], tm=[0, 0], union_lo=0,
              union_hi=5, ref_video=str(v0), log=logs.append)
    work = tmp_path / "work"
    render.render(segments=segs, workdir=work, out_path=tmp_path / "o1.mp4", **kw)
    assert any("reused 0/2" in m for m in logs)
    logs.clear()
    render.render(segments=segs, workdir=work, out_path=tmp_path / "o2.mp4", **kw)
    assert any("reused 2/2" in m for m in logs)
    logs.clear()
    render.render(segments=[segs[0], {"t_start": 3, "t_end": 4, "angle": 1}],
                  workdir=work, out_path=tmp_path / "o3.mp4", **kw)
    assert any("reused 1/2" in m for m in logs)
    assert len(list((work / "segs").glob("*.mp4"))) == 2


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_render_repairs_truncated_segment(tmp_path):
    v0 = _mkvideo(tmp_path / "a0.mp4", 8)
    segs = [{"t_start": 0, "t_end": 3, "angle": 0},
            {"t_start": 3, "t_end": 6, "angle": 0}]
    work = tmp_path / "work"
    kw = dict(videos=[str(v0)], tm=[0], union_lo=0, union_hi=6,
              ref_video=str(v0), log=lambda *_: None)
    render.render(segments=segs, workdir=work, out_path=tmp_path / "o1.mp4", **kw)
    bad = sorted((work / "segs").glob("*.mp4"))[0]
    bad.write_bytes(b"\0" * 261)
    logs = []
    render.render(segments=segs, workdir=work, out_path=tmp_path / "o2.mp4",
                  **{**kw, "log": logs.append})
    assert render._seg_ok(bad)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_slow_motion_replay_keeps_audio_and_video_length(tmp_path):
    a0, a1 = _mkvideo(tmp_path / "a0.mp4", 14), _mkvideo(tmp_path / "a1.mp4", 14)
    segments = [
        {"t_start": 0, "t_end": 4, "angle": 0, "speed": 1},
        {"t_start": 4, "t_end": 8, "t_src_start": 1, "t_src_end": 3,
         "angle": 1, "speed": 0.5, "overlay": "REPLAY"},
        {"t_start": 8, "t_end": 12, "t_src_start": 4, "t_src_end": 8,
         "angle": 1, "speed": 1},
    ]
    out = render.render([str(a0), str(a1)], [0, 0], segments, 0, 12,
                        tmp_path / "work", tmp_path / "out.mp4", str(a0),
                        durations=[14, 14], log=lambda *_: None)
    info = _probe(out)
    assert abs(float(info["format"]["duration"]) - 12) < 0.1
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 12 * 30


def test_mezz_key_changes_for_output_mapping():
    a = render.mezz_key(0, [{"n0": 0, "n1": 30, "frames": 30,
                             "file_start": 1.3, "real": True}])
    b = render.mezz_key(0, [{"n0": 0, "n1": 30, "frames": 30,
                             "file_start": 1.4, "real": True}])
    assert a != b
