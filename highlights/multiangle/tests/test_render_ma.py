"""Render: command building + 3-segment concat smoke with tiny lavfi videos."""

import json
import subprocess
from pathlib import Path

import pytest

from highlights.multiangle import render


def _mkvideo(path: Path, seconds: float = 5.0, size: str = "320x180",
             color: str = "red") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size={size}:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)], check=True)
    return path


def test_segment_cmd_shape():
    cmd = render.segment_cmd("/v/a1.mp4", 12.5, 8.0, "/tmp/seg.mp4")
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "12.500"
    vf = cmd[cmd.index("-vf") + 1]
    assert "scale=1920:1080" in vf and "pad=1920:1080" in vf and "fps=30" in vf
    assert cmd[cmd.index("-crf") + 1] == render.CRF


def test_concat_file(tmp_path):
    lst = render.concat_file(["0000.mp4", "0001.mp4"], tmp_path / "c.txt")
    assert lst.read_text() == "file '0000.mp4'\nfile '0001.mp4'\n"


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg missing")
def test_three_segment_render(tmp_path):
    """Two 5 s clips, director picks alternating angles -> real concat + audio."""
    v0 = _mkvideo(tmp_path / "a0.mp4", 10.0, color="red")
    v1 = _mkvideo(tmp_path / "a1.mp4", 10.0, color="blue")
    segs = [{"t_start": 0.0, "t_end": 3.0, "angle": 0},
            {"t_start": 3.0, "t_end": 5.0, "angle": 1},
            {"t_start": 5.0, "t_end": 8.0, "angle": 0}]
    out = render.render([str(v0), str(v1)], [0.0, 0.0], segs, 0.0, 8.0,
                        tmp_path / "work", tmp_path / "out.mp4", str(v0),
                        log=lambda *a: None)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=width,height,codec_type", "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)
    dur = float(info["format"]["duration"])
    assert 7.0 <= dur <= 9.0
    streams = {s["codec_type"] for s in info["streams"]}
    assert streams == {"video", "audio"}
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1920, 1080)
