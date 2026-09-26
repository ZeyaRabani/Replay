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


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg missing")
def test_segment_cache_reuse_and_prune(tmp_path):
    """Second render of the same segments reuses all cached files; a
    changed plan encodes anew and stale cache entries are pruned."""
    v0 = _mkvideo(tmp_path / "a0.mp4", 10.0)
    v1 = _mkvideo(tmp_path / "a1.mp4", 10.0)
    segs = [{"t_start": 0.0, "t_end": 3.0, "angle": 0},
            {"t_start": 3.0, "t_end": 5.0, "angle": 1}]
    logs: list[str] = []
    kwargs = dict(videos=[str(v0), str(v1)], offsets=[0.0, 0.0],
                  union_lo=0.0, union_hi=5.0, ref_video=str(v0),
                  log=lambda m, *a: logs.append(str(m)))
    work = tmp_path / "work"
    render.render(segments=segs, workdir=work,
                  out_path=tmp_path / "out.mp4", **kwargs)
    assert any("reused 0/2" in m for m in logs)
    n_cached = len(list((work / "segs").glob("*.mp4")))

    logs.clear()
    render.render(segments=segs, workdir=work,
                  out_path=tmp_path / "out2.mp4", **kwargs)
    assert any("reused 2/2" in m for m in logs)

    # one changed segment -> 1 reused, 1 encoded, and the dropped
    # segment's cache file is pruned
    logs.clear()
    segs2 = [segs[0], {"t_start": 3.0, "t_end": 4.5, "angle": 1}]
    render.render(segments=segs2, workdir=work,
                  out_path=tmp_path / "out3.mp4", **kwargs)
    assert any("reused 1/2" in m for m in logs)
    assert len(list((work / "segs").glob("*.mp4"))) == n_cached - 1 + 1


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg missing")
def test_render_repairs_truncated_segment(tmp_path):
    """A cached segment that exists but has no streams (ffmpeg exited 0
    yet left a truncated file) must be re-encoded before concat."""
    v0 = _mkvideo(tmp_path / "a0.mp4", 8.0)
    segs = [{"t_start": 0.0, "t_end": 3.0, "angle": 0},
            {"t_start": 3.0, "t_end": 6.0, "angle": 0}]
    work = tmp_path / "work"
    logs: list[str] = []
    kw = dict(videos=[str(v0)], offsets=[0.0], union_lo=0.0, union_hi=6.0,
              ref_video=str(v0), log=lambda m, *a: logs.append(str(m)))
    render.render(segments=segs, workdir=work,
                  out_path=tmp_path / "out.mp4", **kw)
    # corrupt the first segment the way the real failure looked:
    # a small header-only file with zero streams
    bad = sorted((work / "segs").glob("*.mp4"))[0]
    bad.write_bytes(b"\x00" * 261)
    logs.clear()
    render.render(segments=segs, workdir=work,
                  out_path=tmp_path / "out2.mp4", **kw)
    assert any("invalid segment" in m for m in logs)
    assert render._seg_ok(bad)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(tmp_path / "out2.mp4")],
        capture_output=True, text=True, check=True).stdout)
    assert 5.0 <= float(info["format"]["duration"]) <= 7.0


def test_render_fails_loud_when_segment_stays_bad(tmp_path, monkeypatch):
    """If a re-encoded segment is still streamless, raise before concat
    instead of handing the concat demuxer an empty file."""
    monkeypatch.setattr(render, "run",
                        lambda cmd, log=None: Path(cmd[-1]).write_bytes(b"junk"))
    monkeypatch.setattr(render, "_run_progress",
                        lambda cmd, dur_s, tag, log=None:
                        Path(cmd[-1]).write_bytes(b"junk"))
    monkeypatch.setattr(render, "_seg_ok",
                        lambda p: "mezz" in str(p))
    segs = [{"t_start": 0.0, "t_end": 3.0, "angle": 0}]
    with pytest.raises(RuntimeError, match="still invalid"):
        render.render(videos=["v.mp4"], offsets=[0.0], segments=segs,
                      union_lo=0.0, union_hi=3.0, workdir=tmp_path / "w",
                      out_path=tmp_path / "o.mp4", ref_video="v.mp4",
                      log=lambda *a: None)
    assert not (tmp_path / "o.mp4").exists()


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg missing")
def test_mezzanine_render_exact_frames(tmp_path):
    """20 s source -> mezzanine -> 3 extracted segments: cut boundaries
    land on the 1 s keyframe grid, so total duration and frame count are
    exact."""
    v0 = _mkvideo(tmp_path / "a0.mp4", 20.0, size="640x360")
    segs = [{"t_start": 0.0, "t_end": 5.0, "angle": 0},
            {"t_start": 7.0, "t_end": 12.0, "angle": 0},
            {"t_start": 14.0, "t_end": 19.0, "angle": 0}]
    logs: list[str] = []
    render.render([str(v0)], [0.0], segs, 0.0, 19.0, tmp_path / "work",
                  tmp_path / "out.mp4", str(v0),
                  durations=[20.0], log=lambda m, *a: logs.append(str(m)))
    assert any("mezzanine" in m for m in logs)
    total = sum(s["t_end"] - s["t_start"] for s in segs)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,nb_frames", "-of", "json",
         str(tmp_path / "out.mp4")],
        capture_output=True, text=True, check=True).stdout)
    assert abs(float(info["format"]["duration"]) - total) <= 0.1
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(v["nb_frames"]) == 30 * total
