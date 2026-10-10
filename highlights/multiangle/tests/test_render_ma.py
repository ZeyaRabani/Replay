"""Render: command building + 3-segment concat smoke with tiny lavfi videos."""

import hashlib
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


def test_replay_cmd_shape_and_normal_cache_key():
    cmd = render.replay_cmd(
        "source.mp4", 1.0, 2.0, 0.5, "replay.mp4", "REPLAY")
    vf = cmd[cmd.index("-vf") + 1]
    assert "setpts=PTS/0.5" in vf
    assert cmd[cmd.index("-af") + 1] == "atempo=0.5"
    assert cmd[cmd.index("-frames:v") + 1] == "120"
    assert ("drawtext=" in vf) == render._drawtext_available()
    old_raw = (f"1|2.000|3.000|{render.CANVAS}|{render.CRF}|"
               f"{render.PRESET}|mezz")
    assert render.seg_key(1, 2.0, 3.0, "mezz") == \
        hashlib.sha1(old_raw.encode()).hexdigest()[:16]
    assert render.seg_key(1, 2.0, 3.0, "mezz", 0.5, "REPLAY") != \
        render.seg_key(1, 2.0, 3.0, "mezz")


def test_plan_segments_prefers_live_source_bounds():
    segment = {
        "t_start": 22.0, "t_end": 26.0, "t_src_start": 5.0,
        "t_src_end": 7.0, "angle": 1, "speed": 0.5,
        "overlay": "REPLAY",
    }
    plan = render.plan_segments([segment], [0.0, 80.0], 100.0, 120.0,
                                [200.0, 200.0])[0]
    assert plan == {
        "seg_index": 0, "angle": 1, "t0": 22.0, "t1": 26.0,
        "t_file": 25.0, "dur": 2.0, "dur_out": 4.0,
        "speed": 0.5, "overlay": "REPLAY",
    }


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
    kwargs = dict(videos=[str(v0), str(v1)], tm=[0.0, 0.0],
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
    kw = dict(videos=[str(v0)], tm=[0.0], union_lo=0.0, union_hi=6.0,
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
        render.render(videos=["v.mp4"], tm=[0.0], segments=segs,
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


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg missing")
def test_slow_motion_replay_render(tmp_path):
    a0 = _mkvideo(tmp_path / "a0.mp4", 10.0)
    a1 = _mkvideo(tmp_path / "a1.mp4", 10.0, color="blue")
    segments = [
        {"t_start": 0.0, "t_end": 4.0, "t_src_start": 0.0,
         "t_src_end": 4.0, "angle": 0, "rule": "start", "speed": 1.0},
        {"t_start": 4.0, "t_end": 8.0, "t_src_start": 1.0,
         "t_src_end": 3.0, "angle": 1, "rule": "replay",
         "speed": 0.5, "overlay": "REPLAY"},
        {"t_start": 8.0, "t_end": 12.0, "t_src_start": 4.0,
         "t_src_end": 8.0, "angle": 1, "rule": "cluster", "speed": 1.0},
    ]
    out = render.render(
        [str(a0), str(a1)], [0.0, 0.0], segments, 0.0, 8.0,
        tmp_path / "work", tmp_path / "out.mp4", str(a0), durations=[10, 10],
        log=lambda *_args: None)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=width,height,codec_type,r_frame_rate",
         "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)
    assert abs(float(info["format"]["duration"]) - 12.0) <= 0.5
    streams = {stream["codec_type"] for stream in info["streams"]}
    assert streams == {"video", "audio"}
    video = next(stream for stream in info["streams"]
                 if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1920, 1080)
    assert video["r_frame_rate"] == "30/1"


def test_mezz_cmd_threads():
    cmd = render.mezz_cmd("/v/a.mp4", 0.0, 10.0, "/tmp/m.mp4", threads=1)
    assert cmd[cmd.index("-threads") + 1] == "1"
    cmd = render.mezz_cmd("/v/a.mp4", 0.0, 10.0, "/tmp/m.mp4")
    assert cmd[cmd.index("-threads") + 1] == "2"


def test_mezzanines_build_concurrently(tmp_path, monkeypatch):
    """3 mezzanines run at once, each getting cpu_count/3 ffmpeg threads."""
    import time

    monkeypatch.setattr(render.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(render, "_seg_ok", lambda f: Path(f).exists())
    monkeypatch.setattr(render, "run",
                        lambda cmd, log=print: Path(cmd[-1]).write_bytes(b"x"))
    starts = []

    def fake_progress(cmd, dur_s, tag, log=print):
        starts.append((time.monotonic(), tag))
        Path(cmd[-1]).write_bytes(b"x")
        time.sleep(0.15)

    monkeypatch.setattr(render, "_run_progress", fake_progress)
    videos = [str(_mkvideo(tmp_path / f"a{i}.mp4", 1.0)) for i in range(3)]
    render.render(videos, [0.0, 0.0, 0.0], [], 0.0, 1.0,
                  tmp_path / "work", tmp_path / "out.mp4", videos[0],
                  durations=[1.0, 1.0, 1.0], log=lambda m: None)
    mezz_starts = [t for t, tag in starts if "mezzanine angle" in tag]
    assert len(mezz_starts) == 3
    assert max(mezz_starts) - min(mezz_starts) < 0.15


def test_mezz_threads_value(tmp_path, monkeypatch):
    monkeypatch.setattr(render.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(render, "_seg_ok", lambda f: Path(f).exists())
    monkeypatch.setattr(render, "run",
                        lambda cmd, log=print: Path(cmd[-1]).write_bytes(b"x"))
    cmds = []

    def fake_progress(cmd, dur_s, tag, log=print):
        cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"x")

    monkeypatch.setattr(render, "_run_progress", fake_progress)
    videos = [str(_mkvideo(tmp_path / f"a{i}.mp4", 1.0)) for i in range(3)]
    render.render(videos, [0.0, 0.0, 0.0], [], 0.0, 1.0,
                  tmp_path / "work", tmp_path / "out.mp4", videos[0],
                  durations=[1.0, 1.0, 1.0], log=lambda m: None)
    assert len(cmds) == 3
    assert all(c[c.index("-threads") + 1] == "2" for c in cmds)


def test_extract_cmd_snaps_to_keyframe_grid(tmp_path):
    """Fractional boundaries snap to the 1 s keyframe grid; output length
    equals the snapped window (re-cut 1 s-short bug)."""
    mezz = tmp_path / "mezz.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=duration=45:size=320x180:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=45",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
         "-force_key_frames", "expr:gte(t,n_forced*1)",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", str(mezz)], check=True)
    for t_rel, want_start in ((10.2, 10.0), (10.9, 11.0)):
        out = tmp_path / f"cut_{t_rel}.mp4"
        cmd = render.extract_cmd(str(mezz), t_rel, 20.0, str(out))
        assert float(cmd[cmd.index("-ss") + 1]) == want_start
        subprocess.run(cmd, check=True)
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(out)], capture_output=True, text=True)
        want = round(t_rel + 20.0) - round(t_rel)
        assert abs(float(r.stdout.strip()) - want) < 0.05
