"""Mezzanine start-time regression: the mezz video must start at pts 0 so
its forced 1 s keyframe grid sits on integer seconds — extract_cmd
stream-copies from -ss k0 and snaps to the previous keyframe otherwise.
Reproduces with a VP9 source whose video track starts at ~0.04 (like the
4/9 match sources): after -ss t0 the first decoded frame then lands a
frame after t0, giving pts 0.033. (25 fps libvpx-vp9 + itsoffset 0.037,
t0=1.41 reproduced the bug; plain h264 sources did not.)"""

import json
import subprocess

import pytest

from highlights.multiangle.render import concat_cmd, concat_file, extract_cmd, mezz_cmd, run

FFMPEG = subprocess.run(
    ["which", "ffmpeg"], capture_output=True).returncode == 0
FFPROBE = subprocess.run(
    ["which", "ffprobe"], capture_output=True).returncode == 0


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    vs = next(s for s in d["streams"] if s["codec_type"] == "video")
    n = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())
    return vs, float(d["format"]["duration"]), n


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg missing")
def test_mezzanine_starts_at_zero(tmp_path):
    vsrc = tmp_path / "vsrc.webm"
    src = tmp_path / "src.webm"
    # VP9 video whose track pts starts at ~0.04 (4/9-style source)
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=rate=25:size=640x360", "-t", "12",
         "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "1M",
         "-cpu-used", "8", "-deadline", "realtime", str(vsrc)])
    # no audio track — with a stream starting at 0 alongside, the muxer's
    # baseline normalises the mezz start anyway and the bug doesn't show
    run(["ffmpeg", "-y", "-v", "error", "-itsoffset", "0.037", "-i",
         str(vsrc), "-map", "0:v", "-c:v", "copy", str(src)])
    assert float(_probe(src)[0]["start_time"]) > 0.02

    mezz = tmp_path / "mezz.mp4"
    cmd = [a for a in mezz_cmd(str(src), 1.41, 9.41, str(mezz))
           if a not in ("-nostats", "-progress", "pipe:1")]
    run(cmd)
    vs, _, _ = _probe(mezz)
    assert float(vs["start_time"]) == pytest.approx(0.0, abs=1e-3)

    segs, want = [], [(2.0, 3.0), (5.0, 2.0)]
    for i, (rel, dur) in enumerate(want):
        seg = tmp_path / f"seg{i}.mp4"
        run(extract_cmd(str(mezz), rel, dur, str(seg)))
        _, sdur, nframes = _probe(seg)
        req = round(rel + dur) - round(rel)
        assert sdur == pytest.approx(req, abs=0.05)
        assert nframes == req * 30
        segs.append(str(seg))

    lst = concat_file(segs, tmp_path / "list.txt")
    final = tmp_path / "final.mp4"
    run(concat_cmd(lst, str(final)))
    _, fdur, fframes = _probe(final)
    assert fdur == pytest.approx(sum(w[1] for w in want), abs=0.1)
    assert fframes == sum(w[1] for w in want) * 30
