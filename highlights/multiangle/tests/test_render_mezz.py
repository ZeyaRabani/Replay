"""Output-time mezzanine piece mapping and cache-key tests."""

import json
import subprocess
import wave

import pytest

from highlights.multiangle.render import mezz_key, mezz_pieces

FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


def test_mezz_pieces_fill_gaps_and_later_timemap_wins():
    tm = [[
        {"file_lo": 0.0, "file_hi": 10.0, "offset": 3.37},
        {"file_lo": 10.0, "file_hi": 24.0, "offset": 2.77},
    ]]
    pieces = mezz_pieces(0, tm, 0.0, 20.0, [24.0])
    # initial coverage gap, first mapping, then second mapping wins overlap
    assert not pieces[0]["real"]
    real = [p for p in pieces if p["real"]]
    assert real[0]["file_start"] == 0
    assert real[1]["file_start"] == 10
    # last piece resumes mid-file at shared 13.37 -> file 10.5967
    assert real[-1]["file_start"] == pytest.approx(10.596666666666668)
    assert sum(p["frames"] for p in pieces) == 20 * 30
    assert next(p["n0"] for p in pieces) == 0
    assert pieces[-1]["n1"] == 600


def test_mezz_key_changes_for_mapping_and_has_new_salt():
    base = [{"n0": 0, "n1": 30, "frames": 30,
             "a0": 0, "a1": 48000, "a_start": 1.3,
             "file_start": 1.3, "real": True}]
    changed = [{**base[0], "file_start": 1.4}]
    assert mezz_key(0, base) != mezz_key(0, changed)
    assert mezz_key(0, base) != mezz_key(1, base)



@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_mezz_odd_sar_source_with_filler(tmp_path):
    """2320x1080 anamorphic source (SAR != 1) + trailing black filler piece:
    concat needs identical SAR on all inputs."""
    src = tmp_path / "wide.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=duration=8:size=2320x1080:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-vf", "setsar=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(src)], check=True)
    sar = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=sample_aspect_ratio", "-of", "json",
         str(src)], capture_output=True, text=True, check=True).stdout)
    assert sar["streams"][0]["sample_aspect_ratio"] == "2:1"

    from highlights.multiangle.render import mezz_cmd
    tm = [[{"file_lo": 0.0, "file_hi": 8.0, "offset": 0.0}]]
    pieces = mezz_pieces(0, tm, 0.0, 10.0, [8.0])
    # real piece then a black filler for the uncovered tail
    assert [p["real"] for p in pieces] == [True, False]
    mp4, wav = tmp_path / "m.mp4", tmp_path / "m.wav"
    cmd = mezz_cmd(str(src), pieces, str(mp4), str(wav), threads=1)
    cmd = [c for c in cmd if c not in ("-progress", "pipe:1", "-nostats")]
    subprocess.run(cmd, check=True)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,sample_aspect_ratio",
         "-of", "json", str(mp4)], capture_output=True, text=True,
         check=True).stdout)
    assert int(info["streams"][0]["nb_frames"]) == 300
    assert info["streams"][0]["sample_aspect_ratio"] == "1:1"
    with wave.open(str(wav)) as wf:
        assert wf.getnframes() == sum(
            p["a1"] - p["a0"] for p in pieces)


def _mezz_test_pieces():
    return [{"n0": 0, "n1": 60, "frames": 60,
             "a0": 0, "a1": 96000, "a_start": 0.0,
             "file_start": 0.0, "real": True}]


def test_mezz_cmd_tonemap_only_for_hdr(monkeypatch, tmp_path):
    from highlights.multiangle import render as rd
    from highlights.multiangle.color import SDR_OUT_ARGS, SDR_TAG_VF
    monkeypatch.setattr(rd, "_has_audio", lambda v: False)
    pieces = _mezz_test_pieces()
    monkeypatch.setattr(rd, "is_hdr", lambda v: True)
    cmd_hdr = rd.mezz_cmd("v.mp4", pieces, str(tmp_path / "o.mp4"),
                          str(tmp_path / "o.wav"))
    monkeypatch.setattr(rd, "is_hdr", lambda v: False)
    cmd_sdr = rd.mezz_cmd("v.mp4", pieces, str(tmp_path / "o2.mp4"),
                          str(tmp_path / "o2.wav"))
    f_hdr, f_sdr = cmd_hdr[cmd_hdr.index("-filter_complex") + 1], \
        cmd_sdr[cmd_sdr.index("-filter_complex") + 1]
    assert "zscale=t=linear:npl=203" in f_hdr
    assert "tonemap=tonemap=mobius" in f_hdr
    assert "zscale" not in f_sdr and "tonemap" not in f_sdr
    for cmd in (cmd_hdr, cmd_sdr):
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert SDR_TAG_VF in fc
        for a in ("-colorspace", "-color_primaries", "-color_trc",
                  "-color_range"):
            assert a in cmd
        assert all(a in cmd for a in SDR_OUT_ARGS)


def test_mezz_key_changes_only_for_hdr():
    pieces = _mezz_test_pieces()
    assert mezz_key(0, pieces) != mezz_key(0, pieces, hdr=True)
    assert mezz_key(0, pieces) == mezz_key(0, pieces, hdr=False)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_mezz_hdr_clip_produces_bt709_output(tmp_path):
    """2 s PQ/bt2020 10-bit source through the real mezz path -> tagged
    bt709, yuv420p, exact frames."""
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
    assert is_hdr(str(src))
    from highlights.multiangle.render import mezz_cmd
    cmd = mezz_cmd(str(src), _mezz_test_pieces()[:1],
                   str(tmp_path / "mezz.mp4"), str(tmp_path / "mezz.wav"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    st = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=color_transfer,color_primaries,color_space,pix_fmt",
         "-of", "json", str(tmp_path / "mezz.mp4")],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert st["color_transfer"] == "bt709"
    assert st["color_primaries"] == "bt709"
    assert st["color_space"] == "bt709"
    assert st["pix_fmt"] == "yuv420p"
    nf = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v:0", "-show_entries", "stream=nb_read_frames", "-of",
         "csv=p=0", str(tmp_path / "mezz.mp4")],
        capture_output=True, text=True, check=True).stdout.strip()
    assert nf == "60"


def test_is_hdr_csv_trailing_comma(monkeypatch):
    """prod probe output is 'smpte2084,\\n' — must still parse as HDR."""
    import subprocess as sp

    from highlights.multiangle.color import is_hdr

    def fake(out):
        def run(cmd, **kw):
            return type("P", (), {"returncode": 0, "stdout": out})()
        return run

    for out, want in (("smpte2084,\n", True), ("smpte2084\n", True),
                      ("arib-std-b67\n", True), ("bt709,\n", False),
                      ("bt709\n", False), ("", False),
                      ("unknown\n", False)):
        is_hdr.cache_clear()
        monkeypatch.setattr(sp, "run", fake(out))
        assert is_hdr("v.mp4") == want, out
    is_hdr.cache_clear()
