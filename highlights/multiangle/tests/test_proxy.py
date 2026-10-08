"""480p analysis proxy: offset mapping, fallback chain, duration check."""

import os
import subprocess
from pathlib import Path

import pytest

from highlights.multiangle import proxy
from highlights.multiangle.trackfeat import _frame_reader

FFMPEG = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


def _mkvideo(path: Path, seconds: float = 10.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size=320x180:rate=30",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-an", str(path)], check=True)
    return path


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_transcode_proxy_offset_and_reader(tmp_path):
    """Windowed transcode writes offset; the reader maps proxy time back
    onto the original's file-second grid."""
    src = _mkvideo(tmp_path / "match.mp4", 10.0)
    adir = tmp_path / "a0"
    adir.mkdir()
    logs: list[str] = []
    pi = proxy.ensure_analysis_proxy(
        adir, src, url=None, window=(4.0, 8.0), log=logs.append)
    assert pi.src == "transcode-480p"
    assert pi.offset == pytest.approx(0.0, abs=1e-6) or pi.offset >= 0.0
    # window (4,8) padded 5 each side -> lo 0 -> offset 0, covers 10 s
    assert (adir / "analysis_480p.mp4").exists()
    meta = (adir / "analysis_480p.json").read_text()
    assert '"offset"' in meta
    # reader reports original file seconds even with a shifted proxy
    ts = [t for t, _ in _frame_reader(str(pi.path), 1, 320,
                                      start_s=2.0, end_s=5.0, t_base=7.0)]
    assert ts[0] == pytest.approx(7.0) and len(ts) == 3


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_transcode_full_when_no_window(tmp_path):
    src = _mkvideo(tmp_path / "match.mp4", 10.0)
    adir = tmp_path / "a0"
    adir.mkdir()
    pi = proxy.ensure_analysis_proxy(adir, src, url=None, log=lambda *a: None)
    assert pi.src == "transcode-480p" and pi.offset == 0.0
    assert abs(proxy._probe_duration(pi.path) - 10.0) <= 1.0
    # second call reuses the existing proxy
    pi2 = proxy.ensure_analysis_proxy(adir, src, url=None, log=lambda *a: None)
    assert pi2.path == pi.path and pi2.src == "transcode-480p"


def test_ytdlp_failure_falls_back_to_transcode(tmp_path, monkeypatch):
    """yt-dlp failing must not be fatal: transcode path runs next."""
    def boom(url, dest, cookies, log):
        raise RuntimeError("403 forbidden")
    monkeypatch.setattr(proxy, "_ytdlp_480", boom)
    monkeypatch.setattr(proxy, "_probe_duration",
                        lambda p: 10.0)
    called = []
    monkeypatch.setattr(proxy, "_transcode_480",
                        lambda v, d, lo, hi, log: called.append(
                            d.write_bytes(b"x")) or d)
    adir = tmp_path / "a0"
    adir.mkdir()
    pi = proxy.ensure_analysis_proxy(
        adir, tmp_path / "match.mp4", url="https://youtu.be/x",
        log=lambda *a: None)
    assert pi.src == "transcode-480p" and called


def test_ytdlp_bad_duration_falls_back(tmp_path, monkeypatch):
    """A downloaded proxy whose duration differs >1 s from the source is
    rejected and the transcode runs instead."""
    monkeypatch.setattr(proxy, "_ytdlp_480",
                        lambda u, d, c, log: Path(d).write_bytes(b"x") or d)
    monkeypatch.setattr(
        proxy, "_probe_duration",
        lambda p: 99.0 if p.read_bytes() == b"x" else 10.0)
    monkeypatch.setattr(proxy, "_transcode_480",
                        lambda v, d, lo, hi, log: Path(d).write_bytes(b"y") or d)
    (tmp_path / "match.mp4").write_bytes(b"m")
    logs: list[str] = []
    adir = tmp_path / "a0"
    adir.mkdir()
    pi = proxy.ensure_analysis_proxy(
        adir, tmp_path / "match.mp4", url="https://youtu.be/x",
        log=logs.append)
    assert pi.src == "transcode-480p"
    assert any("failed" in m for m in logs)


def test_ytdlp_iterates_joined_cookiefiles(tmp_path, monkeypatch):
    """An os.pathsep-joined cookies string must try each file, not pass the
    whole joined string as one filename."""
    import yt_dlp

    seen: list = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = dict(opts)
            seen.append(self.opts.get("cookiefile"))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            if self.opts.get("cookiefile") == str(a):
                raise yt_dlp.utils.DownloadError(
                    "Sign in to confirm you're not a bot")
            part = self.opts["outtmpl"].replace("%(ext)s", "mp4")
            Path(part).write_bytes(b"x")

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("y")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    logs: list[str] = []
    dest = tmp_path / "analysis_480p.mp4"
    proxy._ytdlp_480("https://youtu.be/x", dest,
                     os.pathsep.join([str(a), str(b)]), logs.append)
    assert seen == [str(a), str(b)]
    assert dest.exists()
    assert any("trying next" in m for m in logs)


def test_all_failures_fall_back_to_original(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "_ytdlp_480",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(proxy, "_transcode_480",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("y")))
    monkeypatch.setattr(proxy, "_probe_duration", lambda p: 10.0)
    src = tmp_path / "match.mp4"
    adir = tmp_path / "a0"
    adir.mkdir()
    pi = proxy.ensure_analysis_proxy(
        adir, src, url="https://youtu.be/x", log=lambda *a: None)
    assert pi.src == "original" and pi.path == src and pi.offset == 0.0
