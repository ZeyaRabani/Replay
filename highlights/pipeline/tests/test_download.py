from typing import ClassVar

import pytest
import yt_dlp

from highlights.pipeline import download as dl
from highlights.pipeline.errors import PipelineError
from highlights.pipeline.status import StatusWriter


class _FakeYDL:
    """Mimics yt_dlp.YoutubeDL as a context manager."""
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        for hook in self.opts.get("progress_hooks", []):
            hook({"status": "downloading", "downloaded_bytes": 50,
                  "total_bytes": 100, "info_dict": {"height": 1080}})
            hook({"status": "finished", "info_dict": {"height": 1080}})
        out = self.opts["outtmpl"].replace("%(ext)s", "mp4")
        with open(out, "wb") as f:
            f.write(b"video")
        return {"format": "bv*+ba", "width": 1920, "height": 1080,
                "requested_downloads": [{"filepath": out}]}


class _BotYDL(_FakeYDL):
    def extract_info(self, url, download=True):
        raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")


class _Dash403YDL(_FakeYDL):
    """First YoutubeDL instance raises a 403; the HLS retry succeeds."""
    calls: ClassVar[list] = []

    def extract_info(self, url, download=True):
        self.calls.append(self.opts.get("format"))
        if len(self.calls) == 1:
            raise yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")
        return super().extract_info(url, download)


def test_download_403_falls_back_to_hls(tmp_path, monkeypatch):
    _Dash403YDL.calls = []
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _Dash403YDL)
    logs = []
    out = dl.download("http://x", tmp_path, status=None, log=logs.append)
    assert out.name == "match.mp4"
    assert len(_Dash403YDL.calls) == 2
    assert _Dash403YDL.calls[1].startswith("bv*[protocol^=m3u8]")
    assert any("403" in m and "HLS" in m for m in logs)


def test_download_success(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _FakeYDL)
    status = StatusWriter(tmp_path / "status.json")
    out = dl.download("http://x", tmp_path, status=status, log=lambda m: None)
    assert out.name == "match.mp4"
    d = status.status["download"]
    assert d["resolution"] == "1920x1080"
    assert d["filesize"] == 5
    assert status.status["video_path"] == str(out)


class _CaptureYDL(_FakeYDL):
    """Captures the opts dict passed to YoutubeDL."""
    instances: ClassVar[list] = []

    def __init__(self, opts):
        super().__init__(opts)
        self.instances.append(self)


def test_download_pot_provider(tmp_path, monkeypatch):
    _CaptureYDL.instances = []
    monkeypatch.setenv("HL_POT_PROVIDER_URL", "http://bgutil-provider:4416")
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _CaptureYDL)
    logs = []
    out = dl.download("http://x", tmp_path, status=None, log=logs.append)
    assert out.name == "match.mp4"
    opts = _CaptureYDL.instances[0].opts
    assert opts["extractor_args"]["youtubepot-bgutilhttp"]["base_url"] == [
        "http://bgutil-provider:4416"]
    assert any("pot provider" in m for m in logs)


def test_download_no_pot_provider(tmp_path, monkeypatch):
    _CaptureYDL.instances = []
    monkeypatch.delenv("HL_POT_PROVIDER_URL", raising=False)
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _CaptureYDL)
    out = dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    assert out.name == "match.mp4"
    assert "extractor_args" not in _CaptureYDL.instances[0].opts


def test_download_proxy(tmp_path, monkeypatch):
    _CaptureYDL.instances = []
    monkeypatch.setenv("HL_YT_PROXY", "http://user:secret@resi.example:8080")
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _CaptureYDL)
    logs = []
    out = dl.download("http://x", tmp_path, status=None, log=logs.append)
    assert out.name == "match.mp4"
    opts = _CaptureYDL.instances[0].opts
    assert opts["proxy"] == "http://user:secret@resi.example:8080"
    assert any("yt proxy: http://resi.example:8080" in m for m in logs)
    assert not any("secret" in m for m in logs)


def test_download_no_proxy(tmp_path, monkeypatch):
    _CaptureYDL.instances = []
    monkeypatch.delenv("HL_YT_PROXY", raising=False)
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _CaptureYDL)
    out = dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    assert out.name == "match.mp4"
    assert "proxy" not in _CaptureYDL.instances[0].opts


def test_download_bot_check(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _BotYDL)
    with pytest.raises(PipelineError) as ei:
        dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    msg = str(ei.value)
    assert "YouTube blocked" in msg
    assert "upload" in msg.lower()
    assert "cookies" in msg


def test_download_bot_check_with_cookies(tmp_path, monkeypatch):
    """Bot check despite a cookiefile -> cookies-expired message, not the
    generic one."""
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _BotYDL)
    ck = tmp_path / "cookies.txt"
    ck.write_text("youtube.com\tTRUE\t/\tFALSE\t0\tX\tY")
    with pytest.raises(PipelineError) as ei:
        dl.download("http://x", tmp_path, status=None, cookies=str(ck),
                    log=lambda m: None)
    msg = str(ei.value)
    assert "rejected the saved cookies" in msg
    assert "incognito" in msg


class _TransientYDL(_FakeYDL):
    """Raises a transient 503 for the first `fails` extract_info calls."""
    calls: ClassVar[int] = 0
    fails: ClassVar[int] = 0

    def extract_info(self, url, download=True):
        type(self).calls += 1
        if type(self).calls <= type(self).fails:
            raise yt_dlp.utils.DownloadError(
                "HTTP Error 503: Service Unavailable")
        return super().extract_info(url, download)


def _patch_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(dl, "_sleep", sleeps.append)
    return sleeps


def test_download_transient_retries_then_succeeds(tmp_path, monkeypatch):
    _TransientYDL.calls, _TransientYDL.fails = 0, 2
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _TransientYDL)
    sleeps = _patch_sleep(monkeypatch)
    out = dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    assert out.name == "match.mp4"
    assert sleeps == [30, 60]
    assert _TransientYDL.calls == 3


def test_download_transient_gives_up_after_8(tmp_path, monkeypatch):
    _TransientYDL.calls, _TransientYDL.fails = 0, 99
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _TransientYDL)
    sleeps = _patch_sleep(monkeypatch)
    with pytest.raises(PipelineError) as ei:
        dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    assert "after 8 attempts" in str(ei.value)
    assert _TransientYDL.calls == 8
    assert sleeps == [30, 60, 120, 300, 600, 900, 1200]


def test_download_cookies_rejected_retries_without(tmp_path, monkeypatch):
    """Saved cookies rejected -> one retry without cookiefile; a second bot
    check raises the cookies-expired message."""
    class Rec(_BotYDL):
        opts_seen: ClassVar[list] = []

        def __init__(self, opts):
            super().__init__(opts)
            self.opts_seen.append(dict(opts))

    Rec.opts_seen = []
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", Rec)
    sleeps = _patch_sleep(monkeypatch)
    ck = tmp_path / "cookies.txt"
    ck.write_text("youtube.com\tTRUE\t/\tFALSE\t0\tX\tY")
    with pytest.raises(PipelineError) as ei:
        dl.download("http://x", tmp_path, status=None, cookies=str(ck),
                    log=lambda m: None)
    assert "rejected the saved cookies" in str(ei.value)
    assert len(Rec.opts_seen) == 2
    assert "cookiefile" in Rec.opts_seen[0]
    assert "cookiefile" not in Rec.opts_seen[1]
    assert sleeps == []  # cookie fallback does not burn a transient attempt


def test_download_bot_check_not_transient(tmp_path, monkeypatch):
    """Bot-check errors never enter the transient backoff loop."""
    class BotCount(_BotYDL):
        calls: ClassVar[int] = 0

        def extract_info(self, url, download=True):
            type(self).calls += 1
            return super().extract_info(url, download)

    BotCount.calls = 0
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", BotCount)
    sleeps = _patch_sleep(monkeypatch)
    with pytest.raises(PipelineError):
        dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    assert BotCount.calls == 1
    assert sleeps == []


def test_download_tries_next_cookies_file(tmp_path, monkeypatch):
    """Bot check on cookies file A -> tries file B which succeeds."""
    import os
    from pathlib import Path

    a = tmp_path / "a.txt"
    a.write_text("stale")
    b = tmp_path / "b.txt"
    b.write_text("good")
    seen = []

    def fake_extract(opts, formats, url, log):
        seen.append(opts.get("cookiefile"))
        if opts.get("cookiefile") == str(b):
            out = opts["outtmpl"].replace("%(ext)s", "mp4")
            Path(out).write_bytes(b"v")
            return {"requested_downloads": [{"filepath": out}]}
        raise yt_dlp.utils.DownloadError(
            "Sign in to confirm you're not a bot")

    monkeypatch.setattr(dl, "_extract", fake_extract)
    sleeps = _patch_sleep(monkeypatch)
    logs = []
    out = dl.download("http://x", tmp_path / "dst",
                      cookies=os.pathsep.join([str(a), str(b)]),
                      log=logs.append)
    assert out.name == "match.mp4"
    assert seen == [str(a), str(b)]
    assert any("trying next" in m for m in logs)
    assert sleeps == []


def test_download_all_cookies_files_rejected(tmp_path, monkeypatch):
    """Every cookies file + the no-cookies retry bot-checked ->
    COOKIES_REJECTED_MSG."""
    import os

    a = tmp_path / "a.txt"
    a.write_text("stale")
    b = tmp_path / "b.txt"
    b.write_text("also stale")
    seen = []

    def fake_extract(opts, formats, url, log):
        seen.append(opts.get("cookiefile"))
        raise yt_dlp.utils.DownloadError(
            "Sign in to confirm you're not a bot")

    monkeypatch.setattr(dl, "_extract", fake_extract)
    logs = []
    with pytest.raises(PipelineError) as ei:
        dl.download("http://x", tmp_path / "dst",
                    cookies=os.pathsep.join([str(a), str(b)]),
                    log=logs.append)
    assert "rejected the saved cookies" in str(ei.value)
    assert seen == [str(a), str(b), None]  # a, b, then no cookiefile
    assert any("without them" in m for m in logs)
