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


def test_download_success(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _FakeYDL)
    status = StatusWriter(tmp_path / "status.json")
    out = dl.download("http://x", tmp_path, status=status, log=lambda m: None)
    assert out.name == "match.mp4"
    d = status.status["download"]
    assert d["resolution"] == "1920x1080"
    assert d["filesize"] == 5
    assert status.status["video_path"] == str(out)


def test_download_bot_check(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _BotYDL)
    with pytest.raises(PipelineError) as ei:
        dl.download("http://x", tmp_path, status=None, log=lambda m: None)
    msg = str(ei.value)
    assert "YouTube blocked" in msg
    assert "upload" in msg.lower()
    assert "cookies" in msg
