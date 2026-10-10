"""view_ok camera-moved detection on a synthetic clip."""

import shutil
import subprocess

import pytest

from highlights.multiangle.viewcheck import view_ok

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg missing")


@pytest.fixture()
def clip(tmp_path):
    """40 s clip: first half testsrc, second half random noise (new view)."""
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    base = ["-pix_fmt", "yuv420p", "-r", "10", "-g", "10"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=duration=20:size=320x180:rate=10", *base, str(a)],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=gray:size=320x180:rate=10:d=20,noise=alls=80:allf=t",
                    *base, str(b)],
                   check=True)
    lst = tmp_path / "list.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n")
    out = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                   check=True)
    return out


def test_view_ok_detects_camera_move(clip):
    times, ok = view_ok(clip, ref_t=5.0, step=10)
    assert len(ok) >= 3
    # early samples match the reference view; post-flip samples do not
    assert ok[times <= 15].all()
    assert not ok[times >= 25].any()
