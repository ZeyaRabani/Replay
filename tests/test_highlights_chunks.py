import shutil
import subprocess

import pytest

from highlights.chunks import extract_clip

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


@pytest.fixture(scope="module")
def test_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264",
                    "-g", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


def test_extract_clip(test_video, tmp_path):
    dst = tmp_path / "clip.mp4"
    extract_clip(test_video, 1.0, 3.5, dst)
    assert dst.exists()
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(dst)], check=True, capture_output=True, text=True).stdout
    assert abs(float(out.strip()) - 2.5) < 0.5
