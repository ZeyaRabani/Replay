"""Extract mono 16 kHz WAV from a video file with ffmpeg."""
import argparse
import subprocess
from pathlib import Path


def extract(video: Path, wav: Path) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    return wav


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("wav", type=Path)
    a = ap.parse_args()
    print(extract(a.video, a.wav))
