"""Shared fixtures: synthetic clips rendered with ffmpeg (no network, no GPU).

``make_clips`` renders 2-3 short colour videos that share one acoustic scene
(band-limited noise bursts) but start at different, known times, so audio
sync has a ground truth to recover.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pitchworld.sync import SR

FPS = 24
SIZE = (320, 180)


def _scene(rng: np.random.Generator, duration_s: float, sr: int = SR) -> np.ndarray:
    """Quiet background + ~12 sharp, band-limited (1-6 kHz) noise bursts per 10 s."""
    n = int(duration_s * sr)
    x = rng.normal(0, 0.005, n).astype(np.float32)
    n_bursts = max(4, int(duration_s * 1.2))
    for start in rng.uniform(0.2, duration_s - 0.3, n_bursts):
        i = int(start * sr)
        length = int(rng.uniform(0.04, 0.12) * sr)
        t = np.arange(length) / sr
        tone = np.sin(2 * np.pi * rng.uniform(1500, 4500) * t) * np.exp(-t * 25)
        burst = (tone + rng.normal(0, 0.3, length)) * 0.6
        x[i:i + length] += burst[: max(0, min(length, n - i))].astype(np.float32)
    return np.clip(x, -1, 1)


def write_clip(path: Path, audio: np.ndarray, colour: str, sr: int = SR, fps: int = FPS,
               size: tuple[int, int] = SIZE) -> Path:
    duration = len(audio) / sr
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "lavfi", "-i", f"color=c={colour}:s={size[0]}x{size[1]}:r={fps}:d={duration:.3f}",
           "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", str(path)]
    subprocess.run(cmd, input=audio.astype(np.float32).tobytes(), check=True)
    return path


@pytest.fixture(scope="session", autouse=True)
def _require_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")


@pytest.fixture(scope="session")
def make_clips(tmp_path_factory):
    """make_clips(offsets, duration_s=6.0, seed=0) -> list[Path].

    ``offsets[i]`` is the time in clip *i* matching t=0 of clip 0 (the
    ``sync_clips`` convention); offsets[0] must be 0.
    """
    colours = ("green", "blue", "red", "orange")

    def _make(offsets: list[float], duration_s: float = 6.0, seed: int = 0, shared: bool = True) -> list[Path]:
        assert offsets[0] == 0
        rng = np.random.default_rng(seed)
        # clip i starts ``off`` seconds *before* clip 0 on the shared clock
        lead = 1.0 + max(0.0, max(offsets))
        scene = _scene(rng, lead + duration_s + max(0.0, -min(offsets)) + 1.0)
        out = tmp_path_factory.mktemp("clips")
        paths = []
        for i, off in enumerate(offsets):
            if shared or i == 0:
                start = int((lead - off) * SR)
                audio = scene[start:start + int(duration_s * SR)]
            else:
                audio = _scene(np.random.default_rng(seed + 100 + i), duration_s)
            paths.append(write_clip(out / f"cam{i}.mp4", audio, colours[i]))
        return paths

    return _make
