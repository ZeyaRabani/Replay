"""Camera-view check for painted ball zones.

Zones are only valid while the camera still frames the same view as the
still they were drawn on. view_ok samples keyframe-only thumbnails and
correlates each against the reference frame at the draw time; a sustained
drop (pan/rotate/re-framed or a new clip entirely) suspends the zone rule.
"""

import subprocess
from pathlib import Path

import numpy as np

W, H = 64, 36


def _run(cmd: list[str]) -> bytes:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[-300:]}")
    return r.stdout


def _gray_frames(video: str | Path, extra_in: list[str], vf: str,
                 extra_out: list[str] | None = None) -> np.ndarray:
    raw = _run(["ffmpeg", "-v", "error", *extra_in, "-i", str(video),
                "-vf", vf, "-an", "-sn", *(extra_out or []),
                "-f", "rawvideo", "-"])
    n = len(raw) // (W * H)
    return np.frombuffer(raw[: n * W * H], dtype=np.uint8).reshape(n, H, W).astype(float)


def _zncc_shifted(a: np.ndarray, b: np.ndarray, max_shift: int = 2) -> float:
    """Zero-mean normalised cross-correlation of a vs b, max over small shifts."""
    best = -1.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0, y1 = max(0, dy), min(H, H + dy)
            x0, x1 = max(0, dx), min(W, W + dx)
            ay0, ax0 = max(0, -dy), max(0, -dx)
            fa = a[ay0:ay0 + (y1 - y0), ax0:ax0 + (x1 - x0)].ravel()
            fb = b[y0:y1, x0:x1].ravel()
            fa = fa - fa.mean()
            fb = fb - fb.mean()
            denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
            if denom > 0:
                best = max(best, float(fa @ fb) / denom)
    return best


def view_ok(video: str | Path, ref_t: float, step: int = 10,
            thresh: float = 0.6, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(times, ok): per-sample camera-view check vs the frame at ref_t.

    Decoding every keyframe of a long video serially is minutes; the sample
    pass is split into time-range chunks decoded by parallel ffmpeg calls.
    """
    import concurrent.futures as cf
    import json as _json

    dur = float(_json.loads(_run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video)]))["format"]["duration"])
    ref = _gray_frames(video, ["-ss", f"{ref_t:.3f}"],
                       f"scale={W}:{H},format=gray",
                       extra_out=["-frames:v", "1"])[0]

    vf = f"fps={1.0 / step},scale={W}:{H},format=gray"
    n_chunks = max(1, min(workers, int(dur / step / 4) or 1))
    edges = np.linspace(0, dur, n_chunks + 1)
    spans = [(float(edges[i]), float(edges[i + 1]))
             for i in range(n_chunks)]

    def _decode(span: tuple[float, float]) -> np.ndarray:
        s, e = span
        return _gray_frames(video, ["-skip_frame", "nokey",
                                    "-ss", f"{s:.3f}", "-to", f"{e:.3f}"],
                            vf)

    with cf.ThreadPoolExecutor(n_chunks) as ex:
        chunks = list(ex.map(_decode, spans))
    samples = np.concatenate([c for c in chunks if len(c)]) \
        if any(len(c) for c in chunks) else np.zeros((0, H, W))
    times = np.arange(len(samples)) * float(step)
    ok = np.array([_zncc_shifted(f, ref) >= thresh for f in samples])
    return times, ok
