"""Multi-clip temporal alignment via audio cross-correlation.

Every clip is aligned to the first clip's clock. ``offset_s[i]`` is the time in
clip *i* that corresponds to t=0 of clip 0 (positive => clip i started earlier).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = 16000  # Hz
HOP_S = 0.01  # envelope resolution -> offset resolution
# Pitch-side phone/action-cam audio is dominated by wind + handling rumble below
# ~500 Hz which is *uncorrelated* between cameras. Shouts, whistles and ball
# strikes live above 1 kHz and are shared, so we correlate loudness envelopes
# in that band rather than raw waveforms.
BANDS = ((1000, 6000), (800, 4000), (1500, 7500))


@dataclass
class SyncResult:
    offsets_s: list[float]
    confidences: list[float]
    method: list[str]
    common_start_s: list[float]  # per clip: start time (in that clip) of the shared window
    common_duration_s: float
    low_confidence: bool
    notes: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames,r_frame_rate,width,height",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    duration = float(info["format"]["duration"])
    nb = int(video.get("nb_frames", 0) or 0)
    fps = nb / duration if nb else eval(video["r_frame_rate"])
    return {"duration": duration, "fps": fps, "width": int(video["width"]), "height": int(video["height"]),
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"])}


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _envelope(x: np.ndarray, sr: int, band: tuple[int, int], hop_s: float = HOP_S) -> np.ndarray:
    """Normalised log-RMS loudness envelope of a band-passed signal."""
    sos = butter(4, band, btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    h = int(hop_s * sr)
    n = len(y) // h
    e = np.sqrt((y[: n * h].reshape(n, h) ** 2).mean(1))
    e = np.log(e + 1e-6)
    e -= e.mean()
    s = e.std()
    return e / s if s > 0 else e


def _xcorr_env(ea: np.ndarray, eb: np.ndarray, hop_s: float, max_lag_s: float | None, min_overlap_s: float = 3.0):
    corr = fftconvolve(eb, ea[::-1], mode="full")
    lags = np.arange(-len(ea) + 1, len(eb))
    overlap = np.minimum(lags + len(ea), len(eb) - lags).astype(float)
    corr = corr / np.maximum(overlap, 1)
    corr[overlap < min_overlap_s / hop_s] = np.nan
    if max_lag_s is not None:
        corr[np.abs(lags) * hop_s > max_lag_s] = np.nan
    return corr, lags


def xcorr_offset(a: np.ndarray, b: np.ndarray, sr: int = SR, max_lag_s: float | None = None) -> tuple[float, float]:
    """Return (lag_s, confidence). lag_s > 0 means *b* is delayed w.r.t. *a*
    (i.e. an event at time t in *a* is at t+lag in *b*).

    Runs the envelope correlation in several bands; the lag must agree across
    bands (within 60 ms) and stand out from the rest of the correlation function.
    """
    lags_found, scores = [], []
    for band in BANDS:
        corr, lags = _xcorr_env(_envelope(a, sr, band), _envelope(b, sr, band), HOP_S, max_lag_s)
        if not np.isfinite(corr).any():
            continue
        i = int(np.nanargmax(corr))
        guard = int(1.0 / HOP_S)
        rest = np.concatenate([corr[: max(0, i - guard)], corr[i + guard:]])
        rest = rest[np.isfinite(rest)]
        if rest.size == 0:
            continue
        z = (corr[i] - rest.mean()) / (rest.std() + 1e-9)
        ratio = corr[i] / rest.max() if rest.max() > 0 else np.inf
        lags_found.append(lags[i] * HOP_S)
        # z ~ 3 and ratio ~ 1.15 is a typical *real* peak on pitch-side audio; scale so those give ~0.5
        scores.append(float(np.clip(0.5 * (z / 3.0) * np.clip((ratio - 1.0) / 0.15, 0, 2), 0, 1)))
    if not lags_found:
        return 0.0, 0.0
    lag = float(np.median(lags_found))
    spread = float(np.max(lags_found) - np.min(lags_found))
    conf = float(np.mean(scores))
    if spread > 0.06:
        conf *= 0.3  # bands disagree -> not a real acoustic match
    return lag, min(conf, 1.0)


def sync_clips(paths: list[Path], manual_offsets: list[float] | None = None,
               min_conf: float = 0.5, max_lag_s: float | None = None) -> SyncResult:
    infos = [probe(p) for p in paths]
    n = len(paths)
    offsets = [0.0] * n
    confs = [1.0] * n
    method = ["reference"] + ["audio_xcorr"] * (n - 1)
    notes: list[str] = []

    if manual_offsets is not None:
        if len(manual_offsets) != n:
            raise ValueError("--offsets needs one value per clip (first is normally 0)")
        offsets = [float(o) for o in manual_offsets]
        method = ["manual"] * n
    else:
        if not all(i["has_audio"] for i in infos):
            raise ValueError("A clip has no audio track; pass --offsets manually.")
        audio = [load_audio(p) for p in paths]
        pair: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair[(i, j)] = xcorr_offset(audio[i], audio[j], max_lag_s=max_lag_s)
        for i in range(1, n):
            offsets[i], confs[i] = pair[(0, i)]
        # Loop-closure check: for every non-reference pair, the directly measured lag must
        # agree with the one implied by the reference offsets. Agreement is independent
        # evidence (different audio pair), so it raises confidence; disagreement lowers it.
        for i in range(1, n):
            for j in range(i + 1, n):
                lag_ij, conf_ij = pair[(i, j)]
                predicted = offsets[j] - offsets[i]
                err = abs(lag_ij - predicted)
                if err <= 0.08:
                    notes.append(f"clips {i}/{j}: independent lag {lag_ij:+.3f}s closes the loop with reference offsets "
                                 f"(residual {err*1000:.0f} ms) -> sync corroborated")
                    for k in (i, j):
                        confs[k] = min(0.95, confs[k] + 0.35 * conf_ij + 0.15)
                else:
                    notes.append(f"clips {i}/{j}: pairwise lag {lag_ij:+.3f}s disagrees with reference-derived "
                                 f"{predicted:+.3f}s by {err:.3f}s -> sync NOT trustworthy")
                    for k in (i, j):
                        confs[k] *= 0.5
        for i in range(1, n):
            if confs[i] < min_conf:
                notes.append(f"clip {i}: low audio-sync confidence {confs[i]:.2f} (lag {offsets[i]:+.3f}s)")

    # Shared window on the reference clock: [max(-off_i), min(dur_i - off_i)]
    starts_ref = [-o for o in offsets]
    ends_ref = [inf["duration"] - o for inf, o in zip(infos, offsets)]
    t0, t1 = max(starts_ref), min(ends_ref)
    if t1 - t0 < 1.0:
        raise ValueError(f"Synced clips share < 1 s of common time (offsets={offsets}); check the offsets.")
    common_start = [t0 + o for o in offsets]
    low = any(c < min_conf for c in confs[1:]) if manual_offsets is None else False
    return SyncResult(offsets_s=offsets, confidences=confs, method=method, common_start_s=common_start,
                      common_duration_s=t1 - t0, low_confidence=low, notes=notes)


def render_synced(src: Path, dst: Path, start_s: float, duration_s: float, fps: float,
                  height: int | None = None) -> None:
    """Trim + constant-frame-rate re-encode so every synced clip has identical frame indexing."""
    vf = [f"fps={fps}"]
    if height:
        vf.append(f"scale=-2:{height}")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.4f}", "-i", str(src), "-t", f"{duration_s:.4f}",
           "-vf", ",".join(vf), "-vsync", "cfr", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(dst)]
    subprocess.run(cmd, check=True)
