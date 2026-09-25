"""Audio-based sync between camera angles.

Each angle's audio (mono wav extracted by the Option-1 audio stage) is turned
into a 50 Hz onset-strength envelope, whitened to remove mic-gain and speech
differences, then cross-correlated against the reference angle (a0). A
confidence test (peak-to-noise ratio + second-peak ratio) decides whether the
offset is trustworthy; a triangle-consistency check flags inconsistent pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 8000          # resample rate
ENV_HZ = 50        # envelope rate (hop 160 @ 8 kHz)
HOP = 160
WHITE_S = 5.0      # whitening window (s)
REFINE_WIN_S = 0.5
BAND = (300.0, 3000.0)
PNR_MIN = 8.0
R2_MAX = 0.6
TRIANGLE_TOL_S = 0.5


def _load_env(wav_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(whitened onset envelope at ENV_HZ, raw 8 kHz mono waveform)."""
    import librosa

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP).astype(float)
    env = _whiten(env)
    return env, y


def _whiten(env: np.ndarray) -> np.ndarray:
    """Subtract rolling mean, clip at 0, divide by rolling std."""
    k = max(1, int(WHITE_S * ENV_HZ))
    kernel = np.ones(k) / k
    mean = np.convolve(env, kernel, mode="same")
    x = np.clip(env - mean, 0, None)
    var = np.convolve(x * x, kernel, mode="same")
    sd = np.sqrt(np.maximum(var, 1e-12))
    return x / sd


def _bandpass(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(Y, len(y))


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full normalised cross-correlation; index k means a[t+k] vs b[t].

    Peak at lag L means a[t+L] == b[t]: the same event sits at file time
    t-L in a when it's at t in b, i.e. T(b's file time t) = t + L.
    Therefore offset(b) = L / rate (seconds to ADD to b's file time).
    """
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    c = np.fft.irfft(A * np.conj(B), nfft)
    # irfft(A*conj(B))[k] = sum_t a[t+k]b[t]. Positive lags 0..len(a)-1 sit at
    # the front; negative lags -(len(b)-1)..-1 wrap to the END of the nfft
    # buffer (zero-padding), not to positions n-1... .
    pos = c[: len(a)]
    neg = c[nfft - len(b) + 1:]
    full = np.concatenate([neg, pos])
    # full[i] corresponds to lag i - (len(b) - 1)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return full / max(norm, 1e-9)


def _peak_metrics(corr: np.ndarray, lags_s: np.ndarray) -> tuple[float, float, float, float]:
    """(best_lag_s, peak, pnr, r2) from a normalised correlation."""
    i = int(np.argmax(corr))
    peak = float(corr[i])
    far = np.abs(lags_s - lags_s[i]) > 5.0
    noise = float(np.mean(np.abs(corr[far]))) if far.any() else 1e-9
    pnr = peak / max(noise, 1e-9)
    # second peak at least 5 s away
    c2 = corr.copy()
    c2[~far] = -np.inf
    peak2 = float(c2.max()) if np.isfinite(c2).any() else 0.0
    r2 = peak2 / peak if peak > 0 else 1.0
    return float(lags_s[i]), peak, pnr, r2


def _env_lags(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arange(len(a) + len(b) - 1) / ENV_HZ - (len(b) - 1) / ENV_HZ


def estimate_offset(env_a: np.ndarray, env_b: np.ndarray) -> tuple[float, float, float]:
    """Envelope xcorr -> (offset_s for b, pnr, r2)."""
    corr = _xcorr(env_a, env_b)
    lags = _env_lags(env_a, env_b)
    lag_s, _peak, pnr, r2 = _peak_metrics(corr, lags)
    return lag_s, pnr, r2


def _refine(y_a: np.ndarray, y_b: np.ndarray, coarse_off_s: float) -> float:
    """Sub-frame refinement on band-passed raw waveforms around the coarse
    offset. offset(b) = coarse; we search ±REFINE_WIN_S around it."""
    a = _bandpass(y_a, *BAND)
    b = _bandpass(y_b, *BAND)
    # same lag convention as _xcorr: offset = lag / SR
    centre = round(coarse_off_s * SR)
    hw = int(REFINE_WIN_S * SR)
    best_lag, best_v = centre, -np.inf
    n = min(len(a), len(b))
    # direct dot-product scan over the small window (fast enough at 8 kHz)
    for lag in range(centre - hw, centre + hw + 1):
        if lag >= 0:
            aa, bb = a[lag:lag + n], b[:n - lag] if n - lag > 0 else b[:0]
        else:
            aa, bb = a[:n + lag], b[-lag:-lag + n + lag]
        m = min(len(aa), len(bb))
        if m <= 0:
            continue
        v = float(np.dot(aa[:m], bb[:m]))
        if v > best_v:
            best_v, best_lag = v, lag
    return best_lag / SR


def sync_angles(wavs: list[str | Path], durations: list[float],
                manual_offsets: list[float] | None = None) -> dict:
    """Sync all angles against a0. Returns the sync.json dict.

    manual_offsets: len == n angles with offsets[0] == 0 (add to each angle's
    file time to get T). XCorr numbers are still recorded for the record.
    """
    n = len(wavs)
    envs, raws = [], []
    for w in wavs:
        env, y = _load_env(w)
        envs.append(env)
        raws.append(y)

    pairs = []
    offsets = [0.0] * n
    for b in range(1, n):
        off, pnr, r2 = estimate_offset(envs[0], envs[b])
        off = _refine(raws[0], raws[b], off)
        pairs.append({"a": 0, "b": b, "offset": round(off, 3),
                      "pnr": round(pnr, 2), "r2": round(r2, 3),
                      "confident": bool(pnr >= PNR_MIN and r2 <= R2_MAX)})
        offsets[b] = off

    # triangle check a1 vs a2 when >=3 angles
    tri = None
    if n >= 3:
        off12, pnr12, r212 = estimate_offset(envs[1], envs[2])
        off12 = _refine(raws[1], raws[2], off12)
        residual = abs(off12 - (offsets[2] - offsets[1]))
        tri = round(float(residual), 3)
        pairs.append({"a": 1, "b": 2, "offset": round(off12, 3),
                      "pnr": round(pnr12, 2), "r2": round(r212, 3),
                      "confident": bool(pnr12 >= PNR_MIN and r212 <= R2_MAX)})
        if residual > TRIANGLE_TOL_S:
            for pr in pairs:
                pr["consistent"] = False
        else:
            for pr in pairs:
                pr["consistent"] = True

    method = "xcorr"
    needs = [pr["b"] for pr in pairs if pr["a"] == 0 and not pr["confident"]]
    if manual_offsets is not None:
        if len(manual_offsets) != n or manual_offsets[0] != 0:
            raise ValueError("--offsets must have len == n angles, first == 0")
        method = "manual"
        offsets = [float(o) for o in manual_offsets]
        needs = []
        for pr in pairs:
            pr["confident"] = True
            pr["manual"] = True

    offs = np.array(offsets)
    durs = np.asarray(durations, dtype=float)
    coverage = {
        "intersection": [float(max(0.0, (-offs).max())),
                         float((durs - offs).min())],
        "union": [float(min(0.0, (-offs).min())),
                  float((durs - offs).max())],
    }
    return {"reference": 0, "method": method, "offsets": [round(float(o), 3) for o in offsets],
            "pairs": pairs, "triangle_residual_s": tri,
            "needs_manual": sorted(set(needs)), "coverage": coverage}


def write_sync(wavs, durations, out_path, manual_offsets=None) -> dict:
    d = sync_angles(wavs, durations, manual_offsets)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(d, indent=1))
    return d
