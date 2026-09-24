"""Hand-crafted per-window audio features for crowd excitement + whistle detection.

Outputs a dense feature table at a fixed step (1.0 s or 0.5 s). All times are
seconds from the start of the video file.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft

SR = 16000
N_FFT = 1024
HOP = 160  # 10 ms frames


def band_power(spec_pow: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (freqs >= lo) & (freqs < hi)
    return spec_pow[m].sum(axis=0)


def rolling_z(x: np.ndarray, win: int) -> np.ndarray:
    """z-score of x vs a centred rolling mean/std over `win` samples (robust: median/MAD-ish)."""
    mu = uniform_filter1d(x, win, mode="nearest")
    var = uniform_filter1d((x - mu) ** 2, win, mode="nearest")
    sd = np.sqrt(np.maximum(var, 1e-12))
    return (x - mu) / (sd + 1e-6 * (np.abs(mu).max() + 1e-9))


def whistle_score(spec_pow: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame whistle evidence: narrow tonal peak in 2-4.5 kHz.

    Returns (score, peak_freq). Score = peak-bin power / mean power in band
    (tonality ratio) gated by the band's share of total power.
    """
    m = (freqs >= 2000) & (freqs < 4500)
    band = spec_pow[m]
    total = spec_pow.sum(axis=0) + 1e-12
    peak = band.max(axis=0)
    mean = band.mean(axis=0) + 1e-12
    tonality = peak / mean  # ~1 for noise, large for a pure tone
    share = band.sum(axis=0) / total
    score = tonality * np.clip(share * 4, 0, 1)
    peak_freq = freqs[m][band.argmax(axis=0)]
    return score, peak_freq


def detect_whistles(wav: Path, thr: float = 12.0, min_dur: float = 0.25,
                    max_gap: float = 0.15, f_lo: float = 2200, f_hi: float = 4500) -> list[dict]:
    """Frame-level referee-whistle segments.

    A whistle is a run of >= min_dur seconds of frames whose 2-4.5 kHz band is
    strongly tonal (whistle_score > thr) with a stable peak frequency. Short
    dropouts (<= max_gap) inside a run are bridged, so a trilled/double blast
    is one segment.
    """
    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    freqs, _, Z = stft(y, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    P = (np.abs(Z) ** 2).astype(np.float32)
    score, pf = whistle_score(P, freqs)
    rms_db = 10 * np.log10(P.sum(axis=0) / N_FFT + 1e-12)
    on = (score > thr) & (pf >= f_lo) & (pf < f_hi)
    fps = sr / HOP
    gap = int(max_gap * fps)
    # bridge gaps
    on_b = on.copy()
    i = 0
    n = len(on)
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= gap:
                on_b[i:j] = True
            i = j
        else:
            i += 1
    segs = []
    i = 0
    while i < n:
        if on_b[i]:
            j = i
            while j < n and on_b[j]:
                j += 1
            dur = (j - i) / fps
            if dur >= min_dur:
                sel = slice(i, j)
                f = pf[sel][on[sel]]
                segs.append({
                    "t_start": round(i / fps, 2), "t_end": round(j / fps, 2), "duration_s": round(dur, 2),
                    "peak_hz": round(float(np.median(f)), 0), "peak_hz_std": round(float(np.std(f)), 0),
                    "tonality": round(float(np.max(score[sel])), 1),
                    "level_db": round(float(np.max(rms_db[sel])), 1),
                })
            i = j
        else:
            i += 1
    return segs


def compute(wav: Path, step_s: float = 1.0) -> dict:
    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    assert sr == SR, sr
    dur = len(y) / sr

    freqs, _, Z = stft(y, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    P = (np.abs(Z) ** 2).astype(np.float32)  # (freq, frames)
    n_frames = P.shape[1]
    frame_t = np.arange(n_frames) * HOP / sr

    rms_f = np.sqrt(P.sum(axis=0) / N_FFT + 1e-12)
    low = band_power(P, freqs, 0, 300)          # wind / rumble / handling
    speech = band_power(P, freqs, 300, 3000)    # shouts / voices / claps
    high = band_power(P, freqs, 3000, 8000)
    total = P.sum(axis=0) + 1e-12
    # spectral flux on log-magnitude
    L = np.log1p(P)
    flux_f = np.r_[0, np.sqrt(((np.diff(L, axis=1)).clip(min=0) ** 2).sum(axis=0))]
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(P), sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP, units="frames")
    onset_flag = np.zeros(n_frames)
    onset_flag[onsets] = 1
    centroid = (freqs[:, None] * P).sum(axis=0) / total
    flat = librosa.feature.spectral_flatness(S=np.sqrt(P))[0]
    w_score, w_freq = whistle_score(P, freqs)

    # aggregate to windows
    n_win = int(np.floor(dur / step_s))
    idx = np.minimum((frame_t / step_s).astype(int), n_win - 1)

    def agg(v, fn=np.mean):
        if fn is np.mean:
            s = np.bincount(idx, weights=v, minlength=n_win)
            c = np.bincount(idx, minlength=n_win)
            return s / np.maximum(c, 1)
        if fn is np.sum:
            return np.bincount(idx, weights=v, minlength=n_win)
        if fn is np.max:
            out = np.full(n_win, -np.inf)
            np.maximum.at(out, idx, v)
            return out
        raise ValueError

    rms = agg(rms_f)
    rms_db = 20 * np.log10(rms + 1e-6)
    low_db = 10 * np.log10(agg(low) + 1e-9)
    speech_db = 10 * np.log10(agg(speech) + 1e-9)
    high_db = 10 * np.log10(agg(high) + 1e-9)
    speech_ratio = agg(speech / total)
    flux = agg(flux_f)
    onset_density = agg(onset_flag, np.sum) / step_s
    cent = agg(centroid)
    flatness = agg(flat)
    wmax = agg(w_score, np.max)
    wfrac = agg((w_score > 8).astype(float))  # fraction of frames whistle-like
    # frequency of the whistle peak in whistle-like frames (0 if none)
    wf = np.zeros(n_win)
    for k in np.unique(idx[w_score > 8]):
        sel = (idx == k) & (w_score > 8)
        wf[k] = np.median(w_freq[sel])

    w60 = max(3, round(60 / step_s))
    w300 = max(3, round(300 / step_s))
    z60 = rolling_z(rms_db, w60)
    z300 = rolling_z(rms_db, w300)
    zs300 = rolling_z(speech_db, w300)
    # smoothed excitement: mean z over a 5 s window (crowd reaction is sustained)
    k5 = max(1, round(5 / step_s))
    z300_s5 = uniform_filter1d(zs300, k5, mode="nearest")

    t = np.arange(n_win) * step_s
    cols = ["t", "rms_db", "low_db", "speech_db", "high_db", "speech_ratio", "flux",
            "onset_density", "centroid_hz", "flatness", "whistle_max", "whistle_frac",
            "whistle_hz", "z60_rms", "z300_rms", "z300_speech", "z300_speech_s5"]
    mat = np.column_stack([t, rms_db, low_db, speech_db, high_db, speech_ratio, flux,
                           onset_density, cent, flatness, wmax, wfrac, wf, z60, z300, zs300, z300_s5])
    return {
        "source": "audio",
        "step_s": step_s,
        "video_duration_s": float(dur),
        "columns": cols,
        "rows": [[round(float(v), 4) for v in r] for r in mat],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--whistles", type=Path, help="also write frame-level whistle segments here")
    a = ap.parse_args()
    feats = compute(a.wav, a.step)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(feats))
    print(a.out, len(feats["rows"]), "rows")
    if a.whistles:
        segs = detect_whistles(a.wav)
        a.whistles.write_text(json.dumps({"source": "audio", "whistles": segs}, indent=1))
        print(a.whistles, len(segs), "whistle segments")
