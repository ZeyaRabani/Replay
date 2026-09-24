"""Audio loudness envelope and sustained-spike detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def audio_envelope(video: Path, hop_s: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(t, rms_db) at hop_s resolution via pitchworld's ffmpeg audio loader."""
    import librosa

    from pitchworld.sync import load_audio

    x = load_audio(Path(video), sr=16000)
    if x.size == 0:
        return np.array([]), np.array([])
    frame = max(1, int(hop_s * 16000))
    rms = librosa.feature.rms(y=x, frame_length=2 * frame, hop_length=frame)[0]
    db = 20.0 * np.log10(rms + 1e-8)
    t = np.arange(len(db)) * hop_s
    return t, db


def audio_spikes(t: np.ndarray, db: np.ndarray, bin_s: float, baseline_win_s: float = 60.0,
                 thresh_db: float = 6.0, min_dur_s: float = 1.0) -> np.ndarray:
    """Per-bin 0..1: clipped (db - rolling_median)/12 sustained >= min_dur."""
    if len(db) == 0:
        return np.zeros(1)
    from scipy.ndimage import median_filter

    hop = float(t[1] - t[0]) if len(t) > 1 else 0.1
    win = max(1, int(baseline_win_s / hop)) | 1
    baseline = median_filter(db, size=win)
    excess = np.clip((db - baseline) / 12.0, 0, 1)
    loud = db - baseline > thresh_db
    # keep only runs of "loud" >= min_dur
    dur_min = int(min_dur_s / hop)
    keep = np.zeros_like(loud)
    run = 0
    for i, v in enumerate(loud):
        run = run + 1 if v else 0
        keep[i] = run
    # mark samples belonging to long enough runs
    ok = np.zeros_like(loud)
    run_start = -1
    for i in range(len(loud) + 1):
        v = loud[i] if i < len(loud) else False
        if v and run_start < 0:
            run_start = i
        elif not v and run_start >= 0:
            if i - run_start >= dur_min:
                ok[run_start:i] = True
            run_start = -1
    sig = np.where(ok, excess, 0.0)
    n_bins = max(1, int(np.ceil((t[-1] + hop) / bin_s)))
    out = np.zeros(n_bins)
    idx = (t / bin_s).astype(int)
    np.maximum.at(out, idx, sig)
    return out
