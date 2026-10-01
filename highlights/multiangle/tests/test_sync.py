"""Sync: offset recovery from synthetic envelopes + waveforms."""

import numpy as np
import pytest

from highlights.multiangle import sync


def _envelope(n: int, seed: int) -> np.ndarray:
    """Sparse impulse train -> whitened onset envelope-like signal."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    idx = rng.integers(0, n, n // 10)
    e[idx] = rng.uniform(0.5, 1.0, len(idx))
    return e


def test_offset_recovery(tmp_path):
    """Two envelopes with a known 12.34 s offset, different gains + noise."""
    n = int(120 * sync.ENV_HZ)  # 2 min at 50 Hz
    rng = np.random.default_rng(0)
    base = _envelope(n, seed=1)
    shift = round(12.34 * sync.ENV_HZ)  # 617 samples = 12.34 s
    a = base + 0.05 * rng.normal(size=n)
    b = np.roll(base, shift) * 0.7 + 0.08 * rng.normal(size=n)

    off, pnr, r2 = sync.estimate_offset(a, b)
    # off = seconds to ADD to b's file time -> -12.34
    assert abs(off + 12.34) < 0.05
    assert pnr >= sync.PNR_MIN
    assert r2 <= sync.R2_MAX


def test_uncorrelated_not_confident():
    rng = np.random.default_rng(2)
    a = _envelope(6000, seed=3) + 0.05 * rng.normal(size=6000)
    b = _envelope(6000, seed=4) + 0.08 * rng.normal(size=6000)
    _off, pnr, r2 = sync.estimate_offset(a, b)
    assert not (pnr >= sync.PNR_MIN and r2 <= sync.R2_MAX)


def _tone_env(t: np.ndarray, hz: float, seed: int) -> np.ndarray:
    """Noise with a smooth *random* amplitude envelope (non-periodic, so the
    onset envelope has a unique xcorr peak)."""
    rng = np.random.default_rng(seed)
    mod = rng.uniform(0, 1, len(t))
    k = np.ones(int(sync.SR)) / sync.SR  # 1 s smoothing
    mod = np.convolve(mod, k, mode="same")
    return rng.normal(size=len(t)) * (0.2 + mod)


def test_sync_angles_real_wavs(tmp_path, monkeypatch):
    """End-to-end via fake wavs: patch _load_env to skip librosa/ffmpeg."""
    sr, dur = sync.SR, 30.0
    t = np.arange(int(dur * sr)) / sr
    sig = _tone_env(t, 3.7, 5)
    shift_s = 12.34
    b_raw = np.roll(sig, int(shift_s * sr)) * 0.8
    b_raw[int(shift_s * sr):] += np.random.default_rng(6).normal(
        scale=0.1, size=len(b_raw) - int(shift_s * sr))

    # sparse impulse envelopes (like onset_strength) shifted identically via
    # a slice (not np.roll — roll wraps the tail into the head and creates a
    # false second xcorr peak); raws carry the waveforms for _refine
    shift_env = int(shift_s * sync.ENV_HZ)
    n_env = int(dur * sync.ENV_HZ)
    e = _envelope(n_env + shift_env + 100, seed=8)
    # np.roll on the raw delays b, so the env must be delayed too: eb[t]=e[t-shift]
    ea = sync._whiten(e[shift_env:shift_env + n_env]
                      + 0.02 * np.abs(np.random.default_rng(9).normal(size=n_env)))
    eb = sync._whiten(e[:n_env] * 0.7
                      + 0.03 * np.abs(np.random.default_rng(10).normal(size=n_env)))
    calls = iter([(ea, sig), (eb, b_raw)])
    monkeypatch.setattr(sync, "_load_env", lambda w: next(calls))

    out = sync.sync_angles(["a.wav", "b.wav"], [dur, dur])
    assert out["method"] == "xcorr"
    assert abs(out["offsets"][1] + shift_s) < 0.1
    assert out["pairs"][0]["confident"]
    assert out["needs_manual"] == []


def test_manual_offsets(monkeypatch):
    sr, dur = sync.SR, 20.0
    t = np.arange(int(dur * sr)) / sr
    y = _tone_env(t, 2.0, 7)
    calls = [(sync._whiten(np.abs(np.diff(y, prepend=y[0]))
             [: (len(y) // sync.HOP) * sync.HOP].reshape(-1, sync.HOP).mean(axis=1)), y)] * 2
    monkeypatch.setattr(sync, "_load_env", lambda w: next(iter(calls)))

    out = sync.sync_angles(["a.wav", "b.wav"], [dur, dur],
                           manual_offsets=[0.0, -3.2])
    assert out["method"] == "manual"
    assert out["offsets"] == [0.0, -3.2]
    assert out["needs_manual"] == []
    assert all(p["confident"] and p.get("manual") for p in out["pairs"])


def test_refine_subsecond_shift():
    """Refine recovers a sub-second (sample-level) shift on band-limited
    noise via windowed FFT xcorr — fast path, no full-waveform scan."""
    sr, dur = sync.SR, 60.0
    rng = np.random.default_rng(11)
    sig = sync._bandpass(rng.normal(size=int(dur * sr)), *sync.BAND)
    shift_s = 12.3125
    b_raw = np.roll(sig, int(shift_s * sr)) * 0.9
    off, spread = sync._refine(sig, b_raw, -12.34, dur, dur)
    assert abs(off + shift_s) < 0.01
    assert spread <= 0.2


def _stub_sync(monkeypatch, offs):
    """Scripted xcorr/refine: offs maps (a,b) -> (offset, pnr, r2)."""
    monkeypatch.setattr(sync, "_load_env", lambda w: (np.zeros(10), np.zeros(10)))
    # estimate_offset takes envelopes, not indices — key on call order instead
    pairs_called = []
    def est(env_a, env_b):
        idx = len(pairs_called)
        seq = [(0, 1), (0, 2), (1, 2)]
        pairs_called.append(idx)
        return offs[seq[idx]]
    monkeypatch.setattr(sync, "estimate_offset", est)
    monkeypatch.setattr(sync, "_refine", lambda a, b, c, *args: (c, 0.0))


def test_triangle_accepts_weak_pnr(monkeypatch):
    """3 angles, weak pnr but consistent triangle -> all confident."""
    _stub_sync(monkeypatch, {(0, 1): (-748.2, 5.1, 0.9),
                             (0, 2): (-520.2, 3.13, 0.9),
                             (1, 2): (228.0, 2.66, 0.9)})  # -520.2 - -748.2
    out = sync.sync_angles(["a", "b", "c"], [100, 100, 100])
    assert out["method"] == "xcorr+triangle"
    assert out["triangle_residual_s"] <= sync.TRIANGLE_TOL_S
    assert all(p["confident"] and p["accepted_by"] == "triangle"
               for p in out["pairs"])
    assert out["needs_manual"] == []
    assert "triangle" in out["confidence_note"]


def test_triangle_inconsistent_needs_manual(monkeypatch):
    """Weak pnr + 2 s residual -> angles 1 and 2 flagged for manual input."""
    _stub_sync(monkeypatch, {(0, 1): (-748.2, 5.1, 0.9),
                             (0, 2): (-520.2, 3.13, 0.9),
                             (1, 2): (230.0, 2.66, 0.9)})  # off by ~2 s
    out = sync.sync_angles(["a", "b", "c"], [100, 100, 100])
    assert out["method"] == "xcorr"
    assert out["triangle_residual_s"] > sync.TRIANGLE_TOL_S
    assert out["needs_manual"] == [1, 2]


def test_offsets_validation(monkeypatch):
    sr, dur = sync.SR, 10.0
    y = np.zeros(int(dur * sr))
    e = sync._whiten(np.ones(int(dur * sync.ENV_HZ)))
    monkeypatch.setattr(sync, "_load_env", lambda w: (e, y))
    with pytest.raises(ValueError):
        sync.sync_angles(["a", "b"], [dur, dur], manual_offsets=[1.0, 2.0])
