import numpy as np

from highlights.audio import audio_spikes


def test_sustained_spike_detected_blip_ignored():
    hop = 0.1
    t = np.arange(0, 120, hop)
    db = np.full_like(t, -40.0)
    rng = np.random.default_rng(0)
    db += rng.normal(0, 1.0, len(t))
    # 3 s plateau +10 dB at t=60
    db[(t >= 60) & (t < 63)] += 10.0
    # 0.3 s blip +15 dB at t=30
    db[(t >= 30) & (t < 30.3)] += 15.0
    sig = audio_spikes(t, db, bin_s=0.5, baseline_win_s=20, thresh_db=6, min_dur_s=1.0)
    bins = np.where(sig > 0)[0]
    assert len(bins) > 0
    times = bins * 0.5
    assert times.min() >= 59.0 and times.max() <= 64.0
    assert sig[int(30.0 / 0.5)] == 0.0
