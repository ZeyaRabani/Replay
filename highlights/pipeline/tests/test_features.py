import json

import numpy as np

from highlights.pipeline.features import (
    AUDIO_COLS,
    MOTION_COLS,
    SPOTTING_COLS,
    TRACKING_COLS,
    build_features,
    detect_match_window,
)


def _table(cols, n, seed=0):
    rng = np.random.default_rng(seed)
    return {"source": "x", "step_s": 1.0, "video_duration_s": float(n),
            "columns": ["t"] + cols,
            "rows": [[float(t)] + list(rng.normal(size=len(cols)))
                     for t in range(n)]}


def _write(tmp_path):
    audio = tmp_path / "audio.json"
    motion = tmp_path / "motion.json"
    whistles = tmp_path / "whistles.json"
    audio.write_text(json.dumps(_table(
        ["rms_db", "z60_rms", "z300_rms", "z300_speech",
         "onset_density", "whistle_frac"], 300, seed=1)))
    motion.write_text(json.dumps(_table(
        ["motion_total", "motion_goal_roi", "motion_far",
         "near_frac", "net_disturbance"], 300, seed=2)))
    whistles.write_text(json.dumps({"whistles": [
        {"t_start": 10.0, "t_end": 11.0, "duration_s": 1.0}]}))
    return audio, motion, whistles


def test_build_features_columns(tmp_path):
    audio, motion, whistles = _write(tmp_path)
    df = build_features(audio, motion, whistles, 300.0, (30, 270))
    for c in MOTION_COLS + AUDIO_COLS + SPOTTING_COLS + TRACKING_COLS:
        assert c in df.columns
    assert "whistle" in df.columns and "in_match" in df.columns
    assert "motion_total_raw" in df.columns and "rms_db_raw" in df.columns
    assert len(df) == 300
    # tracking/spotting zero-filled
    assert (df[TRACKING_COLS + SPOTTING_COLS].to_numpy() == 0).all()
    # whistle flag set at t=10
    assert df.loc[df["t"] == 10, "whistle"].iloc[0] != 0
    # in_match window applied
    im = df.set_index("t")["in_match"]
    assert im.loc[30] == 1 and im.loc[270] == 1
    assert im.loc[10] == 0 and im.loc[299] == 0
    # z-scored: mean ~0 over present cols
    assert abs(df["motion_total"].mean()) < 1e-9


def test_detect_match_window_short_video(tmp_path):
    audio, _motion, whistles = _write(tmp_path)
    lo, hi, halves, warning = detect_match_window(audio, whistles, 300.0)
    assert (lo, hi) == (0.0, 300.0)
    assert halves == []
    assert warning


def test_detect_match_window_motion_first(tmp_path):
    """quiet 0-600, active 600-3000, quiet 3000-3900, active 3900-6300, quiet."""
    n = 6600
    rng = np.random.default_rng(3)
    motion = np.full(n, 10.0)  # quiet baseline
    for a, b in ((600, 3000), (3900, 6300)):
        motion[a:b] = 100.0
    motion += rng.normal(0, 2, n)
    motion_j = tmp_path / "motion.json"
    motion_j.write_text(json.dumps({
        "source": "x", "step_s": 1.0, "video_duration_s": float(n),
        "columns": ["t", "motion_total"],
        "rows": [[float(t), float(motion[t])] for t in range(n)]}))
    audio_j = tmp_path / "audio.json"
    audio_j.write_text(json.dumps(_table(
        ["rms_db", "z60_rms", "z300_rms", "z300_speech",
         "onset_density", "whistle_frac"], n, seed=4)))
    whistles_j = tmp_path / "whistles.json"
    whistles_j.write_text(json.dumps({"whistles": []}))

    lo, hi, halves, warning = detect_match_window(
        audio_j, whistles_j, float(n), motion_j)
    assert warning is None
    # rolling mean blurs edges by ~150 s; allow that margin
    assert 450 <= lo <= 750
    assert 6150 <= hi <= 6450
    assert len(halves) == 2
    assert 2850 <= halves[0]["end"] <= 3150
