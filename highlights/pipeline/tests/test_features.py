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
