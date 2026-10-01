"""Director style presets: 'fast' cuts more eagerly than 'normal'."""

import numpy as np

from highlights.multiangle.director import STYLES, cut_director


def _alternating_tracks(T=240, n=2, period=5):
    """Best angle alternates every `period` seconds (cluster scores)."""
    tracks = []
    avail = np.ones((n, T), dtype=bool)
    for i in range(n):
        cl = np.array(
            [1.0 if (t // period) % n == i else 0.2 for t in range(T)])
        tracks.append({"ball_conf": np.zeros(T), "ball_size": np.zeros(T),
                       "cluster": cl})
    return tracks, avail


def test_fast_produces_more_cuts_than_normal():
    T = 240
    tracks, avail = _alternating_tracks(T)
    motion = [np.zeros(T), np.zeros(T)]
    d_norm = cut_director(tracks, avail, motion, style="normal")
    d_fast = cut_director(tracks, avail, motion, style="fast")
    assert d_fast["n_cuts"] > d_norm["n_cuts"]
    assert d_norm["style"] == "normal" and d_fast["style"] == "fast"
    # fast actually follows the alternation with short segments
    assert d_fast["median_hold_s"] < d_norm["median_hold_s"]


def test_default_style_is_normal():
    tracks, avail = _alternating_tracks(120)
    motion = [np.zeros(120), np.zeros(120)]
    d = cut_director(tracks, avail, motion)
    assert d["style"] == "normal"


def test_styles_shape():
    assert set(STYLES) == {"normal", "fast"}
    for s in STYLES.values():
        assert s.min_hold > 0 and s.smooth_mean > 0
