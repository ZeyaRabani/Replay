from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pitchworld.pitch import PitchModel

REPO = Path(__file__).resolve().parents[1]


def test_standard_landmarks_geometry():
    p = PitchModel.standard()
    lm = p.landmarks()
    assert lm["corner_A_S"] == (0, 0)
    assert lm["corner_B_N"] == (105.0, 68.0)
    assert lm["centre"] == (52.5, 34.0)
    assert lm["A_post_S"] == pytest.approx((0, 34 - 3.66))
    assert lm["B_post_N"] == pytest.approx((105, 34 + 3.66))
    assert lm["A_box_S"] == pytest.approx((16.5, 34 - 20.16))
    assert lm["B_6yd_N"] == pytest.approx((105 - 5.5, 34 + 9.16))
    assert not any(k.startswith(("A_D", "B_D")) for k in lm)


def test_standard_segments_and_arcs():
    p = PitchModel.standard()
    segs = p.segments()
    assert ((0, 0), (105.0, 0)) in segs
    assert ((52.5, 0), (52.5, 68.0)) in segs
    assert len(segs) == 5 + 6 + 6  # outline + half, two boxes, two 6-yd areas
    arcs = p.arcs()
    assert arcs == [((52.5, 34.0), 9.15, 0, 360)]
    for (a, b) in segs:
        assert p.contains(*a) and p.contains(*b)


def test_small_sided_landmarks_and_arcs():
    p = PitchModel.small_sided(length=50, width=30)
    assert (p.penalty_depth, p.goal_area_depth, p.centre_circle_radius) == (0, 0, 0)
    lm = p.landmarks()
    assert not any("box" in k or "6yd" in k for k in lm)
    r, cy = 6.0, 15.0
    assert lm["A_D_apex"] == (r, cy)
    assert lm["B_D_apex"] == (50 - r, cy)
    assert lm["A_D_S"] == (0, cy - r) and lm["B_D_N"] == (50, cy + r)
    for name, cx in (("A_D_45S", 0), ("A_D_45N", 0), ("B_D_45S", 50), ("B_D_45N", 50)):
        x, y = lm[name]
        assert math.hypot(x - cx, y - cy) == pytest.approx(r)
    assert p.segments() == [((0, 0), (50, 0)), ((50, 0), (50, 30)), ((50, 30), (0, 30)), ((0, 30), (0, 0)),
                            ((25.0, 0), (25.0, 30))]
    assert p.arcs() == [((0, cy), r, -90, 90), ((50, cy), r, 90, 270)]


def test_resolve_and_contains():
    p = PitchModel.small_sided()
    assert p.resolve("centre") == (20.0, 15.0)
    assert p.resolve([1, "2.5"]) == (1.0, 2.5)
    with pytest.raises(KeyError, match="unknown landmark"):
        p.resolve("not_a_landmark")
    assert p.contains(-1.9, 31.9)
    assert not p.contains(-2.1, 15)
    assert not p.contains(20, 32.5)
    assert p.contains(20, 32.5, margin=3)


def test_to_dict_load_round_trip(tmp_path):
    p = PitchModel.small_sided(length=50, width=30, goal_width=3.66, d_radius=6.0)
    d = p.to_dict()
    f = tmp_path / "pitch.json"
    f.write_text(json.dumps(d))
    assert PitchModel.load(f) == p
    assert PitchModel.load(f).to_dict() == d


def test_load_preset_with_overrides(tmp_path):
    f = tmp_path / "pitch.json"
    f.write_text(json.dumps({"preset": "small_sided", "length": 55, "d_radius": 5.5}))
    p = PitchModel.load(f)
    assert p.length == 55 and p.d_radius == 5.5
    assert p.width == 30.0 and p.goal_width == 3.66  # small_sided defaults kept
    assert p.penalty_depth == 0 and p.centre_circle_radius == 0
    f.write_text(json.dumps({"width": 64}))
    q = PitchModel.load(f)
    assert q == PitchModel(width=64)


def test_example_pitch_file_loads():
    p = PitchModel.load(REPO / "examples" / "pitch_small_sided_50x30.json")
    assert (p.length, p.width) == (50, 30)
    assert p.d_radius > 0
    assert "A_D_apex" in p.landmarks()
