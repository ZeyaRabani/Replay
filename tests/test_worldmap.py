import json
import math
from pathlib import Path

import numpy as np
import pytest

from pitchworld import worldmap as wm

DATA = Path(__file__).parent / "data" / "tracking_50frames.json"


@pytest.fixture(scope="module")
def tracking():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def mapped(tracking):
    return wm.map_tracking(tracking, seed_cam=0)


def test_world_frame_is_seed_camera(tracking, mapped):
    pose = tracking["cameras"][0]["calibration"]["pose"]
    wf = mapped["world_frame"]
    assert wf["source"] == "seed_camera_pose"
    assert wf["origin_pitch_m"] == pytest.approx([pose["x"], pose["y"], pose["height"]], abs=1e-3)
    assert wf["scale_is_assumption"] is True
    # the camera's own position maps to the world origin
    frame = wm.world_frame(tracking, 0)
    assert frame.to_world(pose["x"], pose["y"], pose["height"]) == [0.0, 0.0, 0.0]


def test_axes_convention(tracking):
    """x-right / y-down / z-forward: a point straight ahead of the camera on the ground is +z, +y (below eye)."""
    frame = wm.world_frame(tracking, 0)
    pose = tracking["cameras"][0]["calibration"]["pose"]
    yaw = math.radians(pose["yaw_deg"])
    ahead = frame.to_world(pose["x"] + 10 * math.cos(yaw), pose["y"] + 10 * math.sin(yaw), 0.0)
    assert ahead[0] == pytest.approx(0.0, abs=1e-6)
    assert ahead[1] == pytest.approx(pose["height"], abs=1e-3)
    assert ahead[2] == pytest.approx(10.0, abs=1e-6)
    # a point to the camera's right (yaw - 90 deg) has +x
    right = frame.to_world(pose["x"] + 5 * math.cos(yaw - math.pi / 2), pose["y"] + 5 * math.sin(yaw - math.pi / 2))
    assert right[0] == pytest.approx(5.0, abs=1e-6)


def test_per_frame_players(tracking, mapped):
    assert mapped["num_frames"] == len(tracking["frames"]) == 50
    for src, out in zip(tracking["frames"], mapped["frames"]):
        assert len(src["players"]) == len(out["players"])
        for p, q in zip(src["players"], out["players"]):
            assert q["id"] == p["id"]
            assert q["pitch"] == pytest.approx([p["x"], p["y"]], abs=1e-3)
            u, v = q["normalised"]
            assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
            assert u == pytest.approx(min(max(p["x"] / 50, 0), 1), abs=1e-3)
            assert len(q["world"]) == 3


def test_normalised_is_scale_free(tracking):
    a = wm.map_tracking(tracking, seed_cam=0, scale=1.0)
    b = wm.map_tracking(tracking, seed_cam=0, scale=3.0)
    pa, pb = a["frames"][0]["players"][0], b["frames"][0]["players"][0]
    assert pa["normalised"] == pb["normalised"]
    assert np.allclose(np.array(pb["world"]), 3.0 * np.array(pa["world"]), atol=5e-3)


def test_anchors_and_heading_deltas(mapped):
    rec = next(iter(mapped["players"].values()))
    n = len(rec["frames"])
    assert len(rec["anchors"]) == n
    assert len(rec["heading_deltas"]) == max(n - 1, 0)
    for a in rec["anchors"]:
        assert a["pitch"][2] == wm.ANCHOR_HEIGHT_M
        assert len(a["world"]) == 3
    for d in rec["heading_deltas"]:
        assert len(d) == 6
        assert d[0] == 0.0 and d[2] == 0.0  # yaw-only follow camera
        assert d[4] == 0.0  # no vertical step


def test_heading_deltas_pure_translation():
    """A player walking straight along +x for 5 frames: no rotation, translation along camera z."""
    t = {
        "pitch": {"length": 50, "width": 30},
        "cameras": [],
        "quality": {},
        "frames": [{"frame": i, "t": i / 30, "players": [{"id": 1, "x": 1.0 + 0.5 * i, "y": 10.0}]} for i in range(5)],
    }
    out = wm.map_tracking(t)
    d = np.array(out["players"]["1"]["heading_deltas"])
    assert np.allclose(d[:, :3], 0.0, atol=1e-6)
    assert np.allclose(d[:, 5], 0.5, atol=1e-6)
    assert np.allclose(d[:, 3:5], 0.0, atol=1e-6)


def test_fallback_without_calibration_pose(tracking):
    stripped = json.loads(json.dumps(tracking))
    for cam in stripped["cameras"]:
        cam["calibration"].pop("pose", None)
    out = wm.map_tracking(stripped, seed_cam=0)
    assert out["world_frame"]["source"] == "pitch_origin_fallback"
    assert out["world_frame"]["origin_pitch_m"] == [0.0, 0.0, 0.0]
    assert any("no calibration pose" in n for n in out["quality"]["notes"])
    # unknown camera index also falls back rather than crashing
    assert wm.map_tracking(tracking, seed_cam=99)["world_frame"]["source"] == "pitch_origin_fallback"


def test_quality_propagation(tracking, mapped):
    q = mapped["quality"]
    assert q["calibration_low_confidence"] is True
    assert q["mapping_uncertainty_m"] == pytest.approx(
        max(tracking["quality"]["cross_camera_disagreement_m"].values()), abs=1e-3)
    assert q["world_scale_verified"] is False
    no_q = dict(tracking, quality={})
    assert wm.map_tracking(no_q)["quality"]["mapping_uncertainty_m"] == wm.DEFAULT_UNCERTAINTY_M


def test_generic_pitch_dict_and_n_cameras(tracking):
    t = json.loads(json.dumps(tracking))
    t["pitch"] = {"length": 105, "width": 68}
    t["cameras"] = t["cameras"] + [{"index": 3, "calibration": {}}]
    out = wm.map_tracking(t, seed_cam=2)
    assert out["world_frame"]["seed_camera"] == 2
    p = out["frames"][0]["players"][0]
    assert p["normalised"][0] == pytest.approx(p["pitch"][0] / 105, abs=1e-3)


def test_cli(tmp_path):
    out = tmp_path / "wp.json"
    assert wm.main([str(DATA), "--seed-cam", "1", "--out", str(out)]) == 0
    w = json.loads(out.read_text())
    assert w["world_frame"]["seed_camera"] == 1 and w["num_frames"] == 50
