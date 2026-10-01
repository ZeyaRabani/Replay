from pathlib import Path

import cv2
import numpy as np

from pitchworld.calibrate import CameraCalibration
from pitchworld.merge import fuse
from pitchworld.pitch import PitchModel
from pitchworld.postprocess import (
    cluster_teams,
    filter_static_tracks,
    id_stability,
    jersey_feature,
    smooth_tracks,
)

PITCH = PitchModel.load(Path("examples/pitch_small_sided_50x30.json"))
CAL = CameraCalibration(H=np.eye(3).tolist(), method="manual", reproj_error_m=0, reproj_error_px=0,
                        n_points=4, confidence=1.0)


def person(x, y, tid, conf=0.9, h=20, cls=0):
    return {"id": tid, "cls": cls, "conf": conf, "box": [x - 2, y - h, x + 2, y]}


def test_fuse_filters_and_merges_players_and_ball():
    frames = [
        [person(10, 10, 1), person(-10, 10, 2), person(20, 20, 3, h=5), person(15, 15, -1, cls=32)],
    ]
    frames2 = [[person(10.5, 10, 4), person(15.2, 15, -1, cls=32, conf=0.8)]]
    result = fuse([frames, frames2], [CAL, CAL], PITCH, 30, ball=True)
    assert len(result[0]["players"]) == 1
    assert result[0]["players"][0]["cameras"] == [0, 1]
    assert result[0]["ball"]["cameras"] == [0, 1]


def test_smooth_tracks_interpolates_drops_and_computes_speed():
    timeline = []
    for frame in range(20):
        players = []
        if frame != 7:
            players.append({"id": 1, "x": frame, "y": 0, "conf": 1, "cameras": [], "detections": []})
        if frame < 5:
            players.append({"id": 2, "x": 0, "y": 0, "conf": 1, "cameras": [], "detections": []})
        timeline.append({"frame": frame, "t": frame / 30, "players": players})
    out = smooth_tracks(timeline, 30, max_gap=5, min_frames=15)
    p = next(p for p in out[7]["players"] if p["id"] == 1)
    assert p["interpolated"] is True
    assert all(p["id"] != 2 for fr in out for p in fr["players"])
    assert 28.5 < out[10]["players"][0]["speed"] < 31.5


def test_id_stability():
    timeline = []
    for frame in range(13):
        players = []
        if frame <= 10:
            players.append({"id": 1, "x": 10, "y": 10, "cameras": []})
        if frame >= 12:
            players.append({"id": 2, "x": 10.5, "y": 10, "cameras": []})
        timeline.append({"frame": frame, "t": frame / 30, "players": players})
    assert id_stability(timeline)["id_switches_est"] == 1


def test_cluster_teams():
    red = np.array([255, 0, 255, 180.0])
    blue = np.array([0, 255, 255, 180.0])
    black = np.array([0, 0, 0, 20.0])
    features = {i: red for i in range(6)} | {i + 6: blue for i in range(6)} | {12: black}
    result = cluster_teams(features)
    assert sum(team == 0 for team, role in result.values() if role == "player") == 6
    assert sum(team == 1 for team, role in result.values() if role == "player") == 6
    assert result[12] == (None, "other")


def test_jersey_feature():
    image = np.full((100, 100, 3), (40, 110, 40), np.uint8)
    image[20:80, 30:70] = (0, 0, 255)
    feature = jersey_feature(image, [20, 10, 80, 90])
    assert feature is not None and feature[2] > 150
    assert jersey_feature(np.full_like(image, (40, 110, 40)), [20, 10, 80, 90]) is None


def test_filter_static_tracks():
    timeline = []
    for frame in range(100):
        timeline.append({"frame": frame, "t": frame / 30, "players": [
            {"id": 1, "x": -1, "y": 15},
            {"id": 2, "x": 5 + frame * 0.1, "y": 15},
        ]})
    out, dropped = filter_static_tracks(timeline, PITCH, 30)
    assert dropped == [1]
    assert all(p["id"] == 2 for fr in out for p in fr["players"])


def test_jersey_feature_video_import_is_available():
    assert cv2.__version__
