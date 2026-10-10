import json
from pathlib import Path

import numpy as np
import pytest

from highlights.multiangle.goal_aware import end_views, plan_goal_aware
from highlights.multiangle.timemap import (
    from_output_time_with_replays,
    to_output_time_with_replays,
)

DATA = Path(__file__).parent / "data" / "m2808"
OFFSETS = [0.0, -748.2, -520.2]
LO = 1151.8


def _fixture():
    calib = json.loads((DATA / "multiangle/calib_refined.json").read_text())
    events = json.loads((DATA / "project.json").read_text())["candidates"]
    T = int(max(event["t"] for event in events)) + 22
    tracks = []
    available = np.zeros((len(OFFSETS), T), dtype=bool)
    fields = {
        "ball_conf": "ball_conf",
        "ball_size": "ball_size",
        "ball_x": "ball_x",
        "ball_y": "ball_y",
        "players_cx": "players_cx",
        "players_cy": "players_cy",
        "cluster": "cluster_score",
    }
    for angle, offset in enumerate(OFFSETS):
        data = json.loads(
            (DATA / f"angles/a{angle}/track/features_1s.json").read_text())
        columns = data["columns"]
        rows = data["rows"]
        values = {
            key: np.zeros(T, dtype=float)
            for key in fields
        }
        for row in rows:
            shared_t = round(float(row[columns.index("t")]) + offset - LO)
            if not 0 <= shared_t < T:
                continue
            available[angle, shared_t] = True
            for key, column in fields.items():
                values[key][shared_t] = float(row[columns.index(column)])
        tracks.append(values)
    return calib, events, tracks, available


def test_2808_end_view_ranking():
    calib, _events, _tracks, _available = _fixture()
    scores = end_views(calib, 3)
    assert [angle for angle, _score in scores["right"]] == [0, 1, 2]
    assert [angle for angle, _score in scores["left"]] == [2, 0, 1]
    assert [score for _angle, score in scores["right"]] == pytest.approx(
        [0.2048, 0.0730, 0.0085], abs=2e-3)
    assert [score for _angle, score in scores["left"]] == pytest.approx(
        [0.6602, 0.0059, 0.0035], abs=2e-3)


def test_2808_goal_diagnostics_replays_and_time_round_trip():
    calib, events, tracks, available = _fixture()
    plan = plan_goal_aware(events, tracks, available, calib)
    diagnostics = {event["id"]: event for event in plan["events"]}

    for candidate_id in ("c099", "c142"):
        event = diagnostics[candidate_id]
        assert event["end"] == "left"
        assert event["method"] == "calib_ball"
        assert event["hold_angle"] == 2
        assert event["replay_angle"] == 0

    c143 = diagnostics["c143"]
    assert c143["hold_angle"] != c143["replay_angle"]
    assert c143["method"] in ("features", "calib_players")

    goals = [event for event in events if event["type"] == "goal"]
    assert len(plan["replays"]) == len(goals)
    for replay in plan["replays"]:
        assert replay["t_out_end"] - replay["t_out_start"] == pytest.approx(12)

    for event in events:
        t = float(event["t"])
        output_t = to_output_time_with_replays(t, plan["replays"])
        assert from_output_time_with_replays(
            output_t, plan["replays"]) == pytest.approx(t)
