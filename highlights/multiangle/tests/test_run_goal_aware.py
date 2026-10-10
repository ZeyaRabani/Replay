import json

from highlights.multiangle.run import Ctx, confirmed_events
from highlights.multiangle.timemap import assign_output_times


def _replay():
    return assign_output_times([{
        "goal_id": "g",
        "t_goal": 5.0,
        "src_angle": 1,
        "t_src_start": 0.0,
        "t_src_end": 6.0,
        "t_live_at": 2.0,
        "speed": 0.5,
    }])


def test_confirmed_events_maps_review_store_output_times(tmp_path):
    (tmp_path / "project.json").write_text(json.dumps({
        "meta": {"replays_applied": _replay()},
        "candidates": [
            {"id": "pending", "type": "goal", "t": 5.0,
             "status": "pending"},
            {"id": "chance", "type": "chance", "t": 17.0,
             "status": "confirmed"},
            {"id": "goal", "type": "goal", "t": 17.0,
             "status": "confirmed"},
            {"id": "shot", "type": "shot", "t": 5.0,
             "status": "confirmed"},
        ],
    }))
    ctx = Ctx(tmp_path, tmp_path / "multiangle", object())

    assert confirmed_events(ctx) == [
        {"id": "shot", "type": "shot", "t": 2.0},
        {"id": "goal", "type": "goal", "t": 5.0},
    ]


def test_confirmed_events_falls_back_to_pipeline_candidates(tmp_path):
    pipe = tmp_path / "multiangle"
    pipe.mkdir()
    (tmp_path / "project.json").write_text(json.dumps({"candidates": []}))
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "candidates.json").write_text(json.dumps({
        "events": [
            {"id": "goal", "type": "goal", "t": 17.0,
             "status": "confirmed"},
        ],
    }))
    (pipe / "director.json").write_text(json.dumps({"replays": _replay()}))
    ctx = Ctx(tmp_path, pipe, object())

    assert confirmed_events(ctx) == [
        {"id": "goal", "type": "goal", "t": 5.0},
    ]


def test_duration_falls_back_to_feature_rows(tmp_path):
    angle = tmp_path / "angles" / "a0"
    features = angle / "track" / "features_1s.json"
    features.parent.mkdir(parents=True)
    features.write_text(json.dumps({
        "columns": ["t"],
        "rows": [[0.0], [1.0], [4.0]],
    }))
    ctx = Ctx(tmp_path, tmp_path / "multiangle", object(),
              angles=[{"dir": angle}])

    assert ctx.duration(0) == 5.0
