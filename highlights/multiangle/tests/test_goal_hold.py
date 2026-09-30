import numpy as np

from highlights.multiangle import director


def _inputs(T: int = 80):
    tracks = [
        {"ball_conf": np.zeros(T), "ball_size": np.zeros(T),
         "cluster": np.ones(T)},
        {"ball_conf": np.zeros(T), "ball_size": np.zeros(T),
         "cluster": np.ones(T)},
    ]
    available = np.ones((2, T), dtype=bool)
    motion = [np.zeros(T), np.zeros(T)]
    return tracks, available, motion


def test_goal_hold_forces_angle_then_resumes_without_counting_split_as_cut():
    tracks, available, motion = _inputs()
    out = director.cut_director(
        tracks, available, motion,
        goal_windows=[{"t_start": 20, "t_end": 40,
                       "angle": 1, "events": ["g"], "rule": "goal_hold"}],
    )
    hold = next(segment for segment in out["segments"]
                if segment["rule"] == "goal_hold")
    assert hold["angle"] == 1
    assert (hold["t_start"], hold["t_end"]) == (20.0, 40.0)
    assert all(segment["angle"] == 1 for segment in out["segments"]
               if segment["t_start"] < 40 and segment["t_end"] > 20)
    assert any(segment["rule"] == "resume"
               and segment["t_start"] == 40.0 for segment in out["segments"])
    assert out["n_cuts"] == 1
    assert "goal_hold" in out["per_second_rule"]


def test_no_goal_windows_preserves_output():
    tracks, available, motion = _inputs()
    assert director.cut_director(tracks, available, motion) == \
        director.cut_director(tracks, available, motion, goal_windows=None)
