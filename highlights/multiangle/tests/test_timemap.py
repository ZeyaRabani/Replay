from itertools import pairwise

from highlights.multiangle.timemap import (
    assign_output_times,
    events_between,
    events_to_output,
    from_output_time_with_replays,
    output_playlist,
    to_output_time_with_replays,
)


def _replay(t_live_at: float, goal_id: str = "g", t_goal: float = 0.0):
    return {
        "goal_id": goal_id,
        "t_goal": t_goal,
        "src_angle": 1,
        "t_src_start": 10.0,
        "t_src_end": 16.0,
        "t_live_at": t_live_at,
        "speed": 0.5,
    }


def test_time_mapping_round_trip_and_replay_clamp():
    replays = assign_output_times([_replay(20), _replay(50, "h", 40)])
    for t in [0, 19, 20, 21, 35, 50, 60, 100]:
        assert from_output_time_with_replays(
            to_output_time_with_replays(t, replays), replays) == t
    first = replays[0]
    assert to_output_time_with_replays(20, replays) == 32
    assert from_output_time_with_replays(
        first["t_out_start"] + 3, replays) == 20


def test_same_insertion_time_replays_are_sequential():
    replays = assign_output_times([
        _replay(20, "later", 2),
        _replay(20, "earlier", 1),
    ])
    assert [r["goal_id"] for r in replays] == ["earlier", "later"]
    assert [(r["t_out_start"], r["t_out_end"]) for r in replays] == [
        (20.0, 32.0), (32.0, 44.0)]
    assert to_output_time_with_replays(20, replays) == 44


def test_events_mapping_and_timeline_changes():
    replay = assign_output_times([_replay(30)])[0]
    event = {"t": 29.0, "t_start": 28.0, "t_end": 32.0,
             "clip_start": 27.0, "clip_end": 33.0, "type": "goal"}
    mapped = events_to_output([event], [replay])[0]
    assert mapped == {
        "t": 29.0, "t_start": 28.0, "t_end": 44.0,
        "clip_start": 27.0, "clip_end": 45.0, "type": "goal",
    }
    assert events_between([mapped], [replay], [replay]) == [mapped]
    assert events_between([event], [], [replay])[0]["clip_end"] == 45.0
    assert events_between([mapped], [replay], [])[0]["t_end"] == 32.0


def test_output_playlist_invariants_and_empty_replays():
    segments = [
        {"t_start": 0.0, "t_end": 4.0, "angle": 0, "rule": "start"},
        {"t_start": 4.0, "t_end": 8.0, "angle": 1, "rule": "cluster"},
    ]
    replays = assign_output_times([_replay(4), _replay(8, "last", 7)])
    pieces = output_playlist(segments, replays)
    assert [piece["rule"] for piece in pieces] == [
        "start", "replay", "cluster", "replay"]
    assert all(p["t_end"] == q["t_start"] for p, q in pairwise(pieces))
    assert sum(p["t_src_end"] - p["t_src_start"]
               for p in pieces if p["rule"] != "replay") == 8
    assert pieces[-1]["t_end"] == 32
    assert pieces[1]["overlay"] == "REPLAY"
    assert output_playlist(segments, []) == [
        {**segments[0], "t_src_start": 0.0, "t_src_end": 4.0, "speed": 1.0},
        {**segments[1], "t_src_start": 4.0, "t_src_end": 8.0, "speed": 1.0},
    ]


def test_replays_past_live_end_are_dropped():
    pieces = output_playlist(
        [{"t_start": 0.0, "t_end": 8.0, "angle": 0}],
        [_replay(9)],
    )
    assert len(pieces) == 1
    assert pieces[-1]["t_end"] == 8.0
