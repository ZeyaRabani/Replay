"""Live-to-output time mapping for inserted replay segments."""

from __future__ import annotations

from itertools import pairwise

TIME_KEYS = ("t", "t_start", "t_end", "clip_start", "clip_end")


def _duration(replay: dict) -> float:
    return (float(replay["t_src_end"]) - float(replay["t_src_start"])) / float(
        replay.get("speed", 1.0))


def assign_output_times(replays: list[dict]) -> list[dict]:
    """Sort replay records and fill their output-time bounds."""
    ordered = sorted(
        (dict(replay) for replay in replays),
        key=lambda replay: (float(replay["t_live_at"]),
                            float(replay.get("t_goal", 0.0))),
    )
    shift = 0.0
    for replay in ordered:
        start = float(replay["t_live_at"]) + shift
        end = start + _duration(replay)
        replay["t_out_start"] = round(start, 3)
        replay["t_out_end"] = round(end, 3)
        shift += _duration(replay)
    return ordered


def to_output_time_with_replays(t: float, replays: list[dict]) -> float:
    """Map a live-time point to output time, after replays at that point."""
    return float(t) + sum(
        _duration(replay) for replay in replays
        if float(replay["t_live_at"]) <= float(t))


def from_output_time_with_replays(o: float, replays: list[dict]) -> float:
    """Map output time back to live time, clamping points inside replays."""
    shift = 0.0
    for replay in replays:
        start = float(replay["t_out_start"])
        end = float(replay["t_out_end"])
        if float(o) < start:
            break
        if float(o) < end:
            return float(replay["t_live_at"])
        shift += end - start
    return float(o) - shift


def events_to_output(events: list[dict], replays: list[dict]) -> list[dict]:
    """Copy events and map their live-time fields to output time."""
    out = []
    for event in events:
        mapped = dict(event)
        for key in TIME_KEYS:
            if key in mapped and mapped[key] is not None:
                mapped[key] = round(
                    to_output_time_with_replays(float(mapped[key]), replays), 1)
        out.append(mapped)
    return out


def events_between(events: list[dict], old_replays: list[dict],
                   new_replays: list[dict]) -> list[dict]:
    """Remap event times from one output timeline to another."""
    out = []
    for event in events:
        mapped = dict(event)
        for key in TIME_KEYS:
            if key not in mapped or mapped[key] is None:
                continue
            live = from_output_time_with_replays(
                float(mapped[key]), old_replays)
            mapped[key] = round(
                to_output_time_with_replays(live, new_replays), 1)
        out.append(mapped)
    return out


def output_playlist(segments: list[dict], replays: list[dict]) -> list[dict]:
    """Interleave live director segments with inserted replay pieces."""
    if not segments:
        return []
    ordered = assign_output_times(replays)
    live_end = max(float(segment["t_end"]) for segment in segments)
    pending = [replay for replay in ordered
               if float(replay["t_live_at"]) <= live_end]
    pieces: list[dict] = []
    next_replay = 0
    output_cursor = 0.0

    def emit_replays_before(live_t: float) -> None:
        nonlocal next_replay, output_cursor
        while (next_replay < len(pending)
               and float(pending[next_replay]["t_live_at"]) <= live_t):
            replay = pending[next_replay]
            replay_start = float(replay["t_out_start"])
            replay_end = float(replay["t_out_end"])
            piece = {
                "t_start": replay_start,
                "t_end": replay_end,
                "angle": int(replay["src_angle"]),
                "rule": "replay",
                "score": 0.0,
                "runner_up": None,
                "t_src_start": float(replay["t_src_start"]),
                "t_src_end": float(replay["t_src_end"]),
                "speed": float(replay.get("speed", 1.0)),
                "overlay": "REPLAY",
                "replay_of": replay["goal_id"],
            }
            pieces.append(piece)
            output_cursor = replay_end
            next_replay += 1

    for segment in segments:
        start = float(segment["t_start"])
        end = float(segment["t_end"])
        emit_replays_before(start)
        splits = [float(replay["t_live_at"]) for replay in pending[next_replay:]
                  if start < float(replay["t_live_at"]) < end]
        boundaries = [start, *sorted(set(splits)), end]
        for live_start, live_end in pairwise(boundaries):
            emit_replays_before(live_start)
            output_start = output_cursor
            output_end = output_start + live_end - live_start
            pieces.append({
                **segment,
                "t_start": output_start,
                "t_end": output_end,
                "t_src_start": live_start,
                "t_src_end": live_end,
                "speed": 1.0,
            })
            output_cursor = output_end
            emit_replays_before(live_end)
    emit_replays_before(live_end)
    return pieces
