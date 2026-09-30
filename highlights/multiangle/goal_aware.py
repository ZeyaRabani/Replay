"""Goal-oriented camera selection and replay planning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from highlights.analysis.stats import SHOT_TYPES
from highlights.multiangle.timemap import assign_output_times

HOLD_PRE = 6
HOLD_POST = 4
REPLAY_PRE = 5
REPLAY_POST = 1
REPLAY_SPEED = 0.5
END_WINDOW = (-3, 1)
SHOT_FAMILY = SHOT_TYPES


def load_calib(pipe: Path) -> dict | None:
    for name in ("calib_refined.json", "calib.json"):
        try:
            return json.loads((pipe / name).read_text())
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return None
    return None


def _project(H: np.ndarray, x: float, y: float) -> tuple[float, float, float] | None:
    point = np.asarray([x, y, 1.0], dtype=float)
    mapped = H @ point
    w = float(mapped[2])
    if abs(w) < 1e-12:
        return None
    return float(mapped[0] / w), float(mapped[1] / w), w


def _clip_edge(poly: list[tuple[float, float]], edge: int
               ) -> list[tuple[float, float]]:
    if not poly:
        return []

    def inside(point: tuple[float, float]) -> bool:
        x, y = point
        return (x >= 0 if edge == 0 else
                x <= 1 if edge == 1 else
                y >= 0 if edge == 2 else y <= 1)

    def intersection(a: tuple[float, float], b: tuple[float, float]
                     ) -> tuple[float, float]:
        ax, ay = a
        bx, by = b
        if edge < 2:
            x = 0.0 if edge == 0 else 1.0
            t = (x - ax) / (bx - ax)
            return x, ay + t * (by - ay)
        y = 0.0 if edge == 2 else 1.0
        t = (y - ay) / (by - ay)
        return ax + t * (bx - ax), y

    output = []
    previous = poly[-1]
    for current in poly:
        previous_inside, current_inside = inside(previous), inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
    return output


def _area(poly: list[tuple[float, float]]) -> float:
    clipped = poly
    for edge in range(4):
        clipped = _clip_edge(clipped, edge)
    if len(clipped) < 3:
        return 0.0
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(clipped, clipped[1:] + clipped[:1])
    )) / 2


def _clip_pitch_w(poly: list[tuple[float, float]], inverse: np.ndarray,
                  w_sign: float) -> list[tuple[float, float]]:
    if not poly:
        return []
    threshold = 1e-3

    def signed_w(point: tuple[float, float]) -> float:
        return w_sign * float(inverse[2] @ np.asarray([point[0], point[1], 1.0]))

    output = []
    previous = poly[-1]
    previous_w = signed_w(previous)
    previous_inside = previous_w >= threshold
    for current in poly:
        current_w = signed_w(current)
        current_inside = current_w >= threshold
        if current_inside != previous_inside:
            t = (threshold - previous_w) / (current_w - previous_w)
            output.append((
                previous[0] + t * (current[0] - previous[0]),
                previous[1] + t * (current[1] - previous[1]),
            ))
        if current_inside:
            output.append(current)
        previous, previous_w = current, current_w
        previous_inside = current_inside
    return output


def end_views(calib: dict, n_angles: int) -> dict[str, list[tuple[int, float]]]:
    pitch = calib["pitch"]
    length, width = float(pitch["len_m"]), float(pitch["wid_m"])
    scores: dict[str, list[tuple[int, float]]] = {"left": [], "right": []}
    for angle in range(n_angles):
        angle_calib = (calib.get("angles") or {}).get(str(angle)) or {}
        try:
            inverse = np.linalg.inv(np.asarray(angle_calib["H"], dtype=float))
            center = _project(inverse, length / 2, width / 2)
        except (KeyError, ValueError, np.linalg.LinAlgError):
            center = None
            inverse = None
        for end in ("left", "right"):
            score = 0.0
            if inverse is not None and center is not None:
                x0, x1 = ((0.0, length / 3) if end == "left"
                          else (2 * length / 3, length))
                pitch_corners = [
                    (x0, 0.0), (x1, 0.0), (x1, width), (x0, width)]
                w_sign = 1.0 if center[2] > 0 else -1.0
                clipped_pitch = _clip_pitch_w(
                    pitch_corners, inverse, w_sign)
                projected = [
                    _project(inverse, x, y) for x, y in clipped_pitch]
                if all(point is not None for point in projected):
                    score = _area([(point[0], point[1])
                                   for point in projected if point is not None])
            scores[end].append((angle, float(score)))
    for end in scores:
        scores[end].sort(key=lambda item: (-item[1], item[0]))
    return scores


def _in_pitch(calib: dict, angle: int, x: float, y: float
              ) -> tuple[float, float] | None:
    try:
        H = np.asarray(calib["angles"][str(angle)]["H"], dtype=float)
        point = _project(H, x, y)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None
    if point is None:
        return None
    px, py, _ = point
    length = float(calib["pitch"]["len_m"])
    width = float(calib["pitch"]["wid_m"])
    if -2 <= px <= length + 2 and -2 <= py <= width + 2:
        return px, py
    return None


def event_end(calib: dict | None, tracks: list[dict], t: int
              ) -> tuple[str | None, str, float | None]:
    if calib is None:
        return None, "features", None
    length = float(calib["pitch"]["len_m"])
    x_values = []
    for angle, track in enumerate(tracks):
        conf = track.get("ball_conf")
        xs, ys = track.get("ball_x"), track.get("ball_y")
        if conf is None or xs is None or ys is None:
            continue
        for second in range(max(0, t + END_WINDOW[0]),
                            min(len(conf), t + END_WINDOW[1] + 1)):
            if float(conf[second]) > 0:
                point = _in_pitch(
                    calib, angle, float(xs[second]), float(ys[second]))
                if point is not None:
                    x_values.append(point[0])
    if x_values:
        x_med = float(np.median(x_values))
        return ("right" if x_med > length / 2 else "left",
                "calib_ball", x_med)

    x_values = []
    for angle, track in enumerate(tracks):
        xs, ys = track.get("players_cx"), track.get("players_cy")
        if xs is None or ys is None:
            continue
        for second in range(max(0, t + END_WINDOW[0]),
                            min(len(xs), t + END_WINDOW[1] + 1)):
            x, y = float(xs[second]), float(ys[second])
            if x != 0 or y != 0:
                point = _in_pitch(calib, angle, x, y)
                if point is not None:
                    x_values.append(point[0])
    if x_values:
        x_med = float(np.median(x_values))
        end = ("left" if x_med < length / 3 else
               "right" if x_med > 2 * length / 3 else None)
        return end, "calib_players", x_med
    return None, "features", None


def _feature_ranking(tracks: list[dict], available: np.ndarray, t: int,
                     t0: int, t1: int) -> list[int]:
    T = available.shape[1]
    first, last = max(0, t - HOLD_PRE), min(T, t + HOLD_POST)
    p90 = []
    for angle, track in enumerate(tracks):
        cluster = np.asarray(track.get("cluster", np.zeros(T)), dtype=float)
        mask = available[angle, :len(cluster)] & (cluster > 0)
        p90.append(float(np.percentile(cluster[mask], 90)) if mask.any()
                   else 1.0)

    hold_len = max(1, t1 - t0)
    fully_available = [
        angle for angle in range(len(tracks))
        if available[angle, t0:t1].size == hold_len
        and available[angle, t0:t1].all()
    ]
    eligible = fully_available or [
        angle for angle in range(len(tracks))
        if float(available[angle, t0:t1].sum()) / hold_len >= 0.5
    ]
    scored = []
    for angle in eligible:
        track = tracks[angle]
        ball = np.asarray(track.get("ball_conf", np.zeros(T)), dtype=float)
        cluster = np.asarray(track.get("cluster", np.zeros(T)), dtype=float)
        ball_count = int((ball[first:last] > 0).sum())
        values = cluster[first:last]
        mean = float(values.mean()) if len(values) else 0.0
        scored.append((angle, ball_count + mean / (p90[angle] or 1.0)))
    return [angle for angle, _score in
            sorted(scored, key=lambda item: (-item[1], item[0]))]


def rank_cameras(calib: dict | None, tracks: list[dict],
                 available: np.ndarray, t: int, t0: int, t1: int
                 ) -> tuple[list[int], dict]:
    end, method, x_med = event_end(calib, tracks, t)
    hold_len = max(1, t1 - t0)
    fully_available = [
        angle for angle in range(len(tracks))
        if available[angle, t0:t1].size == hold_len
        and available[angle, t0:t1].all()
    ]
    feature_order = _feature_ranking(tracks, available, t, t0, t1)
    if end is not None and calib is not None:
        calibrated = [
            angle for angle, score in end_views(calib, len(tracks))[end]
            if score > 0 and angle in fully_available
        ]
        if calibrated:
            ranking = calibrated + [
                angle for angle in feature_order if angle not in calibrated]
            return ranking, {"end": end, "method": method, "x_med": x_med}
    return feature_order, {"end": end, "method": "features", "x_med": x_med}


def plan_goal_aware(events: list[dict], tracks: list[dict],
                    available: np.ndarray, calib: dict | None) -> dict:
    T = available.shape[1]
    planned = []
    diagnostics = []
    for event in events:
        t = float(event["t"])
        tc = round(t)
        if not 0 <= tc < T:
            continue
        start, end = max(0, tc - HOLD_PRE), min(T, tc + HOLD_POST)
        ranking, diag = rank_cameras(calib, tracks, available, tc, start, end)
        if not ranking:
            continue
        item = {
            "id": event["id"],
            "type": event["type"],
            "t": t,
            "end": diag["end"],
            "method": diag["method"],
            "x_med": diag["x_med"],
            "hold_angle": int(ranking[0]),
            "replay_angle": None,
            "ranking": [int(angle) for angle in ranking],
        }
        entry = {"start": start, "end": end, "angle": int(ranking[0]),
                 "event": item, "replay": None}
        if event["type"] == "goal":
            src_start = max(0, tc - REPLAY_PRE)
            src_end = min(T, tc + REPLAY_POST)
            replay_angle = next((
                angle for angle in ranking[1:]
                if available[angle, src_start:src_end].all()), None)
            item["replay_angle"] = (int(replay_angle)
                                    if replay_angle is not None else None)
            if replay_angle is None:
                item["replay_reason"] = "no second available angle for replay source"
            else:
                entry["replay"] = {
                    "goal_id": event["id"],
                    "t_goal": t,
                    "src_angle": int(replay_angle),
                    "t_src_start": float(src_start),
                    "t_src_end": float(src_end),
                    "t_live_at": 0.0,
                    "speed": REPLAY_SPEED,
                }
        planned.append(entry)
        diagnostics.append(item)

    planned.sort(key=lambda item: item["start"])
    merged: list[dict] = []
    for item in planned:
        if merged and item["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["items"].append(item)
        else:
            merged.append({"start": item["start"], "end": item["end"],
                           "items": [item]})

    windows = []
    for group in merged:
        items = group["items"]
        goal_items = [item for item in items
                      if item["event"]["type"] == "goal"]
        preferred = (goal_items[0] if goal_items else items[0])
        angle = preferred["angle"]
        if not available[angle, group["start"]:group["end"]].all():
            angle = next((
                item["angle"] for item in items
                if available[item["angle"], group["start"]:group["end"]].all()
            ), angle)
        windows.append({
            "t_start": int(group["start"]),
            "t_end": int(group["end"]),
            "angle": int(angle),
            "events": [item["event"]["id"] for item in items],
            "rule": "goal_hold",
        })
        for item in goal_items:
            replay = item["replay"]
            if replay is None:
                continue
            if replay["src_angle"] == angle:
                ranking = item["event"]["ranking"]
                current = ranking.index(replay["src_angle"])
                replay["src_angle"] = next((
                    candidate for candidate in ranking[current + 1:]
                    if candidate != angle and available[
                        candidate,
                        int(replay["t_src_start"]):int(replay["t_src_end"])
                    ].all()), -1)
                if replay["src_angle"] < 0:
                    item["event"]["replay_angle"] = None
                    item["event"]["replay_reason"] = (
                        "no distinct replay angle available")
                    continue
                item["event"]["replay_angle"] = int(replay["src_angle"])
            replay["t_live_at"] = float(min(group["end"], T - 1))

    replays = assign_output_times([
        item["replay"] for group in merged for item in group["items"]
        if item["replay"] is not None and item["replay"]["src_angle"] >= 0
    ])
    diagnostics.sort(key=lambda item: item["t"])
    return {"windows": windows, "replays": replays, "events": diagnostics}
