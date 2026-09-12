"""Generate viewer/sample/tracking.json: 10 synthetic players on a 50x30 pitch, pitchworld schema.

Usage: python viewer/sample/make_sample.py [--seconds 20] [--fps 25]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

PITCH = {"length": 50.0, "width": 30.0, "goal_width": 3.66, "d_radius": 6.0,
         "penalty_depth": 0, "penalty_width": 0, "goal_area_depth": 0, "goal_area_width": 0,
         "centre_circle_radius": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("tracking.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    L, W = PITCH["length"], PITCH["width"]
    n_frames = round(args.seconds * args.fps)

    # ball-ish attractor wandering around the pitch: players drift towards it with team-specific bias
    ball_t = [(L / 2 + 18 * math.sin(0.35 * t) * math.cos(0.11 * t), W / 2 + 10 * math.sin(0.5 * t + 1.0))
              for t in (f / args.fps for f in range(n_frames))]

    players = []
    for i in range(10):
        team = "A" if i < 5 else "B"
        home_x = (10 + 8 * (i % 5)) if team == "A" else (L - 10 - 8 * (i % 5))
        home_y = W / 2 + (rng.random() - 0.5) * 20
        players.append({"id": i + 1, "team": team, "home": (home_x, home_y), "phase": rng.random() * 6.28,
                        "pull": 0.15 + 0.3 * rng.random(), "x": home_x, "y": home_y})

    frames = []
    dt = 1.0 / args.fps
    for f in range(n_frames):
        t = f * dt
        bx, by = ball_t[f]
        recs = []
        for p in players:
            tx = p["home"][0] * (1 - p["pull"]) + bx * p["pull"] + 2.5 * math.sin(0.9 * t + p["phase"])
            ty = p["home"][1] * (1 - p["pull"]) + by * p["pull"] + 2.0 * math.cos(0.7 * t + p["phase"])
            # move towards target with a speed cap (~7 m/s)
            dx, dy = tx - p["x"], ty - p["y"]
            d = math.hypot(dx, dy)
            step = min(d, 7.0 * dt)
            if d > 1e-6:
                p["x"] += dx / d * step
                p["y"] += dy / d * step
            p["x"] = min(max(p["x"], 0.5), L - 0.5)
            p["y"] = min(max(p["y"], 0.5), W - 0.5)
            # occasional dropouts to exercise missing-player handling
            if rng.random() < 0.01 and f > 5:
                continue
            recs.append({"id": p["id"], "team": p["team"], "x": round(p["x"], 3), "y": round(p["y"], 3),
                         "conf": round(0.6 + 0.4 * rng.random(), 3), "cameras": [0, 1] if rng.random() < 0.7 else [1],
                         "detections": [{"camera": 0, "track_id": p["id"], "box": [0, 0, 0, 0]}]})
        frames.append({"frame": f, "t": round(t, 4), "players": recs})

    out = {
        "version": "0.1.0-sample",
        "pitch": PITCH,
        "fps": args.fps,
        "num_frames": n_frames,
        "duration_s": round(n_frames / args.fps, 3),
        "cameras": [{"index": i, "source": f"cam{i}.mp4", "synced_clip": f"synced/cam{i}.mp4",
                     "sync_offset_s": o, "sync_confidence": 0.9, "sync_method": "audio",
                     "trim_start_s": 0.0, "frame_size": [1920, 1080]} for i, o in enumerate([0.0, -0.48, 0.35])],
        "quality": {"sync_low_confidence": False, "calibration_low_confidence": True,
                    "cross_camera_disagreement_m": {"0-1": 1.7, "0-2": 2.6, "1-2": 2.1},
                    "warnings": ["synthetic sample data — not from real footage",
                                 "camera disagreement above 1 m (0-2: 2.6 m)"],
                    "stats": {"players": 10, "frames": n_frames}},
        "frames": frames,
    }
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out} ({n_frames} frames, 10 players)")


if __name__ == "__main__":
    main()
