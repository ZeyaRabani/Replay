"""Run configuration: all tunables in one dataclass, JSON-overridable."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    # remote source (modal://<volume>/<path>)
    source_volume: str = "football-footage"
    # detection
    stride: int = 2                 # process every Nth frame
    chunk_s: float = 30.0           # seconds per detection range (fanned out on Modal)
    gpu: str | None = "A10G"        # Modal GPU tier; None/"" = CPU container
    player_model: str = "yolov8m.pt"
    player_imgsz: int = 1280
    player_conf: float = 0.25
    # ball model spec: "hf:<repo>:<file>" | "coco" (yolov8x class 32) | local path | ultralytics name
    ball_model: str = "hf:AurevinP/PULSE_AI_Models:ball_tracker_200_epoch_92L_25M.pt"
    ball_tile: tuple[int, int] = (960, 960)
    ball_overlap: float = 0.2
    ball_conf: float = 0.2
    ball_keep: int = 3              # top-N ball detections per frame

    # binning / events
    bin_s: float = 0.5
    pre_roll_s: float = 8.0
    post_roll_s: float = 12.0
    merge_gap_s: float = 15.0

    # ball linking / signals
    ball_max_jump_frac: float = 0.08   # of frame width, per processed frame
    ball_min_tracklet: int = 3
    ball_interp_gap_s: float = 0.5
    ball_smooth_window: int = 7
    ball_lost_s: float = 1.0
    v_shot_pitch: float = 8.0          # m/s toward goal to count as a shot (pitch space)
    v_shot_pixel: float = 0.6          # frame-heights/s (pixel space)

    # player signals
    # player signals use pitch (metre) thresholds in both spaces; pixel space is
    # pseudo-metric via box-height scale (1.75 m / box_h)
    v_run_pitch: float = 3.0           # m/s sprint toward goal
    cluster_radius_pitch: float = 5.0  # m
    cluster_min_players: int = 5
    cluster_slow_pitch: float = 1.0    # m/s
    cluster_min_dur_s: float = 2.0
    restart_half_width: float = 12.0   # m either side of centre line
    restart_centre_r: float = 4.0      # m around pitch centre

    # audio
    audio_hop_s: float = 0.1
    audio_baseline_win_s: float = 60.0
    audio_thresh_db: float = 6.0
    audio_min_dur_s: float = 1.0

    # combine weights
    w_attack: float = 0.35
    w_audio: float = 0.25
    w_cluster: float = 0.20
    w_restart: float = 0.15
    w_lost: float = 0.10
    attack_anchor: float = 0.3         # attack signal needed to seed a candidate
    audio_goal_min: float = 0.3        # audio needed to call it a "goal"

    @classmethod
    def load(cls, path: Path | None) -> Config:
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k not in names:
                    raise KeyError(f"unknown config key {k!r}")
                if k == "ball_tile":
                    v = tuple(v)
                setattr(cfg, k, v)
        if os.environ.get("HIGHLIGHTS_GPU") is not None:
            cfg.gpu = os.environ["HIGHLIGHTS_GPU"] or None
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ball_tile"] = list(self.ball_tile)
        return d
