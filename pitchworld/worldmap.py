"""Stage 2 mapping: express Stage 1 pitch-frame player positions in a generated world's coordinate space.

Reactor worlds have **no ground-truth camera and no metric frame** (docs/stage2_reactor.md: ``set_camera_pose``
is a velocity bias, HappyOyster ``move``/``look`` are held controls). The best we can do is a documented,
explicitly approximate similarity transform:

* **world frame = seed-camera frame.** Origin at the seed camera's calibrated pitch position
  (``cameras[i].calibration.pose`` x, y, height, yaw/pitch/roll); axes x-right / y-down / z-forward as in the
  Reactor docs; scale 1 world unit = 1 m *by assumption* (unverifiable: the world is generated, not reconstructed).
* **normalised pitch coords** (u, v) in [0, 1] are scale-free and are what a viewer should really use.
* per player: a camera anchor (1.7 m above the player's feet, yaw along velocity) and ``heading_deltas`` - the
  per-frame ``[rx, ry, rz, tx, ty, tz]`` camera-local deltas a follow camera would send via ``set_camera_pose``.

Usage::

    python -m pitchworld.worldmap tracking.json --seed-cam 0 --out world_positions.json

Numpy-only on purpose (no cv2), so it runs and is testable without the heavy Stage 1 dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ANCHOR_HEIGHT_M = 1.7
DEFAULT_UNCERTAINTY_M = 2.0
MIN_SPEED_FOR_HEADING = 0.02  # m/frame below which the previous heading is kept
HEADING_SMOOTHING = 0.8


# ------------------------------------------------------------------------------------------------
# rotation conventions (copied from posefit.rotation so this module stays numpy-only)
# ------------------------------------------------------------------------------------------------
def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Pitch/world (x along length, y along width, z up) -> camera (x right, y down, z forward)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    r_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return r_roll @ r_pitch @ base @ r_yaw


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (numpy replacement for cv2.Rodrigues)."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # angle ~ pi: take the dominant eigenvector
        w, v = np.linalg.eigh(R)
        axis = v[:, int(np.argmin(np.abs(w - 1.0)))]
        return axis / np.linalg.norm(axis) * angle
    return axis / n * angle


# ------------------------------------------------------------------------------------------------
# world frame
# ------------------------------------------------------------------------------------------------
@dataclass
class WorldFrame:
    """Similarity transform pitch (m) -> world units. ``origin`` is in pitch metres (x, y, z-up)."""

    origin: np.ndarray
    R: np.ndarray  # pitch -> world (camera-style x-right / y-down / z-forward)
    scale: float = 1.0  # world units per metre - ASSUMED, unverifiable
    source: str = "pitch_origin"
    seed_camera: int | None = None
    yaw_deg: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_world(self, x: float, y: float, z: float = 0.0) -> list[float]:
        p = self.R @ (np.array([x, y, z], dtype=np.float64) - self.origin) * self.scale
        return [round(float(v), 3) for v in p]

    def describe(self) -> dict:
        return {
            "definition": "seed-camera frame: origin at the seed camera's calibrated pitch position, "
            "x-right / y-down / z-forward (Reactor camera convention)",
            "source": self.source,
            "seed_camera": self.seed_camera,
            "origin_pitch_m": [round(float(v), 3) for v in self.origin],
            "rotation_pitch_to_world": [[round(float(v), 6) for v in row] for row in self.R],
            "yaw_deg": round(self.yaw_deg, 3),
            "scale_units_per_m": self.scale,
            "scale_is_assumption": True,
            "notes": self.notes,
        }


def _camera_pose(tracking: dict, seed_cam: int) -> dict | None:
    for cam in tracking.get("cameras", []):
        if cam.get("index") == seed_cam:
            pose = (cam.get("calibration") or {}).get("pose")
            if pose and all(k in pose for k in ("x", "y")):
                return pose
            return None
    return None


def world_frame(tracking: dict, seed_cam: int = 0, scale: float = 1.0, level: bool = True) -> WorldFrame:
    """Build the world frame from the seed camera pose; fall back to the pitch origin when there is none.

    ``level=True`` uses the camera's yaw only (pitch/roll zeroed) so world y stays 'down' and z stays
    horizontal; Reactor's camera is what turns/tilts, the frame itself should not.
    """
    pose = _camera_pose(tracking, seed_cam)
    if pose is None:
        wf = WorldFrame(np.zeros(3), rotation(0.0, 0.0, 0.0), scale, "pitch_origin_fallback", seed_cam)
        wf.notes.append(f"camera {seed_cam} has no calibration pose (homography-only or absent): "
                        "origin = pitch corner (0,0,0), z-forward along the pitch length")
        return wf
    yaw = math.radians(pose.get("yaw_deg", 0.0))
    pitch = 0.0 if level else math.radians(pose.get("pitch_deg", 0.0))
    roll = 0.0 if level else math.radians(pose.get("roll_deg", 0.0))
    origin = np.array([pose["x"], pose["y"], pose.get("height", 0.0)], dtype=np.float64)
    return WorldFrame(origin, rotation(yaw, pitch, roll), scale, "seed_camera_pose", seed_cam, math.degrees(yaw))


# ------------------------------------------------------------------------------------------------
# players
# ------------------------------------------------------------------------------------------------
def normalised(pitch: dict, x: float, y: float) -> list[float]:
    length = float(pitch.get("length") or 1.0)
    width = float(pitch.get("width") or 1.0)
    return [round(min(max(x / length, 0.0), 1.0), 4), round(min(max(y / width, 0.0), 1.0), 4)]


def player_tracks(tracking: dict) -> dict[int, list[tuple[int, float, float]]]:
    """pid -> [(frame_index, x, y)] in pitch metres."""
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    for fi, fr in enumerate(tracking["frames"]):
        for p in fr.get("players", []):
            tracks.setdefault(int(p["id"]), []).append((fi, float(p["x"]), float(p["y"])))
    return tracks


def headings(track: list[tuple[int, float, float]]) -> list[float]:
    """Smoothed velocity heading (pitch-frame yaw, radians) per sample; first sample faces +x."""
    out: list[float] = []
    heading = 0.0
    for i, (_, x, y) in enumerate(track):
        if i > 0:
            _, px, py = track[i - 1]
            if math.hypot(x - px, y - py) > MIN_SPEED_FOR_HEADING:
                target = math.atan2(y - py, x - px)
                d = (target - heading + math.pi) % (2 * math.pi) - math.pi
                heading = heading + (1.0 - HEADING_SMOOTHING) * d
        out.append(heading)
    return out


def anchor_poses(track: list[tuple[int, float, float]], wf: WorldFrame) -> list[dict]:
    """Per-sample follow-camera anchor at head height, looking along the player's heading."""
    poses = []
    for (fi, x, y), yaw in zip(track, headings(track)):
        poses.append({
            "frame": fi,
            "pitch": [round(x, 3), round(y, 3), ANCHOR_HEIGHT_M],
            "yaw_deg": round(math.degrees(yaw), 2),
            "world": wf.to_world(x, y, ANCHOR_HEIGHT_M),
            "world_yaw_deg": round(math.degrees(yaw) - wf.yaw_deg, 2),
        })
    return poses


def heading_deltas(track: list[tuple[int, float, float]], wf: WorldFrame) -> list[list[float]]:
    """Per-frame ``[rx, ry, rz, tx, ty, tz]`` camera-local deltas for a camera riding on the player.

    Same convention as reactor_export.keys_to_deltas: rotation = axis-angle of R_a @ R_b^T (camera body
    rotation), translation = the world step in the previous frame's camera axes, scaled to world units.
    Sign of ``ry`` is an unverified hypothesis (docs/stage2_reactor.md I1).
    """
    yaws = headings(track)
    out: list[list[float]] = []
    for i in range(len(track) - 1):
        _, ax, ay = track[i]
        _, bx, by = track[i + 1]
        Ra, Rb = rotation(yaws[i], 0.0, 0.0), rotation(yaws[i + 1], 0.0, 0.0)
        r = rotvec(Ra @ Rb.T)
        t = Ra @ np.array([bx - ax, by - ay, 0.0]) * wf.scale
        out.append([round(float(v), 5) for v in (*r, *t)])
    return out


def mapping_uncertainty_m(tracking: dict) -> float:
    q = tracking.get("quality") or {}
    dis = q.get("cross_camera_disagreement_m")
    if isinstance(dis, dict) and dis:
        vals = [float(v) for v in dis.values() if v is not None]
        if vals:
            return round(max(vals), 3)
    if isinstance(dis, (int, float)):
        return round(float(dis), 3)
    return DEFAULT_UNCERTAINTY_M


def map_tracking(tracking: dict, seed_cam: int = 0, scale: float = 1.0) -> dict:
    wf = world_frame(tracking, seed_cam, scale)
    pitch = tracking.get("pitch") or {}
    q = tracking.get("quality") or {}
    tracks = player_tracks(tracking)

    frames = []
    for fi, fr in enumerate(tracking["frames"]):
        players = []
        for p in fr.get("players", []):
            x, y = float(p["x"]), float(p["y"])
            players.append({
                "id": int(p["id"]),
                "pitch": [round(x, 3), round(y, 3)],
                "normalised": normalised(pitch, x, y),
                "world": wf.to_world(x, y, 0.0),
                "conf": p.get("conf"),
                "cameras": p.get("cameras", []),
            })
        frames.append({"frame": fr.get("frame", fi), "t": fr.get("t"), "players": players})

    per_player = {}
    for pid, track in sorted(tracks.items()):
        per_player[str(pid)] = {
            "frames": [fi for fi, _, _ in track],
            "anchors": anchor_poses(track, wf),
            "heading_deltas": heading_deltas(track, wf),
        }

    return {
        "version": "0.1.0",
        "source_version": tracking.get("version"),
        "fps": tracking.get("fps"),
        "num_frames": len(frames),
        "pitch": pitch,
        "world_frame": wf.describe(),
        "anchor_height_m": ANCHOR_HEIGHT_M,
        "quality": {
            "calibration_low_confidence": bool(q.get("calibration_low_confidence", False)),
            "sync_low_confidence": bool(q.get("sync_low_confidence", False)),
            "mapping_uncertainty_m": mapping_uncertainty_m(tracking),
            "cross_camera_disagreement_m": q.get("cross_camera_disagreement_m"),
            "world_scale_verified": False,
            "notes": [
                "metric accuracy is bounded by Stage 1 cross-camera disagreement (mapping_uncertainty_m)",
                "world scale (1 unit = 1 m) is an assumption: Reactor generates rather than reconstructs, "
                "so it cannot be verified against the world; prefer the normalised (u,v) coordinates",
                *wf.notes,
            ],
        },
        "frames": frames,
        "players": per_player,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracking", type=Path)
    ap.add_argument("--seed-cam", type=int, default=0, help="camera index whose pose defines the world frame")
    ap.add_argument("--scale", type=float, default=1.0, help="world units per metre (assumed, default 1)")
    ap.add_argument("--out", type=Path, default=Path("world_positions.json"))
    a = ap.parse_args(argv)
    tracking = json.loads(a.tracking.read_text())
    out = map_tracking(tracking, a.seed_cam, a.scale)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    q = out["quality"]
    print(f"{a.out}: {out['num_frames']} frames, {len(out['players'])} players, world frame from "
          f"{out['world_frame']['source']} (cam {a.seed_cam}); uncertainty ~{q['mapping_uncertainty_m']} m"
          + (" [calibration_low_confidence]" if q["calibration_low_confidence"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
