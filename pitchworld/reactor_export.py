"""Stage 2 export: tracking.json + synced clips -> Reactor world-model request payloads (no network).

Reactor (https://docs.reactor.inc) hosts *generative, real-time* world models. None of them ingests
multi-view video, camera poses or trajectories: a world is anchored by ONE seed image + a text
prompt, and the virtual camera is steered live with velocity-style commands
(``set_camera_pose``: per-frame ``[rx, ry, rz, tx, ty, tz]`` deltas in the camera-local frame,
LingBot World 2). See docs/stage2_reactor.md for the design and the verified/inferred split.

This module therefore turns Stage 1 output into a *shot*:

* ``seed.jpg``          - a synced frame (letterboxed to the model resolution) that anchors the world
* ``prompt.txt``        - auto-written scene prompt (players, pitch, "camera moves, players stay")
* ``camera_path.json``  - the virtual camera in pitch coordinates, per output frame
* ``manifest.json``     - ordered command list in the exact wire shape of the Python SDK
                          (``set_prompt``, ``set_image``, ``set_seed``, ``start``, ``set_camera_pose`` per chunk)
* ``twin_scene.json``   - fallback "digital twin" (pitch + resampled player tracks + camera path)

CLI (never touches the network)::

    python -m pitchworld.reactor_export out/tracking.json --out out/reactor --anchor-player 3 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .posefit import rotation

# ---- Reactor facts used below (verified against docs.reactor.inc on 2026-09-12) --------------------
MODELS = {
    # slug, seed resolution (w, h), output fps, pixel frames per chunk, usd/s, max floats per set_camera_pose
    "lingbot-world-2": {"slug": "reactor/lingbot-world-2", "seed_size": (1664, 960), "fps": 48,
                        "frames_per_chunk": 12, "usd_per_s": 0.0070, "max_pose_floats": 1536,
                        "camera_command": "set_camera_pose"},
    "helios": {"slug": "reactor/helios", "seed_size": (1280, 768), "fps": 24,
               "frames_per_chunk": 33, "usd_per_s": 0.0017, "max_pose_floats": 0, "camera_command": None},
}
API_KEY_ENV = "REACTOR_API_KEY"
TOKEN_URL = "https://api.reactor.inc/tokens"


@dataclass
class CameraKey:
    """One virtual-camera sample in pitch coordinates (x east, y north, z up; angles in radians)."""

    t: float
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float = 0.0


@dataclass
class Shot:
    name: str
    model: str
    seed_camera: int
    seed_time_s: float
    prompt: str
    keys: list[CameraKey]
    anchor_player: int | None = None
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------
# tracking.json helpers
# ------------------------------------------------------------------------------------------------
def load_tracking(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def player_track(tracking: dict, pid: int) -> list[tuple[float, float, float]]:
    """(t, x, y) samples of one merged player id, in pitch metres."""
    out = []
    for fr in tracking["frames"]:
        for p in fr["players"]:
            if p["id"] == pid:
                out.append((float(fr["t"]), float(p["x"]), float(p["y"])))
    return out


def most_visible_player(tracking: dict) -> int:
    counts: dict[int, int] = {}
    for fr in tracking["frames"]:
        for p in fr["players"]:
            counts[p["id"]] = counts.get(p["id"], 0) + 1
    if not counts:
        raise ValueError("tracking.json has no players")
    return max(counts, key=counts.get)


def interp_track(track: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    ts = np.array([s[0] for s in track])
    return float(np.interp(t, ts, [s[1] for s in track])), float(np.interp(t, ts, [s[2] for s in track]))


def build_prompt(tracking: dict, camera_moves: bool) -> str:
    pitch = tracking["pitch"]
    stats = tracking.get("quality", {}).get("stats", {})
    n = stats.get("mean_players_per_frame") or stats.get("max_players") or stats.get("unique_ids")
    who = f"about {round(n)} football players" if n else "football players"
    txt = (f"Real phone footage of {who} on a {pitch.get('length', 50):.0f} by {pitch.get('width', 30):.0f} metre "
           "small-sided football pitch with white lines and portable goals, daylight, photorealistic, "
           "the players keep playing naturally. ")
    txt += ("The camera glides smoothly around the action while the pitch and goals stay fixed in place."
            if camera_moves else "Fixed tripod camera, no camera motion.")
    return txt


# ------------------------------------------------------------------------------------------------
# virtual camera paths (pitch frame)
# ------------------------------------------------------------------------------------------------
def look_at(cx: float, cy: float, cz: float, tx: float, ty: float, tz: float = 0.9) -> tuple[float, float]:
    """(yaw, pitch) of a camera at c looking at target t; pitch positive = looking down (posefit convention)."""
    dx, dy, dz = tx - cx, ty - cy, tz - cz
    return math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy))


def orbit_path(track: list[tuple[float, float, float]], t0: float, duration: float, fps: float, radius: float,
               height: float, deg_per_s: float, start_deg: float) -> list[CameraKey]:
    keys = []
    n = round(duration * fps)
    for i in range(n):
        t = t0 + i / fps
        px, py = interp_track(track, t)
        a = math.radians(start_deg + deg_per_s * (i / fps))
        cx, cy, cz = px + radius * math.cos(a), py + radius * math.sin(a), height
        yaw, pitch = look_at(cx, cy, cz, px, py)
        keys.append(CameraKey(round(t, 4), cx, cy, cz, yaw, pitch))
    return keys


def follow_path(track: list[tuple[float, float, float]], t0: float, duration: float, fps: float, back: float,
                height: float) -> list[CameraKey]:
    """Chase camera: trails the player along its own direction of motion."""
    keys = []
    n = round(duration * fps)
    prev = None
    heading = 0.0
    for i in range(n):
        t = t0 + i / fps
        px, py = interp_track(track, t)
        if prev is not None and math.hypot(px - prev[0], py - prev[1]) > 0.02:
            heading = 0.8 * heading + 0.2 * math.atan2(py - prev[1], px - prev[0])
        prev = (px, py)
        cx, cy, cz = px - back * math.cos(heading), py - back * math.sin(heading), height
        yaw, pitch = look_at(cx, cy, cz, px, py)
        keys.append(CameraKey(round(t, 4), cx, cy, cz, yaw, pitch))
    return keys


def static_path(cal_pose: dict, t0: float, duration: float, fps: float) -> list[CameraKey]:
    """Re-play a real, calibrated camera (tracking.json['cameras'][i]['calibration']['pose'])."""
    n = round(duration * fps)
    return [CameraKey(round(t0 + i / fps, 4), cal_pose["x"], cal_pose["y"], cal_pose["height"],
                      math.radians(cal_pose["yaw_deg"]), math.radians(cal_pose["pitch_deg"]),
                      math.radians(cal_pose["roll_deg"])) for i in range(n)]


# ------------------------------------------------------------------------------------------------
# pitch-frame camera path -> Reactor set_camera_pose deltas
# ------------------------------------------------------------------------------------------------
def keys_to_deltas(keys: list[CameraKey]) -> np.ndarray:
    """(N-1, 6) per-frame [rx, ry, rz, tx, ty, tz] in the camera-local frame (x right, y DOWN, z forward).

    Rotation deltas are small Euler angles about the local axes; translation is the world step
    expressed in the previous frame's camera axes. posefit.rotation() already maps world (z up) to
    exactly this camera frame, so the same convention serves both calibration and export.
    """
    if len(keys) < 2:
        return np.zeros((0, 6))
    out = np.zeros((len(keys) - 1, 6))
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        Ra, Rb = rotation(a.yaw, a.pitch, a.roll), rotation(b.yaw, b.pitch, b.roll)
        # Rb = R_ab @ Ra  ->  R_ab = Rb @ Ra.T rotates a-axes into b-axes; the camera body rotates by its inverse
        rvec, _ = cv2.Rodrigues(Ra @ Rb.T)
        out[i, 0:3] = rvec.ravel()
        out[i, 3:6] = Ra @ np.array([b.x - a.x, b.y - a.y, b.z - a.z])
    return out


def chunk_commands(deltas: np.ndarray, model: dict, t0: float) -> list[dict]:
    """Pack per-frame deltas into one ``set_camera_pose`` command per model chunk, stamped with the
    stream time at which the command must be *sent* (setters apply at the next chunk boundary)."""
    fpc, fps = model["frames_per_chunk"], model["fps"]
    cmds = []
    for k in range(0, len(deltas), fpc):
        block = deltas[k:k + fpc]
        flat = [round(float(v), 6) for v in block.ravel()]
        if len(flat) > model["max_pose_floats"]:
            raise ValueError("chunk exceeds max_pose_floats")
        cmds.append({"send_at_stream_s": round(max(0.0, k / fps - fpc / fps), 4), "covers_t": [round(t0 + k / fps, 4),
                     round(t0 + (k + len(block)) / fps, 4)], "command": model["camera_command"],
                     "data": {"camera_pose": flat}})
    cmds.append({"send_at_stream_s": round(len(deltas) / fps, 4), "command": model["camera_command"],
                 "data": {"camera_pose": []}, "note": "deactivate pose layer, hand camera back to look axes"})
    return cmds


# ------------------------------------------------------------------------------------------------
# seed frame
# ------------------------------------------------------------------------------------------------
def extract_seed(clip: Path, t: float, size: tuple[int, int], out: Path, max_bytes: int = 2_000_000) -> dict:
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame at {t}s from {clip}")
    W, H = size
    h, w = frame.shape[:2]
    s = min(W / w, H / h)
    rs = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), np.uint8)
    y0, x0 = (H - rs.shape[0]) // 2, (W - rs.shape[1]) // 2
    canvas[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    q = 92
    while True:
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
        if out.stat().st_size <= max_bytes or q <= 40:
            break
        q -= 10
    return {"path": out.name, "width": W, "height": H, "aspect": round(W / H, 3), "bytes": out.stat().st_size,
            "letterbox_scale": round(s, 4), "source_frame_size": [w, h], "source_time_s": t,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


# ------------------------------------------------------------------------------------------------
# export
# ------------------------------------------------------------------------------------------------
def make_shot(tracking: dict, mode: str, model_name: str, anchor: int | None, camera: int, t0: float,
              duration: float, radius: float, height: float, deg_per_s: float, start_deg: float) -> Shot:
    model = MODELS[model_name]
    fps = model["fps"]
    duration = min(duration, tracking["duration_s"] - t0)
    notes = []
    if mode == "static":
        pose = tracking["cameras"][camera].get("calibration", {}).get("pose")
        if not pose:
            raise ValueError(f"cam{camera} has no physical pose (homography-only calibration); use --mode orbit")
        keys = static_path(pose, t0, duration, fps)
        anchor = None
    else:
        anchor = most_visible_player(tracking) if anchor is None else anchor
        track = player_track(tracking, anchor)
        if len(track) < 2:
            raise ValueError(f"player {anchor} has < 2 samples")
        if track[0][0] > t0 or track[-1][0] < t0 + duration:
            notes.append(f"player {anchor} only tracked {track[0][0]:.2f}-{track[-1][0]:.2f}s; path is clamped outside")
        if mode == "orbit":
            keys = orbit_path(track, t0, duration, fps, radius, height, deg_per_s, start_deg)
        elif mode == "follow":
            keys = follow_path(track, t0, duration, fps, radius, height)
        else:
            raise ValueError(mode)
    # seed camera: the real camera whose pose is closest to the first virtual key (or the one requested)
    seed_cam = camera
    if mode != "static":
        best = None
        for c in tracking["cameras"]:
            p = c.get("calibration", {}).get("pose")
            if p:
                d = math.hypot(p["x"] - keys[0].x, p["y"] - keys[0].y)
                if best is None or d < best[0]:
                    best = (d, c["index"])
        if best:
            seed_cam = best[1]
            notes.append(f"seed frame from cam{seed_cam} ({best[0]:.1f} m from first virtual camera position)")
    q = tracking.get("quality", {})
    if q.get("calibration_low_confidence"):
        notes.append("Stage 1 calibration is LOW confidence: virtual camera anchors may be off by metres")
    return Shot(f"{mode}_p{anchor}" if anchor is not None else f"{mode}_cam{camera}", model_name, seed_cam, t0,
                build_prompt(tracking, camera_moves=mode != "static"), keys, anchor, notes)


def export(tracking_path: Path, out: Path, shot: Shot, seed: int = 0, clip_override: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tracking = load_tracking(tracking_path)
    model = MODELS[shot.model]
    cam = tracking["cameras"][shot.seed_camera]
    clip = clip_override or Path(cam["synced_clip"])
    if not clip.exists():  # tracking.json stores absolute paths; fall back to out/synced next to it
        alt = tracking_path.parent / "synced" / Path(cam["synced_clip"]).name
        clip = alt if alt.exists() else clip
    seed_info = extract_seed(clip, shot.seed_time_s, model["seed_size"], out / "seed.jpg")
    (out / "prompt.txt").write_text(shot.prompt)
    deltas = keys_to_deltas(shot.keys)
    pose_cmds = chunk_commands(deltas, model, shot.seed_time_s) if model["camera_command"] else []
    commands = [
        {"send_at_stream_s": 0, "command": "set_prompt", "data": {"prompt": shot.prompt}},
        {"send_at_stream_s": 0, "command": "set_image", "data": {"image": {"$upload_file": "seed.jpg"}},
         "note": "replace $upload_file with the FileRef returned by reactor.upload_file('seed.jpg')"},
        {"send_at_stream_s": 0, "command": "set_seed", "data": {"seed": seed}},
        *pose_cmds[:1],
        {"send_at_stream_s": 0, "command": "start", "data": {}},
        *pose_cmds[1:],
    ]
    (out / "camera_path.json").write_text(json.dumps({
        "frame": "pitch: x east (m), y north (m), z up (m); yaw rad from +x, pitch rad positive = down",
        "fps": model["fps"], "keys": [asdict(k) for k in shot.keys],
        "deltas_camera_local_ydown": [[round(float(v), 6) for v in row] for row in deltas]}))
    twin = {
        "pitch": tracking["pitch"], "fps": model["fps"], "t0": shot.seed_time_s,
        "duration_s": round(len(shot.keys) / model["fps"], 3),
        "players": {str(pid): [[t, x, y] for t, x, y in player_track(tracking, pid)]
                    for pid in sorted({p["id"] for fr in tracking["frames"] for p in fr["players"]})},
        "camera_path": [asdict(k) for k in shot.keys],
        "real_cameras": [{"index": c["index"], "pose": c.get("calibration", {}).get("pose"),
                          "homography_px_to_m": c.get("homography_px_to_m"), "frame_size": c.get("frame_size")}
                         for c in tracking["cameras"]],
        "quality": tracking.get("quality", {}),
    }
    (out / "twin_scene.json").write_text(json.dumps(twin))
    stream_s = len(shot.keys) / model["fps"]
    manifest = {
        "version": 1, "shot": shot.name, "model": model["slug"],
        "auth": {"api_key_env": API_KEY_ENV, "token_url": TOKEN_URL, "sdk": "pip install reactor-sdk",
                 "connect": f"Reactor(model_name='{model['slug']}', api_key=os.environ['{API_KEY_ENV}'])"},
        "files": {"seed.jpg": seed_info, "prompt.txt": {"chars": len(shot.prompt)},
                  "camera_path.json": {"keys": len(shot.keys)}, "twin_scene.json": {"players": len(twin["players"])}},
        "stream": {"fps": model["fps"], "frames": len(shot.keys), "seconds": round(stream_s, 3),
                   "chunks": len(pose_cmds) - 1 if pose_cmds else None,
                   "est_cost_usd": round(stream_s * model["usd_per_s"], 4)},
        "anchor_player": shot.anchor_player, "seed_camera": shot.seed_camera,
        "source": {"tracking_json": str(tracking_path.resolve()),
                   "tracking_sha256": hashlib.sha256(tracking_path.read_bytes()).hexdigest(),
                   "synced_clip": str(clip), "pitchworld_version": tracking.get("version")},
        "quality": {k: v for k, v in tracking.get("quality", {}).items() if k != "stats"},
        "notes": shot.notes,
        "commands": commands,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pitchworld.reactor_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracking", help="tracking.json from `pitchworld run`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="lingbot-world-2", choices=sorted(MODELS))
    ap.add_argument("--mode", default="orbit", choices=["orbit", "follow", "static"])
    ap.add_argument("--anchor-player", type=int, help="merged player id to anchor the camera on (default: most visible)")
    ap.add_argument("--camera", type=int, default=0, help="real camera for --mode static / seed fallback")
    ap.add_argument("--t0", type=float, default=0.0, help="start time in the synced timeline (s)")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--radius", type=float, default=6.0, help="orbit radius / follow distance (m)")
    ap.add_argument("--height", type=float, default=2.5, help="virtual camera height (m)")
    ap.add_argument("--deg-per-s", type=float, default=20.0, help="orbit speed")
    ap.add_argument("--start-deg", type=float, default=-90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", help="override the seed clip path (tracking.json stores absolute paths)")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="write files + manifest only (always on: this module never calls the network)")
    a = ap.parse_args(argv)
    tp = Path(a.tracking)
    shot = make_shot(load_tracking(tp), a.mode, a.model, a.anchor_player, a.camera, a.t0, a.duration, a.radius,
                     a.height, a.deg_per_s, a.start_deg)
    m = export(tp, Path(a.out), shot, a.seed, Path(a.clip) if a.clip else None)
    print(f"[dry-run] {m['shot']} -> {a.out}: model {m['model']}, {m['stream']['frames']} frames "
          f"({m['stream']['seconds']}s @ {m['stream']['fps']} fps, {m['stream']['chunks']} pose chunks, "
          f"~${m['stream']['est_cost_usd']}), {len(m['commands'])} commands, seed cam{m['seed_camera']}")
    for n in m["notes"]:
        print(f"  ! {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
