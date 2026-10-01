# Stage 2 mapping: pitch coordinates -> generated-world coordinates

`pitchworld/worldmap.py` turns Stage 1 `tracking.json` (players in shared pitch metres) into
`world_positions.json`: the same players expressed in the coordinate space we *assume* the generated
Reactor world has. Read the honesty section first.

```
python -m pitchworld.worldmap tracking.json --seed-cam 0 --out world_positions.json
```

## Why this is an approximation, not a calibration

Reactor worlds are generated, not reconstructed (docs/stage2_reactor.md):

* there is **no ground-truth camera**: LingBot World 2 `set_camera_pose` is a per-frame velocity
  *bias* whose translation is max-norm normalised per chunk (magnitude erased); HappyOyster `move`/
  `look` are held controls with no reported pose;
* there is **no metric frame**: nothing in the API returns positions, depth or scale;
* the seed image is the only geometric input, so the only frame we can define at all is the frame of
  the camera that shot the seed image.

Hence the mapping is a documented **similarity transform** with an assumed scale.

## Definition

**World frame = seed-camera frame.**

| item | value |
|---|---|
| origin | seed camera's calibrated pitch position `cameras[i].calibration.pose` (`x`, `y`, `height`) |
| axes | Reactor camera convention: **x right, y down, z forward** (forward = the camera's calibrated `yaw_deg`; pitch/roll are *not* applied so y stays vertical - the generated camera tilts, the frame does not) |
| scale | `1 world unit = 1 m` **by assumption** (`--scale` overrides) |
| fallback | if the seed camera has no `calibration.pose` (homography-only calibration) or the index is unknown: origin = pitch corner `(0,0,0)`, z along the pitch length; recorded as `world_frame.source = "pitch_origin_fallback"` plus a note |

The rotation is the same `posefit.rotation(yaw, 0, 0)` used by Stage 1 calibration and by
`reactor_export.keys_to_deltas`, so all three stages share one convention. The module is numpy-only.

## Output (`world_positions.json`)

```
{
  "world_frame": {source, seed_camera, origin_pitch_m, rotation_pitch_to_world, yaw_deg,
                  scale_units_per_m, scale_is_assumption: true, notes},
  "quality": {calibration_low_confidence, sync_low_confidence, mapping_uncertainty_m,
              cross_camera_disagreement_m, world_scale_verified: false, notes},
  "frames": [{frame, t, players: [{id, pitch: [x, y] m, normalised: [u, v], world: [x, y, z], conf, cameras}]}],
  "players": {"<id>": {frames: [...],
                       anchors: [{frame, pitch: [x, y, 1.7], yaw_deg, world: [x, y, z], world_yaw_deg}],
                       heading_deltas: [[rx, ry, rz, tx, ty, tz], ...]}}
}
```

* `normalised` = `(x / pitch.length, y / pitch.width)` clamped to `[0, 1]`. **Scale-free**; this is
  what the viewer should use to place players in whatever the generated pitch turns out to be.
* `world` = `R_yaw(seed cam) @ (p - origin) * scale`.
* `anchors` = follow-camera pose per player and frame: 1.7 m above the feet, yaw = smoothed velocity
  heading (EMA 0.8, ignores steps < 2 cm/frame so a standing player keeps facing where he last moved).
* `heading_deltas` = per-frame `[rx, ry, rz, tx, ty, tz]` in the *previous frame's* camera axes,
  exactly the form `set_camera_pose` takes (rotation = axis-angle of `R_a R_b^T`; translation = the
  pitch step rotated into camera axes). Yaw-only, so `rx = rz = ty = 0`. Send them chunked as
  `reactor_export.chunk_commands` does. Sign of `ry` is hypothesis I1 in docs/stage2_reactor.md.

Works for any number of cameras (`--seed-cam N`) and any pitch dict with `length`/`width`.

## How accurate is it - honestly

1. **Metric positions are only good to the Stage 1 cross-camera disagreement.** For today's footage
   `quality.cross_camera_disagreement_m` is 1.66-2.57 m between camera pairs and
   `calibration_low_confidence = true` (cam0's fitted height is an implausible 0.67 m; the pitch is a
   50x30 m *guess*). Both flags are copied into the output, and `mapping_uncertainty_m` = max pairwise
   disagreement (2.57 m today; 2.0 m when the quality block has none). Treat every `world` position as
   "somewhere within ~2.5 m of here".
2. **The world scale is unverifiable.** Reactor generates a plausible pitch from one photo; it does not
   reconstruct ours. Whether one generated unit is one metre, or whether the generated pitch is even
   50 m long, cannot be measured through the API (`world_scale_verified: false`). Use `normalised`
   for anything that must line up with what is on screen.
3. **The world origin inherits the seed camera's pose error.** With the seed camera's own calibration
   at confidence 0.3-0.6 the origin/yaw is off by the same order as (1).
4. **`heading_deltas` will drift.** `set_camera_pose` erases translation magnitude per chunk and is a
   bias, not a rig, so a follow camera driven by these deltas keeps the *direction* of motion but not the
   distance; expect the generated camera to detach from the player within seconds (hypothesis I3).

## Tests

`tests/test_worldmap.py` runs on a 50-frame subset of today's real `tracking.json`
(`tests/data/tracking_50frames.json`, frames 100-149, 3 cameras): frame definition and axis
convention, per-frame player mapping, scale-free `normalised`, anchor/delta shapes, pure-translation
deltas on a synthetic straight run, homography-only fallback, quality propagation, generic pitch dict,
CLI.

```
python -m pytest tests/test_worldmap.py -q
ruff check pitchworld tests
```
