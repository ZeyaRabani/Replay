# Replay

Upload 2–4 fixed-camera clips of one football moment → an explorable,
persistent real-time 3D replay: jump to any tracked player or free pitch
viewpoint, look around, at any point in the timeline.

## Run Replay

```bash
pip install -e .
echo "REACTOR_API_KEY=rk_..." > .env     # optional; git-ignored, exchanged server-side
python viewer/server.py                  # -> http://localhost:8080
```

- Live Reactor (HappyOyster) world: `http://localhost:8080/?world=dHpmaEU0lINZtc81EZgHEnya1amxS9MxfbMbSnp2-J4`
- Three.js fallback (no key needed): `http://localhost:8080/`
- Demo video: [`replay-demo.mp4`](replay-demo.mp4) · full run notes and honest
  limitations: [`docs/REPORT.md`](docs/REPORT.md)

Stage 1 (`pitchworld`, below) builds `tracking.json`; Stage 2 (`pitchworld/worldmap.py`,
`reactor-app/`, `docs/stage2_*.md`) maps it into a Reactor world; Stage 3 (`viewer/`)
is the explorable viewer; Stage 4 (`demo/`) cuts the demo video.

# pitchworld

`pitchworld` turns 2–4 football clips from a shared pitch into synchronized
videos and a unified `tracking.json` in pitch coordinates. It synchronizes the
clips, calibrates each camera to the pitch, runs person tracking, and merges
cross-camera observations into stable player IDs.

## Install

Requirements:

- Python 3.10+
- `ffmpeg` on `PATH`
- Modal authentication for GPU tracking

```bash
pip install -e .
modal token set
```

For local CPU tracking, install the optional dependencies with
`pip install -e '.[local]'` and pass `--local`.

## CLI

Run the complete pipeline:

```bash
pitchworld run cam0.mp4 cam1.mp4 [cam2.mp4 cam3.mp4] \
  --out out/ --calib calib.json --pitch examples/pitch_small_sided_50x30.json
```

Reuse existing synchronized clips and cached detections with `--reuse`.
Tracking uses Modal by default; `--local` selects local CPU inference.

Synchronize clips only:

```bash
pitchworld sync cam0.mp4 cam1.mp4 ... --out out/
```

Create or update a calibration interactively:

```bash
pitchworld calibrate cam0.mp4 --camera 0 --pitch pitch.json --out calib.json
```

Render a calibration check image:

```bash
pitchworld check-calib cam0.mp4 --camera 0 --calib calib.json --out check.jpg
```

## Pitch and calibration JSON

A pitch file contains the model dimensions. It may select a preset and
override its fields, for example:

```json
{"preset": "small_sided", "length": 50, "width": 30,
 "goal_width": 3.66, "d_radius": 6.0}
```

Calibration files contain a `cameras` object keyed by camera index. Each
camera can provide any combination of these constraints:

- `points`: `{"world": "landmark_name" | [x, y], "pixel": [u, v]}`
- `lines`: `{"world": "line_name" | {"point": [x, y], "dir": [dx, dy]}, "pixels": [[u, v], ...]}`
- `arcs`: `{"world": "arc_name" | {"centre": [x, y], "radius": r}, "pixels": [[u, v], ...]}`
- `parallels`: `{"world": "line_name", "pixels": [[u, v], ...]}`

Named lines include `goal_line_A`, `goal_line_B`, `touch_S`, `touch_N`, and
`half`. Named arcs include `A_D`, `B_D`, and `centre_circle` when enabled by
the pitch model. Pixel coordinates refer to the calibration frame.
With traced geometry, calibration is fitted as a physical camera pose; with
at least four points and no traced geometry, a homography is fitted directly.

## Output

The output directory contains:

- `sync.json`: offsets, confidence, methods, and the common time window.
- `synced/cam*.mp4`: constant-frame-rate synchronized clips.
- `tracking.json`: the merged tracking result.
- `debug/sync_check.jpg`: synchronization contact sheet.
- `debug/calib_cam*.jpg`: calibration overlays.
- `debug/raw_tracks.json`: cached per-camera detections.
- `debug/pitch_map.mp4` and `debug/overlay_cam*.mp4`: visualization videos.

`tracking.json` includes the pitch model, output `fps`, frame count and
duration, per-camera source/synced paths and calibration metadata, and
`frames`. Each frame has `frame`, `t`, and `players`; each player has `id`,
pitch-coordinate `x`/`y`, `conf`, contributing `cameras`, and source
detections (`camera`, `track_id`, and bounding-box `box`).

The `quality` block contains `sync_low_confidence`,
`calibration_low_confidence`, `cross_camera_disagreement_m` (pairwise median
disagreement in metres), warning strings, and aggregate tracking `stats`.
Treat results as low confidence when calibration is flagged, sync is flagged,
or warnings report missing shared tracks or camera disagreement above 1 metre.
