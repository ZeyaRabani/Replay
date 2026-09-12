# Replay viewer (Stage 3)

Static Three.js web app that renders a `pitchworld` `tracking.json` as an interactive
3D replay with camera anchors on players and around the pitch. No build step —
Three.js is loaded from the jsDelivr CDN via an import map.

## Run

```bash
cd viewer
python -m http.server 8000
# open http://localhost:8000/?src=sample/tracking.json
```

Load data by any of:

- `?src=<url>` — e.g. `?src=../out/tracking.json` (must be served by the same
  server or allow CORS)
- drag & drop a `tracking.json` anywhere on the page
- **Open tracking.json** file picker, or **Load sample**

## What it shows

- **Pitch** built from `tracking.json["pitch"]`: `length`, `width`, `goal_width`,
  `d_radius` (small-sided D arcs), and standard markings (`penalty_*`,
  `goal_area_*`, `centre_circle_radius`) when they are non-zero. Goals are drawn
  as posts + crossbar + translucent net.
- **Players** as capsules (1.7 m tall) with an id label. Colour comes from a
  `team` field if present (`A`/`B`, `home`/`away`, `0`/`1`), otherwise a
  per-id palette. A small arrow shows the finite-difference velocity.
- **Timeline**: scrubber, play/pause (space), speed 0.25–2×, arrow keys step
  frames. Playback uses `fps` from the JSON.
- **Quality banner**: shown when `quality.calibration_low_confidence` /
  `sync_low_confidence` are set, cross-camera disagreement > 1 m, or `warnings`
  is non-empty.

## Camera anchors

- **Player anchors** — click a player in the sidebar: the camera sits at that
  player's head (1.7 m) and looks along their velocity (last known heading when
  stationary). Click again or press `Esc` to release.
- **Free pitch-level anchors** — six buttons: behind goal A, behind goal B,
  corner A/S, corner B/N, halfway S, halfway N. All at 1.7 m looking at the
  centre spot.
- **Mouse-look** — from any anchor, drag in the 3D view to yaw/pitch; wheel
  changes FOV.
- **Free orbit** — OrbitControls (drag = orbit, wheel = zoom, right-drag = pan).

## Coordinates

`pitchworld` world frame: `x` along the pitch from goal A (`x=0`) to goal B
(`x=length`), `y` across from touchline S (`y=0`) to touchline N (`y=width`).
The viewer maps `x → three.X`, `y → three.Z`, up = `three.Y`.

## Sample data

`sample/tracking.json` is synthetic (10 players on a 50×30 pitch, 20 s @ 25 fps,
with `calibration_low_confidence: true` so the banner is exercised). Regenerate
with `python viewer/sample/make_sample.py`.

## Next (Reactor integration)

The player-anchor pose (position + look direction per frame) is exactly what
Stage 2 needs to request a world-model render from a player's point of view;
`anchorPose()` in `app.js` is the seam to hook into.
