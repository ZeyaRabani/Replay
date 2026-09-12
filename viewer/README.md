# Replay viewer (Stage 3)

Build-free web app (plain ES modules, Three.js from a CDN) that turns a `pitchworld`
`tracking.json` + the synced source clips into an explorable replay: jump the camera
to any tracked player or to free pitch viewpoints, look around, at any point on the
timeline. The main viewport is a **live Reactor world stream** (HappyOyster, the
persistent world model) when configured, and a **Three.js render of the same
anchors** otherwise — so the viewer always runs. The active mode is shown in the
badge under the title.

## Run

```bash
# 1. secrets stay server-side; .env is git-ignored, never commit it
echo 'REACTOR_API_KEY=rk_...' > .env            # optional: without it the viewer runs in 3D fallback

# 2. serve viewer/ (reads .env or the env var; also exposes POST /token and GET /config)
python viewer/server.py                          # http://localhost:8080  (PORT=… or `server.py 9000` to change)
```

`viewer/media/tracking.json` is loaded automatically; every `viewer/media/cam*_thumb.mp4`
(or `cam*.mp4`) is shown as a synced source-clip thumbnail, generic for N cameras.
Other data: `?src=<url>`, drag & drop a `tracking.json`, or the file picker.

| URL | Mode |
| --- | --- |
| `http://localhost:8080/` | 3D fallback (Three.js) — always works, no key needed |
| `http://localhost:8080/?world=<encrypted_world_id>` | attach to an existing persistent HappyOyster world and stream it |
| `http://localhost:8080/?create&seed=<public-image-url>` | create a new persistent world from a seed image, then stream it; the URL is rewritten to `?world=<id>` so it can be shared/reloaded |

Reactor notes:

- The browser never sees the API key. `server.py` exchanges `REACTOR_API_KEY` for a
  short-lived JWT at `POST https://api.reactor.inc/tokens` behind `POST /token`.
- HappyOyster fetches the seed image itself, so it must be a **public URL**
  (`?seed=` or `SEED_IMAGE_URL` env var → `GET /config`). `viewer/media/seed.jpg`
  (from `pitchworld reactor_export`) is what we publish; the raw GitHub URL of that
  file works.
- Any failure (no key, token rejected, world build failed, quota) or the **3D fallback**
  button drops to Three.js automatically and shows the reason in the badge.
- Limits from `docs/stage2_reactor.md`: 5 concurrent sessions, 10 sessions/min,
  travel sessions of a few minutes; a world creation takes ~1 min.

## Anchors and camera

- **Player anchors** — every tracked id, sorted by frames seen. Camera at head height
  (1.7 m) on the player's shared-pitch-frame position for the current frame, facing
  the run direction when moving (and it keeps the pitch in view), else the centre.
  Team colour (`team` field) and jersey number (`jersey`/`number`) are shown when
  Stage 1 provides them; today's tracking has neither, so ids get a palette colour.
- **Free anchors** — behind goal A, behind goal B, centre circle, touchline S, touchline N.
- **Three.js mode** — switching anchors is a ~0.4 s eased transition (position lerp +
  look slerp). Mouse drag = look around, wheel = FOV, `Esc` = free orbit.
- **Reactor mode (approximate)** — HappyOyster only exposes *held* move/look controls,
  not an absolute camera pose. A jump is translated into a short burst: turn toward the
  target, walk `distance / 3 m·s⁻¹` (capped at 6 s), turn to the target look direction.
  The turn rate and walk speed are assumptions, so the resulting viewpoint is
  approximate and drift accumulates over several jumps. Mouse drag on the video is
  forwarded as held look commands.

## Timeline

Scrubber over the synced duration (`frames / fps`), play/pause (space), 0.25–2×
speed, arrow keys step frames. The source-clip thumbnails seek with the scrubber and
play/pause with it.

## Honest quality banner

`tracking.json["quality"]` drives the red banner. Today's footage has
`calibration_low_confidence: true`: the three cameras disagree on player positions by
~1.7–2.6 m (median), so **player positions and ids are low confidence** and the
metres shown in the sidebar should not be read as measurements.

## Files

- `index.html`, `style.css` — layout and the black / white / `#FFD600` theme.
- `app.js` — tracking loading, pitch + players, anchors, camera, timeline, thumbnails.
- `reactor.js` — `ReactorWorld`: token → SDK → `attachWorld`/`createWorld` →
  `startTravel`, approximate jumps, look-drag, automatic fallback.
- `server.py` — static files + `/token` + `/config` (stdlib only).
- `media/` — `tracking.json`, `seed.jpg`, `cam*_thumb.mp4` (full `cam*.mp4` are
  git-ignored; drop them in to get full-res thumbnails).

## Coordinates

`pitchworld` world frame: `x` along the pitch from goal A (`x=0`) to goal B
(`x=length`), `y` across from touchline S (`y=0`) to touchline N (`y=width`).
The viewer maps `x → three.X`, `y → three.Z`, up = `three.Y`.
