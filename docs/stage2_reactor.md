# Stage 2 — Reactor world-model integration (design, no API calls yet)

Status: research + export tooling. `pitchworld/reactor_export.py --dry-run` writes every file and the
exact command payloads; nothing in this repo talks to `api.reactor.inc`.

## 1. What Reactor actually is (research summary)

Sources read on 2026-09-12 (all fetched as `.md`):

| # | URL | Used for |
|---|-----|----------|
| R1 | https://docs.reactor.inc/llms.txt | SDK index, connection lifecycle, model list |
| R2 | https://docs.reactor.inc/overview.md | platform pitch, `<1s round-trip latency`, Python quick start |
| R3 | https://docs.reactor.inc/model-api-reference/overview.md | model catalogue (FastH3, HappyOyster, LingBot World 2, LingBot, X2, Helios, LongLive-2.0, SANA-Streaming, Visko Orbis, LTX) |
| R4 | https://docs.reactor.inc/model-api-reference/lingbot-world-2/schema.md | tracks, lifecycle, `set_image`, `set_camera_pose` (full spec) |
| R5 | https://docs.reactor.inc/model-api-reference/lingbot-world-2/overview.md | 1664x960 @ 48 fps, chunk semantics |
| R6 | https://docs.reactor.inc/model-api-reference/happy-oyster/overview.md + `schema.md` | persistent worlds (`createWorld`/`attachWorld`/`startTravel`), 2-3 min travel cap, seed image aspect 1.5-2.0 and <= 2 MB |
| R7 | https://docs.reactor.inc/model-api-reference/helios/schema.md + `overview.md` | 33-frame chunks, 640x384 native / 1280x768 at 2x SR, `set_conditioning`, `set_image_strength` |
| R8 | https://docs.reactor.inc/sdk-reference/python.md | `Reactor(model_name, api_key)`, `send_command`, `upload_file`, `request_clip` |
| R9 | https://docs.reactor.inc/concepts/file-uploads.md | `upload_file` -> `FileRef{upload_id,name,mime_type,size}`; uploads only when status `ready`; URL expires in 15 min |
| R10 | https://docs.reactor.inc/resources/rate-limits.md | 5 concurrent sessions/account, 10 sessions/min (burst 3), token TTL <= 6 h |
| R11 | https://api.reactor.inc/pricing (JSON) | per-second rates (below) |
| R12 | https://www.reactor.inc/models/lingbot-world-2/api | same as R4/R5, marketing page |

### Verified facts

* **Paradigm.** Every Reactor model is a *generative, real-time streaming video* model. A session is a
  WebRTC connection (`disconnected -> connecting -> waiting(GPU) -> ready`), the model publishes a
  `main_video` track, and the client sends JSON **commands** that take effect at the next **chunk**
  boundary. Events (not return values) are the source of truth (`image_accepted`, `chunk_complete`,
  `command_error`, ...). (R1, R4)
* **Inputs a world takes:** a **text prompt** plus **one seed/reference image** (`set_image` with a
  `FileRef` from `upload_file`). No model accepts video files, multi-view images, camera intrinsics /
  extrinsics, point clouds or trajectories as *scene* inputs. The only inbound media tracks are the
  webcam / clip inputs of the *editing* models (SANA-Streaming, X2), which restyle a video rather than
  build a world from it. (R3, R4, R9)
* **Camera control.** LingBot World 2 exposes WASD/look setters and a low-level `set_camera_pose`:
  a flat list of per-frame **deltas** `[rx, ry, rz, tx, ty, tz]` (camera-local frame, **y-down**,
  radians + translation), length a multiple of 6, max 1536 floats; 6 floats = constant velocity for
  the chunk, 6 x chunk_size = one per latent frame, other lengths resampled. It is a **velocity
  profile, not a position path**; translation is max-norm normalised per chunk (magnitude erased,
  direction/shape kept); rotation is not normalised. Chunk = 3 latent frames ~= 12 pixel frames.
  Docs call it "a bias, not a rig": there is **no ground-truth camera** in the world model. (R4)
* **Persistence.** HappyOyster is the only model with persistent worlds: `createWorld({prompt,
  firstFrameImage})` returns an `encrypted_world_id`; `attachWorld(id)` re-opens it later; a travel
  streams for at most 2 min (Adventure, held move/look controls) or 3 min (Directing, text
  instructions). Seed image must be landscape with aspect 1.5-2.0 and <= 2 MB. LingBot World 2 /
  Helios have no persistence: `reset` clears prompt + image; re-`start` regenerates. (R4, R6)
* **Output.** LingBot World 2: 1664x960 @ 48 fps. Helios: 33-frame chunks, 640x384 native,
  1280x768 with default `2x` super-resolution. Python SDK delivers decoded frames as NumPy arrays and
  can `request_clip(duration_seconds)` to download a clip. (R5, R7, R8)
* **Latency.** Marketing: "<1 s round-trip". Mechanically: a setter lands on the *next* chunk, so
  steering granularity is one chunk (~0.25 s at 48 fps for LingBot World 2, ~1.4 s for Helios at 24 fps). (R2, R4, R7)
* **Auth.** API key `rk_...`; server exchanges it for a JWT via `POST https://api.reactor.inc/tokens`
  (header `Reactor-API-Key`). Python: `Reactor(model_name="reactor/lingbot-world-2",
  api_key=os.environ["REACTOR_API_KEY"])` does the exchange itself. Tokens live <= 6 h (session-scoped
  default 1 h, `max_sessions` default 5). (R1, R8, R10)
* **Pricing (USD/second of session, R11):** helios 0.0017, lingbot 0.0050, lingbot-world-2 0.0070,
  fast-h3 0.0070, happy-oyster(-adventure/-director) 0.0139, visko-orbis-stable 0.0097, ltx2 0.0300.
  10 000 credits = $1. An 8 s LingBot World 2 shot ~= $0.06 (plus GPU wait time is billed as session time).
* **Limits (R10):** 5 concurrent sessions per account, 10 new sessions/min (burst 3), `429` with
  `Retry-After`; session duration cap optional via `max_session_duration_seconds`.

### Inferred / unverified (treat as hypotheses; the first live test must check them)

* I1. Sign conventions of `set_camera_pose` rotations (which sign of `ry` is "look left") - docs give
  one example (`[0, 0.04, 0, -0.35, 0, 0]` = "steady yaw with lateral counter-drift", i.e. an orbit)
  but no axis diagram. Our export uses a right-handed x-right / y-down / z-forward frame and
  Rodrigues of the camera-body rotation; flip `ry`/`tx` if the live orbit goes the wrong way.
* I2. Whether "12 pixel frames per chunk" is stable across the deployed LingBot World 2 (docs say
  "current chunk size is 3 latent frames (about 12 pixel frames)").
* I3. Whether a seed frame of a *real* pitch keeps players plausible under camera motion. LingBot is
  a navigation model; people moving independently of the camera is exactly the hard case. Expect
  drift within 2-4 s.
* I4. Helios `set_image` mid-generation + `set_image_strength` could be used to re-anchor every chunk
  with a fresh real frame (a "video-guided" mode). Not documented as a use case; untested.
* I5. Hackathon-specific quota or model allow-list (the "Worlds London" materials were not reachable
  from this session; nothing hackathon-specific was verified).

## 2. Consequence for Replay

Reactor **cannot ingest** what Stage 1 produces (3 synced clips + calibrated poses + tracking). It can
only be *anchored* by one image and *steered* by a camera velocity profile. So Stage 2 is not
"upload the scene", it is **"turn one moment into a camera move the world model can perform"**:

```
tracking.json ──► pick anchor player ──► virtual camera path in pitch coords (orbit / follow / static)
synced/cam*.mp4 ─► seed frame from the real camera nearest the virtual start pose ─► seed.jpg (1664x960)
tracking.json ──► prompt text (n players, pitch size, "camera moves, pitch fixed")  ─► prompt.txt
camera path ────► per-frame [rx,ry,rz,tx,ty,tz] deltas, chunked ──────────────────► set_camera_pose cmds
everything ─────► manifest.json (ordered commands, files, provenance, quality flags, cost estimate)
```

### Mapping of Stage 1 artefacts onto Reactor inputs

| Stage 1 artefact | Reactor input | How |
|---|---|---|
| `synced/cam{i}.mp4` | `set_image(FileRef)` seed | frame at `t0` from the calibrated camera whose pose is nearest the first virtual key, letterboxed to 1664x960 (aspect 1.733, within HappyOyster's 1.5-2.0), JPEG <= 2 MB |
| `tracking.json["frames"]` | camera path target | anchor player id -> `(t,x,y)`; orbit/follow paths are computed around it at 48 fps |
| `tracking.json["cameras"][i]["calibration"]["pose"]` | camera path start / `static` mode | physical pose `(x,y,height,yaw,pitch,roll,f)` -> `CameraKey`; `posefit.rotation()` gives the world->camera matrix used for both calibration and export, so conventions match |
| `tracking.json["pitch"]`, `quality.stats` | `set_prompt` | templated text; the prompt must say the *camera* moves and the *pitch* stays fixed (docs: pose and prompt otherwise fight) |
| `tracking.json["quality"]` | manifest `quality` + notes | LOW-confidence calibration is propagated so Stage 3/4 can show the warning; export never blocks on it |
| `homography_px_to_m` | `twin_scene.json.real_cameras` | kept for the fallback viewer (reproject anything back into real footage) |

### Session flow (what the live Stage 2 runner will do, in order - from manifest.json `commands`)

1. `Reactor("reactor/lingbot-world-2", api_key=$REACTOR_API_KEY).connect()`; wait for `ready`.
2. `ref = await upload_file("seed.jpg")` (only allowed when `ready`).
3. `set_prompt{prompt}` -> `set_image{image: ref}` -> `set_seed{seed}` -> first `set_camera_pose` -> `start`.
4. At each `chunk_complete(chunk_index)` send the next `set_camera_pose` block (`send_at_stream_s` in the
   manifest = one chunk ahead, because setters apply at the next boundary).
5. `request_clip(duration_seconds)` or collect `on_frame` NumPy frames -> `out/reactor/<shot>.mp4` for Stage 4.
6. `set_camera_pose{[]}` then `reset` / disconnect (frees one of the 5 session slots).

Multi-shot demos (several anchors) run sequentially in one session with `reset` between shots
(saves the GPU-assignment wait; sessions/min limit is 10). HappyOyster variant: same seed + prompt via
`createWorld`, keep `encrypted_world_id` so the judges' demo can `attachWorld` instantly; camera then
driven by held `move/look` (coarser than `set_camera_pose`).

## 3. `pitchworld/reactor_export.py`

```
python -m pitchworld.reactor_export out/tracking.json --out out/reactor \
    --mode orbit|follow|static --anchor-player 3 --t0 4.0 --duration 8 \
    --radius 6 --height 2.5 --deg-per-s 20 --model lingbot-world-2 --dry-run
```

Writes `seed.jpg`, `prompt.txt`, `camera_path.json`, `twin_scene.json`, `manifest.json`. `--dry-run` is
the only mode (the flag exists so the Stage 2 runner can share the CLI later). Key numbers on the
synthetic fixture used to test it: 8 s orbit -> 384 keys @ 48 fps, 32 `set_camera_pose` chunks of 72
floats, 37 commands, est. $0.056; `ry` = -0.0073 rad/frame for a 20 deg/s orbit (= 20/48 deg per frame).

`manifest.json` shape (abridged):

```json
{"model": "reactor/lingbot-world-2",
 "auth": {"api_key_env": "REACTOR_API_KEY", "token_url": "https://api.reactor.inc/tokens"},
 "files": {"seed.jpg": {"width": 1664, "height": 960, "bytes": 48549, "sha256": "..."}},
 "stream": {"fps": 48, "frames": 384, "seconds": 8.0, "chunks": 32, "est_cost_usd": 0.056},
 "commands": [
   {"send_at_stream_s": 0, "command": "set_prompt", "data": {"prompt": "..."}},
   {"send_at_stream_s": 0, "command": "set_image",  "data": {"image": {"$upload_file": "seed.jpg"}}},
   {"send_at_stream_s": 0, "command": "set_seed",   "data": {"seed": 0}},
   {"send_at_stream_s": 0, "command": "set_camera_pose", "data": {"camera_pose": [72 floats]}},
   {"send_at_stream_s": 0, "command": "start", "data": {}},
   {"send_at_stream_s": 0.0, "covers_t": [0.25, 0.5], "command": "set_camera_pose", "data": {"camera_pose": [...]}},
   {"send_at_stream_s": 0.25, "covers_t": [0.5, 0.75], "command": "set_camera_pose", "data": {"camera_pose": [...]}},
   ...,
   {"command": "set_camera_pose", "data": {"camera_pose": []}}]}
```

`{"$upload_file": "seed.jpg"}` is the one placeholder: the runner replaces it with the `FileRef`
returned by `upload_file` (a `FileRef` is never constructed by hand, R9).

## 4. Fallback plan (if Reactor cannot give a usable result)

Ordered by how much of the demo survives:

1. **Digital twin from tracking only (no NeRF / no splats).** `twin_scene.json` already contains the
   pitch model, every player track `(t,x,y)`, the real camera poses and the virtual camera path. Stage 3's
   3D viewer renders this directly (flat pitch mesh + billboarded player sprites cropped from the
   synced clips via `homography_px_to_m` / boxes, or capsules). Camera anchors on players are exact
   because they come from the same path that would have driven Reactor. This is deterministic, free,
   and works offline; it is the safe demo.
2. **Reactor as stylist, not simulator.** Render the twin with the virtual camera (Stage 3 offscreen),
   then stream those frames through an *editing* model (SANA-Streaming accepts uploaded clips /
   webcam-style inbound video, R3) or re-anchor Helios every chunk with the rendered frame
   (`set_image` + high `set_image_strength`, I4). Camera control is then exact (it is ours) and Reactor
   only adds photorealism.
3. **Reactor for short "hero" cut-aways.** Use LingBot World 2 exactly as exported but keep shots to
   2-3 s (before drift, I3) and cut back to real footage; Stage 4 stitches. The `static` mode
   (re-play a real camera pose with zero deltas) is the calibration sanity check for this path.
4. **Real-footage-only.** Stage 4 uses the synced clips + `tracking.json` overlays (already produced by
   Stage 1 in `debug/`). Zero Reactor dependency.

Decision rule for the day: run one 8 s orbit shot live as soon as `REACTOR_API_KEY` is available; if the
pitch lines / goals do not stay put under the orbit (I3), switch the primary demo to fallback 1 and use
Reactor only per fallback 3.

## 5. Open items for the live runner (not in this PR)

* `pitchworld/reactor_run.py`: `pip install reactor-sdk`, read `manifest.json`, execute section 2,
  save the clip. ~80 lines; blocked only on a key.
* Verify I1/I2 on the first run; if `ry` sign is wrong, negate `out[:, 0:3]` in `keys_to_deltas`.
* Calibration is LOW confidence on the real footage (1.7-2.6 m cross-camera median). It affects only
  where the virtual camera *starts* relative to the seed frame; the seed frame itself is real, so the
  visual anchor is unaffected. The manifest carries the flag either way.
