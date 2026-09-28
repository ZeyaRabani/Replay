# Stage 2 — generating the pitch world on Reactor (HappyOyster)

Stage 1 gives us synced clips, `tracking.json` and (low-confidence) camera calibration.
Stage 2 turns one real frame of that footage into a **persistent, explorable, real-time
generated world** on Reactor. This document records exactly what was run, what came back,
and what it does and does not prove. Research background is in `docs/stage2_reactor.md`.

## Result

| | |
|---|---|
| Model | `reactor/happy-oyster-adventure` (HappyOyster, the only persistent Reactor model) |
| **encrypted_world_id** | `dHpmaEU0lINZtc81EZgHEgdQ-9pTeoJ79eHHZiW2ZfA` |
| Phase after build | `ready` (build took ~25 s of session time) |
| Perspective | `third_person` |
| Seed | `reactor-app/seeds/seed_cam0.jpg` — cam0 full frame, centre-cropped to 16:9, 1664×936, 489 KB |
| Prompt | `reactor-app/seeds/prompt_cam0.txt` (generated from `tracking.json`, see below) |
| Record | `out/reactor/world.json` (committed; no credentials inside) |
| Tries | 1 world build succeeded first time; no prompt/seed iteration was needed |

Visual verification (first frame returned by Reactor, then a live attach with WASD movement):
the stream shows the artificial pitch, white lines, the blue D arc, the portable white goals,
orange vs yellow bibs, the brick sports hall and floodlight masts from the real footage, and
stays a football pitch while the camera walks forward and turns. Evidence attached to the PR
(`out/reactor/first_frame_adventure.jpg`, `live_view_*.png`, screen recording).

## How it was produced

```bash
cd reactor-app
cp .env.example .env            # put REACTOR_API_KEY=... here; .env is gitignored
export REACTOR_API_KEY=...       # for the Python script

python3 scripts/gen_world.py \
  --tracking /path/tracking.json \
  --frames full0.jpg full1.jpg full2.jpg \        # or --clips cam0.mp4 cam1.mp4 ... --t0 3.0
  --mode adventure \
  --seed-url https://raw.githubusercontent.com/ZeyaRabani/Replay/devin/stage2-worldgen/reactor-app/seeds/seed_cam0.jpg \
  --out out/reactor/world.json
```

`gen_world.py` is generic: any number of cameras; it picks the camera with the most
tracked players per frame unless `--camera N` is given, extracts a frame with `ffmpeg` when
clips are passed, crops/letterboxes to a 1.5–2.0 landscape aspect and compresses under 2 MB,
writes the prompt from the `pitch` dict + `quality.stats` in `tracking.json`, connects with
the Python SDK (`Reactor("reactor/happy-oyster-adventure", api_key)`), sends `create_world`,
waits for `phase == ready` and writes the record. `--dry-run` stops after seed + prompt.
`--mode directing` targets `reactor/happy-oyster-director` (cinematic, 720p, no WASD).

### What conditioning worked

- **Seed by public URL, not upload.** HappyOyster's backend fetches the first frame itself.
  A session `FileRef` from `upload_file` and a tmpfiles.org URL were both rejected with
  `400001 Unable to fetch the image from this URL on the server`. A GitHub raw URL of the
  committed seed worked, so `--seed-url` is the reliable path (`--upload tmpfiles` is kept
  as a best-effort default for other hosts).
- **Prompt describing the real venue** (artificial turf, white lines, blue D arcs, portable
  goals with wheels, brick hall, floodlights, evening light, orange vs yellow bibs, ~28
  players, camera ~2 m high, "pitch stays fixed from every angle"). The generated world kept
  all of these; the seed frame dominates the look, the prompt keeps it on-theme when moving.
- `perspective: third_person` gives a broadcast-like height; first-person is also valid.

### What Reactor cannot take (fidelity gap — read this, judges)

Reactor is a **generative real-time video model, not a reconstruction engine**. It accepts
one image + text. It does **not** ingest the other cameras, the calibration, or the player
trajectories in `tracking.json`. Consequences:

- The world is *plausible*, not *metric*. Distances, line positions and the far half of the
  pitch are hallucinated in the style of the seed; players are generated extras, not the 28
  tracked people, and they are not at the tracked coordinates.
- The world is persistent (same id re-attaches to the same scene) but each travel is a fresh
  generation from the seed; walking far away drifts.
- Stage 1's own calibration is low confidence today (`calibration_low_confidence=true`,
  cameras disagree by 1.7–2.6 m), so even a perfect reconstruction would carry that error.
  We do not hide it: the quality block is copied into `out/reactor/world.json`.

So the honest split for the demo: **Stage 3's deterministic Three.js twin viewer** is the
source of truth for *where players are* (metric, from `tracking.json`); **the Reactor world**
is the immersive "walk onto this pitch" layer whose *look* comes from the real footage.
Player-follow camera paths from Stage 1 can drive HappyOyster's held move/look controls, but
what is rendered is generated, not replayed.

## Cost

One world build (~25 s) plus two attach/travel sessions of ≤2 min. At Reactor's per-minute
streaming pricing this is on the order of a few cents to well under $1 total. Each further
2-minute travel costs one session-minute budget; no build cost is paid again.

## How Stage 3 attaches

The world is permanent. Anyone with the id (no build wait):

- **UI:** `cd reactor-app && pnpm install && pnpm dev`, open http://localhost:3000, scroll to
  *Return to an existing world*, choose *Adventure*, paste
  `dHpmaEU0lINZtc81EZgHEgdQ-9pTeoJ79eHHZiW2ZfA`, click *Attach*. Stream starts in ~10 s;
  WASD moves, arrows look, travel is capped at 2:00 (click *End travel* / start again).
- **TypeScript (SDK, what the app does):**
  ```ts
  const { attachWorld } = useHappyOyster();
  await attachWorld("dHpmaEU0lINZtc81EZgHEgdQ-9pTeoJ79eHHZiW2ZfA"); // then startTravel()
  ```
- **Python:**
  ```python
  await reactor.send_command("attach_world", {"encrypted_world_id": "dHpmaEU0lINZtc81EZgHEgdQ-9pTeoJ79eHHZiW2ZfA"})
  ```

Auth stays server-side: the browser calls `app/api/reactor/token/route.ts`, which exchanges
`REACTOR_API_KEY` for a short-lived scoped JWT (`POST https://api.reactor.inc/tokens`).
The key never reaches the client.

## Not done / fallbacks

- LingBot World 2 short steered clip was **not** run (time budget; HappyOyster succeeded on
  the first attempt and is the model that gives persistence). `docs/stage2_reactor.md`
  documents the LingBot path if a non-persistent clip is wanted.
- Scaffold change: `SnapClip` was removed from the sidebar because it called `useReactor`
  outside the provider and 500'd the page; clip recording is not needed for the demo.
