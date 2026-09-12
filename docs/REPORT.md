# Replay — integration report (Stages 1–4)

Branch `devin/stage2-4-integration` merges every stage branch (stage1-realdata,
stage2-reactor-export, stage2-worldmap, stage2-worldgen, stage3-viewer,
stage3-replay-viewer, stage4-veed-demo-assets, stage4-demo-pipeline, tests-ci,
tracking-fusion-quality). `python -m pytest tests -q` → 39 passed;
`ruff check pitchworld tests` clean.

## Run Replay

```bash
pip install -e .
echo "REACTOR_API_KEY=rk_..." > .env          # git-ignored; key never reaches the browser
cp examples/real_footage/tracking.json viewer/media/tracking.json   # already done on this branch
python viewer/server.py                       # -> http://localhost:8080
```

- Live Reactor world (persistent HappyOyster, seeded from today's cam0 frame):
  `http://localhost:8080/?world=dHpmaEU0lINZtc81EZgHEnya1amxS9MxfbMbSnp2-J4`
- Create a fresh world: `http://localhost:8080/?create&seed=<public seed URL>`
  (HappyOyster only accepts a publicly fetchable `first_frame_image_url`).
- Three.js fallback (no key / Reactor down): `http://localhost:8080/` or the
  "3D fallback" button. The mode badge in the sidebar shows which is active.

Controls: click a player in the list to jump to a head-height camera on that
player (looking along their velocity); "Behind goal A/B", "Centre circle",
"Touchline S/N" are free anchors; drag to look around; space plays the
timeline; the three thumbnails are the synced source clips.

## What was run live (2026-09-12)

1. Server-side token exchange verified (`POST /token` → JWT, 200).
2. Attached to the persistent HappyOyster world: the stream shows the real
   venue look (artificial pitch, blue D arc, portable goals, orange/yellow
   bibs, brick hall, floodlights). Confirmed visually — it is a pitch.
3. Played the timeline, jumped to players 16 → 12 → 23, dragged free-look,
   jumped to the "Behind goal A" free anchor, all in LIVE Reactor mode
   (no fallback was needed).
4. Screen-recorded the walkthrough (57 s) and cut `replay-demo.mp4` with
   `demo/make_demo.py` (title card → 3 original angles → viewer recording with
   6 captions → outro fidelity card). 71.5 s, 1280x720, 9.4 MB.

## Demo

- `replay-demo.mp4` at the repo root (committed, 9.4 MB).
- Reproduce:
  ```bash
  python demo/make_demo.py --recording demo/out/rec.mp4 --jumps demo/out/jumps.json \
    --clips cam0.mp4 cam1.mp4 cam2.mp4 --tracking examples/real_footage/tracking.json \
    --crop 1600:1040:0:85 --out replay-demo.mp4
  ```
  (`demo/record_viewer.py` can capture the recording and write `jumps.json`
  automatically; this run used a manual recording and a hand-written
  `jumps.json`.)

## Limitations (honest)

- **Reactor generates, it does not reconstruct.** HappyOyster is a generative
  world model seeded with one real frame + a prompt built from `tracking.json`.
  The look is the real venue, but the geometry is plausible rather than
  metric, the far half of the pitch is hallucinated, and the people in the
  stream are generated extras — *not* the tracked players. The Three.js twin
  is the source of truth for player positions.
- **Anchor → camera mapping is approximate.** HappyOyster has no absolute
  camera pose; a "jump" is a burst of held move/look commands with assumed
  walk speed / turn rate, so the Reactor viewpoint drifts from the intended
  anchor over time. In fallback mode anchors are exact.
- **Calibration is low confidence.** Cameras disagree on player positions by
  1.7–2.6 m (median, pairwise); cam0's fitted height (0.7 m) is implausible;
  pitch 50x30 m and D radius 6 m are guesses. Only ~10 % of observations are
  multi-camera, so tracks are duplicated (~28 "players"/frame on a 5-a-side
  pitch, 102 ids, 18 stable). No ball track. Modal credentials were absent on
  the day, so detections were reused from the cached run rather than re-run.
- **Seed must be a public URL**: SDK upload and temp hosts were rejected
  (400001), so the seed is committed and served via GitHub raw.
- **No narration** in the demo; WhisperX is wired as an optional `--narration`
  path. VEED OpenEdit cannot render on Linux (`unsupported platform
  linux/x64`) — `replay-demo.wv.html` is emitted for a Mac re-render, the
  actual cut is ffmpeg.
- Persistent world travel budget is 2–3 min per session, 5 concurrent
  sessions; long demos may need re-attach.
