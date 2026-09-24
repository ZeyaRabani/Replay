---
name: replay-highlights-ui-testing
description: Run and test the local Replay Highlights review UI with isolated project state and real media.
---

# Replay Highlights UI testing

Run from the Replay repository root. No application authentication is required.

## Devin Secrets Needed

None for local video review and rendering. Do not require an LLM API key for this UI.

## Runtime setup

- Python dependencies are in `highlights/app/requirements.txt`; ffmpeg/ffprobe must
  be available.
- Frontend dependencies are in `highlights/app/frontend/package.json`.
- If npm is absent from PATH, source `$HOME/.nvm/nvm.sh`, inspect installed versions
  with `nvm ls`, and activate an installed compatible version rather than assuming
  Node is missing. Node 24 worked in this environment.
- Set a fresh, temporary `HL_WORKDIR` before starting the backend so review changes
  do not affect a previous session's state:
  `HL_WORKDIR=/tmp/replay-ui-test uvicorn highlights.app.backend.main:app --host 127.0.0.1 --port 8000`.
- Run `npm run dev -- --host 127.0.0.1` in `highlights/app/frontend`; open
  `http://127.0.0.1:5173`. Vite proxies `/api` to port 8000.
- Backend can serve a prebuilt frontend at port 8000, but prefer Vite when verifying
  the current checkout to avoid stale build artifacts.

## UI workflow and pitfalls

- Load video first via absolute server path, then load candidates by path or upload.
- `cross_validation` is detector/reviewer provenance; `status` is the editable
  review decision. A cross-validation-confirmed event may start review-pending.
- IN/OUT commits happen on blur or Enter. Test invalid edits both immediately and
  after a browser reload; a visible error alone is not proof the state was unchanged.
- Render reel chooses confirmed review candidates; if none are confirmed it chooses
  all non-rejected candidates. Selecting a card only selects playback, not rendering.
- Download links use attachment responses. Open downloaded MP4 files through
  `file://` in Chrome to demonstrate actual playback.
- Use a short real committed clip to test proxy generation without waiting for a
  full-match encode. Clearly distinguish that coverage from full-match proxy tests.
- Source switching retains candidates. Verify candidate windows are appropriate
  for the new duration and that thumbnails, playback source and job display update.
- Check both switching an existing candidate set to a shorter source and importing
  candidates while that shorter source is already loaded. Verify fields immediately
  (before reload), then verify reload persistence; these exercise separate paths.
- Restart uvicorn after backend edits; Vite normally hot-reloads frontend edits.
- Fusion CLI quality presets and the app backend renderer are separate paths;
  do not infer UI quality behavior from a successful fusion CLI render.
