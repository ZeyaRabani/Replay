# Replay Highlights — developer handover

This document is a written reconstruction of the whole Devin build session
(Sept 24 – Oct 1 2026) for whoever continues the project. It covers what the
owner (Zeya) asked for, what was decided and why, what exists, where it runs,
what data is on the server, what did *not* work, and where to pick up.

Session: https://app.devin.ai/sessions/b0bf3e7bdb2e4cbfa64eb85c20aad47a
Main PR (everything below lives on its branch `devin/1790266995-hl4-app-v2`):
https://github.com/ZeyaRabani/Replay/pull/36

> Note: the chat itself is not stored verbatim in the repo; this is a complete,
> faithful summary of it including direct quotes of the owner's key
> instructions. Design specs written during the session are in `docs/specs/`.

---

## 1. What the product is

A free / self-hosted grassroots-football highlights platform for amateur match
footage (one team, 9-a-side, small pitch, 3 fixed phone/camera angles filmed
from the sidelines, published to YouTube).

Core promise, in the owner's words:

> "Option 1 (already built, keep as-is): a single YouTube match video in →
> tight-cut highlights out (goals + chances, 3 s / 5 s pre/post-roll)."

Plus Option 2: 3 camera angles in → one angle-switching "director cut" out.

Hard constraints the owner set:

- **"Do not carry over the ball-tracking-centric detection approach – it's the
  wrong technique for this footage."** Events come from audio + motion + review,
  not from tracking the ball.
- Everything must be free / self-hosted: Oracle Always-Free ARM VM for the
  backend, Vercel free tier for the frontend. **No GPU.**
- Keep source footage at full quality; everything else is regenerable.
- Recoverable history: deleting a cut/project keeps its metadata (Archive).
- Two distinct directing modes must stay separate:
  1. the automatic AI "director cut" (default);
  2. the **"You Direct"** cut, learned from the owner's own manual directing
     sessions, with more frequent "FIFA/game-like" switching.
- **Rejected ideas (do not build):** residential-proxy YouTube ingest, public
  share links, WhatsApp-sized 720p exports, GPU acceleration.
- **"use as many agents as possible on devin and only do everything on 28/8 if
  you are analysing any footage"** — any footage analysis / tuning must use the
  28/8 match only; other matches are only processed, not experimented on.

---

## 2. Chronology (what happened, in order)

### Phase hl3 — automatic single-video highlights (PR #31, merged)
- 8 parallel child sessions: action spotting (SoccerNet, rejected — poor on
  this footage), player tracking (roboflow/sports), **audio excitement +
  whistle**, review UI, and 4 human-style visual ground-truth reviews of the
  match in quarters.
- Fusion script merges signals into ranked `candidates.json`
  (confirmed / rejected / pipeline_only). Tight clips rendered with FFmpeg.
- Learned: audio + motion LR (held-out AUROC 0.80) is the useful detector.

### Phase hl4 — the app (PR #36, still open; this is the main branch)
- Multi-user (name-only login), per-user project history, background pipeline
  runner (detached subprocess, status.json, restart-safe), YouTube ingest via
  yt-dlp best-quality, direct upload, 360p/720p proxies, stats tab,
  one-command Docker deploy (`deploy/install.sh`), Vercel static frontend
  proxying `/api` to Oracle.
- Review UI: three-way verdicts (goal / highlight / reject), trim, mobile
  "Swipe" review, render reel, download.
- Verdict learning: logistic model on the user's verdicts adjusts scoring and
  clip windows (`LEARN_SPEC.md`).
- History/Archive: SQLite events, archive metadata on delete, restart replays
  settings.

### Phase ma — multi-angle director cut (Option 2)
- `highlights/multiangle/`: `sync.py` (audio cross-correlation + FFT refine,
  triangle consistency), `trackfeat.py` (480p proxy YOLO features per angle),
  `director.py` (segment chooser: Normal / Fast styles, ball zones, dead-feed
  recovery, hysteresis), `render.py` (per-angle mezzanines, segment cache,
  concat), `fuse.py`, `cuts.py` (every cut kept as a version), `run.py`.
- Zones: manual ball zones per keyframe still, plus zones **learned** from the
  owner's directing sessions (`LEARNED_ZONES_SPEC.md`).
- **You Direct**: the owner switches cameras live in the browser over 6
  suggested stretches; sessions are compared with the AI cut and a grid search
  learns style knobs + zones + camera preferences (`DIRECTOR_PREFS_SPEC.md`).
  Agreement with the owner's choices after learning: 4/9 47→48 %, 21/8
  52.6→57.1 %, 19/6 44.5→61.1 %.
- Match window / cut range: re-cuts can be restricted to a time window; old
  cuts stay as versions. Owner later asked to keep **only the active directed
  cut** on 4/9, 21/8, 19/6 (done; metadata archived).
- Ops incidents: OOM when 3 matches' angle pipelines ran at once →
  `HL_ANGLE_WORKERS=2`. Disk: ~60 GB of regenerable caches (mezzanines,
  segments) deleted on request; disk 144→86 GB of 194.

### Phase analysis — match stats, players, radar
- `highlights/analysis/`: "Analyse match" stage — team colours, possession
  proxy, territory thirds, momentum, AI text summary (`stats.py`, `summary.py`,
  `teams.py`).
- Players v2: per-angle 1080p tiled YOLO detection with checkpoints
  (`detect_hr.py`, ~12 h CPU for 3 angles of one match), pitch calibration via
  landmark clicks / camera placement (`calib.py`), cross-camera Kalman fusion
  in pitch coordinates (`fuse_tracks.py`), grouping of fragments
  (`groups_v2.py`), top-down **Radar replay** and a Three.js **3D replay**.
- Owner's correction of pitch geometry — **"55 meters by 23 meters"**, no
  penalty box but a **semicircle (D)** with a penalty mark, goals **"4 meters
  wide and 2 meters long"**. Recalibrated; distances changed only ~3 %.

### Phase "make it way better" — four features in parallel (owner picked 5→2→1→4)
1. **Scoreboard/clock overlay** burned into a new cut version from confirmed
   goals (`multiangle/scoreboard.py`).
2. **Goal-aware directing** (PR #44): hold the goal-facing camera 6 s before →
   4 s after each confirmed shot/goal; insert a 0.5× slow-mo REPLAY from a
   second camera after each goal (`multiangle/goal_aware.py`, `timemap.py`).
3. **Whole-match player identity** (PR #43): link fragments into ≤11 identities
   per team with min-cost flow (`analysis/identity.py`), naming UI, merge/split
   editor.
4. **Per-player reels** (`analysis/player_clips.py`).

### Player identity — abandoned as a primary feature
- First cards mixed teammates. Root cause found: per-detection team labels
  were wrong ~40 % of the time (grass in the torso crop), fixed by a grass-safe
  bib-colour relabel pass (`analysis/kit.py`) → team purity now 847/921 tracks
  agree, 0 disagree. A stale-crop bug (track ids reused across runs) also fixed.
- Within-team identity still unreliable: 11 identical bibs, ~30 px tall players
  at the far end, hundreds of occlusions. Stricter linking just fragments into
  50–80 cards. Owner: **"No, let's give up on this idea of player tracking it
  simply doesn't work."** → then **"make it optional"**.
- Current behaviour: team stats + radar/3D with anonymous team-coloured players
  by default; per-player cards/editor/reels behind a
  "Show per-player cards (experimental)" toggle. Nothing deleted on the server.

### 3D replay (latest work)
- Owner's reasoning: *"make a 3D model of the entire game for 1 minute and know
  from each angle exactly where every player is"* — this is what the radar/3D
  already does with anonymous dots; the hard part is identity, not location.
- Implemented: real 55×23 pitch with D, smoothed motion, "Follow the play"
  camera, browser-side WebM export of one goal window (≤60 s). No server
  rendering (owner: *"only for 1 goal so it's not very hard on everything"*).
- Last request: *"have the 3D replay above and then the actual footage below
  with the latest of the changing angles"* → director-cut footage panel under
  the 3D view, synced via `lib/timemap.ts` (commit 2e022a3).
- **Owner's verdict: "No that doesn't work to a good level at all."** Work
  paused. Failure mode not yet identified (see §7).

---

## 3. Repository map (branch `devin/1790266995-hl4-app-v2`)

```
highlights/
  pipeline/      single-video pipeline: download, audio+motion features, scorer, status
  fusion/        candidate fusion/ranking (hl3)
  audio/ motion/ spotting/ tracking/  hl3 experiments (spotting/tracking not used in prod)
  review1..4/    hl3 visual ground-truth tooling
  multiangle/    Option 2: sync, trackfeat, director, zones, learned_zones,
                 manual_direct (You Direct), goal_aware, scoreboard, render,
                 fuse, cuts, timemap, proxy, viewcheck, run
  analysis/      match stats, teams, summary, players v2 (detect_hr, calib,
                 fuse_tracks, groups_v2, identity, kit, player_clips), players_run
  app/backend/   FastAPI: main.py (API), store.py (projects/users/state),
                 players_api.py; tests in app/tests
  app/frontend/  Vite+React+TS: pages/ (Projects, ProjectPage, ProjectReview,
                 ProjectStats, SwipeReview, Login), components/ (DirectorCut,
                 YouDirect, ZoneEditor, RadarReplay, Replay3D, PlayerAnalysis,
                 IdentityCards, MatchAnalysis, CameraCalib, ...), lib/timemap.ts
deploy/          Dockerfile (arm64, CPU torch/ultralytics), docker-compose.yml,
                 install.sh, README.md
docs/specs/      design specs written during the session
pitchworld/      older standalone multi-camera tracking package (not used by the app)
```

Key env vars (compose): `HL_WORKDIR=/data`, `HL_ANGLE_WORKERS=2`,
`HL_POT_PROVIDER_URL`, `HL_YT_PROXY` (unused — proxy rejected),
`HL_YT_COOKIES`, `REPLAY_ADMINS`, `HL_PUBLIC_URL`, `HL_RENDER_WORKERS`,
`HL_TRACK_PROXY`.

Verification used throughout (there is **no CI** on this repo and no frontend
lint script):

```bash
# backend / pipeline
python -m pytest highlights/app/tests highlights/multiangle/tests highlights/analysis/tests
ruff check highlights
# frontend
cd highlights/app/frontend && npx tsc --noEmit -p . && npm run build
# deploy frontend (prebuilt dist, see vercel.json)
npx vercel deploy --prod --yes
```

---

## 4. Production

- Frontend: https://replay-highlights.vercel.app (static dist, `/api` proxied).
- Backend: Oracle Always-Free Ubuntu ARM VM, 4 cores / 24 GB / 194 GB disk,
  http://79.72.77.103, app in `~/replay`, container `deploy-replay-1`,
  token helper `deploy-bgutil-provider-1`, data volume at `/data`.
  Deploy = `cd ~/replay && git pull && docker compose -f deploy/docker-compose.yml up -d --build`.
- Matches on the server (all 3-angle): **28/8** (project `d2e54e589ea4`, the
  only one used for analysis experiments; has calibration, players v2
  detections/fusion/identities, radar, 3D), 4/9, 21/8, 19/6 (You-Direct
  learned cuts, only active cut kept), 7/8, 10/6, 3/7 (Fast AI cuts).
- Disk after cleanup: ~86 GB — 44 GB source footage (keep), 21 GB main
  `match.mp4` per match (keep), 16 GB active cuts (keep); caches regenerate.
- Player detection checkpoints for 28/8 live under its project's analysis
  folder; an older fusion result is kept as a backup folder. **Do not delete
  sources or detection checkpoints without the owner's OK** (re-detection is
  ~12 h of CPU).

### YouTube ingest — known unsolved problem
YouTube blocks Oracle's datacenter IP ("Sign in to confirm you're not a bot").
The bgutil PO-token provider is wired in but does not beat the IP block.
Working path today: the owner pastes cookies into the **YouTube access** panel
(export them from a fresh incognito login, otherwise YouTube rotates them
within minutes). A residential proxy would fix it; the owner **declined** it.
Direct upload always works.

---

## 5. Things learned about the footage / numbers to trust

- Pitch: 55 × 23 m, D-shaped area, 4 m goals; three ~1080p side cameras.
- Audio excitement + motion is the reliable event detector; action-spotting
  models trained on broadcast football are not.
- 28/8 fused tracking: 921 tracks after relabel (1,249 before), median 18–22
  players visible at once (correct for 9-a-side + subs), ~67–72 % of the match
  covered per identity. Team-level numbers are sound; per-player identity is
  not.
- The detector is partly sequential; it uses ~1.7 of 4 cores and nothing
  speeds it up on this box except fewer frames/angles.
- Director output time ≠ live time once slow-mo replays are inserted; always
  map with `multiangle/timemap.py` / `frontend/src/lib/timemap.ts`.

---

## 6. Owner's working preferences (worth knowing)

- Wants parallelism ("use as many agents as possible") and very short status
  updates; asks "where are we now?" often — keep a running status.
- Accepts honest caveats; dislikes features that are "not to a good level".
- Approves destructive server actions only explicitly (cache deletion, cut
  deletion were approved; nothing else is).
- Frontend changes should be deployed to the Vercel URL, not just pushed.

---

## 7. Open items / where to pick up

1. **3D replay + synced footage** (`components/Replay3D.tsx`): owner rejected
   the result but did not say why. First step: ask/observe which of these it
   is — (a) sync drift between video and dots, (b) footage shows the wrong
   moment (output-vs-live mapping, cut range `cutLo`, replay inserts),
   (c) layout/usability (video too small, autoplay muted, stutter from seeking
   a 3.8 GB mp4 — the proxy is used only if ready). Reproduce on the 28:40
   goal of 28/8.
2. **PR #36** is huge (200+ commits) and open; decide whether to merge as-is
   or squash. PRs #43/#44 were merged into its branch.
3. YouTube ingest without cookies — no accepted solution yet.
4. Player identity: kept optional; a stronger appearance model (shorts/skin/
   hair) was estimated at ~half a day with uncertain gain.
5. Oracle has no CI/CD; deploys are manual `git pull && compose up --build`.
