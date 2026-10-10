# HL4 — Replay Highlights v2: shared design spec

Repo: github.com/ZeyaRabani/Replay (branch off `main`, which now contains the merged
`highlights/` tree from PR #31). Python 3.10, FastAPI backend at
`highlights/app/backend/`, React/Vite/Tailwind frontend at `highlights/app/frontend/`,
existing detection tracks at `highlights/{audio,motion,tracking,spotting,fusion}/`.
Read `highlights/app/README.md` and `highlights/fusion/REPORT.md` first.

Everything runs locally on CPU. No cloud/paid inference. No auth providers.

Three independent workstreams (A, B, C) run in parallel on separate machines and
must NOT edit each other's directories. They communicate only through the
contracts below. Each opens its own PR against `main`; the lead integrates.

- A: `highlights/pipeline/**`               (new)          — pipeline runner CLI
- B: `highlights/app/backend/**`, `highlights/app/tests/**`, `highlights/app/README.md`
- C: `highlights/app/frontend/**`

Lint/test commands (must pass in each PR):
    ruff check highlights
    python -m pytest highlights/app/tests highlights/fusion highlights/pipeline -q
    cd highlights/app/frontend && npm ci && npm run build

---------------------------------------------------------------------------
## Product requirements (from the user)

1. Input either a YouTube URL or a local video upload; the software does everything.
2. Always download the BEST quality YouTube offers (video+audio, merged). Clips are
   rendered from the native source at CRF 14 / slow / yuv420p / AAC 256k (already
   the "high" preset in `highlights/fusion/render.py` and app `ffmpeg.py`) — never
   downscale, never upscale.
3. More highlights and more stats: a timeline graph of the match (the user liked the
   existing Timeline), event histograms, activity/audio-excitement curves,
   detected halves, top moments, per-type counts.
4. Long jobs run in the background and survive the user closing the tab, getting
   disconnected, or the backend restarting. Progress is visible on return.
5. Super-simple profiles: login with just a display name (no password, no Google).
   Each user sees a history of their projects.

---------------------------------------------------------------------------
## Contract 1 — Project directory layout (owned by B, consumed by A and B)

HL_WORKDIR (env, default `~/.replay_highlights`) /
  users.json                          {"users":[{"name":"zeya","created_at":...}]}
  projects/<project_id>/
    project.json                      see Contract 3 (B owns)
    source/                           uploaded file or yt-dlp output lives here
    match.mp4|.mkv                    canonical video path recorded in project.json
    pipeline/                         A writes everything here
      status.json                     Contract 2
      log.txt
      audio/features_1s.json, whistles.json, audio.wav
      motion/features_1s.json
      features_1s.parquet
      candidates.json                 Contract 4 (app-compatible)
      stats.json                      Contract 5
    renders/<job_id>/...              existing render outputs
    proxy.mp4

`project_id` = 12 hex chars (uuid4 hex[:12]).

---------------------------------------------------------------------------
## Contract 2 — Pipeline CLI + status file (A owns, B consumes)

    python -m highlights.pipeline.run --project-dir <projects/<id>> \
        [--youtube-url URL | --video PATH] [--stages download,probe,audio,motion,features,score,candidates,stats] \
        [--cookies FILE]

- Runs as a plain detached subprocess; must be safe to `Popen(..., start_new_session=True)`
  from the backend and keep running if the backend dies. Idempotent: rerunning skips
  stages whose outputs already exist unless `--force`.
- Writes `pipeline/status.json` atomically (tmp+rename) at least every 2 s while
  running:

    {
      "state": "queued|running|done|failed",
      "stage": "download|probe|audio|motion|features|score|candidates|stats|done",
      "progress": 0.0-1.0,            # overall
      "stage_progress": 0.0-1.0,
      "message": "human readable, e.g. 'Downloading 1080p (45%)'",
      "error": null | "string incl. actionable hint",
      "started_at": epoch, "updated_at": epoch, "finished_at": epoch|null,
      "pid": int,
      "video_path": "/abs/path/match.mp4" | null,   # set after download/probe
      "video": {"duration_s":..,"width":..,"height":..,"fps":..} | null,
      "download": {"format":"...","resolution":"1920x1080","filesize":...} | null
    }

- download stage: yt-dlp, `-f "bestvideo*+bestaudio/best" --merge-output-format mp4
  --remux-video mp4` (fallback mkv if mp4 merge impossible), `--js-runtimes node`
  when node is present, `--cookies FILE` if provided or env `HL_YT_COOKIES` set,
  progress hook -> status. Record the chosen format/resolution in status.download.
  On "Sign in to confirm you're not a bot" error, `error` must say exactly that
  YouTube blocked the automated download and the user should upload the file
  instead (or provide a cookies file). Never substitute another video.
- probe: ffprobe -> status.video.
- audio: reuse `highlights/audio/extract_audio.py` + `features.py` (+ whistles).
- motion: reuse `highlights/motion/motion.py`.
- features: generic version of `highlights/fusion/build_features.py` that works with
  only audio+motion present (tracking/spotting columns optional, filled 0). Match
  window = detected from audio (kickoff/half-time/full-time heuristics already in
  `highlights/audio/events.py`) rather than the hard-coded 1050–4990; fall back to
  [0, duration] with a warning in status.message.
- score: rule score over available RULE_COLS (see fusion/score.py) + a learned
  logistic model. Train the learned model ONCE on the existing match
  (`highlights/fusion/outputs/features_1s.parquet` + `labels.json`) using only the
  audio+motion columns, save it as `highlights/pipeline/models/audio_motion_lr.joblib`
  (commit it; it's tiny), and apply it to new videos. Report held-out AUROC in the
  commit message/README honestly.
- candidates: NMS 12 s, top 60 peaks inside the match window -> Contract 4 with
  `cross_validation:"pipeline_only"`, type "chance" (or "shot" when motion_goal_roi
  z>2 — document the rule), confidence = calibrated learned prob (0–0.9 cap; never
  0.99 — that is reserved for visually confirmed goals). Also emit the top 15 as
  `status:"pending"`, rest also pending — the user confirms in the UI.
- stats: Contract 5.

---------------------------------------------------------------------------
## Contract 3 — project.json (B owns)

    {"id":"...", "owner":"zeya", "title":"...", "created_at":..,
     "source": {"kind":"youtube|upload|path", "url":..., "filename":...},
     "video": VideoInfo|null, "candidates_version":n, "candidates":[...],
     "proxy_complete":bool, "proxy_source":"", "pipeline_state":"none|queued|running|done|failed"}

B refactors the existing single `ProjectStore` into one store per project dir
(`ProjectStore(root=projects/<id>)`) with a registry; the existing behaviour
(validation, revalidate_windows, render, proxy) is kept per project.

---------------------------------------------------------------------------
## Contract 4 — candidates.json (unchanged app schema)

    {"source":"pipeline", "video_duration_s":..., "events":[
      {"type":"goal|shot|chance|excitement|other","t":..,"t_start":..,"t_end":..,
       "confidence":0-1,"signals":{...},"notes":"","cross_validation":"pipeline_only|confirmed|visual_only|rejected"}]}

---------------------------------------------------------------------------
## Contract 5 — stats.json (A produces; B serves at GET /api/projects/{id}/stats; C renders)

    {
      "duration_s": ..,
      "match_window": [lo, hi],                 # detected play window
      "halves": [{"start":..,"end":..}, {...}] | [],
      "bin_s": 30,
      "timeline": [ {"t":0, "motion":0.12, "audio":0.3, "excitement":0.2, "events":1}, ... ],
          # one row per bin; motion/audio/excitement normalised 0–1 over the match window
      "events_by_type": {"goal":1,"shot":28,"chance":16},
      "events_per_10min": [ {"t":0,"goal":0,"shot":2,"chance":1}, ... ],
      "top_moments": [ {"t":2736,"type":"goal","confidence":0.99,"reason":"..."} , ... top 10 ],
      "whistles": [t, t, ...],
      "activity": {"mean_motion":..,"peak_motion_t":..,"loudest_t":..,"quietest_stretch":[a,b]},
      "pipeline": {"model":"audio_motion_lr v1","auroc_reference":0.xx,"notes":"..."}
    }
For the demo match, B computes the same shape from the committed fusion outputs
(`highlights/fusion/outputs/features_1s.parquet`, `candidates.json`) so stats work
even without rerunning the pipeline. A ships `highlights/pipeline/stats.py` with a
pure function `compute_stats(features_df, candidates_events, duration, match_window, halves, whistles) -> dict`
that B imports — so A must land this function early with the exact signature above,
and B may vendor a copy if A's PR is not yet merged (lead will dedupe).

---------------------------------------------------------------------------
## Contract 6 — HTTP API (B implements, C consumes). All JSON.

Users
  POST /api/users {name}                      -> {name, created_at}  (idempotent; name trimmed, 1–40 chars)
  GET  /api/users                             -> [ {name, created_at, n_projects} ]
Header `X-User: <name>` on all project routes; 401 if missing/unknown.

Projects
  GET  /api/projects                          -> [ProjectSummary]  (owner's projects, newest first)
      ProjectSummary = {id,title,created_at,source:{kind,url?,filename?},pipeline_state,
                        progress,stage,message,video:{duration_s,width,height,fps}|null,
                        n_candidates,n_confirmed,thumb_url|null}
  POST /api/projects  JSON {title?, youtube_url}                 -> ProjectSummary (state queued; spawns pipeline)
  POST /api/projects  multipart file=<video> [title]             -> ProjectSummary (saved to source/, spawns pipeline without download stage)
  POST /api/projects  JSON {title?, path}                         -> same, local path (for CLI/dev)
  GET  /api/projects/{id}                     -> ProjectSummary + {pipeline: status.json contents}
  DELETE /api/projects/{id}                   -> 204
  POST /api/projects/{id}/pipeline/run  {stages?:[], force?:bool} -> status.json  (re-run)
  POST /api/projects/{id}/pipeline/cancel                          -> status.json  (kill pid)
  GET  /api/projects/{id}/pipeline            -> status.json (+ last 50 log lines as "log")
  GET  /api/projects/{id}/stats               -> Contract 5

Project-scoped versions of every existing route (same bodies/responses as today):
  /api/projects/{id}/video, /video/proxy, /video/proxy/status, /video/proxy.mp4, /video/source.mp4
  /api/projects/{id}/candidates, /candidates/load, /candidates/{cid}, /candidates/{cid}/reset, /candidates/{cid}/thumb.jpg
  /api/projects/{id}/render, /render/{job_id}, /files/{job_id}/{name}
  /api/projects/{id}/project  (legacy shape)

Background/durability
  - Pipeline runs are detached subprocesses (Popen start_new_session=True, stdout->log.txt).
  - On backend startup: scan projects; any status.json with state running whose pid is
    dead -> mark failed with message "interrupted; click Re-run to resume" (rerun is
    idempotent so it resumes from completed stages).
  - When status.json reaches done: backend loads pipeline/candidates.json into the
    project store (only if the project has no candidates yet or candidates_version==0)
    and registers status.video_path as the video. Do this lazily in GET /api/projects/{id}
    and in the summary list (cheap file mtime check), not with a thread.
  - Demo: on first start, if HL_WORKDIR has no projects and the committed
    `highlights/fusion/outputs/candidates.json` exists, create project "Demo match
    (5qj_nsQSzvQ)" owned by user "demo" with those candidates and video path from env
    `HL_DEMO_VIDEO` (default /home/ubuntu/match/match.mp4) if that file exists; stats
    from committed fusion outputs. GET /api/users must list "demo".

---------------------------------------------------------------------------
## Contract 7 — Frontend (C)

Routes (react-router-dom):
  /login            name field -> POST /api/users -> store name in localStorage -> /projects
  /projects         history list (cards: title, source badge YouTube/Upload, created, status
                    pill with progress bar + stage message live-polled every 2 s while
                    queued/running, thumbnail, counts). "New project" panel: tabs
                    "YouTube link" | "Upload file". Delete. Switch user (top right).
  /projects/:id     tabs: Review (the existing UI: VideoPlayer, Timeline, CandidateList,
                    CandidateCard, RenderBar — moved under project-scoped API) and Stats.
                    While pipeline running show a progress screen with stage list + live log
                    tail and a note "you can close this tab; processing continues".
  Stats tab: recharts (add dependency `recharts`, version published ≥7 days ago):
    - Match timeline area chart: motion + audio + excitement per bin, event markers
      (goal/shot/chance colours), half-time band shaded, click -> seek in Review tab.
    - Events per 10 min stacked bars.
    - Events by type donut/bars.
    - Top moments list (clickable).
    - Activity KPIs (mean motion, peak, loudest moment, halves detected, whistles).
  Keep the existing look (dark Tailwind theme, lucide icons). `api.ts` gets a
  `projectApi(id)` factory adding `X-User` header from localStorage and the
  `/api/projects/{id}` prefix. Until B's backend is merged, C develops against a tiny
  mock (e.g. `vite.config.ts` proxy to a `mock/server.mjs` implementing Contract 6 with
  fake data) — commit the mock under `highlights/app/frontend/mock/`.
