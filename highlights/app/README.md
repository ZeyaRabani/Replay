# Replay Highlights

Product shell for a football-highlights review + render tool. FastAPI backend
(multi-user, multi-project, detached pipeline subprocess), React/Vite
frontend, FFmpeg rendering, and a CLI.

## Setup

```bash
# from repo root
pip install -r highlights/app/requirements.txt

cd highlights/app/frontend
npm install
npm run build
```

## Run

```bash
# backend (serves the built frontend at / when dist/ exists)
uvicorn highlights.app.backend.main:app --host 127.0.0.1 --port 8000

# dev frontend with HMR (proxies /api -> 127.0.0.1:8000)
cd highlights/app/frontend && npm run dev
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `HL_WORKDIR` | `~/.replay_highlights` | data root (users.json + projects/) |
| `HL_DEMO_VIDEO` | `/home/ubuntu/match/match.mp4` | demo project's local match file |
| `HL_PIPELINE_CMD` | `python -m highlights.pipeline.run` | pipeline runner command (shlex-split) |
| `HL_YT_COOKIES` | unset | cookies file passed to the runner for YouTube downloads |

## Auth & projects

Every request except media GETs carries an `X-User` header naming a user
created via `POST /api/users` (`demo` is always present). Users only see
their own projects (404 otherwise). Media GET routes (`video/source.mp4`,
`video/proxy.mp4`, candidate/project thumbnails, render file downloads) are
unauthenticated so `<video src>` / `<img src>` work without headers.

Directory layout (Contract 1):

```
$HL_WORKDIR/
  users.json
  projects/<id>/            # id = uuid hex[:12]
    project.json            # Contract 3 state
    source/                 # uploaded source file
    proxy.mp4               # low-res proxy
    thumbs/                 # thumbnail cache (keyed by video path+mtime)
    renders/<job_id>/       # clips/, reel.mp4, stats.json, manifest.json
    pipeline/
      status.json           # runner heartbeat
      log.txt               # runner stdout/stderr
      candidates.json       # runner output, imported lazily
      stats.json            # Contract 5 stats
```

## API

### Users

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/users` | `{"name"}` -> `{name, created_at}` (idempotent) |
| GET | `/api/users` | `[{name, created_at, n_projects}]` |

### Projects (X-User required)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/projects` | multipart `file`+`title`, or JSON `{title?, youtube_url?, path?, run_pipeline?}` |
| GET | `/api/projects` | `[ProjectSummary]` for the caller |
| GET | `/api/projects/{id}` | summary + `pipeline` status |
| DELETE | `/api/projects/{id}` | cancel pipeline, remove jobs, delete dir -> 204 |
| POST | `/api/projects/{id}/pipeline/run` | `{stages?, force?}` -> status (409 if busy) |
| POST | `/api/projects/{id}/pipeline/cancel` | SIGTERM process group -> failed status |
| GET | `/api/projects/{id}/pipeline` | status + `log` tail |
| GET | `/api/projects/{id}/stats` | Contract 5 stats JSON |
| GET | `/api/projects/{id}/thumb.jpg` | project thumbnail (public) |

Project sources: `youtube` (runner downloads), `upload` (multipart file,
streamed to `source/` in 1 MiB chunks), `path` (local file). Uploads and
local paths run the pipeline with `--stages
probe,audio,motion,features,score,candidates,stats` (no download stage);
`run_pipeline=false` skips the pipeline entirely (dev/test shortcut).

### Scoped review routes (`/api/projects/{id}/...`, X-User required)

| Method | Path | Purpose |
|---|---|---|
| POST/GET | `/video` | register / get video info (`proxy_ready`) |
| POST | `/video/proxy` | build low-res proxy in background |
| GET | `/video/proxy/status` | `{ready, progress}` |
| GET | `/video/proxy.mp4` | proxy file (public, Range) |
| GET | `/video/source.mp4` | original file (public, Range) |
| POST | `/candidates/load` | multipart file or JSON `{"path"}` |
| GET | `/candidates?sort=confidence\|time` | list |
| PATCH | `/candidates/{cid}` | edit status/clip window/type/notes |
| POST | `/candidates/{cid}/reset` | restore default clip window |
| GET | `/candidates/{cid}/thumb.jpg?t=` | cached thumbnail (public) |
| GET | `/project` | legacy project shape |
| POST | `/render` | `{"ids"?, "overlay"?, "reencode"?}` -> `{job_id}` |
| GET | `/render/{job_id}` | job status/progress/outputs |
| GET | `/files/{job_id}/{name}` | render artifacts (public) |

### Legacy routes

The old unscoped `/api/video`, `/api/candidates`, `/api/render`, `/api/stats`,
`/api/project`, `/api/files/...` etc. still exist and map to user `demo`'s
newest project (404 if none). `GET /api/stats` keeps the old stats shape;
the scoped `/stats` returns Contract 5.

## CLI

```bash
python -m highlights.app.render --video match.mp4 --candidates candidates.json --out reel/
```

Options: `--only {confirmed|all}` (default: cross_validation != rejected),
`--types goal,shot`, `--min-confidence 0.5`, `--pad-goal 5`, `--pad-default 3`,
`--no-overlay`, `--reencode`, `--max-clips N`.

Outputs: `out/clips/*.mp4`, `out/reel.mp4`, `out/stats.json`, `out/manifest.json`.

## candidates.json schema

```json
{"source": "<track>", "video_duration_s": 5340.0,
 "events": [{"type": "goal|shot|chance|excitement|other", "t": 0.0,
   "t_start": 0.0, "t_end": 0.0, "confidence": 0.0, "signals": {},
   "notes": "", "cross_validation": "confirmed|pipeline_only|visual_only|rejected"}]}
```

## Durability

- The pipeline is a detached `Popen` (`start_new_session`, stdout ->
  `pipeline/log.txt`); a backend restart does not kill it.
- `pipeline/status.json` is the runner's heartbeat. On registry (re)build and
  on each project read, a queued/running status whose pid is gone is
  reconciled to `failed` ("interrupted; click Re-run to resume").
- When a run finishes, the project lazily imports `video_path` (ffprobe) and
  `pipeline/candidates.json` on the next read.
- Cancel sends SIGTERM to the runner's process group, escalating to SIGKILL
  after ~3 s.
- `project.json` and `status.json` are written atomically (tmp + rename).

## Demo project

On an empty workdir the backend seeds a `demo` user project "Demo match
(5qj_nsQSzvQ)" from the committed `highlights/fusion/outputs/` (47 events,
match_window, precomputed stats). `HL_DEMO_VIDEO` points it at the local
match file so video/proxy/thumbnails work.

## Stats (Contract 5)

`backend/stats.py` bins per-second features into 30 s timeline rows
(motion/audio normalised 0–1 over the match window, excitement = 0.5+0.5
mix), counts non-rejected events per bin and per 10 min block, lists the top
10 moments by confidence, and reports activity (mean/peak motion, loudest
time, quietest 5-min stretch). `demo_stats()` computes this from the
committed fusion outputs; the half-time break is cut from the demo video, so
`halves` is empty there.

## Testing

```bash
python -m pytest highlights/app/tests highlights/fusion -q
```

Tests point `HL_PIPELINE_CMD` at `tests/fake_pipeline.py` — a stub runner
that writes `argv.json`, cycles status queued -> running -> done, and emits
two candidates. `FAKE_PIPELINE_HANG=1` keeps it running (cancel tests);
`FAKE_PIPELINE_CRASH=1` exits mid-run.

## Defaults

- Clip padding: 5 s around goals, 3 s otherwise; clamped to [0, duration].
- Overlay (default on): one-line `"{TYPE}  {mm:ss}"` drawtext, bottom-left,
  white on a semi-transparent box. Re-encodes affected clips to libx264.
- Stream copy is used whenever overlay and reencode are off.

## Runtime

Measured on an 8-core CPU box (no GPU), ffmpeg 4.4.2, Python 3.10, Node 20:

- CLI on the real 89-min match (1280x576@30fps, 850 MB) with the 12-event
  `sample/candidates.json`: 11 clips (78 s total) + `reel.mp4`, overlay on:
  **8.7 s wall** (`user 31.8 s`).
- Same on the 120 s synthetic sample, 4 clips: 1.8 s.
- UI render job, 3 goal clips (30 s) with overlay: ~26 s including job polling.
- Thumbnail (single ffmpeg frame seek into the 89-min file): well under 1 s.
- Low-res proxy (360p/15 fps) of the full match: several minutes; it is
  optional — the player streams the source with HTTP range requests until
  the proxy is ready.

Outputs of the real-match CLI run (stats + manifest) are in `outputs/`.

## Known limitations

- Stream-copy cuts snap to the nearest keyframe, so clip boundaries are
  approximate.
- `X-User` is a plain header (no passwords/tokens) — trusted-network auth only.
- drawtext overlay skipped (with a warning) if ffmpeg lacks drawtext or no
  font is found.
