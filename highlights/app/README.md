# Replay Highlights

Product shell for a football-highlights review + render tool. FastAPI backend,
React/Vite frontend, FFmpeg rendering, and a CLI. No detection logic, no auth.

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

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/video` | register video `{"path"}` (ffprobe) |
| GET | `/api/video` | video info + `proxy_ready` |
| POST | `/api/video/proxy` | build low-res proxy in background |
| GET | `/api/video/proxy/status` | `{ready, progress}` |
| GET | `/api/video/proxy.mp4` | proxy file (Range) |
| GET | `/api/video/source.mp4` | original file (Range) |
| POST | `/api/candidates/load` | multipart file or JSON `{"path"}` |
| GET | `/api/candidates?sort=confidence|time` | list |
| PATCH | `/api/candidates/{id}` | edit status/clip window/type/notes |
| POST | `/api/candidates/{id}/reset` | restore default clip window |
| GET | `/api/candidates/{id}/thumb.jpg?t=` | cached thumbnail |
| POST | `/api/render` | `{"ids"?, "overlay"?, "reencode"?}` -> `{job_id}` |
| GET | `/api/render/{job_id}` | job status/progress/outputs |
| GET | `/api/files/{job_id}/{name}` | download render artifacts |
| GET | `/api/stats` | stats JSON |

## State & outputs

- Project state persists to `highlights/app/workdir/project.json` (override with `HL_WORKDIR`).
- Renders land in `workdir/renders/{job_id}/` (`clips/`, `reel.mp4`, `stats.json`, `manifest.json`).
- Thumbnails cached in `workdir/thumbs/`; proxy at `workdir/proxy.mp4`.

## Defaults

- Clip padding: 5 s around goals, 3 s otherwise; clamped to [0, duration].
- Overlay (default on): one-line `"{TYPE}  {mm:ss}"` drawtext, bottom-left, white
  on a semi-transparent box. Re-encodes affected clips to 720p libx264.
- Stream copy is used whenever overlay and reencode are off.

## Runtime

Measured on an 8-core CPU box (no GPU), ffmpeg 4.4.2, Python 3.10, Node 20:

- CLI on the real 89-min match (1280x576@30fps, 850 MB) with the 12-event
  `sample/candidates.json`: 11 clips (78 s total) + `reel.mp4`, overlay on
  (720p re-encode): **8.7 s wall** (`user 31.8 s`).
- Same on the 120 s synthetic sample, 4 clips: 1.8 s.
- UI render job, 3 goal clips (30 s) with overlay: ~26 s including job polling.
- Thumbnail (single ffmpeg frame seek into the 89-min file): well under 1 s each.
- Low-res proxy (360p/15 fps) of the full match: several minutes; it is
  optional — the player streams the source with HTTP range requests until the
  proxy is ready.

Outputs of the real-match CLI run (stats + manifest) are in `outputs/`.

## Known limitations

- Stream-copy cuts snap to the nearest keyframe, so clip boundaries are approximate.
- No auth; single shared project state.
- drawtext overlay skipped (with a warning) if ffmpeg lacks drawtext or no font is found.
