# Option 2 — multi-angle full-match assembly (design spec)

Addition to the existing single-camera tool. New package `highlights/multiangle/`.
Everything self-hosted (ffmpeg, numpy/scipy/librosa, ultralytics YOLO CPU). No cloud APIs.

## Data layout — a multi-angle project is a normal ProjectStore with extras

```
<project>/
  project.json                 # source_info = {"kind": "multiangle", "angles": [...]}
  angles/
    a0/ match.<ext> + pipeline/   # each angle = a full Option-1 sub-project dir run with
    a1/ ...                       #   highlights.pipeline.run (download, probe, audio, motion,
    a2/ ...                       #   features, score, candidates)  + track/features_1s.json (new)
  multiangle/
    status.json                  # StatusWriter, same shape as pipeline/status.json
    log.txt
    sync.json                    # offsets + confidences (see below)
    director.json                # cut decisions + rule ratios
    fused_candidates.json        # CandidatesFile schema, events carry angle info in signals
    score.json                   # final score (see fuse)
  match.mp4                      # THE DIRECTOR CUT — canonical video so the existing
  pipeline/                      #   review/render/stats UI works unchanged:
    candidates.json (= fused)    #   candidates.json, match_window.json, stats.json,
    stats.json, match_window.json#   features_1s.parquet copied/derived from angle 0 (ref)
```

Angle spec (input): `{"url": str|null, "filename": str|null, "label": str}`; label is
user-given ("Green end right"). Angle order = input order; a0 is the reference angle
(timeline zero = a0's file time). Shared timeline T = a0 file time.
`offset[i]` = seconds to ADD to angle i's file time to get T (offset[0] = 0).

## Stage list — `highlights/multiangle/run.py` (CLI, detached like pipeline.run)

```
python -m highlights.multiangle.run --project-dir DIR [--cookies F] [--force]
     [--stages download,angles,sync,track,director,render,fuse,stats]
     [--offsets 0,12.5,-3.2]   # manual override -> sync stage writes these with method=manual
```
Stages, each skipped when outputs exist unless --force (mirror pipeline.run Stage pattern,
reuse StatusWriter, same log format, `_stdout_is`).

1. **download** — for each angle with a url and no `angles/aN/match.*`: call the existing
   `highlights.pipeline.download` best-quality path exactly as pipeline.run stage_download
   does (with cookies). Uploaded files are already placed in `angles/aN/match.<ext>` by the
   backend. Fail loudly with the same bot-block error text as Option 1.
2. **angles** — run `highlights.pipeline.run` as a subprocess per angle (stages
   probe,audio,motion,features,score,candidates — NOT stats), sequentially (4-core box),
   forwarding progress: overall progress = (i + sub_progress)/N. Reuse each angle's
   `pipeline/status.json` for sub_progress.
3. **sync** — `sync.py`:
   - input: each angle's `pipeline/audio/audio.wav` (already extracted by stage_audio).
   - Load mono, resample to 8 kHz. Compute an onset-strength envelope at 50 Hz
     (`librosa.onset.onset_strength`, hop = 160 @ 8 kHz), then whiten: subtract 5 s
     rolling mean, clip at 0, divide by rolling std. This removes speech/mic-gain differences.
   - Pairwise (ref a0 vs each ai) GCC-PHAT style normalised cross-correlation via FFT over the
     full envelopes (zero-padded), lag search range ±(max(len) ) seconds.
   - Confidence = peak / (mean of |xcorr| outside ±5 s of the peak) -> "peak-to-noise ratio"
     (PNR). Also compute second-highest peak ratio r2 = peak2/peak (peaks ≥ 5 s apart).
     `confident = PNR >= 8 and r2 <= 0.6`.
   - Refine: after coarse lag at 50 Hz, re-estimate on raw 8 kHz waveform ±0.5 s around
     coarse lag (cross-correlation of band-passed 300–3000 Hz) to get sub-frame offset.
   - Consistency check: also correlate a1 vs a2; `residual = |off(a1,a2) - (off(a2)-off(a1))|`
     recorded; if residual > 0.5 s, mark pair confidence "inconsistent".
   - Output `sync.json`:
     ```json
     {"reference": 0, "method": "xcorr"|"manual",
      "offsets": [0.0, 12.34, -3.21],
      "pairs": [{"a": 0, "b": 1, "offset": 12.34, "pnr": 14.2, "r2": 0.31, "confident": true},
                {"a": 0, "b": 2, ...}, {"a": 1, "b": 2, ...}],
      "triangle_residual_s": 0.08, "needs_manual": [1]  // angle indexes not confident
     }
     ```
   - If any angle `needs_manual` and no `--offsets` given: write sync.json, set status
     `state="needs_input"`, `message="Sync confidence low for angle(s) X — enter offsets"`,
     and EXIT 0 without running later stages. (Never silently guess.)
   - `--offsets` given: method=manual, all confident=true (manual), still compute + record
     xcorr numbers for the record.
   - Shared timeline coverage: `t_lo = max_i(-offset_i)`, `t_hi = min_i(dur_i - offset_i)`
     → the window where ALL angles exist; also union window. Director cut spans the UNION but
     where only some angles exist, only those are eligible. Write `coverage` in sync.json.
4. **track** — `trackfeat.py`: per angle, YOLO (ultralytics, `yolov8n.pt` default, download
   once into `highlights/multiangle/models/`, `imgsz=960`, `conf=0.25`,
   `classes=[0 (person), 32 (sports ball)]`) at **1 fps**, frames pulled via ffmpeg pipe at
   960 width (reuse detect_track.frame_reader pattern). Per second write row:
   ```
   t, n_players, ball_conf (max, 0 if none), ball_size (box height / frame height),
   ball_x, ball_y (normalised 0..1), players_cx (mean x norm), players_cy,
   players_spread (std x), players_height_mean (norm, proxy for closeness),
   cluster_score = n_players * (1 - |players_cx - 0.5|*2*0.5) * players_height_mean
   ```
   → `angles/aN/track/features_1s.json` `{"fps":1,"model":..,"rows":[...]}` plus
   `ball_rate` (fraction of seconds with ball_conf ≥ 0.35) in a `meta` dict.
   Progress via ctx.status like stage_motion. Must run on CPU ARM (torch cpu wheels).
   Benchmark first with `--max-seconds 120`; if >0.6 s/frame on this box, drop imgsz to 640.
5. **director** — `director.py` (pure numpy, unit-testable):
   - Build a common 1 Hz timeline over the union window; for each angle map T -> file time
     (T - offset_i), sample its track row (nearest second) and motion `motion_total` from
     `pipeline/motion/features_1s.json` (for cut-point search); mark unavailable when outside
     angle's file range.
   - Per second, candidate score per available angle:
     - **Rule BALL**: if any angle has `ball_conf ≥ 0.35` within a ±1 s window → eligible
       angles = those with a ball; score = ball_size (bigger = closer to camera).
       Rule = "ball".
     - else **Rule CLUSTER**: score = cluster_score; rule = "cluster".
     - if no angle has any player rows (dead time / no coverage) → keep current, rule="hold".
   - Smoothing (broadcast style): state machine over seconds.
     `MIN_HOLD = 8 s`. A switch to angle j is *proposed* when j's smoothed score (3 s median)
     exceeds current's by `MARGIN = 25 %` (ball rule) or `40 %` (cluster rule) for
     `CONFIRM = 3` consecutive seconds. When proposed and hold satisfied, choose the cut
     point = the second within [t-2, t+2] with minimum summed motion (natural break), then
     switch. A ball-rule proposal may override MIN_HOLD once hold ≥ 4 s.
   - Also: hard cut to any angle when the current one becomes unavailable.
   - Output `director.json`:
     ```json
     {"segments": [{"t_start": 0.0, "t_end": 41.0, "angle": 2, "rule": "ball"|"cluster"|"hold"|"coverage",
                    "score": .., "runner_up": {"angle":0,"score":..}}, ...],
      "per_second_rule": {"ball": 812, "cluster": 4310, "hold": 220},
      "ratios": {"ball": 0.152, "cluster": 0.807, "hold": 0.041},
      "n_cuts": 133, "mean_hold_s": 40.2, "angle_share": {"0": 0.35, "1": 0.31, "2": 0.34}}
     ```
     Every segment is a logged cut decision (rule fired + scores) — this is the audit log.
6. **render** — `render.py`: ffmpeg. Output canvas 1920x1080 30 fps (or the max common
   height), each angle normalised with `scale=1920:1080:force_original_aspect_ratio=decrease,
   pad=1920:1080:(ow-iw)/2:(oh-ih)/2` — no cropping. Audio: ALWAYS from the reference angle
   a0 (continuous, avoids level jumps), delayed/trimmed by the union window start; where a0
   has no coverage use the next available angle's audio (rare, at ends). Encode per segment
   in parallel-ish (sequential on 4 cores is fine) to `multiangle/segs/NNNN.mp4` with
   libx264 preset veryfast crf 18 (source is already lossy; 90 min must finish in a few hours
   on 4 ARM cores), then concat demuxer (`-c copy`) → `match.mp4` in project root, then
   mux the a0 audio track (aac 192k) in one pass. Progress = rendered seconds / total.
   Register the resulting file as the project video (write `pipeline/probe.json` the way
   pipeline.run stage_probe does, so the UI shows resolution).
7. **fuse** — `fuse.py`: events from each angle's `pipeline/candidates.json`, mapped to T.
   Greedy cluster within ±4 s. Fused event: t = confidence-weighted mean; type = highest
   TYPE_PRIO among members (goal>shot>chance>excitement>other); confidence = 
   `1 - Π(1 - c_i)` capped 0.95 then ×1.0 if ≥2 angles else ×0.85;
   `signals.angles = [i,...]`, `signals.angle_conf = {i: c}`,
   `cross_validation = "confirmed"` when ≥2 angles agree, `"pipeline_only"` when single-angle;
   `notes` = "cross-confirmed by angles A,B" / "single-angle (angle N: label)". Also
   `signals.disputed=true` when member types disagree (e.g. goal vs chance) — list types.
   t_start/t_end = 3 s pre / 5 s post like Option 1. Write `multiangle/fused_candidates.json`
   AND copy to `pipeline/candidates.json` (CandidatesFile schema, source="multiangle").
   Also copy a0's `pipeline/match_window.json` shifted by offset (0 for a0) and
   `features_1s.parquet` (needed by stats) with t shifted to T (t column) — clip to union.
8. **stats** — run `highlights.pipeline.stats.compute_stats` exactly as pipeline.run
   stage_stats but on the fused candidates; then add
   `stats["multiangle"] = {sync: sync.json summary, director: director.json ratios/n_cuts,
   confirmation: {cross: n, single: n, disputed: n}, score: score.json}`.
   `score.json`: `{"home": {"label": str, "goals": n}, "away": {...}, "basis":
   "confirmed goals with team set" }` — teams come from goal candidates' `signals.team`
   ("home"/"away") which are set by review (UI patch) or by the fuse stage when a goal
   candidate has `team` in notes; default unknown → counted under `"unassigned"`. The
   score is recomputed by the backend from CONFIRMED goal candidates at request time
   (`GET /api/projects/{id}/stats` merges), so human review drives the final score.

## Backend (FastAPI) additions
- `POST /api/projects/multiangle` JSON `{title, angles: [{url,label}], cookies_text?}` →
  creates ProjectStore with `source_info.kind="multiangle"`, `angles/aN/` dirs, spawns
  `highlights.multiangle.run` detached (new `pipeline.spawn_multiangle`, same status/pid
  handling but status path `multiangle/status.json`; make `read_status` pick the right
  file by kind). Cookies: same `_user_default_cookies` precedence as Option 1.
- Upload variant: `POST /api/projects/multiangle/upload` multipart with `files[]` +
  `labels[]` + `title`.
- `GET /api/projects/{id}/multiangle` → `{sync, director summary (no per-segment list),
  angles: [{label, url, duration, status}], score}`; `GET .../multiangle/director` → full
  segments; `GET .../multiangle/angle/{i}/video` streams angle file (for side-by-side check).
- `PUT /api/projects/{id}/multiangle/offsets` `{offsets:[...]}` → respawn from stage sync
  with `--offsets` (manual fallback).
- `POST .../pipeline/run` for multiangle projects must respawn multiangle.run.
- `CandidatePatch` gains `team: Literal["home","away"] | None`, `EventType` gains "tackle".
- Summary/list: include `mode: "single"|"multiangle"`, `n_angles`.

## Frontend
- Projects page: a mode toggle **"Single camera" | "Multi-angle (director cut)"** in New
  project. Multi-angle form: 2–4 rows of (YouTube URL, label) + Add angle; same cookie panel.
- Project card shows "Multi-angle · 3 angles" and stage message (download angle 2/3, sync,
  tracking angle 1/3, director, render 43%…).
- Review page for multiangle projects gains a **"Director cut"** tab: sync table (pairs,
  offset, PNR, confident badge, triangle residual), offsets entry form when
  `state == needs_input` (or "Override offsets"), rule ratios (ball vs cluster vs hold) as a
  bar, angle-share bar, cut timeline strip (colour per angle, hover shows rule/score), and
  the final score box (editable via goal candidates' team select). Candidate list shows a
  "cross-confirmed"/"single-angle"/"disputed" badge and, for goals, a Home/Away select.
- Everything else (video player, trims, render, stats charts) is unchanged: it plays
  `match.mp4` = director cut.

## Deploy
- Dockerfile: add `highlights/multiangle/requirements.txt` (ultralytics, torch CPU via
  `--extra-index-url https://download.pytorch.org/whl/cpu`) — must build on arm64.
  Pre-download yolov8n.pt into the image.
