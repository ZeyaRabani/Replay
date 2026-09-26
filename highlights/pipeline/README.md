# highlights.pipeline

End-to-end highlight pipeline: one command turns a full-match video (local
file or YouTube URL) into candidates.json + stats.json using **audio and
motion signals only** — no tracking, spotting, PANNs, or human labels.

## Usage

```bash
source ~/.nvm/nvm.sh          # node >= 18 helps yt-dlp solve JS challenges
pip install -r highlights/pipeline/requirements.txt

python -m highlights.pipeline.run --project-dir /path/to/proj --video match.mp4
python -m highlights.pipeline.run --project-dir /path/to/proj \
    --youtube-url https://www.youtube.com/watch?v=XXXX --cookies cookies.txt
```

Flags: `--stages download,probe,...` (subset, always run in pipeline order),
`--force` (re-run even if outputs exist), `--cookies FILE` (or env
`HL_YT_COOKIES`) for YouTube bot-check walls.

Video resolution order: `--video` is linked/copied to `<project>/match.<ext>`;
otherwise an existing `match.<ext>` is reused; otherwise a single video file
in `<project>/source/` is adopted; otherwise `--youtube-url` is downloaded.

## Stages

| stage | weight | what it does | outputs |
|---|---|---|---|
| download | 0.35 | yt-dlp `bestvideo*+bestaudio`, remux to mp4 | `match.mp4` (or `.mkv`) |
| probe | 0.02 | ffprobe duration/size/fps | `pipeline/probe.json` |
| audio | 0.10 | 16 kHz wav + per-second features + whistle segments | `pipeline/audio/*` |
| motion | 0.35 | frame-diff features at 5 fps 320x144 (subprocess) | `pipeline/motion/features_1s.json` |
| features | 0.05 | join to 1-s frame, detect match window, z-score | `features_1s.parquet`, `match_window.json` |
| score | 0.05 | trained LR model -> learned prob (3-s smooth) + rule signal | `scores.parquet` |
| candidates | 0.03 | top-60 peaks, 12 s NMS | `candidates.json` |
| stats | 0.05 | Contract 5 stats via `compute_stats` | `stats.json` |

A stage is skipped when all its outputs already exist (unless `--force`).
Status is written atomically to `pipeline/status.json` at least every 2 s
(heartbeat thread) plus on every stage transition; logs go to
`pipeline/log.txt` and stdout.

## status.json schema

`state` queued/running/done/failed, `stage`, `progress` (0-1 overall,
weighted as above), `stage_progress`, `message`, `error`, `started_at`,
`updated_at`, `finished_at`, `pid`, `video_path`, `video` (probe dict),
`download` ({format, resolution, filesize}).

On SIGTERM the run marks `state=failed, error="cancelled"` and exits 1.

## Model

`train_model.py` trains on the committed fusion features of the demo match
(`5qj_nsQSzvQ`): rolling mean+max over ±3 s of the 11 audio/motion columns +
whistle (24 features), positives within ±4 s of a labelled event, negatives
in-match and ≥15 s away, StandardScaler + LogisticRegression(balanced,
C=0.3). Temporal 2-fold held-out AUROC (split at t=3020): **0.802** —
respectable for audio+motion only, well below the full fusion model that
also sees tracking/spotting columns. Saved as
`models/audio_motion_lr.joblib` (+ `.json` metadata).

## Candidates

Peaks: greedy argmax of the smoothed learned probability over in-match
seconds, 12 s NMS, top 60. Type: `shot` when the robust z of
`motion_goal_roi` at the peak exceeds 2, else `chance` — the pipeline cannot
distinguish a goal from a strong attack without visual review, so every
event is `cross_validation: "pipeline_only"`, `status: "pending"`, and
confidence is capped at 0.9.

## Match window

`detect_match_window` uses referee whistles (first whistle in the first 40 %
of the video → kickoff, last in the last 40 % → full time) plus the audio
`half_time_gap` heuristic for the halves. If the window would cover <30 %
of the video, or the video is <10 min, it falls back to the full video with
a warning recorded in `match_window.json` / `stats.json`.

## Known limitations

- **motion.py ROI is camera-specific**: the goal ROI / far region constants
  are hard-coded for the demo camera (1280x576). Input of any size is scaled
  to 320x144 so the stage runs, but `motion_goal_roi` only means "near goal"
  for that camera setup.
- **YouTube bot check**: YouTube may answer automated downloads with
  "Sign in to confirm you're not a bot". The pipeline reports this as an
  actionable error (upload the file or supply `--cookies`/`HL_YT_COOKIES`);
  it never substitutes a different video.
- **JS runtime**: recent yt-dlp solves YouTube's JS challenges via an
  external runtime. `node` (nvm: `source ~/.nvm/nvm.sh`, v24) is enabled
  automatically when on PATH; without it some videos may fail.
- The learned model is trained on a single grassroots match; treat scores
  on other footage as a ranking signal, not calibrated probabilities.

## Tests

```bash
python -m pytest highlights/pipeline -q   # fast, no network
```
