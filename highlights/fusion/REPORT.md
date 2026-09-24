# Highlight fusion — final report

## Headline result

- **1 goal directly observed**: video time **00:45:36.000 (t = 2736 s)**,
  near goal. Ball visible in the near net; keeper enters the net to
  retrieve it (~2738–2742). Independent corroboration: the
  `net_occupancy` deep-in-net detector flags **2738.5–2741.5**.
- **No other goal was visually confirmed.** Far-goal events remain hard
  to verify from this ground-level near-end camera, and **one or more
  far goals cannot be ruled out**.

## Coverage

- Four independent coarse-to-fine visual passes (`review1`–`review4`)
  covered the full video at varying tile densities.
- Actual match play: approximately **1105–4986 s**. Warmup before
  ~1020 s and the file tail after full time were excluded from the
  match window [1050, 4990].

## Pipeline honesty notes

- **SoccerNet E2E-Spot failed domain transfer** on this footage:
  background argmax 100%, Goal max ≈ 0.002, Shots max ≈ 0.01, zero
  events ≥ 0.5. Only its ball-out-of-play score was retained, as a weak
  feature.
- **Tracking is secondary**: YOLOv8s person detection + ByteTrack-ish
  tracking at 2 fps, manual/self-calibrated geometry, roughly ±10 m
  error at the far end, no ball tracking.
- **Audio**: effectively no crowd; speech dominates. Used only as a
  secondary signal (whistle flags, RMS z-scores).

## Pipeline metrics (held-out temporal CV vs the reviewer event list)

These are **candidate-retrieval** metrics (does a peak land near a
reviewed event), **not** event-classification quality — visual review
remains necessary.

- Learned fusion held-out **AUROC 0.780**.
- Learned @ K=40: precision .400, recall .410 (high-confidence reviewer
  recall .462).
- Rule @ K=40: precision .475, recall .487 (high-confidence reviewer
  recall .462).
- Self-assessed goal precision: 1/1 selected goals confirmed = 100%
  among selected goals. **Goal recall is unknown** — far-goal
  visibility prevents proving that 1 is exhaustive; we do not claim
  100% recall.

## Candidates

The final `candidates.json` contains **47 candidates**:
3 goals (1 confirmed, 2 rejected tracking-only), 28 shots, 16 chances;
by cross-validation: 32 confirmed, 7 visual_only, 6 pipeline_only,
2 rejected. **27 events are selected** and rendered to clips.

## Output files

- `highlights/fusion/outputs/features_1s.parquet` — fused 1 s feature
  table (motion, audio, spotting, tracking, whistle flag, in_match).
- `highlights/fusion/outputs/labels.json` — deduplicated reviewer labels.
- `highlights/fusion/outputs/peaks_{rule,learned}.json` — top-60 peaks.
- `highlights/fusion/outputs/net_occupancy_1s.json` + `net_detections.json`
  — deep-in-net occupancy timeseries + raw YOLO detections.
- `highlights/fusion/outputs/candidates.json` — canonical candidate list
  (mirrored to `highlights/app/outputs/candidates.json` for the app).
- `highlights/fusion/outputs/rendered/` — the **earlier preview-quality
  deliverable** (CRF28, max width 960, AAC 96k). Superceded for quality by:
- `highlights/fusion/outputs/rendered_high/clips/*.mp4` — 27 selected
  clips at **source resolution 1280×576** (libx264 CRF14 preset slow,
  yuv420p, AAC 256k, faststart). Visually transparent; the honest maximum
  resolution — the source is 1280×576 so it cannot be increased.
- `rendered_high/reel.mp4` — concatenated high-quality reel (156 MB;
  exceeds GitHub's 100 MB file limit so it is not committed — see local
  copy at `/home/ubuntu/match/reel_high.mp4`).
- `rendered_high/manifest.json` — `{quality, codec_settings, clips}` map.
- `--quality max` exists for lossless archival (CRF0 veryslow + FLAC in
  MKV, `outputs/rendered_max`); a 10 s proof clip of the confirmed goal
  is at `/home/ubuntu/match/goal_2736_lossless.mkv` (42 MB for 10 s —
  a full lossless reel would be many GB).

## Rerun

```bash
python3 highlights/motion/motion.py --video match.mp4 \
    --out-json highlights/motion/outputs/features_1s.json
python3 highlights/fusion/build_features.py
python3 highlights/fusion/labels.py
python3 highlights/fusion/score.py
python3 highlights/fusion/net_occupancy.py --video /home/ubuntu/match/match.mp4
python3 highlights/fusion/fuse.py
python3 highlights/fusion/render.py            # high quality (default)
python3 highlights/fusion/render.py --quality max   # lossless archival MKV
```
