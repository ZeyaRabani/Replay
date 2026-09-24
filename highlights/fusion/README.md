# highlights/fusion

Fuses the per-track 1 s features into a scored candidate list and renders
highlight clips. CPU-only, local execution.

## Pipeline

| script | what it does |
|---|---|
| `build_features.py` | joins motion/audio/spotting/tracking `features_1s.json` on t∈[0,5337], adds `whistle` overlap flag + `in_match` window [1050,4990], z-scores, NaN→0 → `outputs/features_1s.parquet` |
| `labels.py` | keeps goal/shot/chance events from the four reviewer JSONs inside the match window → `outputs/labels.json` |
| `score.py` | (a) rule score = mean of clipped robust-z of six near-goal signals, 3 s smoothing, 12 s NMS → top 60; (b) learned = LogisticRegression on [mean,max] windowed features, temporal 2-fold held-out CV → top 60; prints precision/recall@K + AUROC → `peaks_{rule,learned}.json` |
| `net_occupancy.py` | 2 fps YOLOv8s person detection on the near-goal crop; deep-in-net = bbox bottom-centre in the net polygon, x_center≥1070, y_bottom≥290; emits `net_occupancy_1s.json`, raw `net_detections.json`, and occupancy runs |
| `fuse.py` | canonical candidate list: deduped reviewer events (+ authored 2736 goal correction), nearest peaks attached, confidence formula, pipeline-only learned peaks, rejected tracking goals → `outputs/candidates.json` + `app/outputs/candidates.json` |
| `render.py` | renders `selected` candidates to `outputs/rendered/clips/*.mp4` (x264 CRF 28, w≤960, AAC 96k, faststart) + `reel.mp4` (concat, stream copy) + `manifest.json` |

## Dependencies

numpy, pandas, pyarrow (features_1s.parquet), scikit-learn (score.py),
ultralytics + opencv (net_occupancy.py), matplotlib (motion.py plot),
ffmpeg/ffprobe on PATH. All inference is CPU — YOLOv8s on 400×320 crops
at 2 fps takes ~4 min for the match window.

## Commands

```bash
python3 highlights/fusion/build_features.py
python3 highlights/fusion/labels.py
python3 highlights/fusion/score.py
python3 highlights/fusion/net_occupancy.py
python3 highlights/fusion/fuse.py
python3 highlights/fusion/render.py
python3 -m pytest highlights/fusion/test_fuse.py -q
```

## Limitations

- Fixed near-end ground-level camera: far-goal events are hard to
  verify; goal recall is unknown.
- SoccerNet spotting did not transfer (all-background output); retained
  features are weak priors only.
- Tracking has ±10 m error at the far end and no ball tracking; audio
  is speech-dominated (no crowd).
- Scores are candidate-retrieval aids — visual review is still the
  source of truth for event confirmation.
