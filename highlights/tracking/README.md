# highlights/tracking

Secondary signals for the hl3 highlight pipeline: player detection +
ByteTrack on a fixed broadcast-style phone camera, ORB stabilisation to a
reference frame, self-calibration to a rough pitch model, per-second cluster
features and candidate event extraction. These are weak, approximate signals
meant as secondary evidence for the parent pipeline — not precise tracking.

## Setup

```bash
pip install -r highlights/tracking/requirements.txt
# CPU build of torch:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# download the match video (720p is plenty)
yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
  --merge-output-format mp4 -o ~/match/match.mp4 https://youtu.be/5qj_nsQSzvQ
```

## Run (in order)

```bash
# 1. detect + track + stabilise  (~19 min on an 8-core CPU for 89 min of video)
python -m highlights.tracking.detect_track \
  --video ~/match/match.mp4 --out ~/match/out/tracks.npz \
  --fps 2 --model yolov8s.pt --imgsz 1280 --conf 0.3 --ref-time 600

# 2. self-calibration (pixels in the reference frame at ref_time)
python -m highlights.tracking.calibrate \
  --tracks ~/match/out/tracks.npz --out ~/match/out/calib.json \
  --near-goal 950 300 --far-goal 315 147 --length 65

# 3. per-second features
python -m highlights.tracking.features \
  --tracks ~/match/out/tracks.npz --calib ~/match/out/calib.json \
  --out ~/match/out/features_1s.json --duration 5337.154

# 4. candidate events
python -m highlights.tracking.events \
  --features ~/match/out/features_1s.json \
  --out ~/match/out/events.json --duration 5337.154

# 5. debug video for the top near-goal chance (or pass --t-start)
python -m highlights.tracking.render_debug \
  --video ~/match/match.mp4 --tracks ~/match/out/tracks.npz \
  --calib ~/match/out/calib.json --events ~/match/out/events.json \
  --features ~/match/out/features_1s.json --out-dir ~/match/out/debug
```

`detect_track.py` supports `--max-seconds` for benchmarking — always
benchmark on 120 s first:

```bash
python -m highlights.tracking.detect_track --video ~/match/match.mp4 \
  --out /tmp/bench.npz --fps 2 --model yolov8s.pt --imgsz 1280 \
  --conf 0.3 --ref-time 60 --max-seconds 120
```

## Detector choice

We use **ultralytics yolov8s.pt, COCO `person` class at full 1280 px width**
(`--imgsz 1280`). This is not a football-specific model — it was chosen for
far-end recall: far players in this footage are only 15-25 px tall and need
the full input resolution. Football-specific weights from Roboflow
Universe/HF (`uisikdag_players_best.pt`, `gianpaj_players_best.pt`; classes
ball/goalkeeper/player/referee) were compared but not used.

The `roboflow/sports` package (sports 0.1.0) is installed but unused: its
pitch keypoint model cannot work on this corner ground-level view (no visible
pitch keypoints — confirmed with the parent) and its player-detector weights
require a Roboflow account.

Raw-inference benchmarks on this machine (8-core CPU, no GPU):
yolov8n@960 ≈ 17 ms/frame, yolov8s@640 ≈ 25 ms, yolov8s@960 ≈ 45 ms.

## Measured runtime (8-core CPU, no GPU)

- `detect_track.py`: 1139 s for 10674 frames at 2 fps = **9.37 frames/s**
  end-to-end (decode + yolov8s@1280 + ByteTrack + ORB homography).
- `calibrate.py`, `features.py`, `events.py`: < 1 min each.
- `render_debug.py`: ~30 s for a 30 s clip.

## Stabilisation and calibration

**Stabilisation** (`detect_track.py`): every frame is warped into a reference
frame (t = 600 s) via an ORB homography — 3000 features, keypoints restricted
to the static upper band y < 260 px (houses/fence/trees/goals; the grass has
no features and players move), player boxes masked, BFMatcher Hamming + 0.75
ratio test + RANSAC 5 px. A frame needs ≥ 25 inliers and a non-degenerate H,
else the previous H is reused (`stab_ok=0`).

**Self-calibration** (`calibrate.py`): the pitch has no usable markings for a
keypoint model, so the horizon line is fitted from player pixel heights:
`h_px = k*(v - a - b*u)` — fitted a=77.87, b=0.1486, k=1.206, residual 2.9 px,
60892 inliers — giving camera height ≈ 1.45 m assuming 1.75 m players. Player
depth comes from box height `Z = f*1.75/h_px` with an **assumed** focal
length f = 950 px, because foot-row depth is unreliable at the far end.
World frame: origin at the near-goal centre pixel (950,300), x toward the
far goal (315,147). Pitch length was overridden to 65 m (foot-row estimate
gave 52 m; player heights suggest 60-80 m). **Treat all far-end positions as
±10 m.**

**Team colour**: coarse torso-HSV thresholds (lime / orange / grey) instead
of SigLIP+UMAP+KMeans — CPU budget, and the bibs are small. Counts are weak;
secondary evidence only.

## Feature columns (`features_1s.json`, one row per second)

- `t` — second since video start
- `n_players` — mean on-pitch player count
- `n_lime`, `n_orange` — mean per-team counts (coarse torso colour)
- `n_raw` — mean raw detections (incl. off-pitch/spectators)
- `stab_ok` — fraction of frames this second with a fresh homography
- `cx`, `cy` — mean player position (m), x along 0..L goal-to-goal
- `spread` — mean distance of players to the centroid (m)
- `mean_speed` — median per-track speed (m/s)
- `frac_near_third`, `frac_mid_third`, `frac_far_third` — player fraction per pitch third
- `n_near_box`, `n_far_box` — mean count inside each box (12 m deep, ±9 m)
- `n_centre` — mean count inside the centre circle (r=9 m)
- `rush_near_3s`, `rush_far_3s` — mean per-track x displacement over the last 3 s (m), toward each goal
- `restart_flag` — tight stationary cluster of ≥5 players (kickoff heuristic)
- `centre_cluster_flag` — ≥6 players, ≥4 in centre circle, low speed (kickoff heuristic)
- `keeper_x` — min player x (m; proxy for the near keeper's depth)

## Event rules (`events.py`)

- **chance**: the player cluster rushed ≥ `RUSH_M` (6 m) toward a goal in 3 s
  while that box was crowded (near: `BOX_N` = 3, far: `FAR_BOX_N` = 2) and
  ≥5 players were on the pitch. Far-end chances get a confidence haircut.
- **goal**: a chance followed within `GOAL_WINDOW_S` (10-75 s) by a
  centre-cluster/kickoff formation.
- **warm-up**: anything before 1020 s is a warm-up firing — emitted as
  `type=other`, `signals.kind=warmup` (mostly `near_box_crowd` during drills).
- Remaining runs are emitted as `type=other` restarts/kickoffs.

## Results on this match (5337 s)

81 events: 19 `chance` (14 near / 5 far), 8 `goal` candidates (confidence
≤ 0.6 — weak inference), 8 warm-up firings (all `near_box_crowd` during the
418-909 s drills), the rest `other` restarts. See `outputs/events.json`.

## Known limitations

- Far-end precision is ±10 m (assumed focal length, assumed 1.75 m players).
- COCO `person` picks up spectators and players on the neighbouring pitch.
- Team colours are weak (small bibs, dusk lighting at the end).
- Restart/kickoff detection is noisy — a 7v7 kickoff is not a tight cluster.
- Goal candidates are weak inference (attack + later centre formation).
- No ball usage anywhere in the pipeline.
- 669 frames over the match reused the previous homography (`stab_ok=0`,
  mostly low-texture / dusk frames at the end).

## Outputs

`outputs/` holds a copy of `meta.json`, `calib.json`, `features_1s.json`,
`events.json` and three debug frames for the top near-goal chance
(t = 2351-2381 s). `tracks.npz` is large and not committed.
