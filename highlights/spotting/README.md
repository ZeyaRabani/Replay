# hl3 / spotting — pretrained SoccerNet action spotting (E2E-Spot) on CPU

Runs the released **E2E-Spot** checkpoint (`soccer_challenge_rny008gsm_gru_rgb`, Hong et al. ECCV'22,
RegNetY-008 + Gated Shift Modules + bidirectional GRU, trained on SoccerNet-v2 train+val+test, 17 classes)
over a full match on CPU and emits per-frame class probabilities, NMS'd event candidates in the shared
schema, and a dense per-second feature file.

## Model choice

| candidate | weights obtainable? | CPU cost | decision |
|---|---|---|---|
| E2E-Spot (jhong93/spot) | yes — `github.com/jhong93/e2e-spot-models`, 50 MB, BSD-3 | 2 fps × 398×224, RegNetY-008: **~5–7× realtime on 8 cores** | **used** |
| COMEDIAN | weights not published in an easily runnable form; needs pre-extracted features + heavy ViT backbone | high | skipped |
| NetVLAD++ / CALF feature baselines | yes, but need Baidu/ResNet feature extraction first; worse transfer to amateur footage per "Beyond the Premier" | high | not needed |
| dshin13/autohighlight | ball/possession heuristics, no pretrained spotting weights | – | skipped |

The model is re-implemented in `e2e_model.py` (module names identical to the original so the checkpoint
loads with `strict=True`) because the upstream code needs `torch==1.11`, old `timm` and hard-codes
`torch.cuda.FloatTensor` inside the GSM module. No upstream code is imported.

## Setup

```bash
cd highlights/spotting
bash setup.sh            # torch-cpu, timm, tqdm, pillow + clones the 50 MB checkpoint into third_party/
sudo apt-get install -y ffmpeg   # if missing
```

## Run

```bash
python3 run_spotting.py /path/to/match.mp4 \
    --warmup_end 1020 \          # seconds; events before this are typed "other" with signals.kind="warmup"
    --work_dir ~/match/spotting_work \   # frame cache (JPEG @2fps 398x224 + uint8 .npy)
    --out_dir outputs
```

Options: `--threshold 0.1` (saved-list threshold), `--nms_s 10` (per-class NMS window), `--no_flip_tta`
(halves runtime), `--topk_per_class 10`, `--max_seconds 120` (throughput test), `--reuse_scores`
(re-run only the post-processing from `outputs/scores.npz`).

Review clips / contact sheets for the top-N candidates:

```bash
python3 extract_clips.py /path/to/match.mp4 outputs/events.json -n 10 -o ~/match/review
python3 extract_clips.py /path/to/match.mp4 outputs/events_topk.json -n 40 --match_only -o ~/match/review_topk
```

## Outputs (`outputs/`)

* `events.json` — shared schema; per-class NMS peaks with `confidence >= 0.1`. `signals = {class, score, top3, background, kind}`
  where `kind` is `warmup` (t < `--warmup_end`) or `match`. Class → type: Goal→`goal`, Shots on/off target→`shot`,
  Corner/Free-kick/Penalty→`chance`, **Ball out of play→`excitement`** (see findings), others→`other`.
* `events_topk.json` — top-10 NMS peaks per highlight-relevant class *regardless of threshold* (confidences are raw
  and mostly tiny). Provided because the domain gap pushes the real shot/goal classes far below 0.1.
* `features_1s.json` — `{"source":"spotting","step_s":1.0,"columns":["t","p_goal",...,"p_foreground_max"],"rows":[...]}`,
  per-second max of each class probability.
* `scores.npz` — raw averaged softmax, `probs[N_frames, 18]` at 2 fps (`t`, `classes` arrays included).
* `summary.json` — runtime, counts at 0.1/0.3/0.5, per-class max/mean, and **per-class score baselines**
  (p50/p90/p99/p99.9/max, `max_over_p99`) computed separately over the warm-up period and the match period.

## Measured runtime (8-core CPU, no GPU, 31 GB RAM)

Video: 1280×576 @30 fps, 5337 s (89 min).

| step | time |
|---|---|
| ffmpeg frame extraction (2 fps, 398×224 JPEG) | 118 s |
| JPEG → uint8 tensor cache (10 674 frames, 2.9 GB in RAM) | 8 s |
| inference, 100-frame clips, 50 % overlap, h-flip TTA, batch 2 | **989 s** (10.8 frames/s, 5.4× realtime) |
| post-processing | < 2 s |
| **total** | **~18.5 min** (≈9.5 min without flip TTA) |

Peak RAM ≈ 5 GB.

## Results on this match — honest assessment

**The pretrained model essentially does not transfer to this footage.** The argmax is `background` on
100 % of frames.

Per-class baselines over the match period (t ≥ 1020 s), from `summary.json`:

| class | p50 | p99 | max | max/p99 |
|---|---|---|---|---|
| Ball out of play | 0.0090 | 0.055 | **0.482** | 8.8 |
| Clearance | 0.0013 | 0.015 | 0.049 | 3.4 |
| Corner | 0.0009 | 0.016 | 0.045 | 2.9 |
| Throw-in | 0.0016 | 0.015 | 0.044 | 2.9 |
| Indirect free-kick | 0.0006 | 0.006 | 0.042 | 6.8 |
| Foul | 0.0008 | 0.007 | 0.037 | 5.3 |
| **Shots on target** | 0.0005 | 0.002 | **0.010** | 4.6 |
| **Shots off target** | 0.0004 | 0.002 | **0.009** | 4.8 |
| **Goal** | 0.0002 | 0.0007 | **0.002** | 2.5 |

* Events with confidence ≥ 0.1: **12** (10 in match period, 2 in warm-up); ≥ 0.3: **1**; ≥ 0.5: **0**.
* All 12 are `Ball out of play` except one `Clearance` (0.14, warm-up). **Goal, Shots on target and Shots off
  target never exceed 0.01** — these channels are noise-level and should not be used by the parent.
  The Goal channel's peak is only 2.5× its own p99; the shot channels ~4.7×.
* Warm-up period (0–1020 s, shooting drills into the near goal): the model does *not* light up either
  (Goal max 0.004, shots max 0.006), which confirms it is not reacting to the near-goal shooting action as a
  broadcast-trained model would.

**One useful signal did emerge.** I extracted 6 s clips around the top-10 candidates and inspected frames
(`extract_clips.py`). Every one of the top `Ball out of play` peaks in the match period
(1486 s p=0.48, 1570 s p=0.29, 2488 s p=0.20, 3504 s p=0.17, 1352 s p=0.17, 3827 s, 2639 s, 1242 s, 1547 s, 2801 s)
coincides with an **attack on the near goal**: the keeper (lime bib) comes off his line / crouches / dives, a
cluster of orange+lime players converges on the near penalty arc, and in 1352 s the ball is visibly in the air
above the goal. None of them looked like an actual ball-out-of-play moment. Presumably the model has learned
"players bunched around a goal + ball leaving the visible area" as a correlate of that class. So on this camera
the `Ball out of play` channel behaves as a weak *near-goal-action detector*, which is why `events.json` maps it
to type `excitement`. It is weak (10 peaks ≥ 0.1 across 72 min of play, and the p99 is 0.055 so lower peaks are
indistinguishable from noise), and it says nothing about the far goal.

The `Shots on/off target` top-k peaks (p≈0.005–0.01, `events_topk.json`) partially overlap the same moments
(1484 s, 1351 s, 1248 s, 2677 s look like near-goal attacks; 4501 s is the keeper distributing the ball) but
their scores are within ~5× of their own p99, so I would treat them as tie-breakers at most.

**Recommendation for fusion:** use `features_1s.json` column `p_ball_out_of_play` (or the `excitement`
events) as one weak feature for near-goal action; ignore `p_goal`, `p_shots_*`; do not threshold this track
on its own.

## Known limitations

* Domain gap: model trained on broadcast (moving, zoomed main camera, 16:9). Here: fixed wide phone camera
  at ground level behind one corner, 2.22:1 aspect ratio, squashed to 398×224 (the model's expected input;
  I resize without cropping so the far goal stays in frame, which distorts geometry further).
* Runs at the model's native 2 fps, so temporal precision is ±0.5 s at best.
* Far-goal events are ~20 px tall in the source; after downscaling to 224p they are a handful of pixels and
  the model has no chance of seeing them.
* Only one checkpoint evaluated (the strongest released one). The RegNetY-002 variant was not tried; it would
  be ~2× faster but is unlikely to transfer better.
* `third_party/` (checkpoint) is git-ignored; `setup.sh` fetches it.
