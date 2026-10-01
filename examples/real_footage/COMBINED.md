# Combined real-footage calibration (cam0 + cam1 + cam2)

Files:

- `calib_combined.json` – pitch block + `cameras["0"|"1"|"2"]` manual constraints, merged from
  `cam0_constraints.json` (PR #7), `cam1_constraints.json` (PR #10), `cam2_constraints.json` (PR #19).
- `pitch_combined.json` – the reconciled pitch (`--pitch`).
- `combined_calib_cam{0,1,2}.jpg`, `combined_sync_check.jpg`, `combined_pitch_map_frames.jpg` – debug output of the run below.

Run (cached sync / synced clips / raw_tracks.json reused, no GPU):

```
python -m pitchworld.cli run <142936.mp4> <143058.mp4> <143219.mp4> --out out/ --reuse \
    --calib examples/real_footage/calib_combined.json --pitch examples/real_footage/pitch_combined.json [--joint-refine]
```

## 1. Pitch geometry reconciliation

| source | length | width | goal width | D radius |
|---|---|---|---|---|
| cam0 notes | ~60 (±8, from far-goal pixel width) | 40 (placeholder) | 3.66 | 7.32 (post ruler 7.3, people scale 6.8) |
| cam1 notes | 50 (preset) | 30 (preset) | 3.66 | 6.0 (people scale crosses 1.0 at ~6.3) |
| cam2 notes | 50 (preset) | 30 (preset) | 3.66 (unverified) | 6 kept; people scale says 11–12 (with a 4.4 m camera) |
| pitch_geometry.md (all 3 clips) | 37 (32–45) | 30 (25–40) | 4.88 (4.4–5.3) | 6 (4.5–8) |

Chosen (`pitch_combined.json`): **length 37 m, width 30 m, goal_width 3.66 m, d_radius 6.6 m**.

- **Layout**: the geometry analysis is the only source that used all three clips together and it matches what the
  median frames show: cam0 films one portable goal (backdrop: blue building + warehouse), cam1 and cam2 film the
  *other* goal (backdrop: houses, same bag pile behind the goal). The goals stand on two parallel white lines of the
  big 3G pitch with a blue D in front of each; cam1's note "all three cameras see the same goal" is contradicted by
  the backdrops and was not followed.
- **Length 37 m**: taken from pitch_geometry.md. cam0's ~60 m is derived from the far-goal pixel width at a
  fitted focal length and is very sensitive to the fitted horizon (I tried projecting the far goal posts through the
  fitted poses: with a 1.1–1.5 m camera the far goal line sits ~30 px below the horizon and the implied length swings
  from 60 to 300 m for a 5 px change, so that estimate is not usable). Scanning L in {33,37,41,45,50,55} changes the
  cross-camera medians only through which player pairs fall inside the 3 m match gate, not systematically, so L is
  essentially unconstrained by the current constraints; 37 is kept as the best-argued value.
- **Width 30 m**: no touchline of the cross-pitch is visible in any camera; 30 is the common placeholder. cam0's
  explicit post coordinates were re-expressed for width 30 (y = 15 − 2.76 and 15 + 0.89; the goal is ~1 m S of the D
  centre, as cam0 measured).
- **goal_width 3.66**: cam0 measured the two post ground contacts at 3.65 units apart at r = 7.32, i.e. the goal
  width is tied to the chosen D radius (r = 6.6 ⇒ ~3.3 m). The geometry session's 4.88 m comes from post *height*
  ratios and would need r ≈ 9.8 to be consistent with cam0's post ruler, which both cam0 and cam1 people-scale
  checks exclude. 3.66 (standard portable goal) is kept; it only affects the `*_post_*` landmarks used by cam0/cam1.
- **d_radius 6.6**: compromise between cam0 (7.32; people scale says 6.8), cam1 (6.0, people scale 6.3) and the
  geometry estimate (6, range 4.5–8). At 6.6 cam0 and cam1 fit with 0.24 m ground rms each; at 6.0 cam0 degrades to
  0.67 m and at 7.3 cam1 degrades to 0.29 m / cam2 to 0.88 m. cam2's "11–12 m" is rejected: it requires a 4.4 m camera
  for a hand-held phone and a D radius no other camera supports; cam2's people-scale is more likely explained by its
  ~0.8 m ground residual / wrong principal point (see its notes).

## 2. Landmark renaming

cam1 and cam2 authored their constraints against goal **A**; since they film the goal opposite cam0, every `A_*`
landmark was renamed to `B_*` (`goal_line_A → goal_line_B`, `A_D → B_D`, `A_post_N → B_post_S`, `A_D_N → B_D_S`,
`A_D_S → B_D_N`). Point S/N flips because the B end is viewed from the opposite direction (cam1/2 look towards −x, so
"left" in the image is the +y side); `touch_S`/`goal_line_*` *parallel* directions are unchanged (a direction has no
side). cam0 keeps `A_*`. No constraint was dropped.

## 3. Results (`tracking.json["quality"]`)

### Plain run (`--reuse`, no joint refine)

| | cam0 | cam1 | cam2 |
|---|---|---|---|
| height (m) | 1.25 | 1.28 | 2.25 |
| position (x, y) | (−3.4, 6.8) | (40.9, 5.9) | (42.8, 29.8) |
| focal (px) | 1065 | 1128 | 878 |
| reprojection | 3.3 px / 0.24 m | 2.5 px / 0.24 m | 3.0 px / 0.80 m |
| confidence | 0.76 | 0.76 | 0.20 |

`cross_camera_disagreement_m`: **cam0-1 2.43, cam0-2 1.79, cam1-2 2.37** → `calibration_low_confidence = true`.
Only 7.6 % of fused players are seen by ≥2 cameras.

### `--joint-refine`

Refinement (players as shared ground points, D radius free) converges to d_radius 7.41 and reports
cam0-1 1.42, cam0-2 1.24, cam1-2 0.09 m — but at the cost of physically implausible poses: cam0 f = 2245 px, h = 2.1 m,
26 px reprojection; cam2 h = 6.8 m, f = 4512 px, 43 px / 3.2 m reprojection on the traced lines. The lower
disagreement is obtained by moving the cameras off the painted markings, so it is **not** trusted and the flag stays
`calibration_low_confidence = true`. The plain-run calibration is the one to use.

## 4. Honest assessment / most likely cause

The cameras still disagree by ~1.8–2.4 m (median), so the combined file does not yet define one shared pitch frame.
Most likely causes, in order:

1. **Depth is unconstrained.** Every camera is calibrated from a single goal line + D arc, i.e. markings within
   ~8 m of the camera. A hand-held phone at 1.1–1.5 m has a horizon ~30 px above the far goal; a 1–2 px error on the
   traced arc changes the pitch tilt enough to move a player 20 m away by several metres along the viewing direction.
   The three fits are individually excellent (0.24 m rms) but each one is only reliable near its own goal — exactly
   where the other cameras are far away, so the pairwise comparison samples the worst region of each.
2. **Pitch length is a free guess** (37 m). cam0 and cam1/2 are anchored to opposite goal lines; any error in L is
   added directly to the cam0 vs cam1/2 disagreement along x. The scan over L did not find a clear minimum, which is
   consistent with (1) dominating.
3. **cam2** (0.80 m rms, conf 0.20, wrong principal point because of the 2113/2316 px black bar) is the weakest
   camera; cam0-2 is nevertheless the best pair (1.79 m), which again points at depth/L rather than a single bad camera.
4. Adding my own far-goal post pixels (goal B in cam0, goal A in cam1/cam2) as extra points made every fit *worse*
   (cam2 exploded to a 60 m camera), i.e. under the current model the near markings and the far goal cannot both be
   satisfied — consistent with an unmodelled focal/principal-point error or with the goals not being where the
   template puts them (the portable goals are not centred on the D, see cam0).

What would fix it: a marking that ties the two ends together (the big pitch's halfway line runs through both goal
centres per pitch_geometry.md — trace it in all three cameras as a `parallel` to `touch_S` and, in cam0/cam1, its
intersection with the goal line as an explicit point), plus fixing the principal point for the letterboxed frames.
