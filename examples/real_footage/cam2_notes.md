# Camera 2 calibration constraints - notes

Files: `cam2_constraints.json` (full calibration file, camera "2" only), `cam2_check.jpg` (CLI check-calib render),
`cam2_traces.jpg` / `cam2_traces_far_crop.jpg` (all traced markings on a 4-frame temporal median of the original clip),
`cam2_median_grid_far.jpg` (median, far-field crop with pixel grid), `cam2_context_cam1_same_arc.jpg` (cam1 median, the same arc).

All pixel coordinates are in the ORIGINAL 2316x1080 frame of `Screen_Recording_20260912_143219_YouTube.mp4`.
Traces were drawn by hand on frames at t = 0.5, 3, 8, 14, 20 s, verified against a temporal median (players removed),
then refined with `pitchworld.calibrate.snap_polyline_to_mask` (white mask for L1, blue mask for the arc). The snapped
points agree with the hand trace to < 1 px; a straight-line fit to L1 has 0.47 px RMS (`v = 0.3478 u + 207.6`).

## What is visible in cam2 (stable markings, players excluded)

| id | pixels (approx) | description |
|----|-----------------|-------------|
| L1 | (697,450) -> (2091,933) | long, bright white straight line, nearest the camera; tarmac/bags on the camera side. Continues off-frame at both ends. |
| blue arc | (963,536) -> (1121,493) | blue curved marking; its lower end meets L1 at (963,536); its upper end reaches L2a at about (1150,465). Bulges toward the camera/right. |
| L2a | (550,347) -> (1030,441) | fainter white straight line, starts at the parked goal, ends at corner K=(1057,449). Parallel to L1 (vanishing point (212,281) shared with L1 and the fence base). |
| L2b | (1090,451) -> (1450,464) | fainter white line from corner K toward the far fence. Only exists on the right of K; L2a only on the left of K (so K is a corner, not a crossing). |
| fence base | (760,359) -> (1900,532) | far edge of the turf, parallel to L1. |
| blue line | (1795,535) -> (2110,735) | blue straight-ish line at the far right, identity unknown. |
| goals | parked portable goal, near post ground contact C=(652,437) on L1; white goal at K (a goalkeeper stands in it); small orange goal ~(1480,500); white goal at the fence ~(1745,520) | none of these can be tied to a named goal of a single modelled pitch with confidence. |

The same arc / L1 / L2a / corner-K geometry is visible in cam1 (see `cam2_context_cam1_same_arc.jpg`), so cam1 and cam2
observe the same physical features.

## Pitch layout / dimensions - what the evidence supports

* L1 is the shared reference line of all three cameras. The blue arc is centred ON L1: a pose fit with the arc centre
  constrained to L1 (`A_D`-style) fits the pixels at 2.3-3.0 px RMS with a 2.0-2.2 m camera height, whereas the
  alternative "semicircle spanning L1 -> L2a" needs a 20 m camera and 3.5-16 px RMS. So "L1 = goal line, arc = D" is
  the geometry the pixels prefer; this matches the identity already used in the repo (`goal_line_A` + `A_D`).
* The D radius is NOT 6 m if the people are 1.75 m: `metres_per_unit_from_people` on the raw cam2 tracks
  (1170 boxes, conf > 0.5) gives scale 1.92 with d_radius = 6 m. Re-fitting with d_radius = 10 / 12 / 14 m gives
  scale 1.14 / 0.95 / 0.81 and camera height 3.7 / 4.4 / 5.1 m (pixel RMS unchanged, 2.3 px). Best estimate of the
  arc radius from people scale: ~11-12 m (with h ~ 4.4 m; the camera clearly is above head height - the horizon from
  the L1/L2a/fence vanishing point is at v ~ 281, well above the roofs of the 2-storey houses behind the fence).
  A 12 m radius arc is not a small-sided D; it is more like a penalty arc / centre circle of a larger pitch, or a
  training marking. I did NOT change the shared `pitch` block (kept 50 x 30 m, goal 3.66 m, d_radius 6 m) because
  other cameras' constraint files depend on it, but the coordinator should consider d_radius ~ 12 m (or a joint fit
  of d_radius across cameras): with 6 m every projected metre is ~1.9x too small and cameras will disagree.
* L2a is parallel to L1 (direction is certain; the offset is not). Under the D-on-L1 hypothesis the arc's far end
  reaches L2a, so L2a is at |x| ~ d_radius from L1 if it is a line of the same pitch - but if L1 were a goal line,
  a parallel line 6-12 m inside it with a goal standing in its corner K is hard to explain with the small-sided
  template. L2a/L2b/K may belong to an adjacent training grid. Hence L2a is used only as a `parallels` constraint.
* The fence base is parallel to L1 (`parallels`, direction only).
* L2b, corner K, the parked goal contact C and the far-right blue line are recorded under
  `_unverified_traces_cam2` in the JSON and are NOT used as constraints (identity unknown).

## Constraints used (cameras["2"])

* `points`: `A_D_S` = the arc's endpoint on L1, pixel (962.7, 535.5). Under the fitted pose this pixel projects to
  (0.21, 9.04) m vs template (0, 9) m, so it is consistent with the line+arc constraints.
* `lines`: `goal_line_A` = L1, 12 snapped points.
* `arcs`: `A_D` = blue arc, 9 snapped points.
* `parallels`: L2a (9 pts) and fence base (5 pts), both direction `goal_line_A`.

dof = 2 + 2 + 5 + 1 + 1 = 11 (>= 9, so the "thin constraints" warning is gone).

## Validation (pitch 50x30, d_radius 6, `manual_calibrate` + `check-calib`)

```
pose: cam @ (-5.3, 1.5) m, height 2.05 m, yaw 41.5 deg, pitch 11.8 deg, roll 5.2 deg, f = 878 px
reprojection: 3.0 px RMS (0.73 m RMS on the ground, dominated by the far end of L1)
people scale (metres_per_unit_from_people, cam 2, 1170 boxes): 1.92  <- pitch geometry ~1.9x too small, see above
confidence reported by manual_calibrate: 0.27
ruff check pitchworld: clean (no source changes)
```

Height 2.05 m looks "plausible" only because the 6 m radius shrinks the world; with people-consistent scale the camera is
~4.4 m up, which matches the horizon position (camera above the goal crossbars and roughly level with tree tops).

## Remaining uncertainty / things the coordinator should know

1. Which goal L1 belongs to (A vs B) and which touchline is S vs N cannot be determined from cam2 alone; I kept the
   repo's convention (`goal_line_A`, `A_D`, arc end on the S side) so that cam0/cam1 files stay compatible. If the
   coordinator's cam0/cam1 sessions pick the mirrored identity (`A_D_N`), flip `A_D_S` -> `A_D_N` here.
2. The pitch template (50x30, d 6 m) is very likely wrong for this arc (see people scale). The pixel residual is
   insensitive to d_radius (only the metric scale changes), so cross-camera agreement in metres requires the correct
   radius. Recommend a joint estimate of `d_radius` from `metres_per_unit_from_people` over all three cameras
   (cam2 alone says ~11.5 m).
3. The video content occupies x in [0, ~2113] of the 2316-wide frame (black bar on the right, see `cam2_check.jpg`), so
   the true principal point is at u ~ 1056, not 1158. `posefit` assumes the image centre; this ~100 px offset biases the
   pose (mostly yaw/roll and height) for all three cameras. Consider cropping the clips to the content or adding a
   principal-point offset in `posefit` (not changed here: Python source is out of scope for this task).
4. No goal-post ground contacts were used: the parked goal on L1 is not a goal of the modelled pitch, and the goal at
   corner K / the far goals cannot be tied to a template goal.
