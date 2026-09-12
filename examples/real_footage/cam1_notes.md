# Camera 1 calibration notes (real footage)

Frame size 2316x1080, 30 fps, ~24 s. All pixel coordinates below are in the synced `cam1.mp4`
(same framing as the original screen recording; the phone is static, markings do not move between
t = 1, 5, 10, 15, 20 s).

Files: `cam1_constraints.json` (calibration file, cameras {"1"}), `cam1_check.jpg`
(`check-calib` overlay), `cam1_traces.jpg` (snapped traces on the t = 1 s frame), `cam1_goal.jpg`
(zoom on the portable goal used to pick the post ground contact).

## What the venue is

Looking at all three t = 1 s frames together (full0/1/2.jpg):

* It is one full-size 3G pitch (white lines, floodlights, fence, pavement along the near edge; full-size
  11-a-side goals visible in the far distance in cam0 and cam2). The game is being played *across*
  it on a smaller area marked in **blue**: a blue "D" is painted in front of the portable goal and
  cam2 shows a second blue D further along the same white line, i.e. the blue markings are cross-pitches
  whose goal lines coincide with a white line of the big pitch. All three cameras see the **same**
  portable wheeled goal and the **same** blue D:
  * cam0 stands behind/left of the goal (D to the right of the goal, line running away to the right),
  * cam1 stands off-pitch on the pavement beside the goal, looking along the line (goal at the right
    edge of frame, D in the foreground left of it, line running away to the left),
  * cam2 stands off-pitch on the other side (goal at left, D to its right, bags on the pavement).
* World frame (pitchworld/pitch.py): the white line the portable goal stands on is `goal_line_A`
  (x = 0), goal A centred on the blue D at y = width/2 = 15, the pitch extends into +x. The blue D is
  `A_D`. In cam1 the camera is on the +y (north) side of the goal, so the near post is `A_post_N` and the
  visible end of the D on the goal line is `A_D_N`.

## Pitch dimensions

Nothing in the footage fixes the length or width of the blue cross-pitch (the far blue goal line is not
visible in any camera), so I keep the shared 50 x 30 m preset (`examples/pitch_small_sided_50x30.json`).
Only the **scale** (D radius) and the near goal line matter for cam1's homography near the play.

Scale evidence, with the fit run at several `d_radius` values (people-scale = 1.75 m / median implied
height of 7264 YOLO boxes from raw_tracks.json cam 1; this sweep was run with an earlier constraint set
that still included the rejected far line, hence the higher rms - the trend is what matters):

| d_radius | rms resid | cam height | people-scale |
|---|---|---|---|
| 5.5 | 0.39 m | 1.03 m | 1.23 |
| **6.0** | 0.40 m | 1.12 m | 1.09 |
| 6.5 | 0.42 m | 1.22 m | 0.97 |
| 7.0 | 0.44 m | 1.31 m | 0.87 |

People-scale crosses 1.0 at d ~ 6.3 m; 6.0 m (the value shared with the other cameras / the preset) is
within ~10 % and gives a plausible hand-held phone height (~1.1-1.2 m, chest/waist level). I keep
d_radius = 6.0 so all cameras share one model; if the coordinator wants a single global rescale, ~6.3 m
is the cam1 estimate. Goal width 3.66 m is assumed (standard portable mini-goal); the two rear wheels of
the goal project ~3.1-3.7 m apart depending on the fit, consistent with it.

## Constraints used (all in `cam1_constraints.json`)

* `lines: goal_line_A` - 48 points snapped onto the white mask from x = 5 to the near post (x ~ 1690),
  plus 3 points where the line re-emerges right of the goal (x > 2180). Straight-line rms 0.36 px.
* `arcs: A_D` - 29 points snapped onto the blue mask, from where the D meets the goal line (1085, 546)
  round to (1130, 375). The rest of the D is hidden behind the goalkeeper / goal.
* `points: A_D_N` (1088, 556) - intersection of the D trace with the goal line.
* `points: A_post_N` (1716, 568) - ground contact of the near post (see cam1_goal.jpg). The portable
  goal was placed by hand so it need not be exactly centred on the D; the fit puts it at y = 16.5
  vs 16.83 in the model (0.3 m), which is the dominant residual. Kept because it is the only physical
  point shared with cam0/cam2 (both see the same post) and it anchors the goal-mouth region.
* `parallels: goal_line_A` - white line "P", a long white line ~6.4 m in front of the goal line and
  parallel to it (shares its vanishing point at ~(2850, 620)). 40 points (far part + part near the
  keeper). I could not identify what it is on the big pitch (a 6-yard line would be at 5.5 m), so it is a
  direction-only constraint.
* `parallels: touch_S` - white line "Q", from the T-junction with P at the keeper (1340, 395) away to
  the far mini-goals (415, 115). A fit *without* Q predicts the vanishing point of the touchline
  direction 24 px from Q's line (~1.5 deg), so Q is perpendicular to the goal line; identity unknown, so
  direction-only.

## Rejected

* Far white line at the top-left (0, 88)-(455, 175) looked parallel to the goal line in the image but
  projects 30 deg off the goal-line direction in every fit; removed (it is probably the 18-yard line of
  the big pitch at a different angle, or a 9v9 marking).
* Far post `A_post_S`: hidden behind the net; the rear wheels are not on the goal line (the ground frame
  goes back ~1.5 m) so they are not usable as post points.
* `A_D_S` / `A_D_apex`: occluded by keeper and goal.
* Distant mini-goals and the row of far players: on line Q, but not pitch markings.

## Validation (final file, d = 6.0)

`python -m pitchworld.cli check-calib cam1.mp4 --camera 1 --calib cam1_constraints.json --pitch
../pitch_small_sided_50x30.json --out cam1_check.jpg`:

* method `manual_pose`, rms residual **0.22 m / 2.5 px**, confidence 0.78 (2 points + 1 line + 1 arc
  + 2 parallels; the goal-line and arc traces alone were 0.20 m in the old thin calibration, so the added
  constraints cost almost nothing in residual while fixing the direction of the perpendicular axis).
* pose: camera at (x, y) = (-3.4, 23.1) m, i.e. 3.4 m behind the goal line and ~8 m north of the goal
  centre, on the pavement - matches where the phone visibly is; **height 1.13 m** (hand-held, waist/chest),
  focal 1122 px (~92 deg horizontal FOV at 2316 px - a wide/0.6x phone lens, consistent with the visible
  barrel-like stretch at the frame edges), roll 12 deg, pitch 14 deg.
* people-scale 1.11 (n = 7264 boxes) -> the model is ~10 % small; see the d_radius table above.
* Projected constraints: goal line x rms 0.03 m; D radius 5.98-6.02 m along the trace; P projects to
  x = 6.3-6.9 m with direction 2 deg from the goal-line direction; Q direction 2.4 deg from the touchline
  direction. Sanity: dropping `A_post_N` moves the pose by < 0.1 m and changes people-scale by 0.03.
* `cam1_check.jpg`: red = traced pixels, yellow = projected 50 x 30 model outline.

## Source observations (not changed - no Python edits allowed for this task)

* `snap_polyline_to_mask` averages *all* mask pixels within `band` of each station, so it silently
  drops sections where the line is thicker than the `white` "thin" mask or where the rough polyline is
  more than `band` px off; it also uses the whole-image pixel list (slow, O(N) per station).
* `manual_calibrate`'s parallels are soft: with a poorly conditioned far segment the fitted pose can
  leave a "parallel" 30 deg off without a large residual - a per-constraint residual report from
  `check-calib` would make this visible.
* `check-calib` prints "auto: found 2 long white line(s)..." even when a manual calibration is given.
