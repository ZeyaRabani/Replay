# Camera 0 calibration notes (real footage, 2316x1080)

Files: `cam0_constraints.json` (full calibration file: `pitch` + `cameras["0"]`), `cam0_pitch.json` (the same
pitch block on its own, for `--pitch`), `cam0_annotated.jpg` (traces + landmarks on a median frame),
`cam0_check.jpg` (output of `pitchworld check-calib`).

Reproduce the check:

```
python -m pitchworld.cli check-calib cam0.mp4 --camera 0 \
    --calib examples/real_footage/cam0_constraints.json \
    --pitch examples/real_footage/cam0_pitch.json --out cam0_check.jpg
```

All pixel coordinates are in the original/synced 2316x1080 frame (the synced clip is a pure time crop, same
geometry). Traces were made on a temporal median of ~20 frames spread over the clip (removes the players) and then
snapped to the white / blue colour masks with `snap_polyline_to_mask`.

## What cam0 sees

Cam0 is a phone held ~1.5 m high, standing ~4 m *behind* the goal line and ~9 m to the S (right, from the camera's
point of view) of the goal, looking diagonally across the goal mouth into the pitch.

* Portable wheeled goal (3.66 m mouth) in the left foreground, seen from behind/through the side net. The visible
  right-hand post stands exactly on the white line; the left post is visible through the net.
* A white line runs from under that post down to the bottom-right of the frame -> **goal line A** (`goal_line_A`).
  Goal on it + D attached to it = it is a goal line, not a touchline.
* A blue semicircle attached to that line -> **the D of goal A** (`A_D`). Only its S half is clearly visible; the
  N half passes behind the goal and is faint.
* A second white line farther into the pitch, running parallel to the goal line about 7.8 m into the pitch,
  with a right-angle corner near the top-centre of the frame, from which a long white line runs away from the
  camera perpendicular to the goal line (towards goal B). World identity of these two is unknown (they look like the
  boundary of an overlaid smaller training pitch, see below) so they are used as **parallels only**.
* Far goal (goal B) small in the frame, roughly 60 m away; other portable goals / mini goals scattered on the right
  belong to other pitches and are not used.

## Pitch geometry decision

`PitchModel.small_sided(length=60, width=40, goal_width=3.66, d_radius=7.32)`.

* **D radius 7.32 m (8 yd).** This is the only quantity the pitch coordinates actually depend on for cam0
  (everything visible is measured relative to the D + goal line; length/width only place the far end). It was
  derived, not guessed, from two independent metric rulers applied to a pose fit made without any point constraints:
  * *Goal width:* the two post ground contacts project to 3.65 model units apart at r = 7.32 -> 3.66 m goal
    implies r = 7.3 m. (At r = 6 they were 3.0 units apart, at r = 8 they were 4.0.)
  * *People height:* `metres_per_unit_from_people` on the cam0 raw tracks (9 796 boxes, conf > 0.5, 1.75 m assumed)
    gives 0.93 m/unit at r = 7.32 (-> r = 6.8 m); 1.23 at r = 6; 0.92 at r = 8. Player-height scale is biased by
    crouching / running poses and box padding, so the goal-width ruler is preferred; both agree r = 7 +/- 0.4 m.
* **Goal is not centred on the D.** From the lines+arc fit the posts sit at y = 17.24 and 20.89 while the D centre
  is y = 20 (= width/2): the portable goal stands ~1.0 m S of the D centre. That is normal for wheeled goals and the
  cam0 image plainly shows the D's S end much closer to the S post than to the (hidden) N end. Because of this the
  posts are given explicit `[x, y]` world coordinates rather than the `A_post_S/N` landmark names (which assume a
  centred goal). Anyone using `A_post_*` for cam1/cam2 should apply the same offset (goal centre at y = width/2 - 1.0).
* **Length ~60 m (uncertain +/-8).** Goal B's frame is ~63 px wide in cam0 at focal 1112 px -> ~65 m from the camera,
  camera is 4 m behind goal line A, bearing ~6 deg off the pitch axis -> goal line B at x ~ 60, y ~ 17 (i.e. roughly
  in line with goal A). Cam1/cam2, which look along the pitch, should pin this down better.
* **Width 40 m is a placeholder.** No touchline of this pitch is identified in cam0; the only role of `width` here is
  to fix the D centre at y = 20. If another camera fixes the width, keep the *relative* offsets above (goal centre at
  width/2 - 1.0). Previous guess 50x30 with r = 6 m is inconsistent with the goal-width ruler (it makes the goal 3.0 m
  wide and the camera 2.0 m high with people scale 1.23).
* The parallel line 7.8 m in front of the goal line ends at a right-angle corner at (7.8, ~21.6) from which a
  perpendicular line runs 55+ m along the pitch at y ~ 22. Since our goal centre is at y ~ 19 this cannot be a
  touchline of *this* pitch; most likely it is the boundary of an overlaid smaller training pitch. Kept as
  `parallels` (direction only).

## Constraints in `cam0_constraints.json`

| type | world | pixels | confidence |
|---|---|---|---|
| line | `goal_line_A` | 28 pts, (560,491)->(2302,842) | high |
| arc | `A_D` | 29 pts, faint N part (980..1100) + snapped S part down to the line | high (S half), medium (N part) |
| point | `A_D_S` (arc meets goal line) | (1215.9, 606.5) | high |
| point | `[0, 17.24]` S post ground contact | (545, 488) | high pixel, y from fit (see above) |
| point | `[0, 20.89]` N post ground contact | (242, 430) | medium (seen through net, +/-8 px) |
| parallel | `goal_line_A` | 36 pts, the line 7.8 m in front of the goal | high direction, unknown identity |
| parallel | `touch_S` | 17 pts, the long line at y ~ 22 | medium-high direction (shares a visible right-angle corner with the previous line) |

## Validation

Pose fit (`manual_calibrate`, frame 2316x1080):

* reprojection residual **0.06 m / 2.8 px** RMS over all constraints, confidence 0.94, no notes
* camera pose: **height 1.48 m**, position (-4.1, 10.7) = 4 m behind goal line A, 9 m S of the D centre,
  yaw 28 deg, pitch 14 deg down, roll 2 deg, focal 1112 px (~ 25 mm eq. for a phone main camera, plausible)
* people scale 0.93 (see above; 1.0 would correspond to r ~ 6.8 m)
* without the three point constraints the fit is 8 dof (thin, conf 0.6): height 1.45 m, err 0.04 m. Adding the posts
  and `A_D_S` makes it 12 dof, changes the pose by < 0.1 m / 0.4 deg, and raises conf to 0.94 - i.e. the points are
  consistent with the lines, they are not fighting them.
* `cam0_check.jpg`: the reprojected goal line and D lie on the painted white line and blue arc across the whole frame.

## Caveats / for the other cameras

* Mirror ambiguity: a goal line + D + parallel fit alone has a second solution rotated 180 deg about the D centre
  (pitch on the wrong side of the goal line, camera height and residual identical). The post points remove it here;
  cam1/cam2 traces made only from a line + arc should check the sign of the projected arc x.
* The D radius / scale is the dominant cross-camera error source: a 1 m error in r moves every projected player by
  ~14 %. If cam1/cam2 can see a full D or the goal B width they should confirm r = 7.3.

## Possible source issues noticed (not edited)

* `pitchworld/cli.py` `--pitch` insists on a bare pitch JSON; passing the full calibration file (which has a `pitch`
  key, as the README's calibration format suggests) fails with `PitchModel.__init__() got an unexpected keyword
  argument 'pitch'`. Reading `data["pitch"]` when present would be a 2-line fix.
* `pip install -e .` fails (setuptools backend without PEP 660 `build_editable`); `pip install .` or `PYTHONPATH=.` works.
