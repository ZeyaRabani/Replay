# Real-footage pitch geometry (three phone clips, 2316x1080, ~24 s)

Result file: `pitch_estimated.json` (loadable by `PitchModel` via the `small_sided` preset).
Annotated frames (t = 18 s): `geometry_cam0_annotated.jpg`, `geometry_cam1_annotated.jpg`,
`geometry_cam2_annotated.jpg`. Labels below (E0/E1/E2, K, C, beta) refer to those images.

Everything here comes from hand/Hough-measured pixel positions in extracted frames plus
projective reasoning (horizon / vanishing points / cross-ratios) and two absolute rulers
(portable-goal height, people ~1.75 m). Nothing was surveyed; treat all numbers as estimates
with the ranges given.

## 1. What is on the ground (layout)

```
              goal line A (white, PitchModel x = 0)
   C ----+========[ goal A ]========-----------------------
         |           ( D_A, blue )                          axis line "beta" (white) runs from
         |               |                                  goal A to goal B along the pitch axis
         |          beta |  (perp. to both goal lines)      (it is the halfway line of the big
         |               |                                  white 11-a-side pitch this small pitch
         |               K  <- D_B apex, keeper stands here  is laid *across*)
         |           ( D_B, blue )       ( D of next cross-pitch, blue, further along the line )
   ------+========[ goal B ]========-----------------------------------------------------------
              goal line B (white, PitchModel x = L)
```

* **Target pitch = a cross-pitch laid across a full-size white-lined 3G pitch.** The two portable
  goals stand on two long *parallel* white lines (cam0 near line = goal line A; cam1/cam2 near
  line = goal line B). In cam2 the far goal A's crossbar is parallel to the near goal line B
  (their image lines meet at the horizon, ~(32, 221) px), confirming the goals face each other on
  parallel lines.
* **Blue arcs are D's in front of each portable goal** (not centre-circle parts, not unrelated).
  In cam0 the arc leaves goal line A at E0 and bulges *towards the pitch*; in cam1/cam2 the arc
  leaves goal line B at E1/E2 and bulges towards the pitch, and its apex K is exactly where the
  goalkeeper stands. Both arcs are the same colour and radius. There is a further blue arc along
  goal line B (right edge of cam2) belonging to the neighbouring cross-pitch — **ignore it**.
* **The white line "beta"** (cam0 far line; cam1/cam2 line that passes under goal A and through
  K) is perpendicular to the goal lines and passes through/near both goal centres and the D_B
  apex. In cam0 it meets goal line A at a right angle (checked with the horizon: the two image
  lines meet 1.2 px off the horizon estimate, i.e. the meeting point is a real ground point, not a
  vanishing point). It is the big pitch's halfway line; the small pitch has **no white
  touchlines** of its own (its width is unmarked).
* Other white lines (the corner **C** left of goal A in cam1, the second line meeting beta at K,
  lines behind goal B in cam1) belong to the big pitch / another overlay and are *not* small-pitch
  markings. Do not use them as pitch landmarks unless the big pitch is modelled explicitly.
* No centre circle, no centre spot, no penalty spot and no rectangular goal area were seen.

## 2. Ratios (scale-free, from projective reasoning)

| quantity | method | value | range |
|---|---|---|---|
| goal-line A ⟂ beta | cam0: line intersection vs horizon | 90° | +-3° |
| goal-line B ∥ goal A crossbar | cam2: image lines meet at horizon | parallel | - |
| D radius / goal width (r/g) | cam2 cross-ratio on goal line B (post_L, post_R, E2, VP) | 0.9-1.0 | 0.85-1.6 (cam0 back-projection gives 1.4-1.6) |
| goal A height / goal B height in cam2 | 68 px / 176 px | 0.39 | +-0.04 |
| far-goal-A / goal-B pixel height in cam1 | 43 px / 300 px | 0.14 | - |

The r/g estimates from cam0 and cam2 disagree (cam0 assumes a horizon row and camera height
that are only known to ~+-15 %, cam2 depends on which white post is the front-left one). Their
overlap, and the fact that a 6 m D is the standard mini-soccer/futsal-style marking, gives the
central value below.

## 3. Absolute scale

Rulers: portable goal height (12x6 ft goal = 1.83 m, 16x7 ft = 2.13 m; assumed 2.0 +-0.15 m),
people 1.75 m. cam2 is letterboxed 1920x1080 in a 2316 canvas -> native phone main camera,
focal length ~1400 px (1250-1450). cam1 is much wider (goal B 300 px tall at ~5-8 m) -> ultrawide,
f ~ 850-950 px; cam0 f ~ 1600-1700 px.

* **Goal width g**: post spacing vs post height in cam0/cam1 gives an aspect of 2.3-2.6 (width /
  height). With height 1.83-2.13 m -> g = 4.4-5.3 m. Best match to a catalogue size: **4.88 m
  (16 ft) x 2.0 m** ("16x7 ft" 7-a-side goal). 5.0x2.0 m fits equally; 3.66 m is excluded
  (aspect 2.0), 2.44 m excluded.
* **Pitch length L (goal line A to goal line B)**: cam2 sees both goals. Goal B (176 px) is at
  15-17 m, goal A (68 px) at 35-45 m; cam2 stands ~16 m off the axis on goal line B, so
  L = sqrt(dA^2 - 16^2) = **37 m**, range **32-45 m**. The far goalkeeper (80 px) at ~31 m is
  consistent (6-8 m in front of goal A).
  Cross-check: 37 m is exactly the width of a 60x40 yd (55x37 m) 7-a-side pitch, and a 5-a-side
  40x30 m pitch laid across it would also give 37-40. The previous guess of 50 m is at the very top
  of the range and probably too long.
* **D radius r**: r/g x g -> 4.4-8 m; **6.0 m** adopted (matches the standard 6 m D of
  mini-soccer pitches and the value the current calibration already uses).
* **Pitch width W**: unmarked. Players in all three clips stay within ~+-15 m of the axis.
  Adopt **30 m** (range 25-40 m). W only matters for the y-extent of the rendered pitch; it does
  not constrain calibration because there are no touchline landmarks.

## 4. PitchModel fields to set (pitchworld/pitch.py)

```json
{"preset": "small_sided", "length": 37.0, "width": 30.0, "goal_width": 4.88, "d_radius": 6.0}
```

* `length = 37.0` (32-45) - goal line A is x = 0, goal line B is x = L.
* `width = 30.0` (25-40) - cosmetic; goals are centred at y = W/2 as the model assumes.
* `goal_width = 4.88` (4.4-5.3; 5.0 equally plausible).
* `d_radius = 6.0` (4.5-8).
* `penalty_*`, `goal_area_*`, `centre_circle_radius` = 0 (the `small_sided` preset already zeroes
  them). No centre circle / spot exists on the footage.

Landmarks already provided by `PitchModel.landmarks()` that are actually visible and should be
traced for calibration:

* `A_post_S/N`, `B_post_S/N` (base of the front posts). cam1 and cam2 both see **both** goals;
  the far goal's posts give a long-baseline constraint that the current fits lack.
* `A_D_end_S/N`, `B_D_end_S/N` (E0, E1, E2) and `A_D_apex` / `B_D_apex` (K in cam1, the keeper's
  feet). The apex is a real point on the axis line, so it can be traced as a point, not only via
  the arc.
* Goal-line segments `goal_line_A` / `goal_line_B` (already used).

Markings **not** in the model that would help:

1. **Axis line** (0, W/2) - (L, W/2): a long white line visible in all three cameras, perpendicular
   to the goal lines; would add a second line direction per camera (currently every traced line
   is parallel to a goal line, which is why the pose is weakly constrained). Suggest an optional
   `axis_line: bool` in `PitchModel.segments()` (one additive line).
2. The neighbouring cross-pitch D along goal line B (cam2 right edge) if its spacing is ever
   measured; not needed now.
3. No penalty spot / centre circle / second arc per goal needed.

## 5. Uncertainty summary

| parameter | adopted | range | dominant error |
|---|---|---|---|
| length L | 37 m | 32-45 m | goal height (1.83-2.13 m), cam2 focal length |
| width W | 30 m | 25-40 m | unmarked - judgement |
| goal width g | 4.88 m | 4.4-5.3 m | goal height, post identification |
| D radius r | 6.0 m | 4.5-8 m | cam0 horizon/camera height, cam2 post identification |
| goals coaxial on beta | yes | +-1.5 m lateral | line-fit extrapolation |

If Stage 1 gets a chance to refine: jointly fitting L and r with both goals' posts plus the axis
line across all three cameras should collapse the L range to +-2 m; W cannot be determined from
markings and should stay a cosmetic parameter.
