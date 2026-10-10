# Learned zones from You-Direct sessions

Goal: the user's saved you-direct sessions replace hand-drawn zones. For each
camera we learn, from the seconds the user chose camera k, WHERE in camera i's
picture the ball / player feet were — and turn "cells where the user preferred
camera i" into zone polygons in the exact v2 zones.json shape the director
already consumes. Drawn zones stay optional (fallback).

## 1. `highlights/multiangle/learned_zones.py` (new)

```python
GRID_W, GRID_H = 8, 5          # cells over the normalised frame [0,1]x[0,1]
MIN_VOTES = 2                  # seconds of evidence a cell needs
MIN_SHARE = 0.60               # share of votes for camera i in camera i's cell
BALL_CONF = 0.2                # ZONE_BALL_OK
DENS_W = 0.5                   # weight of a density vote relative to a ball vote

def learn_zones(sessions: list[dict], tracks: list[dict], avail: np.ndarray,
                range_lo: float, offsets: list[float],
                durations: list[float]) -> dict | None:
    """Returns zones.json v2 {"version": 2, "learned": True,
    "angles": [[{"t": file_t, "zones": [poly..]}, ...] per angle],
    "cells": [[GRID_H x GRID_W share-or-None] per angle], "n_votes": [...]}
    or None when no camera gets any cell."""
```

Algorithm, for every session s and every shared second t in
[t_start, t_end) with a user choice u (carry the last choice forward inside
the session like learn_prefs does; skip seconds before the first choice):

- idx = int(t - range_lo); skip if idx out of [0, T).
- For every camera i with avail[i, idx]:
  - ball vote: if tracks[i].ball_conf[idx] >= BALL_CONF and ball_x, ball_y > 0:
    cell = (floor(ball_y*GRID_H), floor(ball_x*GRID_W)) clipped;
    votes[i][cell][u] += 1.0
  - density votes: for each foot (fx, fy) in tracks[i].players_xy[idx] (if
    present): cell as above; votes[i][cell][u] += DENS_W / len(feet)
    (so one second of feet counts at most DENS_W total).

Cell (r, c) of camera i is "camera i's zone" when
total = sum(votes[i][cell].values()) >= MIN_VOTES and
votes[i][cell][i] / total >= MIN_SHARE.

Polygons: for each row, merge horizontal runs of selected cells into one
rectangle [[x0,y0],[x1,y0],[x1,y1],[x0,y1]] in normalised coords. Then also
merge vertically adjacent rectangles with identical x-range into one taller
rectangle (simple pass). No other polygon simplification.

Keyframes: for each camera i, one keyframe per session at
file_t = clip(s.t_start - offsets[i], 0, durations[i]) (so the director's
viewcheck compares each stretch against a still from that stretch); every
keyframe of camera i carries the SAME learned polygon list (votes are pooled
across sessions). Sort by t. Cameras with no selected cells get [].

Return None if every camera has [].

## 2. `highlights/multiangle/run.py`

`load_director_inputs(ctx)`: zone source order:
1. `ctx.pipe / "zones_learned.json"` if it exists and parses (log
   "director: zones learned from N sessions").
2. else `ctx.pipe / "zones.json"` (unchanged behaviour).
Factor the existing zones block into `_load_zone_inputs(ctx, zd, ...)` so both
files go through the identical normalise + viewcheck code. Add
`"zone_source": "learned" | "drawn" | None` to the returned dict; stage_director
writes it into director.json as `zone_source`.

Also export a helper `director_input_meta(ctx)` is NOT needed — but
`load_director_inputs` must expose `offsets` and `durations` in its return
dict (needed by the learner).

## 3. `highlights/app/backend/main.py` — `_run_direct_learn`

Before the grid search:
```python
lz = learn_zones(sessions, inp["tracks"], inp["avail"], inp["lo"],
                 inp["offsets"], inp["durations"])
```
Evaluate agreement (same compare as learn_prefs baseline, default style, no
prefs) for three zone variants: drawn (inp zones, may be None), learned
(if lz), none. To do that, build zone inputs for `lz` via the factored
`_load_zone_inputs` (it needs a Ctx — `_direct_ctx(p)` already exists).
Pick the variant with the highest baseline agreement; on ties prefer learned
> drawn > none. Then run `learn_prefs` with that variant's zones (replay/cand
closures use the chosen zones/zone_ok/zone_kf).

Persist:
- `multiangle/zones_learned.json` = lz (always written when lz is not None,
  even if not chosen — it's diagnostics), plus `"chosen": bool`.
- If the chosen variant is NOT learned, delete/rename
  `zones_learned.json` -> keep the file but the loader must skip it: add
  `"active": true/false` to the file; `load_director_inputs` uses it only
  when `active` is true.
- Add to the learn result: `"zone_source"`, `"zone_agreement": {"drawn": x,
  "learned": y, "none": z}` (None where a variant was unavailable), and
  `"learned_cells": lz["cells"]` for the UI.

`post_direct_learn` response/`direct_learn.json` include those fields.
Add `GET /multiangle/zones/learned` returning zones_learned.json (404 if
absent) — for the frontend to draw the learned areas.

## 4. Frontend (`ZoneEditor.tsx` or `YouDirect.tsx`, keep small)

In the You direct card, after an analysis, show one line:
"Learned camera areas from your sessions: agreement drawn X% · learned Y% ·
none Z% → using <source>". If learned zones are active, in the Zones card
overlay the learned rectangles on each camera still in a distinct dashed
colour with the label "learned from your directing" (read-only). Add
`api.ts` client + types.

## 5. Tests — `highlights/multiangle/tests/test_learned_zones.py`

- Synthetic 2 cameras, T=60: ball in camera 0 at x≈0.2 for seconds 0–29
  (user chose 0), at x≈0.8 for 30–59 (user chose 1). Camera 1 sees the ball
  at x≈0.8 during 30–59 too. Assert: camera 0 gets a polygon covering
  (0.2, y) and none covering (0.8, y); camera 1 gets one covering (0.8, y).
- MIN_SHARE: alternate choices 0/1 on the same cell → no zone.
- Return None when no votes.
- Keyframes: 2 sessions → 2 keyframes per camera, t = t_start - offset.
- `load_director_inputs` picks zones_learned.json (active) over zones.json
  (use a tmp project layout if an existing test fixture exists for it; else
  unit-test only `_load_zone_inputs` source selection).

Verification: `pytest highlights/multiangle -q`, `ruff check highlights`,
frontend `npx tsc --noEmit` + `npm run build`.
