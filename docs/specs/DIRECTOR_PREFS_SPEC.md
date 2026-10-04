# Director learned preferences (per match) + 3-stretch suggestions

Evidence (4/9 sessions, /multiangle/direct/sessions): user switches 9–10×
per 75 s, director 2–4×; "event" rule disagreed 13/20, 17/22, 17/24 s;
zone rule pointed at cam 2 while user held cam 1 for 35 s. The five
Style knobs can't express "prefer camera k when the signal says j" or
"don't obey the event rule", so learning moved agreement only 41.6→43.4.

## 1. director.py — `prefs` argument

```python
def cut_director(..., style_overrides=None, prefs: dict | None = None) -> dict
```
`prefs` schema (all optional; defaults = today's behaviour):
```json
{"event_rule": true,
 "angle_remap": {"zone": {"2": 1}, "ball": {}, "cluster": {}},
 "angle_weight": [1.0, 1.0, 1.0]}
```
Apply inside `per_second` (add the same `prefs` kwarg there; cut_director
forwards it):
- `event_rule=False`: skip the `ev.max() > 0` branch entirely (fall
  through to zone/ball/cluster).
- `angle_weight[i]`: multiply `S[i, t]` for the **ball and cluster**
  branches only (not zone/event) before the argmax.
- `angle_remap[rule][str(j)] = k`: after the branch picks `j` under rule
  `zone|ball|cluster`, if `available[k, t]` then `best_a[t] = k` and set
  `S[k, t] = max(S[k, t], S[j, t])` (so downstream margin/hold logic
  sees the remapped angle as the leader). If k is not available, keep j.
Record `prefs` in the returned dict (`"prefs": dict(prefs or {})`).
Keep `per_second`'s return signature; `cand_a`/`cand_r` are what
cut_director already consumes.

## 2. manual_direct.py — learning

New function:
```python
def learn_prefs(sessions, replay_fn, cand_fn, range_lo, n_angles) -> dict
```
- `cand_fn(overrides, prefs) -> (cand_a, cand_r)` from a baseline
  `per_second` run (range-relative, 1 Hz like director.json).
- `replay_fn(overrides, prefs) -> segments` (extend the existing replay
  signature with `prefs`).
Procedure:
1. Baseline candidates with defaults. For every user-directed second t
   (shared seconds → index `t - range_lo`), with candidate (r, j) and
   user angle u (skip None): tally `votes[r][j][u]`.
2. `angle_remap[r][j] = u*` when `u* != j`, votes ≥ 5 s, and u* has
   ≥ 60 % of votes for that (r, j). Only for r in zone/ball/cluster.
3. `event_rule = False` if, over seconds where r == event, the user
   agreed with j < 40 % of the time and there were ≥ 8 such seconds.
4. Stage A: existing `LEARN_GRID` (72 combos) × {prefs from steps 2–3,
   empty prefs} → score with the existing agreement scorer; keep best.
5. Stage B: with best of A fixed, grid `angle_weight ∈ {0.7,1.0,1.5}^n`
   (27 for 3 angles) → keep best; tie → closest to all-1.0.
Return `{"best": overrides, "prefs": prefs, "agreement_pct_before",
"agreement_pct_after", "n_sessions", "n_seconds", "votes": votes}`.
Keep `learn()` as-is (tests) and have `learn_prefs` reuse its scorer.

## 3. main.py `_run_direct_learn` / `post_direct_learn`

- Build `cand_fn` and `replay_fn` from `load_director_inputs`; call
  `learn_prefs`.
- Persist `director_params.json` as
  `{"style_overrides": res["best"], "prefs": res["prefs"]}`.
- `direct_learn.json` gets the full result.

## 4. run.py

Where `director_params.json` is read and `style_overrides` is passed to
`cut_director`, also pass `prefs=params.get("prefs")`.

## 5. 3-stretch suggestions (backend + YouDirect)

Backend: `GET /multiangle/direct/suggest3` → `{"stretches": [payload,
payload, payload]}` where each payload is `_direct_stretch(p, s, e)`
for the busiest **60 s** window in each third of `[lo, hi]`. Busiest =
max over 60-s windows (step 5 s) of
`n_candidates_in_window + 0.5 * n_director_switches_in_window`
(candidates from fused_candidates.json / candidates.json, switches
from director.json segments). Windows overlapping a saved session's
span score 0 unless nothing else is left. Reuse `_direct_stretch`
for the payload.
Frontend YouDirect.tsx: replace the single "Suggest a stretch" button
with "Suggest 3 stretches (early / mid / late)" showing three rows
`Early 31:54–32:54 · Load`, etc.; Load sets the from/to fields and
loads the stretch exactly as the manual path does. Keep manual from/to.
Banner text: "Directed N of 3 stretches" and show "Analyse my
directing & re-cut" when N ≥ 2 (as today).

## Verification (mandatory, scoped)
- `pytest highlights/multiangle highlights/app/tests -q -k "direct or director or manual"`
  plus new tests: (a) `event_rule=False` never yields rule "event";
  (b) remap applied when k available, not when unavailable;
  (c) `learn_prefs` on a synthetic session where the user always picks
  angle 1 when the zone rule says 2 returns `angle_remap.zone["2"] == 1`
  and agreement_after ≥ agreement_before.
- `ruff check highlights`; `cd highlights/app/frontend && npx tsc --noEmit && npm run build`.
