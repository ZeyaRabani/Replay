# Verdict learning + dynamic clip windows — spec

Branch: devin/1790266995-hl4-app-v2 (PR #36). Commit style as before. Scope: code + tests up to push; deploy is a separate handoff.

## 1. Training data & model  (`highlights/pipeline/learn.py`, new)

Data root: `LEARN_DIR = workdir()/"learning"` (workdir from app store; on Oracle `/data/learning`).

### collect_project_examples(p: ProjectStore) -> pd.DataFrame
For a project with `pipeline/features_1s.parquet` and a video:
- rows = candidates with status in {confirmed, rejected} and `t <= duration + 1` (out-of-range are excluded — they were rejected because the footage was trimmed).
- columns: `project_id, cand_id, t, label` (1 confirmed, 0 rejected), `is_goal` (status confirmed & type goal), `ai_type` (signals.ai_type or type), `pre` = t - clip_start, `post` = clip_end - t, `edited_window` (True iff (pre,post) differs from default_clip_window by >0.25 s), `verdict_at` (now), plus the model features: `rolling_X(features_1s, MODEL_COLS, window=3)` at the row nearest to t (24 columns named `f00..f23`), and `learned_prob` from `pipeline/scores.parquet` column `learned` at that row (0 if missing).
- Reuse `highlights.pipeline.score.rolling_X` and `highlights.pipeline.features.MODEL_COLS`.

### save_examples(df)
Upsert into `LEARN_DIR/examples.parquet` keyed on (project_id, cand_id) — later verdicts overwrite. This is the persistent store so deleted projects keep contributing.

### train(examples) -> dict meta
- Need ≥ 20 rows and both classes present, else return `{"trained": False, "reason": ...}` and do NOT overwrite an existing model.
- X = f00..f23 + learned_prob (25 cols); y = label. `StandardScaler` + `LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000)`.
- Leave-one-project-out AUROC when ≥2 projects (mean over folds; skip a fold if its held-out set has one class); else `None`.
- Window prior: over rows with label==1: `pre_median`, `post_median` overall and per `is_goal` (goal vs not); fall back to 3/3 and 5/5 when <3 rows.
- Save `LEARN_DIR/verdict_lr.joblib` = {"scaler","clf","columns","window":3,"version":"verdict_lr v1"} and `LEARN_DIR/verdict_lr.json` meta = {trained_at, n_examples, n_pos, n_neg, n_goals, projects: {id: {pos,neg}}, auroc_lopo, window_prior:{goal:{pre,post}, other:{pre,post}}, version}.
- Fully deterministic; no network.

### train_from_registry(reg, project: ProjectStore | None) -> meta
Under a module-level `threading.Lock` (non-blocking: if already training, return {"trained": False, "reason": "busy"}): if `project` given, collect+save its examples; then load examples.parquet and `train`. Also, for robustness, on every call also collect+save examples of every other project in the registry that has verdicts (cheap: a few hundred rows) so nothing gets lost when a user deletes a project without rendering.

### load_verdict_model() -> dict | None
Loads joblib + json if present. `HL_LEARN_DIR` env var overrides LEARN_DIR (tests).

## 2. Applying it in the pipeline  (`highlights/pipeline/run.py`, `candidates.py`)

In `stage_candidates`: `vm = load_verdict_model()`. If present:
- `vp = verdict_prob(df, scores.learned, vm)` per second (same rolling_X + learned column → predict_proba[:,1]), smoothed with `smooth3`.
- blended `learned2 = 0.5*learned + 0.5*vp` → passed to `make_candidates` instead of `learned`.
- each event gets `signals["verdict_prob"] = round(vp[i],4)` and `signals["model"] = "verdict_lr v1"`; also record `res["learning"] = {"model": version, "n_examples", "auroc_lopo"}`.
- Log `ctx.log(f"candidates: verdict model {version} ({n_examples} verdicts, AUROC {auroc}) applied")`; without a model log "no verdict model, base scoring".
Implement `verdict_prob` in `learn.py`. Multiangle projects run `highlights.pipeline.run` on match.mp4 (confirm in multiangle/run.py — the stages after `angles` should reach stage_candidates through the same code path; if it calls make_candidates directly anywhere, route it through the same helper).

## 3. Dynamic clip windows  (`candidates.py` + `app/backend/store.py`)

New `dynamic_window(i, t, learned, motion, duration, etype, prior) -> (t_start, t_end)` in candidates.py:
- `prior` = window_prior from the verdict model (or default goal 5/5, other 3/3). `min_pre, min_post` = prior values; `MAX_PRE = 12`, `MAX_POST = 12` (goal: MAX_POST 15).
- activity a[j] = max(learned[j]/learned[i], motion[j]/motion[i]) with motion[i] guarded > 0; threshold `ACT = 0.45`.
- start: walk j from i backwards while `t[i]-t[j] < MAX_PRE` and a[j] >= ACT (allow one single-second dip); t_start = min(t[j_last], t[i]-min_pre). Same forward for end with min_post. Clamp to [0, duration]; enforce 4 s ≤ length ≤ 30 s (shrink symmetric toward t if longer).
- Events get `t_start`, `t_end` from this (replacing the fixed -8/+6) and `signals["window"] = "dynamic"`.

`store.make_candidates`: if the event has t_start/t_end with `t_start < t < t_end` and 4 ≤ len ≤ 30 → use them as clip_start/clip_end; else default_clip_window as now. Keep `default_clip_window` (still used for manual resets and single-camera fallback).

## 4. Trigger on Render reel  (`app/backend/main.py`)

In `_start_render`, after the render thread starts: `threading.Thread(target=_train_after_render, args=(p,), daemon=True).start()` where `_train_after_render` calls `learn.train_from_registry(get_registry(), p)` and writes `_hist(p, "model_trained", **{k: meta[k] for k in (n_examples, n_pos, n_neg, auroc_lopo, version)})` when trained, or `_hist(p, "model_train_skipped", reason=...)`. Never raise into the request.

New GET `/api/learning` (admin-only like other admin routes, or any user — pick whatever the existing admin routes use) → verdict_lr.json meta or `{"trained": False}`.

## 5. Stats from verdicts  (`main.py _stats5`)
If the project has any confirmed candidates: override `st["goals"]` = number of confirmed type=goal, add `st["highlights"]` = confirmed non-goal, `st["basis_events"] = "user verdicts"`. Otherwise unchanged.

## 6. Tests (pytest, highlights/app/tests + highlights/pipeline/tests)
- learn: synthetic features_1s (200 s, MODEL_COLS random), 30 candidates with labels correlated to one feature → train returns trained=True, files written to tmp HL_LEARN_DIR, meta counts right, window_prior computed; verdict_prob shape == len(df); out-of-range and pending candidates excluded; upsert overwrites same (project,cand).
- candidates: dynamic_window — flat activity → (t-min_pre, t+min_post); a plateau extending 8 s before the peak → t_start ≈ t-8; never > MAX; length clamp.
- store: make_candidates uses event t_start/t_end when sane; falls back otherwise.
- main: POST render → history contains model_trained or model_train_skipped (mock the training thread or join it); GET /api/learning.
- Run `pytest highlights/app/tests highlights/pipeline/tests -q` with PYTHONPATH as before; ruff on changed files. Existing tests must keep passing (fix only if my change broke them, tell me which).
