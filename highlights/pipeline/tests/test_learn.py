import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from highlights.pipeline import learn
from highlights.pipeline.features import MODEL_COLS


def _features(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"t": np.arange(n, dtype=float),
         **{c: rng.normal(size=n) for c in MODEL_COLS},
         "in_match": np.ones(n, dtype=bool)})
    return df


def _project(tmp_path, n_cand=30, seed=0, duration=200.0):
    """Stub ProjectStore: pipeline/features_1s.parquet + candidates."""
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    df = _features(int(duration), seed)
    # make f00-ish signal: motion_total high at confirmed t's
    cands = []
    for k in range(n_cand):
        t = 10 + 6 * k
        pos = k % 2 == 0
        df.loc[df["t"] == t, "motion_total"] = 5.0 if pos else -5.0
        start, end = t - 4.0, t + 4.0
        cands.append(SimpleNamespace(
            id=f"c{k:03d}", t=float(t), type="goal" if k % 5 == 0 else "shot",
            status="confirmed" if pos else "rejected",
            clip_start=start, clip_end=end, signals={}))
    df.to_parquet(pipe / "features_1s.parquet")
    pd.DataFrame({"t": df["t"], "learned": np.linspace(0, 1, len(df))}).to_parquet(
        pipe / "scores.parquet")
    return SimpleNamespace(id="proj_0000000000", pipeline_dir=pipe,
                           video=SimpleNamespace(duration_s=duration),
                           candidates=cands)


def test_collect_and_train(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    p = _project(tmp_path)
    df = learn.collect_project_examples(p)
    assert len(df) == 30
    assert set(df["label"]) == {0, 1}
    assert df["is_goal"].sum() > 0
    assert all(f"f{i:02d}" in df.columns for i in range(24))
    assert "learned_prob" in df.columns
    learn.save_examples(df)
    meta = learn.train(pd.read_parquet(learn.learn_dir() / "examples.parquet"))
    assert meta["trained"] is True
    assert meta["n_examples"] == 30 and meta["n_pos"] == 15 and meta["n_neg"] == 15
    assert set(meta["window_prior"]) == {"goal", "other"}
    assert (learn.learn_dir() / "verdict_lr.joblib").is_file()
    assert json.loads((learn.learn_dir() / "verdict_lr.json").read_text())["version"] == \
        learn.VERSION
    vm = learn.load_verdict_model()
    vp = learn.verdict_prob(_features(), np.zeros(200), vm)
    assert vp.shape == (200,)
    assert np.all((vp >= 0) & (vp <= 1))


def test_train_too_few(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    df = learn.collect_project_examples(_project(tmp_path)).head(10)
    assert learn.train(df)["trained"] is False
    assert not (learn.learn_dir() / "verdict_lr.joblib").exists()


def test_excludes_pending_and_out_of_range(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    p = _project(tmp_path)
    p.candidates[0].status = "pending"
    p.candidates[1].t = 9999.0  # beyond duration + 1
    df = learn.collect_project_examples(p)
    ids = {c.id for c in (p.candidates[0], p.candidates[1])}
    assert not (set(df["cand_id"]) & ids)


def test_upsert_overwrites(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    p = _project(tmp_path)
    df1 = learn.collect_project_examples(p)
    learn.save_examples(df1)
    p.candidates[0].status = "rejected"
    df2 = learn.collect_project_examples(p)
    learn.save_examples(df2)
    all_df = pd.read_parquet(learn.learn_dir() / "examples.parquet")
    assert len(all_df) == len(df1)  # same key overwritten, not duplicated
    row = all_df[all_df["cand_id"] == p.candidates[0].id].iloc[0]
    assert row["label"] == 0


def test_window_prior_from_verdicts(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_LEARN_DIR", str(tmp_path / "learn"))
    p = _project(tmp_path)
    for i, c in enumerate(p.candidates):
        if c.status == "confirmed":
            c.clip_start, c.clip_end = c.t - 7.0, c.t + 9.0
    df = learn.collect_project_examples(p)
    prior = learn._window_prior(df)
    goal = prior["goal"]
    assert goal["pre"] == pytest.approx(7.0)
    assert goal["post"] == pytest.approx(9.0)
