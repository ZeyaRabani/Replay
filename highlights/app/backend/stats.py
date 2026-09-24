"""Stats computation (Contract 5): re-exports the shared pipeline
implementation and adapts `demo_stats()` to the committed fusion outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from highlights.pipeline.stats import compute_stats as _compute_stats

REPO_ROOT = Path(__file__).resolve().parents[3]
FUSION_OUT = REPO_ROOT / "highlights" / "fusion" / "outputs"


def compute_stats(
    features_df: pd.DataFrame,
    candidates_events: list[dict],
    duration: float,
    match_window: list | None,
    halves: list | None,
    whistles: list | None,
    *,
    model: str = "unknown",
    auroc_reference: float | None = None,
    notes: str = "",
) -> dict:
    """Compatibility wrapper: pipeline compute_stats takes a `pipeline` dict
    instead of separate model/auroc/notes kwargs."""
    return _compute_stats(
        features_df,
        candidates_events,
        duration,
        match_window,
        halves,
        whistles,
        pipeline={"model": model, "auroc_reference": auroc_reference,
                  "notes": notes},
    )


def demo_stats() -> dict:
    """Stats for the bundled demo match from committed fusion outputs."""
    df = pd.read_parquet(FUSION_OUT / "features_1s.parquet")
    cf = json.loads((FUSION_OUT / "candidates.json").read_text())
    whistles: list[float] = []
    wpath = REPO_ROOT / "highlights" / "audio" / "outputs" / "whistles.json"
    if wpath.is_file():
        whistles = [w["t_start"] for w in json.loads(wpath.read_text()).get("whistles", [])]
    return compute_stats(
        df,
        cf["events"],
        float(cf["video_duration_s"]),
        cf.get("match_window"),
        [],  # half-time break is cut from the demo video
        whistles,
        model="fusion learned+rule (committed outputs)",
        auroc_reference=0.78,
        notes="Precomputed from highlights/fusion/outputs for the demo match; halves not detected (break cut from video)",
    )
