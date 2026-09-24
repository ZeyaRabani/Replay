"""Tests for the fused candidate list built by fuse.py."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CANDS = os.path.join(HERE, "outputs", "candidates.json")


@pytest.fixture(scope="module")
def doc():
    subprocess.run([sys.executable, os.path.join(HERE, "fuse.py")],
                   check=True, cwd=REPO, capture_output=True)
    with open(CANDS) as f:
        return json.load(f)


def test_2736_correction(doc):
    g = [c for c in doc["candidates"] if abs(c["t"] - 2736.0) <= 2.0]
    assert len(g) == 1
    g = g[0]
    assert g["type"] == "goal"
    assert g["cross_validation"] == "confirmed"
    assert g["confidence"] == 0.99
    assert g["clip_end"] - g["clip_start"] == pytest.approx(10.0)  # 5s pad
    assert g["signals"]["net_retrieval"] is True
    assert g["selected"] is True


def test_warmup_excluded(doc):
    for c in doc["candidates"]:
        assert c["t"] >= 1050.0
        assert c["t"] <= 4990.0


def test_rejected_tracking_goals_capped(doc):
    for c in doc["candidates"]:
        if c["cross_validation"] == "rejected":
            assert c["confidence"] <= 0.25
            assert abs(c["t"] - 2736.0) > 12.0


def test_pipeline_only_not_selected(doc):
    for c in doc["candidates"]:
        if c["cross_validation"] == "pipeline_only":
            assert c["selected"] is False
            assert c["confidence"] <= 0.49


def test_ids_ranks_unique_sorted(doc):
    cands = doc["candidates"]
    ids = [c["id"] for c in cands]
    ranks = [c["rank"] for c in cands]
    assert len(ids) == len(set(ids))
    assert ranks == list(range(1, len(cands) + 1))
    confs = [c["confidence"] for c in cands]
    assert confs == sorted(confs, reverse=True)


def test_clip_bounds(doc):
    for c in doc["candidates"]:
        pad = 5.0 if c["type"] == "goal" else 3.0
        assert c["clip_start"] == pytest.approx(max(0.0, c["t"] - pad),
                                              abs=0.01)
        assert c["clip_end"] == pytest.approx(min(5337.153, c["t"] + pad),
                                              abs=0.01)
        assert c["clip_start"] < c["t"] < c["clip_end"]
