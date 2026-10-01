"""Player grouping: fingerprints, cannot-link, constrained clustering."""

import numpy as np

from highlights.analysis.groups import (
    FEAT_DIM,
    cannot_link,
    group_tracklets,
    tracklet_fingerprint,
)


def _tr(tid, t0, t1, team="A", crops=None):
    return {"id": tid, "team": team, "t_start": float(t0),
            "t_end": float(t1), "n_samples": int(t1 - t0),
            "distance_m": 10.0, "sprints": 1,
            "crops": crops or [], "path": []}


def _coloured_feats(colours: dict[int, np.ndarray], ids):
    """Hand-made unit-norm fingerprints: each colour = one hot region
    bucket (ids share buckets -> cosine distance 0 between same colour)."""
    feats = np.zeros((len(ids), FEAT_DIM))
    for r, tid in enumerate(ids):
        f = np.zeros(FEAT_DIM)
        f[colours[tid]] = 1.0
        feats[r] = f
    return feats


def test_three_colours_three_groups():
    ids = [1, 2, 3, 4, 5, 6]
    # disjoint time spans so nothing cannot-links
    trs = [_tr(t, t * 10, t * 10 + 5) for t in ids]
    colours = {1: 0, 2: 0, 3: 50, 4: 50, 5: 100, 6: 100}
    feats = _coloured_feats(colours, ids)
    groups = group_tracklets(trs, feats)
    assert len(groups) == 3
    sizes = sorted(len(g["tracklet_ids"]) for g in groups)
    assert sizes == [2, 2, 2]
    assert all(g["cohesion"] == 1.0 for g in groups)


def test_overlap_never_merges():
    # identical fingerprints but overlapping in time -> cannot link
    trs = [_tr(1, 0, 60), _tr(2, 10, 70)]
    feats = np.ones((2, FEAT_DIM)) / np.sqrt(FEAT_DIM)
    groups = group_tracklets(trs, feats)
    assert len(groups) == 2
    assert cannot_link(trs[0], trs[1])
    assert not cannot_link(_tr(1, 0, 5), _tr(2, 10, 15))


def test_teams_never_merge():
    trs = [_tr(1, 0, 5, team="A"), _tr(2, 0, 5, team="B")]
    feats = np.ones((2, FEAT_DIM)) / np.sqrt(FEAT_DIM)
    groups = group_tracklets(trs, feats)
    assert len(groups) == 2
    assert {g["team"] for g in groups} == {"A", "B"}


def test_max_per_team_cap():
    # 15 singletons, all distinct colours, non-overlapping -> cap to 11
    ids = list(range(1, 16))
    trs = [_tr(t, t * 10, t * 10 + 5) for t in ids]
    feats = np.eye(FEAT_DIM)[:15]          # all distance ~1.4 apart
    groups = group_tracklets(trs, feats, max_per_team=11)
    assert len(groups) <= 11


def test_determinism():
    ids = [1, 2, 3, 4]
    trs = [_tr(t, t * 10, t * 10 + 5) for t in ids]
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(4, FEAT_DIM))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    g1 = group_tracklets(trs, feats)
    g2 = group_tracklets(trs, feats)
    assert [g["tracklet_ids"] for g in g1] == [g["tracklet_ids"]
                                             for g in g2]


def test_fingerprint_blank_crops(tmp_path):
    f = tracklet_fingerprint([tmp_path / "missing.jpg"])
    assert f.shape == (FEAT_DIM,)
    assert not f.any()
