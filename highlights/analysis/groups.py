"""Player grouping: cluster tracklets into likely-same-player groups.

Each tracklet is fingerprinted by region-wise HSV histograms of its
review crops (head/torso/shorts/legs — kit colours dominate), then
constrained average-linkage agglomerative clustering merges the closest
groups while respecting hard "cannot be the same person" constraints:
two tracklets visible at the same time on the same camera are different
people, and the team vote (A/B) is never crossed.

Pure numpy + cv2; deterministic (closest pair, tie-break by member ids).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

CROP_W, CROP_H = 32, 96
REGION_SPLITS = ((0.0, 0.15), (0.15, 0.50), (0.50, 0.75), (0.75, 1.0))
H_BINS, S_BINS = 12, 4
FEAT_DIM = len(REGION_SPLITS) * H_BINS * S_BINS  # 192

MAX_PER_TEAM = 11
DIST_THRESH = 0.45
MIN_OVERLAP_S = 1.0


def _region_hist(img_bgr: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """L1-normalised HxS joint histogram of a horizontal band."""
    import cv2
    band = img_bgr[y0:y1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hist, _, _ = np.histogram2d(
        hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(),
        bins=(H_BINS, S_BINS), range=((0, 180), (0, 256)))
    s = hist.sum()
    return (hist / s).ravel() if s > 0 else hist.ravel()


def _fingerprint_one(img_bgr: np.ndarray) -> np.ndarray:
    import cv2
    img = cv2.resize(img_bgr, (CROP_W, CROP_H))
    h = img.shape[0]
    parts = [
        _region_hist(img, round(h * a), max(round(h * b), round(h * a) + 1))
        for a, b in REGION_SPLITS]
    return np.concatenate(parts)


def tracklet_fingerprint(crop_paths: list[Path]) -> np.ndarray:
    """Mean region-histogram fingerprint over the tracklet's crops
    (first/best/last). Returns zeros(FEAT_DIM) when no crop decodes."""
    import cv2
    feats = []
    for p in crop_paths:
        img = cv2.imread(str(p))
        if img is not None:
            feats.append(_fingerprint_one(img))
    if not feats:
        return np.zeros(FEAT_DIM)
    f = np.mean(np.stack(feats), axis=0)
    n = float(np.linalg.norm(f))
    return f / n if n > 0 else f


def _overlap_s(a: dict, b: dict) -> float:
    lo = max(a["t_start"], b["t_start"])
    hi = min(a["t_end"], b["t_end"])
    return max(0.0, hi - lo)


def cannot_link(a: dict, b: dict, min_overlap_s: float = MIN_OVERLAP_S) -> bool:
    """Two tracklets overlapping in time on the same camera are two
    different people."""
    return _overlap_s(a, b) >= min_overlap_s


def _pairwise_cosine(feats: np.ndarray) -> np.ndarray:
    """Cosine distance matrix [n, n]."""
    sim = feats @ feats.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def group_tracklets(tracklets: list[dict], feats: np.ndarray,
                    *, max_per_team: int = MAX_PER_TEAM,
                    dist_thresh: float = DIST_THRESH) -> list[dict]:
    """Constrained average-linkage clustering, per team pool ("A", "B",
    None = own pool). Average link = mean pairwise cosine distance of the
    union's members; a merge is legal only if no member pair overlaps in
    time. Pools larger than max_per_team keep merging the closest legal
    pairs below the threshold requirement until the cap is met or no
    legal pair remains."""
    n = len(tracklets)
    dist = _pairwise_cosine(np.asarray(feats))
    tids = [int(t["id"]) for t in tracklets]
    team_of = {i: (tracklets[i].get("team") or "-") for i in range(n)}

    # tracklet-level cannot-link matrix: same camera + overlapping in time
    starts = np.array([t["t_start"] for t in tracklets])
    ends = np.array([t["t_end"] for t in tracklets])
    overlap = (np.minimum(ends[:, None], ends[None, :])
               - np.maximum(starts[:, None], starts[None, :]))
    clink = overlap >= MIN_OVERLAP_S

    # per team pool: cluster distance matrix D (average link) + cluster
    # conflict matrix C (any member pair cannot-link). Merge updates:
    # D[new,c] = (|c1|·D[c1,c] + |c2|·D[c2,c])/(|c1|+|c2|); C[new] = OR.
    clusters: list[list[int]] = []
    for team in ("A", "B", "-"):
        idx = [i for i in range(n) if team_of[i] == team]
        if not idx:
            continue
        D = dist[np.ix_(idx, idx)].copy()
        C = clink[np.ix_(idx, idx)].copy()
        cl: list[list[int]] = [[i] for i in idx]
        sizes = np.ones(len(idx))
        while len(cl) > 1:
            Dm = np.where(C, np.inf, D)
            np.fill_diagonal(Dm, np.inf)
            over_cap = len(cl) > max_per_team
            flat = int(np.argmin(Dm))
            a, b = flat // Dm.shape[1], flat % Dm.shape[1]
            d = Dm[a, b]
            if not np.isfinite(d) or (d >= dist_thresh and not over_cap):
                break
            # merge b into a
            w = (sizes[a] * D[a] + sizes[b] * D[b]) / (sizes[a] + sizes[b])
            D[a] = w
            D[:, a] = w
            C[a] |= C[b]
            C[:, a] |= C[b]
            cl[a] += cl[b]
            sizes[a] += sizes[b]
            keep = [i for i in range(len(cl)) if i != b]
            D = D[np.ix_(keep, keep)]
            C = C[np.ix_(keep, keep)]
            cl = [cl[i] for i in keep]
            sizes = sizes[keep]
        clusters += cl

    out = []
    team_label = {"A": "A", "B": "B", "-": "?"}
    counters: dict[str, int] = {}
    for c in sorted(clusters,
                    key=lambda c: -sum(
                        tracklets[i]["t_end"] - tracklets[i]["t_start"]
                        for i in c)):
        team = team_of[c[0]]
        counters[team] = counters.get(team, 0) + 1
        dur = sum(tracklets[i]["t_end"] - tracklets[i]["t_start"]
                  for i in c)
        members = sorted((tracklets[i] for i in c), key=lambda t: t["id"])
        # crops: best (index 1) first, spread across members
        crops: list[str] = []
        for pref in (1, 0, 2):
            for tr in members:
                if len(crops) >= 6:
                    break
                cps = tr.get("crops") or []
                if pref < len(cps):
                    crops.append(cps[pref])
            if len(crops) >= 6:
                break
        # cohesion: 1 - mean intra-cluster distance
        if len(c) > 1:
            vals = [dist[i, j] for i in c for j in c if i < j]
            cohesion = round(1.0 - float(np.mean(vals)), 4)
        else:
            cohesion = 1.0
        out.append({
            "id": f"g{team_label[team]}{counters[team]}",
            "team": None if team == "-" else team,
            "tracklet_ids": sorted(tids[i] for i in c),
            "duration_s": round(dur, 1),
            "distance_m": round(sum(float(tracklets[i].get("distance_m", 0.0))
                                    for i in c), 1),
            "sprints": sum(int(tracklets[i].get("sprints", 0)) for i in c),
            "crops": crops,
            "cohesion": cohesion,
        })
    return out


def build_groups(players_dir: Path) -> dict:
    """Read tracklets.json + crops under players_dir, cluster, write
    groups.json. Returns the written document."""
    players_dir = Path(players_dir)
    doc = json.loads((players_dir / "tracklets.json").read_text())
    tracklets = doc.get("tracklets") or []
    crops_dir = players_dir / "crops"
    feats = np.stack([
        tracklet_fingerprint(
            [crops_dir / c for c in (t.get("crops") or [])])
        for t in tracklets]) if tracklets else np.zeros((0, FEAT_DIM))
    groups = group_tracklets(tracklets, feats)
    out = {
        "groups": groups,
        "n_tracklets": len(tracklets),
        "n_grouped": sum(len(g["tracklet_ids"]) for g in groups),
        "generated_at": time.time(),
    }
    from highlights.io import write_json_atomic
    write_json_atomic(players_dir / "groups.json", out, indent=1)
    return out
