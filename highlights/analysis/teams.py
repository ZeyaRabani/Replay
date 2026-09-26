"""Team identification pass: cluster shirt colours on the reference angle.

Decodes the reference-angle video (480p analysis proxy when present) at
0.5 fps, detects persons + ball with YOLO, builds an HSV torso
descriptor per detection and k-means them into two clusters (plus an
outlier class for referee/keeper kits). Player foot-x positions are
aggregated into one row per shared-T second for stats.

    python -m highlights.analysis.teams --video match.mp4 --out teams.json \
        --window-file 0 5400 --shared-offset 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from highlights.io import write_json_atomic
from highlights.multiangle.trackfeat import (
    CLASSES,
    CONF,
    FRAME_W,
    _ensure_model,
    _frame_reader,
)

FPS = 0.5
MIN_DESCRIPTORS = 20
OUTLIER_FRAC = 0.10
N_SEEDS = 5

COLS = ["t_shared", "ball_x", "ball_y", "ball_conf",
        "teamA_xs", "teamB_xs", "teamA_n", "teamB_n"]


def torso_descriptor(frame_bgr: np.ndarray, xyxy) -> np.ndarray | None:
    """[med_h, med_s, med_v, dom_frac] of the upper-torso crop, or None.

    Uses the middle 60% of the box width over the top 45% of its height.
    When saturated pixels (S>40 & V>40) cover >=20% of the crop they are
    used alone; otherwise all pixels are kept (white/black/grey kits).
    dom_frac is the share of coloured pixels in the dominant of 12 hue
    bins (0 when the crop is achromatic)."""
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 + 0.2 * bw))
    cx2 = int(min(w, x2 - 0.2 * bw))
    cy1 = int(max(0, y1))
    cy2 = int(min(h, y1 + 0.45 * bh))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    coloured = (hsv[:, 1] > 40) & (hsv[:, 2] > 40)
    keep = hsv[coloured] if coloured.mean() >= 0.2 else hsv
    if keep.size == 0:
        return None
    med = np.median(keep, axis=0)
    dom_frac = 0.0
    cols = hsv[coloured]
    if cols.size:
        hist = np.histogram(cols[:, 0] / 179.0 * 360.0, bins=12,
                            range=(0, 360))[0]
        dom_frac = float(hist.max() / cols.shape[0])
    return np.array([med[0], med[1], med[2], dom_frac])


def descriptor_features(desc: np.ndarray) -> np.ndarray:
    """[cos(h)*s, sin(h)*s, v] — hue circularity-safe, achromatic kits
    collapse to the v axis."""
    h = math.radians(float(desc[0]) / 179.0 * 360.0)
    s = float(desc[1]) / 255.0
    v = float(desc[2]) / 255.0
    return np.array([math.cos(h) * s, math.sin(h) * s, v])


def kmeans2(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Tiny numpy k-means with k=2 (k-means++-ish seeding)."""
    rng = np.random.default_rng(seed)
    i0 = int(rng.integers(len(X)))
    d0 = ((X - X[i0]) ** 2).sum(axis=1)
    i1 = int(np.argmax(d0)) if d0.max() > 0 else (i0 + 1) % len(X)
    cent = X[[i0, i1]].astype(float).copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(50):
        d = ((X[:, None, :] - cent[None]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                cent[k] = X[labels == k].mean(axis=0)
    return labels, cent


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Label agreement under the best permutation of b."""
    same = float((a == b).mean())
    return max(same, 1.0 - same)


def cluster_teams(descs: list[np.ndarray]) -> dict:
    """Two shirt-colour clusters + outliers over descriptor list.

    The 10% of points farthest from their centroid are outliers
    (referee/keepers) and get label -1; the remaining points are
    reclustered over N_SEEDS seeds. confidence = stability * separation.
    Clusters are sorted by centroid hue so team A is deterministic."""
    X = np.array([descriptor_features(d) for d in descs])
    labels, cent = kmeans2(X, 0)
    dist = ((X - cent[labels]) ** 2).sum(axis=1) ** 0.5
    n_out = max(1, round(len(X) * OUTLIER_FRAC))
    order = np.argsort(-dist)
    outlier_mask = np.zeros(len(X), dtype=bool)
    outlier_mask[order[:n_out]] = True

    inliers = ~outlier_mask
    Xi = X[inliers]
    if Xi.shape[0] < 4 or len(np.unique(Xi, axis=0)) < 2:
        Xi = X
        inliers = np.ones(len(X), dtype=bool)
        outlier_mask = np.zeros(len(X), dtype=bool)
    base_labels, _ = kmeans2(Xi, 0)
    stability = 1.0
    cents = []
    for seed in range(N_SEEDS):
        lab, c = kmeans2(Xi, seed)
        cents.append(c)
        if seed:
            stability = min(stability, _agreement(base_labels, lab))
    cent2 = cents[0]
    within = []
    for k in (0, 1):
        pts = Xi[base_labels == k]
        within.append(float(pts.std()) if len(pts) else 0.0)
    sep = float(np.linalg.norm(cent2[0] - cent2[1]) /
                max(1e-6, 2.0 * np.mean(within)))
    sep = float(np.clip(sep, 0.0, 1.0))
    confidence = round(stability * sep, 3)

    full_labels = np.full(len(X), -1, dtype=int)
    full_labels[inliers] = base_labels
    # order clusters by centroid hue: project centroids back to hue angle
    hues = np.degrees(np.arctan2(cent2[:, 1], cent2[:, 0])) % 360.0
    remap = np.argsort(hues)  # cluster k -> rank
    rank = {int(k): r for r, k in enumerate(remap.tolist())}
    relabel = np.array([rank.get(int(v), -1) if v >= 0 else -1
                        for v in full_labels])
    return {
        "labels": relabel,
        "centroids": cent2[remap],
        "confidence": confidence,
        "outlier_mask": outlier_mask,
    }


def colour_name(hsv) -> str:
    h, s, v = [float(x) for x in hsv[:3]]
    deg = h / 179.0 * 360.0
    if s < 50:
        if v > 170:
            return "white"
        if v < 70:
            return "black"
        return "grey"
    if deg < 15 or deg >= 345:
        return "red"
    if deg < 40:
        return "orange"
    if deg < 70:
        return "yellow"
    if deg < 170:
        return "green"
    if deg < 200:
        return "sky blue"
    if deg < 260:
        return "blue"
    if deg < 300:
        return "purple"
    return "pink"


def hsv_to_hex(hsv) -> str:
    import cv2

    px = np.uint8([[[int(hsv[0]), int(hsv[1]), int(hsv[2])]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def _team_info(centroid_feat: np.ndarray) -> dict:
    """Recover a representative HSV from a feature-space centroid."""
    s = float(np.linalg.norm(centroid_feat[:2]))
    deg = float(np.degrees(np.arctan2(centroid_feat[1], centroid_feat[0])) % 360.0)
    hsv = [deg / 360.0 * 179.0, s * 255.0, float(centroid_feat[2]) * 255.0]
    return {"hsv": [round(x, 1) for x in hsv], "hex": hsv_to_hex(hsv)}


def run_teams_pass(video: str | None, out_path: Path, *,
                   window_file: tuple[float, float],
                   proxy_offset: float = 0.0,
                   shared_offset: float = 0.0,
                   model_path: Path | None = None,
                   fps: float = FPS,
                   imgsz: int = 960,
                   log=print,
                   progress_file: Path | None = None,
                   detector=None,
                   frames=None) -> dict:
    """Detect persons+ball over window_file (ref-angle file seconds),
    cluster shirt colours, aggregate per shared-T second.

    detector(frame) -> (persons: [xyxy], balls: [(xyxy, conf)]); default
    builds YOLO via _ensure_model. `frames` (iterable of (t_file, bgr))
    bypasses video decode entirely — used by tests."""
    if detector is None:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "0")) or
                              (os.cpu_count() or 1))
        model = YOLO(str(_ensure_model(model_path or
                                       Path(__file__).parent.parent
                                       / "multiangle" / "models" / "yolov8n.pt")))

        def detector(frame):
            res = model.predict(frame, imgsz=imgsz, conf=CONF,
                                classes=CLASSES, verbose=False)[0]
            persons, balls = [], []
            boxes = res.boxes
            if boxes is not None and len(boxes):
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                for k, c in enumerate(cls):
                    if c == 0:
                        persons.append(xyxy[k])
                    else:
                        balls.append((xyxy[k], float(conf[k])))
            return persons, balls
        model_name = str(model_path)
    else:
        model_name = "injected"

    if frames is None:
        ds = max(0.0, window_file[0] - proxy_offset)
        de = window_file[1] - proxy_offset
        frames = _frame_reader(str(video), fps, FRAME_W, ds, de,
                               t_base=window_file[0])

    samples = []          # (t_file, descs, foot_xs, ball)
    descs_all: list[np.ndarray] = []
    n_frames = 0
    for t, frame in frames:
        persons, balls = detector(frame)
        descs, foot_xs = [], []
        w = frame.shape[1]
        hgt = frame.shape[0]
        for b in persons:
            d = torso_descriptor(frame, b)
            if d is not None:
                descs.append(d)
                foot_xs.append(float((b[0] + b[2]) / 2 / w))
        ball = (0.0, 0.0, 0.0)
        if balls:
            bx, bconf = max(balls, key=lambda b: b[1])
            ball = (float((bx[0] + bx[2]) / 2 / w),
                    float((bx[1] + bx[3]) / 2 / hgt), float(bconf))
        samples.append((float(t), descs, foot_xs, ball))
        descs_all.extend(descs)
        n_frames += 1  # noqa: SIM113 — t comes from the generator, not the index
        if n_frames % 30 == 0:
            log(f"teams @{t:.0f}s")
            if progress_file is not None:
                progress_file.write_text(
                    json.dumps({"t": round(float(t), 3), "frames": n_frames}))

    if len(descs_all) < MIN_DESCRIPTORS:
        raise RuntimeError("too few players detected to identify teams")

    cl = cluster_teams(descs_all)
    labels = cl["labels"]
    di = 0
    per_frame = []        # (t, teamA_xs, teamB_xs, ball)
    for t, descs, foot_xs, ball in samples:
        ta, tb = [], []
        for j in range(len(descs)):
            lab = int(labels[di]); di += 1
            if lab == 0:
                ta.append(foot_xs[j])
            elif lab == 1:
                tb.append(foot_xs[j])
        per_frame.append((t, ta, tb, ball))

    # aggregate: one row per shared second; when 2+ frames share a second
    # keep the one with more persons
    best: dict[int, tuple] = {}
    for t, ta, tb, ball in per_frame:
        sec = int(t + shared_offset)
        cur = best.get(sec)
        if cur is None or len(ta) + len(tb) > len(cur[0]) + len(cur[1]):
            best[sec] = (ta, tb, ball)
    rows = []
    for sec in sorted(best):
        ta, tb, ball = best[sec]
        rows.append([sec, round(ball[0], 3), round(ball[1], 3),
                     round(ball[2], 3),
                     [round(x, 3) for x in ta],
                     [round(x, 3) for x in tb],
                     len(ta), len(tb)])

    teamA = _team_info(cl["centroids"][0])
    teamB = _team_info(cl["centroids"][1])
    teamA["name"] = colour_name(teamA["hsv"])
    teamB["name"] = colour_name(teamB["hsv"])
    if teamA["name"] == teamB["name"]:
        if teamA["hsv"][2] >= teamB["hsv"][2]:
            teamA["name"] += " (light)"
            teamB["name"] += " (dark)"
        else:
            teamA["name"] += " (dark)"
            teamB["name"] += " (light)"

    out = {
        "fps": fps, "columns": COLS, "rows": rows,
        "teams": {"A": teamA, "B": teamB},
        "team_confidence": cl["confidence"],
        "n_descriptors": len(descs_all),
        "n_outliers": int(cl["outlier_mask"].sum()),
        "model": model_name, "imgsz": imgsz,
        "video": str(video) if video else None,
        "window_file": [round(window_file[0], 3), round(window_file[1], 3)],
        "window_shared": [round(window_file[0] + shared_offset, 3),
                          round(window_file[1] + shared_offset, 3)],
        "shared_offset": shared_offset,
        "generated_at": round(time.time(), 1),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, out, indent=0)
    log(f"teams: {teamA['name']} vs {teamB['name']} "
        f"(conf {cl['confidence']}) -> {out_path}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.teams")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window-file", nargs=2, type=float, required=True,
                    metavar=("LO", "HI"))
    ap.add_argument("--proxy-offset", type=float, default=0.0)
    ap.add_argument("--shared-offset", type=float, default=0.0)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=FPS)
    args = ap.parse_args(argv)
    run_teams_pass(args.video, args.out,
                   window_file=(args.window_file[0], args.window_file[1]),
                   proxy_offset=args.proxy_offset,
                   shared_offset=args.shared_offset,
                   model_path=args.model, fps=args.fps, imgsz=args.imgsz)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
