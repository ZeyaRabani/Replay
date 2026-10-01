"""Kit-colour relabelling for fused v2 tracks.

detect_hr's per-detection team label comes from the median hue inside the
torso box; grass pixels leak in and flip ~40% of detections on some
pitches. The crops saved during fusion are tighter, so each track can be
re-voted from its own crops: count the fraction of saturated pixels near
each team's kit hue in the torso band of every crop and take a majority
vote. Tracks with no confident colour majority get team=None and fall
into the identities' unassigned bucket.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from highlights.io import write_json_atomic

MIN_SAT = 60  # teams.json hsv[1] below this = white/black kit, can't vote
MIN_FRAC = 0.12  # per-crop: this share of pixels must match a kit hue
DOM_RATIO = 2.0  # per-crop: winner must beat the loser by this factor
MAJ_SHARE = 0.6  # majority-vote share needed to assign a track team


def kit_hues(teams: dict) -> tuple[float, float] | None:
    """(hue_a, hue_b) when both kits are saturated colours, else None."""
    kits = (teams or {}).get("teams") or {}
    ha = (kits.get("A") or {}).get("hsv")
    hb = (kits.get("B") or {}).get("hsv")
    if (not ha or not hb or len(ha) < 2 or len(hb) < 2
            or ha[1] < MIN_SAT or hb[1] < MIN_SAT):
        return None
    return ha[0], hb[0]


def classify_fracs(fa: float, fb: float) -> str:
    """Kit vote for one crop's colour fractions: 'A', 'B' or '' (no call)."""
    if fa > MIN_FRAC and fa > DOM_RATIO * fb:
        return "A"
    if fb > MIN_FRAC and fb > DOM_RATIO * fa:
        return "B"
    return ""


def kit_fracs(img_bgr: np.ndarray, hue_a: float, hue_b: float,
              half: int = 10) -> tuple[float, float]:
    """Fraction of torso-band pixels matching each kit hue.

    hue_a/hue_b are OpenCV hues (0..179) from teams.json teams[A|B].hsv[0].
    Returns (frac_a, frac_b) over all torso-band pixels (saturated or not).
    """
    h, w = img_bgr.shape[:2]
    tor = img_bgr[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)]
    hsv = cv2.cvtColor(tor, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.int32), hsv[..., 1], hsv[..., 2]
    sat = (S > 70) & (V > 70)
    n = max(1, tor.shape[0] * tor.shape[1])

    def _frac(hue: float) -> float:
        d = np.abs(H - round(hue))
        d = np.minimum(d, 180 - d)  # hue wraps at 0/179
        return float(((d <= half) & sat).sum() / n)

    return _frac(hue_a), _frac(hue_b)


def classify_crops(paths: list[Path], hue_a: float,
                   hue_b: float) -> tuple[str | None, float]:
    """Majority-vote a track's team from its saved crops.

    Each crop votes 'A', 'B', or abstains. Returns (team, confidence)
    where confidence is majority share * (voting crops / total crops);
    team is None when fewer than MAJ_SHARE of votes agree.
    """
    votes: list[str] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        v = classify_fracs(*kit_fracs(img, hue_a, hue_b))
        if v:
            votes.append(v)
    if not votes:
        return None, 0.0
    top = max(votes.count("A"), votes.count("B"))
    share = top / len(votes)
    conf = share * (len(votes) / max(1, len(paths)))
    if share < MAJ_SHARE:
        return None, conf
    return ("A" if votes.count("A") >= votes.count("B") else "B"), conf


def relabel_tracks(v2_dir: Path, teams: dict, log=print) -> dict:
    """Rewrite tracks.json with kit-voted team labels.

    Keeps the detector's original label in "team_det" (set once, never
    overwritten on re-run), sets "team" to the crop-voted label (None
    when unclassifiable) and "kit_conf" to the vote confidence.
    Skipped entirely when either kit is unsaturated (white/black).
    Returns {changed, unclassified, total}.
    """
    hues = kit_hues(teams)
    if hues is None:
        log("kit relabel: skipped (kits not both saturated colours)")
        return {"changed": 0, "unclassified": 0, "total": 0}
    hue_a, hue_b = hues

    tracks_path = v2_dir / "tracks.json"
    doc = json.loads(tracks_path.read_text())
    tracks = doc.get("tracks") or []
    crops_dir = v2_dir / "crops"
    changed = unclassified = 0
    for tr in tracks:
        paths = [crops_dir / c for c in tr.get("crops") or []]
        team, conf = classify_crops(paths, hue_a, hue_b)
        if "team_det" not in tr:
            tr["team_det"] = tr.get("team")
        if tr.get("team") != team:
            changed += 1
        if team is None:
            unclassified += 1
        tr["team"] = team
        tr["kit_conf"] = round(conf, 3)
    write_json_atomic(tracks_path, doc, indent=1)
    log(f"kit relabel: {changed}/{len(tracks)} tracks relabelled, "
        f"{unclassified} unclassified")
    return {"changed": changed, "unclassified": unclassified,
            "total": len(tracks)}


def relabel_detections(video: str | Path, npz_path: Path, teams: dict,
                       *, fps: float, start_s: float = 0.0,
                       end_s: float | None = None, frames=None,
                       log=print) -> dict:
    """Re-label a saved det_a*.npz from the source video, decode only.

    Reads the same frames detect_angle did (_frame_reader at `fps`),
    matches each yielded t to the npz rows within half a frame interval,
    crops every box, and overwrites `team` with the kit-colour vote.
    Original labels are kept once in `team_det` (never overwritten on a
    re-run). Atomic write, same .tmp.npz dance as detect_hr._flush.
    Returns {n_dets, changed, blank}; skipped when kits aren't both
    saturated colours.
    """
    hues = kit_hues(teams)
    if hues is None:
        log("kit relabel-dets: skipped (kits not both saturated colours)")
        return {"n_dets": 0, "changed": 0, "blank": 0}
    hue_a, hue_b = hues

    z = np.load(npz_path, allow_pickle=False)
    if "t" not in z or not len(z["t"]):
        log(f"kit relabel-dets: {npz_path.name} empty — skipped")
        return {"n_dets": 0, "changed": 0, "blank": 0}
    rows_t = z["t"]
    team = z["team"].tolist()
    team_det = (z["team_det"].tolist() if "team_det" in z
                else list(team))

    if frames is None:
        from highlights.multiangle.trackfeat import _frame_reader, _probe_dims
        w, _h = _probe_dims(str(video))
        frames = _frame_reader(str(video), fps, w, start_s, end_s)

    tol = 0.5 / fps
    n_changed = n_blank = 0
    for t, frame in frames:
        idx = np.nonzero(np.abs(rows_t - t) <= tol)[0]
        if not len(idx):
            continue
        h, w = frame.shape[:2]
        for i in idx:
            x1 = int(np.clip(z["x1"][i], 0, w - 1))
            y1 = int(np.clip(z["y1"][i], 0, h - 1))
            x2 = int(np.clip(z["x2"][i], x1 + 1, w))
            y2 = int(np.clip(z["y2"][i], y1 + 1, h))
            v = classify_fracs(*kit_fracs(frame[y1:y2, x1:x2],
                                          hue_a, hue_b))
            if v != team[i]:
                n_changed += 1
            if not v:
                n_blank += 1
            team[i] = v
    tmp = npz_path.with_suffix(".tmp.npz")
    np.savez(tmp, t=rows_t, x1=z["x1"], y1=z["y1"], x2=z["x2"],
             y2=z["y2"], conf=z["conf"], team=np.asarray(team),
             team_det=np.asarray(team_det))
    os.replace(tmp, npz_path)
    n = len(rows_t)
    log(f"kit relabel-dets: {npz_path.name} {n_changed}/{n} changed, "
        f"{n_blank} blank")
    return {"n_dets": n, "changed": n_changed, "blank": n_blank}
