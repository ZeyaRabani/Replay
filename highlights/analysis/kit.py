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
from pathlib import Path

import cv2
import numpy as np

from highlights.io import write_json_atomic

MIN_SAT = 60  # teams.json hsv[1] below this = white/black kit, can't vote
MIN_FRAC = 0.12  # per-crop: this share of pixels must match a kit hue
DOM_RATIO = 2.0  # per-crop: winner must beat the loser by this factor
MAJ_SHARE = 0.6  # majority-vote share needed to assign a track team


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
        fa, fb = kit_fracs(img, hue_a, hue_b)
        if fa > MIN_FRAC and fa > DOM_RATIO * fb:
            votes.append("A")
        elif fb > MIN_FRAC and fb > DOM_RATIO * fa:
            votes.append("B")
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
    kits = (teams or {}).get("teams") or {}
    ha = (kits.get("A") or {}).get("hsv")
    hb = (kits.get("B") or {}).get("hsv")
    if (not ha or not hb or len(ha) < 2 or len(hb) < 2
            or ha[1] < MIN_SAT or hb[1] < MIN_SAT):
        log("kit relabel: skipped (kits not both saturated colours)")
        return {"changed": 0, "unclassified": 0, "total": 0}

    tracks_path = v2_dir / "tracks.json"
    doc = json.loads(tracks_path.read_text())
    tracks = doc.get("tracks") or []
    crops_dir = v2_dir / "crops"
    changed = unclassified = 0
    for tr in tracks:
        paths = [crops_dir / c for c in tr.get("crops") or []]
        team, conf = classify_crops(paths, ha[0], hb[0])
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
