"""kit relabel: fused tracks re-vote their team from saved crops."""

import json

import cv2
import numpy as np

from highlights.analysis.kit import classify_crops, kit_fracs, relabel_tracks

# OpenCV hues: orange ~11, yellow-green ~36, grass green ~60
HUE_A, HUE_B = 11.0, 36.0


def _crop(path, torso_bgr, bg_bgr=(40, 110, 50)):
    """160x80 BGR image: grass-green background, coloured torso band."""
    img = np.full((160, 80, 3), bg_bgr, dtype=np.uint8)
    img[24:88, 16:64] = torso_bgr
    cv2.imwrite(str(path), img)


def _tracks_doc(ids_teams):
    return {"step": 0.5, "t0": 0.0, "tracks": [
        {"id": tid, "team": team, "start": 0.0, "end": 1.0,
         "xy": [[0, 0]], "dist_m": 0.0, "sprints": 0,
         "crops": [f"c{tid}.jpg"]} for tid, team in ids_teams]}


def test_kit_fracs_picks_torso_hue(tmp_path):
    p = tmp_path / "a.jpg"
    _crop(p, (30, 100, 220))  # orange torso (hue ~11)
    fa, fb = kit_fracs(cv2.imread(str(p)), HUE_A, HUE_B)
    assert fa > 0.5 and fb < fa / 4


def test_classify_crops_orange_is_A(tmp_path):
    p = tmp_path / "o.jpg"
    _crop(p, (30, 100, 220))  # orange
    assert classify_crops([p], HUE_A, HUE_B)[0] == "A"


def test_classify_crops_yellowgreen_is_B(tmp_path):
    p = tmp_path / "y.jpg"
    _crop(p, (30, 220, 182))  # yellow-green kit (hue ~36)
    assert classify_crops([p], HUE_A, HUE_B)[0] == "B"


def test_classify_crops_grass_is_None(tmp_path):
    p = tmp_path / "g.jpg"
    # no coloured torso at all — solid grass
    img = np.full((160, 80, 3), (40, 110, 50), dtype=np.uint8)
    cv2.imwrite(str(p), img)
    team, conf = classify_crops([p], HUE_A, HUE_B)
    assert team is None and conf == 0.0


def test_classify_crops_majority_two_of_three(tmp_path):
    a1, a2, b1 = (tmp_path / n for n in ("a1.jpg", "a2.jpg", "b1.jpg"))
    _crop(a1, (30, 100, 220))
    _crop(a2, (30, 100, 220))
    _crop(b1, (30, 220, 182))
    team, conf = classify_crops([a1, a2, b1], HUE_A, HUE_B)
    assert team == "A" and 0.0 < conf < 1.0


def test_classify_crops_split_vote_is_None(tmp_path):
    a1, b1 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _crop(a1, (30, 100, 220))
    _crop(b1, (30, 220, 182))
    assert classify_crops([a1, b1], HUE_A, HUE_B)[0] is None


def test_relabel_tracks_rewrites_teams(tmp_path):
    v2 = tmp_path / "players_v2"
    (v2 / "crops").mkdir(parents=True)
    _crop(v2 / "crops" / "c1.jpg", (30, 100, 220))  # orange -> A
    _crop(v2 / "crops" / "c2.jpg", (30, 220, 182))  # yel-grn -> B
    img = np.full((160, 80, 3), (40, 110, 50), dtype=np.uint8)
    cv2.imwrite(str(v2 / "crops" / "c3.jpg"), img)   # grass -> None
    doc = _tracks_doc([(1, "B"), (2, "B"), (3, "A")])
    (v2 / "tracks.json").write_text(json.dumps(doc))
    teams = {"teams": {"A": {"hsv": [HUE_A, 150, 160]},
                       "B": {"hsv": [HUE_B, 100, 130]}}}
    out = relabel_tracks(v2, teams, log=lambda _m: None)
    assert out == {"changed": 2, "unclassified": 1, "total": 3}
    tracks = {t["id"]: t for t in
              json.loads((v2 / "tracks.json").read_text())["tracks"]}
    assert tracks[1]["team"] == "A" and tracks[1]["team_det"] == "B"
    assert tracks[2]["team"] == "B" and tracks[2]["team_det"] == "B"
    assert tracks[3]["team"] is None and tracks[3]["team_det"] == "A"
    # re-run keeps the original detector label, not the relabelled one
    relabel_tracks(v2, teams, log=lambda _m: None)
    tracks = {t["id"]: t for t in
              json.loads((v2 / "tracks.json").read_text())["tracks"]}
    assert tracks[1]["team_det"] == "B" and tracks[3]["team_det"] == "A"


def test_relabel_tracks_skips_unsaturated_kit(tmp_path):
    v2 = tmp_path / "players_v2"
    v2.mkdir(parents=True)
    (v2 / "tracks.json").write_text(json.dumps(_tracks_doc([(1, "A")])))
    white = {"teams": {"A": {"hsv": [0, 20, 240]},
                       "B": {"hsv": [HUE_B, 100, 130]}}}
    out = relabel_tracks(v2, white, log=lambda _m: None)
    assert out == {"changed": 0, "unclassified": 0, "total": 0}
    t = json.loads((v2 / "tracks.json").read_text())["tracks"][0]
    assert t["team"] == "A" and "team_det" not in t
