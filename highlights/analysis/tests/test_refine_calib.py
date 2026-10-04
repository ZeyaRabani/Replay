"""refine_calib: joint per-camera H refinement from cross-camera
anchor pairs; effective_h preference; reresolve after renumbering."""

import numpy as np

from highlights.analysis import anchors as anch
from highlights.analysis.calib import _dlt, apply_h, effective_h, landmark_xy
from highlights.analysis.refine_calib import refine

PITCH = {"len_m": 55.0, "wid_m": 23.0, "template": "small",
         "goal_w_m": 4.0, "d_radius_m": 6.0}
L, W = 55.0, 23.0

# plausible normalised image quads for the 4 pitch corners
# (0,0) (L,0) (L,W) (0,W) — three phone viewpoints
QUADS = [
    [[0.55, 0.35], [0.95, 0.42], [0.80, 0.95], [0.30, 0.80]],
    [[0.10, 0.40], [0.60, 0.30], [0.75, 0.85], [0.25, 0.95]],
    [[0.30, 0.25], [0.85, 0.38], [0.70, 0.90], [0.15, 0.70]],
]
CORNERS = np.array([[0, 0], [L, 0], [L, W], [0, W]], float)

LM_NAMES = [n for n in landmark_xy(L, W, PITCH)
            if not n.endswith("_apex")]


def _true_h(i):
    return _dlt(np.array(QUADS[i], float), CORNERS)


def _img_of(H, x, y):
    """pitch metres -> normalised image coords through inv(H)."""
    v = np.linalg.inv(H) @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _scene(rng):
    """calib dict + anchors doc + stabs + dets for 3 cameras."""
    Htrue = [_true_h(i) for i in range(3)]
    lm_xy = landmark_xy(L, W, PITCH)
    angles = {}
    for a in range(3):
        pts = []
        for n in LM_NAMES:
            fx, fy = _img_of(Htrue[a], *lm_xy[n])
            pts.append({"name": n, "fx": fx, "fy": fy})
        angles[str(a)] = {"pts": pts, "H": Htrue[a].tolist(),
                          "rms_m": 0.5}
    # corrupt cameras 1 and 2: swap two landmarks + heavy jitter
    for a in (1, 2):
        pts = angles[str(a)]["pts"]
        for p in pts:
            p["fx"] += float(rng.normal(0, 0.08))
            p["fy"] += float(rng.normal(0, 0.08))
        pts[0]["name"], pts[1]["name"] = pts[1]["name"], pts[0]["name"]
        angles[str(a)]["H"] = _dlt(
            np.array([[p["fx"], p["fy"]] for p in pts]),
            np.array([lm_xy[p["name"]] for p in pts])).tolist()
    # anchors: 3 moments x 8 players, visible in all 3 cameras
    players = np.column_stack([rng.uniform(5, L - 5, 24),
                               rng.uniform(3, W - 3, 24)])
    moments = [{"id": "start", "t": 100.0}, {"id": "mid", "t": 200.0},
               {"id": "end", "t": 300.0}]
    clicks = []
    for m_i, m in enumerate(moments):
        for j in range(8):
            x, y = players[8 * m_i + j]
            for a in range(3):
                fx, fy = _img_of(Htrue[a], x, y)
                clicks.append({"id": f"m{m_i}p{j}a{a}", "moment": m["id"],
                               "angle": a, "fx": fx, "fy": fy,
                               "team": "A", "label": f"p{j}",
                               "box": [fx - 0.01, fy - 0.04, fx + 0.01,
                                       fy]})
    t = np.array([100.0, 200.0, 300.0])
    stabs = {a: {"t": t, "H": np.tile(np.eye(3), (3, 1, 1)),
                 "step_s": 1.0} for a in range(3)}
    ts, feet, confs, teams, boxes = [], [], [], [], []
    for tt in t:
        for x, y in players:
            ts.append(tt)
            feet.append([x / L, y / W])
            confs.append(0.9)
            teams.append("A")
            boxes.append([0, 0, 20, 40])
    dets = {a: {"t": np.array(ts), "foot": np.array(feet),
                "conf": np.array(confs), "team": np.array(teams),
                "box": np.array(boxes), "w": 1.0, "h": 1.0}
            for a in range(3)}
    # dets in IMAGE coords: feet are img pts, not pitch coords
    for a in range(3):
        dets[a]["foot"] = np.array(
            [_img_of(Htrue[a], x, y) for x, y in players] * 3)
        dets[a]["t"] = np.array(sorted(ts))
    calib = {"pitch": PITCH, "angles": angles}
    anchors_doc = {"moments": moments, "clicks": clicks}
    return calib, anchors_doc, stabs, dets, players, Htrue


def test_refine_recovers_cross_camera_agreement():
    rng = np.random.default_rng(11)
    calib, doc, stabs, dets, players, Htrue = _scene(rng)
    out = refine(calib, doc, stabs, [0.0, 0.0, 0.0], dets)
    assert out is not None
    assert out["ref_angle"] == 0
    assert out["pair_median_m_after"] < 1.0
    assert out["pair_median_m_after"] < out["pair_median_m_before"]
    # each camera's refined H recovers true player positions
    errs = []
    for a in range(3):
        for x, y in players:
            fx, fy = _img_of(Htrue[a], x, y)
            px, py = apply_h(out["H"][a], fx, fy)
            errs.append(np.hypot(px - x, py - y))
    assert float(np.median(errs)) < 1.5


def test_refine_needs_eight_pairs():
    calib = {"pitch": PITCH,
             "angles": {"0": {"H": np.eye(3).tolist(), "pts": []}}}
    doc = {"moments": [{"id": "mid", "t": 10.0}],
           "clicks": [{"id": "c", "moment": "mid", "angle": 0,
                       "label": "p", "team": "A",
                       "box": [0.4, 0.5, 0.42, 0.55]}]}
    assert refine(calib, doc, {}, [0.0]) is None


def test_effective_h_prefers_refined():
    assert effective_h({"H": [[1]], "H_refined": [[2]]}) == [[2]]
    assert effective_h({"H": [[1]]}) == [[1]]
    assert effective_h({}) is None


def test_reresolve_updates_track_ids(tmp_path):
    # anchors saved against old track ids get re-resolved to the new
    # nearest track after a re-fuse renumbers them
    v2d = tmp_path / "players_v2"
    v2d.mkdir()
    (tmp_path / "multiangle").mkdir()
    (tmp_path / "multiangle" / "sync.json").write_text(
        '{"offsets": [0.0]}')
    cal = {"angles": {"0": {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}
    n = round(200 / 0.5) + 1
    tracks = {"tracks": [{"id": 200001, "team": "A", "start": 0,
                          "end": 200, "xy": [[40, 10]] * n}]}
    (v2d / "tracks.json").write_text(__import__("json").dumps(tracks))
    anch.save(v2d, {"moments": [{"id": "mid", "t": 100.0}],
                    "clicks": [{"id": "c", "moment": "mid", "angle": 0,
                                "fx": 40.0, "fy": 10.0, "team": "A",
                                "label": "Rui", "track_id": 99999,
                                "xy": [40.0, 10.0]}]})
    out = anch.reresolve(tmp_path, v2d, cal)
    assert out is not None
    assert out["clicks"][0]["track_id"] == 200001
    assert out["clicks"][0]["track_team"] == "A"


def test_rejection_on_degenerate_landmarks():
    from highlights.analysis.refine_calib import rejection_reason
    calib = {"angles": {"0": {"rms_m": 0.7}, "1": {"rms_m": 0.7}}}
    good = {"lm_rms_m": {"0": 0.6, "1": 1.2}, "pair_median_m_before": 12.0,
            "pair_median_m_after": 1.9}
    assert rejection_reason(good, calib) is None
    bad = dict(good, lm_rms_m={"0": 37.5, "1": 0.6})
    assert "a0 landmarks 37.5 m" in rejection_reason(bad, calib)
    worse = dict(good, pair_median_m_after=12.5)
    assert "did not improve" in rejection_reason(worse, calib)
