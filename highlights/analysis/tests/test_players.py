"""Players pass tests: synthetic frames + injected tracker."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from highlights.analysis.players import (
    assign_team,
    default_roster,
    players_stats,
    run_players_pass,
    strip_for_api,
    validate_roster,
)

GREEN = (60, 160, 40)          # BGR grass
ORANGE = (0, 100, 255)         # BGR
WHITE = (235, 235, 235)
DARK = (30, 30, 40)

TEAMS = {"teams": {"A": {"hsv": [10, 220, 230]},
                   "B": {"hsv": [0, 25, 235]}}}


def _box(x, y, w=24, h=60):
    return np.array([x, y, x + w, y + h], dtype=float)


def _paint(f, box, colour):
    x1, y1, x2, y2 = box.astype(int)
    f[y1:y1 + 27, x1:x2] = colour                 # torso
    f[y1 + 27:y2, x1:x2] = DARK                   # legs


def _script():
    """40 frames at 2 fps (20 s): tid 1 orange moves x 30->200, tid 2
    white static, tid 3 orange present for 3 s only (dropped)."""
    frames, dets = [], []
    for i in range(40):
        t = i * 0.5
        f = np.full((240, 320, 3), GREEN, dtype=np.uint8)
        b1 = _box(30 + i * (170 / 39), 60)
        b2 = _box(250, 70)
        _paint(f, b1, ORANGE)
        _paint(f, b2, WHITE)
        det = [(1, b1), (2, b2)]
        if i < 6:                                 # tid 3: 0..2.5 s
            b3 = _box(140, 140)
            _paint(f, b3, ORANGE)
            det.append((3, b3))
        frames.append((t, f))
        dets.append(det)
    return frames, dets


def _fake_tracker(dets):
    it = iter(dets)
    return lambda frame: next(it, [])


def test_assign_team():
    orange = np.array([10, 220, 230, 0.8])
    white = np.array([0, 25, 235, 0.0])
    assert assign_team(orange, TEAMS) == "A"
    assert assign_team(white, TEAMS) == "B"
    # referee yellow: hue far from both centroids -> None
    yellow = np.array([30, 240, 240, 0.9])
    assert assign_team(yellow, TEAMS) is None
    assert assign_team(orange, {"teams": {}}) is None


def test_run_players_pass(tmp_path):
    frames, dets = _script()
    doc = run_players_pass(
        None, tmp_path,
        window_file=(0.0, 20.0), shared_offset=0.0,
        teams=TEAMS, frames=iter(frames), tracker=_fake_tracker(dets),
        log=lambda m: None)
    trs = {t["id"]: t for t in doc["tracklets"]}
    assert set(trs) == {1, 2}                    # tid 3 dropped (<6 s)
    assert trs[1]["team"] == "A" and trs[2]["team"] == "B"
    assert trs[1]["distance_m"] > 20             # 170/320 * ~100 m
    assert trs[2]["distance_m"] < 2
    assert trs[1]["path"] and trs[1]["n_samples"] == 40
    # crops: 3 per kept tracklet, height <= 160
    import cv2
    for tid in (1, 2):
        assert trs[tid]["crops"] == [f"{tid}_{i}.jpg" for i in range(3)]
        for name in trs[tid]["crops"]:
            img = cv2.imread(str(tmp_path / "crops" / name))
            assert img is not None and img.shape[0] <= 160
    # dropped tracklet leaves no crops
    assert not (tmp_path / "crops" / "3_0.jpg").exists()
    stripped = strip_for_api(doc)
    assert "path" not in stripped["tracklets"][0]
    assert stripped["tracklets"][0]["duration_s"] > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_run_players_pass_mp4(tmp_path):
    """Real mp4 via cv2.VideoWriter; fake tracker finds the rects by
    colour — proves the _frame_reader decode path."""
    import cv2
    video = tmp_path / "synthetic.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                         10, (320, 240))
    for i in range(100):                          # 10 s
        f = np.full((240, 320, 3), GREEN, dtype=np.uint8)
        _paint(f, _box(30 + i * 1.5, 60), ORANGE)
        vw.write(f)
    vw.release()

    def tracker(frame):
        # 960px-wide proxy frame: find the orange pixels' bounding box
        mask = ((frame[:, :, 2] > 200) & (frame[:, :, 1] < 160) &
                (frame[:, :, 0] < 60))
        ys, xs = np.where(mask)
        if xs.size < 50:
            return []
        return [(1, np.array([xs.min(), ys.min(), xs.max(), ys.max()],
                            dtype=float))]

    doc = run_players_pass(
        str(video), tmp_path / "out",
        window_file=(0.0, 10.0), shared_offset=0.0,
        teams=TEAMS, tracker=tracker, log=lambda m: None)
    assert len(doc["tracklets"]) == 1
    tr = doc["tracklets"][0]
    assert tr["team"] == "A" and tr["distance_m"] > 5
    assert tr["t_end"] - tr["t_start"] >= 9.0


def test_validate_roster():
    tids, cids = {1, 2, 3}, {"c1", "c2"}
    good = {"players": [{"id": "p1", "name": "Nine", "team": "A",
                         "tracklet_ids": [1, 2]},
                        {"id": "p2", "name": "Ten", "team": "B",
                         "tracklet_ids": [3]}],
            "scorers": {"c1": "p1"}}
    out = validate_roster(good, tids, cids)
    assert out["players"][0]["tracklet_ids"] == [1, 2]
    assert out["scorers"] == {"c1": "p1"}
    assert validate_roster(default_roster(), tids, cids) == default_roster()
    with pytest.raises(ValueError, match="object"):
        validate_roster(["x"], tids, cids)
    with pytest.raises(ValueError, match="duplicate"):
        validate_roster({"players": [dict(good["players"][0]),
                                     dict(good["players"][0])]},
                        tids, cids)
    with pytest.raises(ValueError, match="unknown tracklet"):
        validate_roster({"players": [{"id": "p", "name": "x",
                                      "tracklet_ids": [99]}]}, tids, cids)
    with pytest.raises(ValueError, match="two players"):
        validate_roster({"players": [
            {"id": "a", "name": "x", "tracklet_ids": [1]},
            {"id": "b", "name": "y", "tracklet_ids": [1]}]}, tids, cids)
    with pytest.raises(ValueError, match="unknown candidate"):
        validate_roster({"players": [], "scorers": {"nope": "p1"}},
                        tids, cids)
    with pytest.raises(ValueError, match="known player"):
        validate_roster({"players": [{"id": "p", "name": "x"}],
                         "scorers": {"c1": "ghost"}}, tids, cids)


def _tr(tid, team, t0, t1, dist, sprints=0):
    return {"id": tid, "team": team, "t_start": t0, "t_end": t1,
            "n_samples": 10, "distance_m": dist, "sprints": sprints,
            "crops": [], "path": []}


def test_players_stats():
    tracklets = [_tr(1, "A", 0, 10, 50.0, 2), _tr(2, "A", 10, 18, 30.0),
                 _tr(3, "B", 0, 12, 80.0, 1), _tr(4, None, 0, 8, 5.0)]
    candidates = [{"id": "c1", "status": "confirmed", "type": "goal"},
                  {"id": "c2", "status": "pending", "type": "goal"}]
    roster = {"players": [{"id": "p1", "name": "Nine", "team": "A",
                           "tracklet_ids": [1, 2]},
                          {"id": "p2", "name": "Ten", "team": "B",
                           "tracklet_ids": [3]}],
              "scorers": {"c1": "p1", "c2": "p2"}}
    st = players_stats(roster, tracklets, candidates, TEAMS)
    by_id = {p["id"]: p for p in st["players"]}
    assert by_id["p1"]["goals"] == 1             # only confirmed count
    assert by_id["p2"]["goals"] == 0
    assert by_id["p1"]["n_tracklets"] == 2
    assert by_id["p1"]["tracked_s"] == 18.0
    assert by_id["p1"]["distance_m"] == 80
    assert st["teams"]["A"]["goals"] == 1
    assert st["teams"]["A"]["sprints"] == 2
    assert st["teams"]["B"]["distance_m"] == 80
    assert st["unassigned"]["n_tracklets"] == 1
    assert st["unassigned"]["tracked_s"] == 8.0
    assert [p["team"] for p in st["players"]] == ["A", "B"]
    assert "not official" in st["caveat"]


def test_tracklet_ids_unique_when_tracker_reuses_ids(tmp_path):
    """ByteTrack reuses a tid after the gap: two finished tracklets must
    still get distinct ids (crops and roster keys would collide)."""
    frames, dets = [], []
    for i in range(60):                     # 30 s at 2 fps
        t = i * 0.5
        f = np.full((240, 320, 3), GREEN, dtype=np.uint8)
        det = []
        if i < 20 or i >= 40:               # tid 1: 0-10s and 20-30s
            b = _box(30 + i, 60)
            _paint(f, b, ORANGE)
            det.append((1, b))
        frames.append((t, f))
        dets.append(det)
    doc = run_players_pass(
        None, tmp_path,
        window_file=(0.0, 30.0), shared_offset=0.0,
        teams=TEAMS, frames=iter(frames), tracker=_fake_tracker(dets),
        log=lambda m: None)
    ids = [t["id"] for t in doc["tracklets"]]
    assert len(ids) == len(set(ids)) == 2   # same tracker id, two tracklets
    assert all(t["track_id"] == 1 for t in doc["tracklets"])
    # crops named by the fresh ids, not the recycled tracker id
    names = sorted(p.name for p in (tmp_path / "crops").glob("*.jpg"))
    assert names == sorted(
        f"{t['id']}_{i}.jpg" for t in doc["tracklets"] for i in range(3))
