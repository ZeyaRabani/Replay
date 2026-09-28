"""Learned zones from you-direct sessions."""
import numpy as np

from highlights.multiangle.learned_zones import GRID_H, GRID_W, _rects, learn_zones


def _track(ball_cell, T, conf=1.0, feet=None):
    """Track with ball parked at cell (r, c) centre for T seconds."""
    r, c = ball_cell
    return {
        "ball_conf": np.full(T, conf),
        "ball_x": np.full(T, (c + 0.5) / GRID_W),
        "ball_y": np.full(T, (r + 0.5) / GRID_H),
        "players_xy": feet if feet is not None else [[]] * T,
    }


def _sess(t0, t1, angle):
    return {"t_start": t0, "t_end": t1,
            "choices": [{"t": t0, "angle": angle}]}


def test_ball_votes_learn_cell():
    """3 s of choosing camera 1 while its ball sits in cell (2,3)
    must learn that cell for camera 1 only."""
    T, lo = 40, 0
    avail = np.ones((3, T), dtype=bool)
    tracks = [_track((0, 0), T), _track((2, 3), T), _track((4, 7), T)]
    sess = [_sess(5, 8, 1)]
    zd = learn_zones(sess, tracks, avail, lo, [0.0] * 3, [300.0] * 3)
    assert zd is not None and zd["learned"]
    polys = [kf["zones"] for kf in zd["angles"][1]]
    # one rect = the cell (row 2, col 3)
    p = polys[0][0]
    xs = sorted({pt[0] for pt in p}); ys = sorted({pt[1] for pt in p})
    assert abs(xs[0] - 3 / GRID_W) < 1e-6 and abs(xs[1] - 4 / GRID_W) < 1e-6
    assert abs(ys[0] - 2 / GRID_H) < 1e-6 and abs(ys[1] - 3 / GRID_H) < 1e-6
    assert all(not zd["angles"][i][0]["zones"] for i in (0, 2))
    assert zd["cells"][1][2][3] == 1.0


def test_below_min_votes_and_share_learns_nothing():
    """One second of votes (< MIN_VOTES) or split votes (< MIN_SHARE)
    produce no zones."""
    T = 40
    avail = np.ones((2, T), dtype=bool)
    tracks = [_track((1, 1), T), _track((2, 2), T)]
    # 1 s for camera 0 -> below MIN_VOTES
    zd = learn_zones([_sess(5, 6, 0)], tracks, avail, 0,
                     [0.0] * 2, [300.0] * 2)
    assert zd is None or not any(zd["angles"][0][0]["zones"])
    # split: user alternates 0/1 while ball sits in camera 0's cell ->
    # share for camera 0 = 50% < 60%
    sess = [{"t_start": 5, "t_end": 9,
             "choices": [{"t": 5, "angle": 0}, {"t": 6, "angle": 1},
                         {"t": 7, "angle": 0}, {"t": 8, "angle": 1}]}]
    zd = learn_zones(sess, tracks, avail, 0, [0.0] * 2, [300.0] * 2)
    assert zd is None


def test_density_votes_share_of_feet():
    """players_xy adds DENS_W/feet per foot, so a cell's weight is the
    share of feet standing in it."""
    T = 40
    avail = np.ones((2, T), dtype=bool)
    tr0 = _track((0, 0), T, conf=0.0)  # no ball votes
    tr0["players_xy"] = [
        [[0.3, 0.3], [0.301, 0.301], [0.7, 0.9]]  # cells (1,2) x2, (4,5)
    ] * T
    zd = learn_zones([_sess(5, 9, 0)], [tr0, _track((0, 0), T)], avail,
                     0, [0.0] * 2, [300.0] * 2)
    # cell (1,2): 4 s * 2/3 = 2.67 >= MIN_VOTES; cell (4,5): 1.33 < MIN_VOTES
    assert zd is not None
    assert len(zd["angles"][0][0]["zones"]) == 1
    assert zd["cells"][0][1][2] == 1.0 and zd["cells"][0][4][5] == 1.0
    # a single second is never enough evidence
    zd = learn_zones([_sess(5, 6, 0)], [tr0, _track((0, 0), T)], avail,
                     0, [0.0] * 2, [300.0] * 2)
    assert zd is None


def test_rects_merge_runs_and_columns():
    cells = {(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (1, 6)}
    polys = _rects(cells)
    assert len(polys) == 2  # 3x2 block + lone (1,6)
    big = [p for p in polys if len({pt[0] for pt in p}) == 2]
    # the 3-wide x 2-tall block covers x in [1/8, 4/8], y in [0, 2/5]
    bp = next(p for p in polys
              if abs(max(pt[0] for pt in p) - min(pt[0] for pt in p))
              > 0.3)
    assert abs(min(pt[0] for pt in bp) - 1 / 8) < 1e-9
    assert abs(max(pt[0] for pt in bp) - 4 / 8) < 1e-9
    assert abs(min(pt[1] for pt in bp) - 0.0) < 1e-9
    assert abs(max(pt[1] for pt in bp) - 2 / 5) < 1e-9
    assert len(big) == 2


def test_keyframes_convert_to_file_t():
    """One keyframe per session, t in file seconds = t_start - offset
    clipped into [0, duration], sorted by t."""
    T = 400
    avail = np.ones((2, T), dtype=bool)
    tracks = [_track((0, 0), T), _track((2, 3), T)]
    sess = [_sess(50, 54, 1), _sess(200, 204, 1)]
    zd = learn_zones(sess, tracks, avail, 0, [0.0, 30.0], [600.0] * 2)
    ts = [kf["t"] for kf in zd["angles"][1]]
    assert ts == [20.0, 170.0]
    # same zones on every keyframe
    zs = [kf["zones"] for kf in zd["angles"][1]]
    assert zs[0] == zs[1]


def test_off_camera_seconds_do_not_vote():
    T = 40
    avail = np.ones((2, T), dtype=bool)
    avail[1, 10:30] = False  # camera 1 unavailable mid-session
    tracks = [_track((0, 0), T), _track((2, 3), T)]
    zd = learn_zones([_sess(5, 40, 1)], tracks, avail, 0,
                     [0.0] * 2, [300.0] * 2)
    # only seconds 30..39 voted for camera 1 -> 10 s, still learns cell
    assert zd["angles"][1][0]["zones"]
