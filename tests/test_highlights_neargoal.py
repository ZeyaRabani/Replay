import numpy as np

from highlights.config import Config
from highlights.neargoal import neargoal_candidates


def _chunks(obs_frames, fps=30.0, blobs_per_frame=None, net_motion=None):
    """obs_frames: {frame_idx: [(x, y, kind)]} -> one chunk dict."""
    ch = {"frame0": 0, "fps": fps, "t0": 0.0, "ball": [], "blobs": [],
          "persons": [], "net_motion": []}
    for fi, obs in obs_frames.items():
        for x, y, kind in obs:
            if kind == "ball":
                ch["ball"].append([fi, x, y, 8, 8, 0.5])
            else:
                ch["blobs"].append([fi, x, y, 50.0])
    for fi in range(max(obs_frames) + 1):
        v = net_motion.get(fi, 0.0) if net_motion else 0.0
        ch["net_motion"].append([fi, v])
        for b in (blobs_per_frame or {}).get(fi, []):
            ch["blobs"].append([fi, b[0], b[1], 40.0])
    return [ch]


def _cfg():
    return Config()


def test_goal_via_mouth_then_net():
    cfg = _cfg()
    obs = {}
    # ball path: outside -> inside mouth -> inside net, ~20px/frame (~600px/s)
    path = [(1250 + 15 * k, 420) for k in range(40)]  # x 1250->1835 crosses mouth then net (y inside both)
    for k, (x, y) in enumerate(path):
        obs[k] = [(x, y, "ball")]
    cands = neargoal_candidates(_chunks(obs), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands) == 1
    c = cands[0]
    assert c.type == "goal" and c.anchor == "neargoal" and c.goal == "A"
    assert c.confidence >= 0.6


def test_fast_pass_beside_mouth_is_chance():
    cfg = _cfg()
    obs = {}
    # fast track at y=850 (below mouth, bottom edge ~y=581-604), >150px away -> nothing
    path = [(1200 + 25 * k, 850) for k in range(20)]
    for k, (x, y) in enumerate(path):
        obs[k] = [(x, y, "blob")]
    cands = neargoal_candidates(_chunks(obs), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands) == 0
    # now closer: y=700 -> nearest mouth vertex ~96-150px -> chance
    obs2 = {k: [(1200 + 25 * k, 700, "blob")] for k in range(20)}
    cands2 = neargoal_candidates(_chunks(obs2), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands2) == 1 and cands2[0].type == "chance"


def test_slow_track_nothing():
    cfg = _cfg()
    obs = {k: [(1500 + k, 400, "blob")] for k in range(20)}  # 30px/s crawl
    cands = neargoal_candidates(_chunks(obs), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands) == 0
