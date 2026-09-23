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
    path = [(1250 + 12 * k, 400) for k in range(40)]  # x 1250->1718 crosses mouth (1400-1625) then net
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
    # fast track at y=800 (below mouth y<570), passes x=1400-1625 region, within 150px? 800-570=230 > 150
    path = [(1200 + 25 * k, 800) for k in range(20)]
    for k, (x, y) in enumerate(path):
        obs[k] = [(x, y, "blob")]
    cands = neargoal_candidates(_chunks(obs), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands) == 0 or all(c.type == "chance" for c in cands)
    # now closer: y=700 -> dist 130 < 150 -> chance
    obs2 = {k: [(1200 + 25 * k, 700, "blob")] for k in range(20)}
    cands2 = neargoal_candidates(_chunks(obs2), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands2) == 1 and cands2[0].type == "chance"


def test_slow_track_nothing():
    cfg = _cfg()
    obs = {k: [(1500 + k, 400, "blob")] for k in range(20)}  # 30px/s crawl
    cands = neargoal_candidates(_chunks(obs), np.zeros(20000), 0.5, cfg, 0.0, 600.0)
    assert len(cands) == 0
