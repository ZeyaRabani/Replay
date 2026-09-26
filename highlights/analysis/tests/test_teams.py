"""Teams pass tests: synthetic frames + injected detector."""

from __future__ import annotations

import numpy as np

from highlights.analysis.teams import (
    cluster_teams,
    colour_name,
    run_teams_pass,
    torso_descriptor,
)

GREEN = (60, 160, 40)          # BGR grass
ORANGE = (0, 100, 255)         # BGR
WHITE = (235, 235, 235)
YELLOW = (0, 220, 220)
DARK = (30, 30, 40)


def _box(x, y, w=24, h=60):
    return np.array([x, y, x + w, y + h], dtype=float)


def _frame(shift=0):
    """320x240 pitch: 3 orange + 3 white players, 1 yellow ref, 1 ball."""
    f = np.full((240, 320, 3), GREEN, dtype=np.uint8)
    boxes = []
    xs_o = [30, 60, 95]
    xs_w = [200, 240, 280]
    for i, x in enumerate(xs_o):
        y = 60 + i * 40
        b = _box(x + shift, y)
        x1, y1, x2, y2 = b.astype(int)
        f[y1:y1 + 27, x1:x2] = ORANGE           # torso
        f[y1 + 27:y2, x1:x2] = DARK             # legs
        boxes.append(b)
    for i, x in enumerate(xs_w):
        y = 70 + i * 40
        b = _box(x, y)
        x1, y1, x2, y2 = b.astype(int)
        f[y1:y1 + 27, x1:x2] = WHITE
        f[y1 + 27:y2, x1:x2] = DARK
        boxes.append(b)
    b = _box(150, 150)                            # referee
    x1, y1, x2, y2 = b.astype(int)
    f[y1:y2, x1:x2] = YELLOW
    boxes.append(b)
    f[200:206, 160:166] = (255, 255, 255)         # ball
    ball = (np.array([160, 200, 166, 206], dtype=float), 0.9)
    return f, boxes, ball


def _detector(f_and_boxes):
    def det(frame):
        return list(f_and_boxes[1]), [f_and_boxes[2]]
    return det


def test_colour_name():
    assert colour_name([0, 200, 200]) == "red"
    assert colour_name([10, 200, 200]) == "orange"   # 20 deg
    assert colour_name([60, 200, 200]) == "green"    # 120 deg
    assert colour_name([105, 200, 200]) == "blue"    # 210 deg
    assert colour_name([0, 30, 200]) == "white"
    assert colour_name([0, 30, 40]) == "black"
    assert colour_name([0, 30, 120]) == "grey"
    assert colour_name([165, 200, 200]) == "pink"    # 330 deg


def test_torso_descriptor():
    f, boxes, _ = _frame()
    d = torso_descriptor(f, boxes[0])
    assert d is not None
    assert 5 < d[0] < 25            # orange hue ~15-20 deg -> H ~8-12
    assert d[1] > 100 and d[3] > 0.5
    d = torso_descriptor(f, boxes[3])
    assert d is not None and d[1] < 60    # white -> low saturation
    assert torso_descriptor(f, np.array([0, 0, 3, 3])) is None


def test_run_teams_pass(tmp_path):
    frames = []
    for i in range(30):
        f, boxes, ball = _frame(shift=0)
        frames.append((float(i), f))
    det = _detector((None, boxes, ball))
    out = run_teams_pass(None, tmp_path / "teams.json",
                         window_file=(0.0, 30.0), shared_offset=0.0,
                         frames=iter(frames), detector=det, log=lambda m: None)
    names = {out["teams"]["A"]["name"], out["teams"]["B"]["name"]}
    assert names == {"orange", "white"}
    assert out["team_confidence"] >= 0.6
    assert len(out["rows"]) == 30
    assert out["n_outliers"] >= 1
    # referee is yellow — should land in the outlier set for most frames
    assert out["rows"][0][7] + out["rows"][0][6] <= 7


def test_cluster_teams_splits():
    rng = np.random.default_rng(0)
    orange = np.tile(np.array([10, 220, 230, 0.8]), (20, 1)) + rng.normal(0, 3, (20, 4))
    white = np.tile(np.array([0, 25, 235, 0.0]), (20, 1)) + rng.normal(0, 3, (20, 4))
    descs = [np.array(r) for r in np.vstack([orange, white])]
    cl = cluster_teams(descs)
    assert cl["confidence"] >= 0.5
    labs = cl["labels"]
    assert len(set(labs[:20]) - {-1}) == 1
    assert len(set(labs[20:]) - {-1}) == 1
