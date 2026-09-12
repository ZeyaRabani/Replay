"""Headless tests for pitchworld.calib_tool.CalibSession."""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from pitchworld.calib_tool import CalibSession, render_screenshot
from pitchworld.pitch import PitchModel
from pitchworld.posefit import Pose, homography_world_to_pixel

W, H = 1920, 1080
PITCH = PitchModel.load(Path("examples/pitch_small_sided_50x30.json"))
POSE = Pose(x=-8, y=15, z=6, yaw=0, pitch=math.radians(20), roll=0, f=1500)
HWP = homography_world_to_pixel(POSE, W, H)


def proj(xy):
    u, v, w = HWP @ np.array([xy[0], xy[1], 1.0])
    return (u / w, v / w)


def make_session(frame=None, existing=None):
    if frame is None:
        frame = np.zeros((H, W, 3), np.uint8)
    return CalibSession(frame, PITCH, 0, "video.mp4", 1.0, existing)


def test_trace_line_arc_fit():
    s = make_session()
    s.set_mode("line")
    s.select_element(s.element_list.index("goal_line_A"))
    for y in np.linspace(3, 27, 8):
        s.add_point(*proj((0, y)))
    assert s.commit_trace()
    s.set_mode("arc")
    s.select_element(s.element_list.index("A_D"))
    for t in np.radians(np.linspace(-80, 80, 8)):
        s.add_point(*proj((6 * math.cos(t), 15 + 6 * math.sin(t))))
    assert s.commit_trace()
    s.set_mode("line")
    s.select_element(s.element_list.index("touch_S"))
    for x in np.linspace(5, 30, 8):
        s.add_point(*proj((x, 0)))
    assert s.commit_trace()
    assert s.fit is not None, s.fit_error
    assert s.fit.reproj_error_m < 0.3
    assert abs(s.fit.pose["height"] - 6) < 1.0


def test_point_mode_and_undo():
    s = make_session()
    names = ["A_post_S", "A_post_N", "A_D_S", "A_D_N"]
    start_idx = s.elem_idx["point"]
    assert s.element_list[start_idx] == "A_post_S"
    for name in names:
        assert s.current_element == name
        s.add_point(*proj(PITCH.landmarks()[name]))
    assert len(s.points) == 4
    assert s.element_list[s.elem_idx["point"]] == "A_D_apex"
    s.undo()
    assert len(s.points) == 3
    assert s.element_list[s.elem_idx["point"]] == "A_D_N"  # stepped back
    # undo pops trace points first
    s.set_mode("line")
    s.add_point(10, 10)
    s.add_point(20, 20)
    assert len(s.trace) == 2
    s.undo()
    assert len(s.trace) == 1
    s.undo()
    assert len(s.trace) == 0


def test_element_cycling_and_keys():
    s = make_session()
    assert s.handle_key(ord("l")) is None
    assert s.mode == "line"
    first = s.current_element
    s.handle_key(9)  # Tab
    assert s.current_element != first
    s.handle_key(ord("["))
    assert s.current_element == first
    s.handle_key(ord("3"))
    assert s.elem_idx["line"] == 2
    assert s.snap is False
    s.handle_key(ord("n"))
    assert s.snap is True
    assert s.handle_key(13) == "save"  # empty trace -> save
    s.add_point(10, 10)
    s.add_point(20, 20)
    assert s.handle_key(13) is None  # commits the trace
    assert len(s.lines) == 1
    assert s.handle_key(ord("q")) == "quit"


def test_snap_to_mask():
    frame = np.full((H, W, 3), (60, 160, 60), np.uint8)
    a, b = (400, 500), (1200, 700)  # true white line
    cv2.line(frame, a, b, (255, 255, 255), 3)
    s = make_session(frame)
    s.set_mode("line")
    s.toggle_snap()
    # trace offset ~5px perpendicular
    d = np.array([b[0] - a[0], b[1] - a[1]], float)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    for t in np.linspace(0, 1, 8):
        p = np.array(a, float) + t * d + 5 * n
        s.add_point(p[0], p[1])
    assert s.commit_trace()
    px = np.array(s.lines[0]["pixels"])
    assert len(px) >= 3
    dist = np.abs((px - np.array(a)) @ n)
    assert dist.max() < 2.0


def test_save_load_roundtrip(tmp_path):
    out = tmp_path / "calib.json"
    out.write_text(json.dumps({"cameras": {"9": {"points": [{"world": "centre", "pixel": [1, 2]}]}}}))
    s = make_session()
    s.set_mode("line")
    s.select_element(0)
    for y in np.linspace(3, 27, 6):
        s.add_point(*proj((0, y)))
    s.commit_trace()
    s.set_mode("arc")
    s.select_element(0)
    for t in np.radians(np.linspace(-80, 80, 6)):
        s.add_point(*proj((6 * math.cos(t), 15 + 6 * math.sin(t))))
    s.commit_trace()
    s.save(out)
    data = json.loads(out.read_text())
    assert "9" in data["cameras"]  # other cameras preserved
    s2 = make_session(existing=data["cameras"]["0"])
    assert s2.lines == s.lines and s2.arcs == s.arcs
    assert s2.points == s.points and s2.parallels == s.parallels
    # fit state recomputed (dof < 7 here -> no fit, but no crash either)
    assert s2.dof == s.dof


def test_render_headless():
    s = make_session()
    img = render_screenshot(s, 1600, 900)
    assert img.shape == (900, 1600, 3)
    s.set_mode("line")
    for y in np.linspace(3, 27, 8):
        s.add_point(*proj((0, y)))
    s.commit_trace()
    s.set_mode("arc")
    for t in np.radians(np.linspace(-80, 80, 8)):
        s.add_point(*proj((6 * math.cos(t), 15 + 6 * math.sin(t))))
    s.commit_trace()
    s.set_mode("line")
    s.select_element(2)  # touch_S
    for x in np.linspace(5, 30, 8):
        s.add_point(*proj((x, 0)))
    s.commit_trace()
    assert s.fit is not None
    img2 = render_screenshot(s, 1600, 900)
    assert img2.shape == (900, 1600, 3)
