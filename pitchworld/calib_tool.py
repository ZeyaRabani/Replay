"""Interactive landmark clicker (OpenCV window).

    pitchworld calibrate VIDEO --camera 0 --pitch pitch.json --out calib.json
                        [--landmarks A_post_N,A_post_S,A_D_S,A_D_apex] [--time 1.0]

Keys:  click = place point for the current landmark
       u = undo last point        s = skip current landmark
       +/- = zoom (around mouse)  arrows / drag = pan
       enter = save               q/esc = abort
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .calibrate import manual_calibrate, read_frame
from .pitch import PitchModel

DEFAULT_ORDER = [
    "A_post_S", "A_post_N", "A_D_S", "A_D_N", "A_D_apex", "corner_A_S", "corner_A_N",
    "B_post_S", "B_post_N", "B_D_S", "B_D_N", "B_D_apex", "corner_B_S", "corner_B_N",
    "half_S", "half_N", "centre",
]


class _State:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0])
        self.mouse = (0, 0)
        self.clicks: list[tuple[str, tuple[float, float]]] = []
        self.drag_start = None

    def to_view(self, pt):
        return ((np.asarray(pt) - self.pan) * self.zoom).astype(int)

    def to_image(self, pt):
        return tuple((np.asarray(pt, dtype=float) / self.zoom + self.pan).tolist())

    def render(self, current: str, view_w: int, view_h: int) -> np.ndarray:
        M = np.array([[self.zoom, 0, -self.pan[0] * self.zoom], [0, self.zoom, -self.pan[1] * self.zoom]])
        view = cv2.warpAffine(self.frame, M, (view_w, view_h))
        for name, pt in self.clicks:
            p = tuple(self.to_view(pt))
            cv2.drawMarker(view, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(view, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        txt = f"click: {current}   [u]ndo [s]kip [+/-] zoom [enter] save [q]uit   points={len(self.clicks)}"
        cv2.rectangle(view, (0, 0), (view_w, 28), (0, 0, 0), -1)
        cv2.putText(view, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return view


def run_click_tool(video: Path, camera_index: int, pitch: PitchModel, out: Path, landmarks: list[str] | None,
                   frame_time: float = 1.0, view_w: int = 1600, view_h: int = 900) -> None:
    frame = read_frame(video, frame_time)
    order = [l for l in (landmarks or DEFAULT_ORDER) if l in pitch.landmarks()]
    st = _State(frame)
    st.zoom = min(view_w / frame.shape[1], view_h / frame.shape[0])
    idx = 0
    win = f"pitchworld calibrate cam{camera_index}"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, _):
        st.mouse = (x, y)
        nonlocal idx
        if event == cv2.EVENT_LBUTTONDOWN:
            st.drag_start = (x, y, st.pan.copy())
        elif event == cv2.EVENT_LBUTTONUP:
            if (st.drag_start is not None and abs(x - st.drag_start[0]) < 3 and abs(y - st.drag_start[1]) < 3
                    and idx < len(order)):
                st.clicks.append((order[idx], st.to_image((x, y))))
                idx += 1
            st.drag_start = None
        elif event == cv2.EVENT_MOUSEMOVE and st.drag_start is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            sx, sy, pan0 = st.drag_start
            st.pan = pan0 - np.array([x - sx, y - sy]) / st.zoom

    cv2.setMouseCallback(win, on_mouse)
    while True:
        cur = order[idx] if idx < len(order) else "(done - press enter)"
        cv2.imshow(win, st.render(cur, view_w, view_h))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if k == ord("u") and st.clicks:
            st.clicks.pop()
            idx -= 1
        elif k == ord("s") and idx < len(order):
            idx += 1
        elif k in (ord("+"), ord("=") , ord("-")):
            f = 1.25 if k != ord("-") else 0.8
            m = np.array(st.mouse, dtype=float)
            img_pt = m / st.zoom + st.pan
            st.zoom *= f
            st.pan = img_pt - m / st.zoom
        elif k in (13, 10):
            break
    cv2.destroyAllWindows()
    points = [{"world": n, "pixel": [round(p[0], 1), round(p[1], 1)]} for n, p in st.clicks]
    cal = manual_calibrate(points, pitch)
    data = json.loads(out.read_text()) if out.exists() else {"cameras": {}}
    data["cameras"][str(camera_index)] = {"video": str(video), "frame_time": frame_time, "points": points}
    out.write_text(json.dumps(data, indent=2))
    print(f"saved {len(points)} points for camera {camera_index} -> {out}")
    print(f"reprojection error: {cal.reproj_error_px:.1f}px / {cal.reproj_error_m:.2f}m  confidence={cal.confidence:.2f}")
    for n in cal.notes:
        print("  note:", n)
