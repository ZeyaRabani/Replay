"""Interactive calibration tool (OpenCV window) + headless session state machine.

    pitchworld calibrate VIDEO --camera 0 --pitch pitch.json --out calib.json
                        [--landmarks A_post_N,A_post_S] [--time 1.0]
                        [--fresh] [--screenshot out.jpg]

``CalibSession`` holds all constraint/editing state and never touches a GUI,
so tests and the ``--screenshot`` mode can drive it headless. ``run_click_tool``
wires it to an OpenCV window.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .calibrate import (
    _constraint_dof,
    colour_mask,
    manual_calibrate,
    read_frame,
    snap_polyline_to_mask,
    white_line_mask,
)
from .pitch import PitchModel
from .posefit import resolve_arc
from .viz import _pitch_grid_pixels

DEFAULT_ORDER = [
    "A_post_S", "A_post_N", "A_D_S", "A_D_N", "A_D_apex", "corner_A_S", "corner_A_N",
    "B_post_S", "B_post_N", "B_D_S", "B_D_N", "B_D_apex", "corner_B_S", "corner_B_N",
    "half_S", "half_N", "centre",
]

LINE_NAMES = ["goal_line_A", "goal_line_B", "touch_S", "touch_N", "half"]
ARC_NAMES = ["A_D", "B_D", "centre_circle"]

KEY_HELP = ("[p]oint [l]ine [a]rc pa[r]allel   Tab/[/]/1-9 element   "
            "space commit trace   enter save   [n] snap   [u]ndo   [s]kip   [f] refit   [q]uit")

_MODES = ("point", "line", "arc", "parallel")


class CalibSession:
    """Headless calibration state: constraint lists, current mode/element, trace, fit."""

    MODES = _MODES

    def __init__(self, frame: np.ndarray, pitch: PitchModel, camera_index: int, video: Path | str,
                 frame_time: float, existing: dict | None = None):
        self.frame = frame
        self.pitch = pitch
        self.camera_index = camera_index
        self.video = video
        self.frame_time = frame_time
        self.frame_size = (frame.shape[1], frame.shape[0])
        existing = existing or {}
        self.points = list(existing.get("points") or [])
        self.lines = list(existing.get("lines") or [])
        self.arcs = list(existing.get("arcs") or [])
        self.parallels = list(existing.get("parallels") or [])
        self.history: list[str] = []
        for kind, lst in (("points", self.points), ("lines", self.lines),
                          ("arcs", self.arcs), ("parallels", self.parallels)):
            self.history += [kind] * len(lst)
        lm = pitch.landmarks()
        names = [n for n in DEFAULT_ORDER if n in lm] + sorted(n for n in lm if n not in DEFAULT_ORDER)
        arcs = []
        for a in ARC_NAMES:
            try:
                resolve_arc(pitch, a)
                arcs.append(a)
            except (KeyError, ValueError):
                pass
        self.elements = {"point": names, "line": list(LINE_NAMES), "arc": arcs, "parallel": list(LINE_NAMES)}
        self.elem_idx = {m: 0 for m in self.MODES}
        self.mode = "point"
        self.trace: list[list[float]] = []
        self.snap = False
        self._snap_mask: np.ndarray | None = None
        self.fit = None
        self.fit_error: str | None = None
        self.refit()

    # ---- elements / modes ----------------------------------------------
    @property
    def element_list(self) -> list[str]:
        return self.elements[self.mode]

    @property
    def current_element(self) -> str:
        lst = self.element_list
        return lst[self.elem_idx[self.mode]] if lst else "(none)"

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            return
        if len(self.trace) >= 2:
            self.commit_trace()
        else:
            self.trace.clear()
        self.mode = mode

    def next_element(self, step: int = 1) -> None:
        lst = self.element_list
        if lst:
            self.elem_idx[self.mode] = (self.elem_idx[self.mode] + step) % len(lst)

    def select_element(self, i: int) -> None:
        if 0 <= i < len(self.element_list):
            self.elem_idx[self.mode] = i

    # ---- editing ---------------------------------------------------------
    def toggle_snap(self) -> None:
        self.snap = not self.snap

    def snap_mask(self) -> np.ndarray:
        if self._snap_mask is None:
            self._snap_mask = cv2.bitwise_or(white_line_mask(self.frame), colour_mask(self.frame, "blue"))
        return self._snap_mask

    def add_point(self, u: float, v: float) -> None:
        if self.mode == "point":
            self.points.append({"world": self.current_element, "pixel": [round(u, 1), round(v, 1)]})
            self.history.append("points")
            self.next_element()
            self.refit()
        else:
            self.trace.append([u, v])

    def commit_trace(self) -> bool:
        need = 3 if self.mode == "arc" else 2
        if len(self.trace) < need or self.mode == "point":
            return False
        pixels = self.trace
        if self.snap:
            snapped = snap_polyline_to_mask(self.snap_mask(), self.trace)
            if len(snapped) >= need:
                pixels = snapped
        rec = {"world": self.current_element, "pixels": [[round(u, 1), round(v, 1)] for u, v in pixels]}
        kind = {"line": "lines", "arc": "arcs", "parallel": "parallels"}[self.mode]
        getattr(self, kind).append(rec)
        self.history.append(kind)
        self.trace.clear()
        self.refit()
        return True

    def undo(self) -> None:
        if self.trace:
            self.trace.pop()
            return
        if not self.history:
            return
        kind = self.history.pop()
        lst = getattr(self, kind)
        if lst:
            lst.pop()
        if kind == "points" and self.elements["point"]:
            self.elem_idx["point"] = (self.elem_idx["point"] - 1) % len(self.elements["point"])
        self.refit()

    # ---- fit -------------------------------------------------------------
    @property
    def dof(self) -> int:
        return _constraint_dof(self.points, self.lines, self.arcs, self.parallels)

    @property
    def n_constraints(self) -> int:
        return len(self.points) + len(self.lines) + len(self.arcs) + len(self.parallels)

    def refit(self) -> None:
        self.fit = None
        self.fit_error = None
        if self.dof >= 7:
            try:
                self.fit = manual_calibrate(self.entry(), self.pitch, self.frame_size)
            except Exception as e:
                self.fit_error = str(e)[:80]

    # ---- persistence -------------------------------------------------------
    def entry(self) -> dict:
        return {"video": str(self.video), "frame_time": self.frame_time,
                "points": self.points, "lines": self.lines, "arcs": self.arcs, "parallels": self.parallels}

    def save(self, out: Path) -> dict:
        data = json.loads(out.read_text()) if out.exists() else {"cameras": {}}
        data.setdefault("cameras", {})[str(self.camera_index)] = self.entry()
        out.write_text(json.dumps(data, indent=2))
        return self.entry()

    # ---- status / keys -----------------------------------------------------
    def status_lines(self) -> list[str]:
        n = len(self.element_list)
        idx = self.elem_idx[self.mode]
        l1 = (f"mode={self.mode}  element={self.current_element} ({idx + 1}/{n})  "
              f"snap={'ON' if self.snap else 'off'}  trace={len(self.trace)}  "
              f"constraints={self.n_constraints} dof={self.dof}")
        if self.fit is not None:
            fit = self.fit
            l2 = f"fit: residual {fit.reproj_error_m:.2f}m / {fit.reproj_error_px:.1f}px"
            if fit.pose:
                l2 += f"  cam height {fit.pose['height']:.1f}m"
            l2 += f"  conf {fit.confidence:.2f}"
            if fit.notes:
                l2 += f"   ! {fit.notes[0][:60]}"
        elif self.fit_error:
            l2 = f"fit failed: {self.fit_error}"
        else:
            l2 = f"need >= 7 dof for a fit (have {self.dof}); trace a line/arc or click landmarks"
        return [l1, l2, KEY_HELP]

    def handle_key(self, k: int) -> str | None:
        """Keyboard state machine. Returns 'save' / 'quit' / None."""
        if k == 27 or k == ord("q"):
            return "quit"
        if k in (13, 10):
            if self.trace:
                self.commit_trace()
                return None
            return "save"
        if k == 9:
            self.next_element()
            return None
        if k == ord("["):
            self.next_element(-1)
            return None
        if ord("1") <= k <= ord("9"):
            self.select_element(k - ord("1"))
            return None
        if k == ord("0"):
            self.select_element(9)
            return None
        if k == 32:  # space
            self.commit_trace()
            return None
        c = chr(k) if 0 <= k < 256 else ""
        if c == "p":
            self.set_mode("point")
        elif c == "l":
            self.set_mode("line")
        elif c == "a":
            self.set_mode("arc")
        elif c == "r":
            self.set_mode("parallel")
        elif c == "n":
            self.toggle_snap()
        elif c == "u":
            self.undo()
        elif c == "s":
            self.next_element()
        elif c == "f":
            self.refit()
        return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class ViewState:
    """Zoom/pan/mouse state for the tool window (image px <-> view px)."""

    def __init__(self, frame: np.ndarray, view_w: int, view_h: int):
        self.frame = frame
        self.zoom = min(view_w / frame.shape[1], view_h / frame.shape[0])
        self.pan = np.array([0.0, 0.0])
        self.mouse = (0, 0)
        self.drag_start = None

    def to_view(self, pt):
        return ((np.asarray(pt) - self.pan) * self.zoom).astype(int)

    def to_image(self, pt):
        return tuple((np.asarray(pt, dtype=float) / self.zoom + self.pan).tolist())


_TRACE_COLORS = {"line": (0, 255, 255), "arc": (255, 255, 0), "parallel": (255, 0, 255)}


def render_view(session: CalibSession, view: ViewState, view_w: int, view_h: int) -> np.ndarray:
    z, pan = view.zoom, view.pan
    M = np.array([[z, 0, -pan[0] * z], [0, z, -pan[1] * z]])

    def tv(pts) -> np.ndarray:
        return ((np.asarray(pts, dtype=np.float64).reshape(-1, 2) - pan) * z).astype(np.int32)

    img = cv2.warpAffine(session.frame, M, (view_w, view_h))
    for p in session.points:
        pt = tuple(tv(p["pixel"])[0])
        cv2.drawMarker(img, pt, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, str(p["world"]), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for kind, trs in (("line", session.lines), ("arc", session.arcs), ("parallel", session.parallels)):
        col = _TRACE_COLORS[kind]
        for tr in trs:
            pts = tv(tr["pixels"])
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, 2)
            for pt in pts:
                cv2.circle(img, tuple(pt), 4, col, 1)
            cv2.putText(img, str(tr["world"]), tuple(pts[0] + [6, 14]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    if session.trace:
        pts = tv(session.trace)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(img, tuple(pt), 3, (0, 255, 0), -1)
        cv2.line(img, tuple(pts[-1]), view.mouse, (0, 255, 0), 1)
    if session.fit is not None:
        w, h = session.frame_size
        for seg in _pitch_grid_pixels(session.pitch, session.fit, w, h):
            cv2.polylines(img, [tv(seg.reshape(-1, 2)).reshape(-1, 1, 2)], False, (0, 165, 255), 2)
        fit = session.fit
        col = (0, 200, 0) if fit.confidence >= 0.6 else (0, 0, 255)
        pose = f"  h={fit.pose['height']:.1f}m" if fit.pose else ""
        cv2.putText(img, f"fit {fit.reproj_error_m:.2f}m / {fit.reproj_error_px:.1f}px conf={fit.confidence:.2f}{pose}",
                    (10, view_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.rectangle(img, (0, 0), (view_w, 74), (0, 0, 0), -1)
    for i, line in enumerate(session.status_lines()):
        cv2.putText(img, line, (8, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def render_screenshot(session: CalibSession, view_w: int = 1600, view_h: int = 900) -> np.ndarray:
    return render_view(session, ViewState(session.frame, view_w, view_h), view_w, view_h)


# --------------------------------------------------------------------------
# Window loop
# --------------------------------------------------------------------------
def run_click_tool(video: Path, camera_index: int, pitch: PitchModel, out: Path, landmarks: list[str] | None,
                   frame_time: float = 1.0, view_w: int = 1600, view_h: int = 900,
                   load_existing: bool = True, screenshot: Path | None = None) -> None:
    frame = read_frame(video, frame_time)
    existing = None
    if load_existing and out.exists():
        existing = (json.loads(out.read_text()).get("cameras") or {}).get(str(camera_index))
    session = CalibSession(frame, pitch, camera_index, video, frame_time, existing)
    if landmarks:
        rest = [n for n in session.elements["point"] if n not in landmarks]
        session.elements["point"] = [n for n in landmarks if n in pitch.landmarks()] + rest
    if screenshot is not None:
        cv2.imwrite(str(screenshot), render_screenshot(session, view_w, view_h))
        for line in session.status_lines():
            print(line)
        print(f"screenshot -> {screenshot}")
        return

    view = ViewState(frame, view_w, view_h)
    win = f"pitchworld calibrate cam{camera_index}"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, _):
        view.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            view.drag_start = (x, y, view.pan.copy())
        elif event == cv2.EVENT_LBUTTONUP:
            if view.drag_start is not None and abs(x - view.drag_start[0]) < 3 and abs(y - view.drag_start[1]) < 3:
                session.add_point(*view.to_image((x, y)))
            view.drag_start = None
        elif event == cv2.EVENT_RBUTTONUP:
            session.commit_trace()
        elif event == cv2.EVENT_MOUSEMOVE and view.drag_start is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            sx, sy, pan0 = view.drag_start
            view.pan = pan0 - np.array([x - sx, y - sy]) / view.zoom

    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, render_view(session, view, view_w, view_h))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("+"), ord("="), ord("-")):
            f = 1.25 if k != ord("-") else 0.8
            m = np.array(view.mouse, dtype=float)
            img_pt = m / view.zoom + view.pan
            view.zoom *= f
            view.pan = img_pt - m / view.zoom
            continue
        action = session.handle_key(k)
        if action == "quit":
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if action == "save":
            break
    cv2.destroyAllWindows()
    entry = session.save(out)
    n = sum(len(entry[k]) for k in ("points", "lines", "arcs", "parallels"))
    print(f"saved {n} constraints for camera {camera_index} -> {out}")
    if session.fit is not None:
        cal = session.fit
        print(f"reprojection error: {cal.reproj_error_px:.1f}px / {cal.reproj_error_m:.2f}m  "
              f"confidence={cal.confidence:.2f}")
        for note in cal.notes:
            print("  note:", note)
    else:
        print("  warning: no fit possible (need >= 7 dof); constraints saved anyway")
        if session.fit_error:
            print("  last fit error:", session.fit_error)
