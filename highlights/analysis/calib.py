"""Per-angle pitch calibration for players v2.

The user clicks known pitch landmarks on a still from each camera; a
least-squares homography maps normalised frame coords (fx, fy in
[-1, 2], off-frame allowed for corner cameras) to pitch metres
(x along 0..len_m, y across 0..wid_m, y=0 = far touchline).

Stored as multiangle/calib.json:
    {"angles": {"0": {"pts": [{"name","fx","fy"}], "H": [[..]*3],
                      "rms_m": x}}, "pitch": {"len_m", "wid_m"}}
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# landmark name -> (x, y) metres as a fraction of (len_m, wid_m);
# pitch coords match the radar: x 0..L near->far attack, y 0..W
# far->near touchline. 100x64 pitch proportions (box dims in metres).
_LM: dict[str, tuple[float, float]] = {
    "corner_near_left": (0.0, 1.0),
    "corner_near_right": (1.0, 1.0),
    "corner_far_right": (1.0, 0.0),
    "corner_far_left": (0.0, 0.0),
    "halfway_near": (0.5, 1.0),
    "halfway_far": (0.5, 0.0),
    "centre_spot": (0.5, 0.5),
    "pen_area_l_goal_near": (0.0, 0.814844),
    "pen_area_l_goal_far": (0.0, 0.185156),
    "pen_area_l_edge_near": (0.165, 0.814844),
    "pen_area_l_edge_far": (0.165, 0.185156),
    "pen_area_r_goal_near": (1.0, 0.814844),
    "pen_area_r_goal_far": (1.0, 0.185156),
    "pen_area_r_edge_near": (0.835, 0.814844),
    "pen_area_r_edge_far": (0.835, 0.185156),
    "six_l_goal_near": (0.0, 0.642969),
    "six_l_goal_far": (0.0, 0.357031),
    "six_l_edge_near": (0.055, 0.642969),
    "six_l_edge_far": (0.055, 0.357031),
    "six_r_goal_near": (1.0, 0.642969),
    "six_r_goal_far": (1.0, 0.357031),
    "six_r_edge_near": (0.945, 0.642969),
    "six_r_edge_far": (0.945, 0.357031),
    "pen_spot_l": (0.11, 0.5),
    "pen_spot_r": (0.89, 0.5),
    "goalpost_l_near": (0.0, 0.557188),
    "goalpost_l_far": (0.0, 0.442812),
    "goalpost_r_near": (1.0, 0.557188),
    "goalpost_r_far": (1.0, 0.442812),
}

_LABELS = {
    "corner": "Corner", "halfway": "Halfway", "centre_spot": "Centre spot",
    "pen_area_l": "Penalty box L", "pen_area_r": "Penalty box R",
    "six_l": "Six-yard L", "six_r": "Six-yard R",
    "pen_spot_l": "Penalty spot L", "pen_spot_r": "Penalty spot R",
    "goalpost_l": "Goal post L", "goalpost_r": "Goal post R",
    "d_l": "D left", "d_r": "D right",
}

# small-sided (9-a-side) template: marked touchlines + halfway + centre
# circle, portable goals, and a semicircular "D" arc (radius d_radius_m)
# centred on each goal centre — no boxes/pen spots. Extra landmarks:
# name -> (x_m, y_m) expression over (L, W, goal_w, r).
def _small_lm(L: float, W: float, goal_w: float, r: float
              ) -> dict[str, tuple[float, float]]:
    cy = W / 2
    return {
        "corner_near_left": (0.0, W),
        "corner_near_right": (L, W),
        "corner_far_right": (L, 0.0),
        "corner_far_left": (0.0, 0.0),
        "halfway_near": (L / 2, W),
        "halfway_far": (L / 2, 0.0),
        "centre_spot": (L / 2, cy),
        "goalpost_l_near": (0.0, cy + goal_w / 2),
        "goalpost_l_far": (0.0, cy - goal_w / 2),
        "goalpost_r_near": (L, cy + goal_w / 2),
        "goalpost_r_far": (L, cy - goal_w / 2),
        # arc meets the goal line
        "d_l_near": (0.0, cy + r),
        "d_l_far": (0.0, cy - r),
        "d_r_near": (L, cy + r),
        "d_r_far": (L, cy - r),
        # arc apex
        "d_l_apex": (r, cy),
        "d_r_apex": (L - r, cy),
    }


def landmarks_for(pitch: dict) -> list[dict]:
    """[{"name","label","x","y"}] for a pitch dict
    {"len_m","wid_m","template","goal_w_m","d_radius_m"} — "full"
    is the standard 29-landmark table, "small" the 9-a-side one."""
    template = pitch.get("template", "full")
    L, W = float(pitch["len_m"]), float(pitch["wid_m"])
    if template == "small":
        goal_w = float(pitch.get("goal_w_m") or 3.66)
        r = float(pitch.get("d_radius_m") or 9.0)
        lm = _small_lm(L, W, goal_w, r)
        return [{"name": n, "label": _label(n), "x": x, "y": y}
                for n, (x, y) in lm.items()]
    return landmarks(L, W)


def _label(name: str) -> str:
    for prefix, lab in _LABELS.items():
        if name.startswith(prefix):
            tail = name[len(prefix):].strip("_").replace("_", " ")
            return f"{lab} - {tail}" if tail else lab
    return name


def landmarks(len_m: float, wid_m: float) -> list[dict]:
    """[{"name","label","x","y"}] for a len_m x wid_m pitch (full)."""
    L, W = float(len_m), float(wid_m)
    return [{"name": n, "label": _label(n),
             "x": fx * L, "y": fy * W} for n, (fx, fy) in _LM.items()]


def landmark_xy(len_m: float, wid_m: float,
                pitch: dict | None = None) -> dict[str, list[float]]:
    if pitch is not None:
        return {l["name"]: [l["x"], l["y"]] for l in landmarks_for(pitch)}
    return {l["name"]: [l["x"], l["y"]] for l in landmarks(len_m, wid_m)}


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4+ point DLT homography (src->dst) via SVD, float64. Both sides
    isotropic-normalised for conditioning; raises ValueError when the
    system is degenerate (collinear/coincident points)."""

    def _norm(P):
        c = P.mean(axis=0)
        s = (P - c)
        scale = math.sqrt(2.0) / max(1e-12, float(np.hypot(*s.T).mean()))
        return np.array([[scale, 0, -scale * c[0]],
                         [0, scale, -scale * c[1]], [0, 0, 1.0]])

    Ts, Td = _norm(src), _norm(dst)
    A = []
    s_n = np.c_[src, np.ones(len(src))] @ Ts.T
    d_n = np.c_[dst, np.ones(len(dst))] @ Td.T
    for (x, y, _w), (u, v, _w2) in zip(s_n, d_n):
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    _u, s, vt = np.linalg.svd(A)
    if s[-2] < 1e-10:                      # rank deficient -> degenerate
        raise ValueError("points are collinear or degenerate")
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def apply_h(H, fx: float, fy: float) -> tuple[float, float]:
    H = np.asarray(H, dtype=float)
    v = H @ np.array([fx, fy, 1.0])
    if abs(v[2]) < 1e-12:
        return (math.inf, math.inf)
    return float(v[0] / v[2]), float(v[1] / v[2])


def solve_homography(pts: list[dict], len_m: float, wid_m: float,
                     pitch: dict | None = None) -> tuple[list, float]:
    """Least-squares H (frame->pitch) from >=4 named landmarks.

    pts: [{"name","fx","fy"}]. Returns (H 3x3 list, rms_m). Raises
    ValueError on unknown names, <4 pts or a singular system. `pitch`
    (full pitch dict incl. template) overrides len_m/wid_m when given."""
    lm = landmark_xy(len_m, wid_m, pitch)
    src, dst = [], []
    for p in pts:
        name = p.get("name")
        if name not in lm:
            raise ValueError(f"unknown landmark {name!r}")
        fx, fy = p.get("fx"), p.get("fy")
        if not (isinstance(fx, (int, float)) and isinstance(fy, (int, float))
                and -1.0 <= fx <= 2.0 and -1.0 <= fy <= 2.0):
            raise ValueError(f"{name}: fx,fy must be in [-1, 2]")
        src.append([float(fx), float(fy)])
        dst.append(lm[name])
    if len(src) < 4:
        raise ValueError("need at least 4 landmarks")
    H = _dlt(np.array(src), np.array(dst))
    rms = math.sqrt(float(np.mean(
        [((apply_h(H, s[0], s[1])[0] - d[0]) ** 2
          + (apply_h(H, s[0], s[1])[1] - d[1]) ** 2)
         for s, d in zip(src, dst)])))
    return H.tolist(), rms


def load_calib(project_dir: Path) -> dict | None:
    from .run import _load_json
    return _load_json(Path(project_dir) / "multiangle" / "calib.json")
