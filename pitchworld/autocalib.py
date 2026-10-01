"""Click-free calibration for small-sided pitches (goal line + blue "D" + portable goal).

Pipeline (one calibration frame):
  1. colour masks: white on/near grass (lines), blue (the "D" arc), raw white (goal frame)
  2. arc: largest elongated blue component below the horizon -> centre-line samples
  3. lines: HoughLinesP on the white-line mask, merged; the goal line is the long line that passes
     through both arc end-points; other long lines become "parallel to the goal line" candidates
  4. goal: two near-vertical white bars whose feet sit near the goal line, joined by a crossbar ->
     ground contacts (post feet) as candidate ``A_post_S`` / ``A_post_N`` points
  5. hypotheses: {arc + goal line} x {with/without each parallel} x {with/without posts, both post orders}
     -> ``calibrate.manual_calibrate`` (physical pose fit) -> scored by residual, camera height and, when
     person boxes are given, the implied person height (``metres_per_unit_from_people``).

Everything is a *candidate*: the result carries an honest confidence and notes; callers should fall back to
manual constraints when confidence is low. Nothing here changes behaviour for cameras that do not see a D.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import (
    CameraCalibration,
    colour_mask,
    manual_calibrate,
    metres_per_unit_from_people,
    snap_polyline_to_mask,
)
from .pitch import PitchModel


@dataclass
class GoalLineCandidate:
    seg: list[float]  # [x1,y1,x2,y2]
    end_dist: float  # distance of the nearest arc end to this line (px)
    pixels: list[list[float]] = field(default_factory=list)  # samples on the line
    other_lines: list[list[list[float]]] = field(default_factory=list)  # samples on other long white lines
    posts: list[list[float]] = field(default_factory=list)  # ground contacts of goal posts (0, 1 or 2)
    post_segs: list[list[float]] = field(default_factory=list)  # the post segments [x1,y1,x2,y2]
    notes: list[str] = field(default_factory=list)


@dataclass
class Features:
    horizon_y: float
    arc: list[list[float]] = field(default_factory=list)  # centre-line samples of the blue D
    arc_ends: list[list[float]] = field(default_factory=list)  # two end points of the arc (on the goal line)
    candidates: list[GoalLineCandidate] = field(default_factory=list)  # goal-line candidates, best first
    chosen: int = -1
    notes: list[str] = field(default_factory=list)

    def select(self, i: int) -> None:
        self.chosen = i

    @property
    def goal_line(self) -> list[list[float]]:
        return self.candidates[self.chosen].pixels if self.chosen >= 0 else []

    @property
    def other_lines(self) -> list[list[list[float]]]:
        return self.candidates[self.chosen].other_lines if self.chosen >= 0 else []

    @property
    def posts(self) -> list[list[float]]:
        return self.candidates[self.chosen].posts if self.chosen >= 0 else []

    @property
    def post_segs(self) -> list[list[float]]:
        return self.candidates[self.chosen].post_segs if self.chosen >= 0 else []

    @property
    def all_notes(self) -> list[str]:
        out = list(self.notes)
        for i, c in enumerate(self.candidates):
            out.append(f"goal line candidate {i}: {c.end_dist:.0f}px from an arc end, {len(c.posts)} post(s); " + "; ".join(c.notes))
        return out


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------
def grass_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (30, 40, 30), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return g


def horizon_row(grass: np.ndarray, frac: float = 0.25) -> float:
    """First image row from the top where at least ``frac`` of the pixels are grass."""
    rows = grass.mean(1) / 255.0
    idx = np.nonzero(rows > frac)[0]
    return float(idx[0]) if len(idx) else 0.0


def line_mask(frame: np.ndarray, grass: np.ndarray) -> np.ndarray:
    """White pixels on the grass (thick lines are kept, unlike ``calibrate.white_line_mask``)."""
    white = colour_mask(frame, "white")
    near = cv2.dilate(grass, np.ones((21, 21), np.uint8))
    m = cv2.bitwise_and(white, near)
    # clutter: regions dense in white (goal nets, kits, buildings) -> pitch lines are sparse thin strokes
    dens = cv2.boxFilter(white, -1, (101, 101), normalize=True)
    clutter = cv2.dilate((dens > 0.35 * 255).astype(np.uint8) * 255, np.ones((41, 41), np.uint8))
    m[clutter > 0] = 0
    # drop blobs that are clearly not lines (people, bags): large *and* not elongated
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        _x, _y, bw, bh, area = stats[i]
        fill = area / max(bw * bh, 1)
        elong = max(bw, bh) / max(min(bw, bh), 1)
        keep[i] = area < 400 or elong > 4 or fill < 0.25
    return np.where(keep[lab], 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# arc
# --------------------------------------------------------------------------
def thin(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen thinning (vectorised) of a binary uint8 mask -> 1-px wide skeleton."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]; p3 = img[:-2, 2:]; p4 = img[1:-1, 2:]; p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]; p7 = img[2:, :-2]; p8 = img[1:-1, :-2]; p9 = img[:-2, :-2]
            p1 = img[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int16) for n in nb)
            a = sum(((nb[k] == 0) & (nb[(k + 1) % 8] == 1)).astype(np.int16) for k in range(8))
            c1 = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) if step == 0 else (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (p1 == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return (img[1:-1, 1:-1] * 255).astype(np.uint8)


def _component_centreline(comp: np.ndarray, step: int = 8) -> np.ndarray:
    """Skeleton samples (every ``step`` px along the skeleton) of a thick curved band given as a binary image."""
    sk = thin(comp)
    ys, xs = np.nonzero(sk)
    if len(xs) < 5:
        return np.zeros((0, 2))
    pts = _order_along(np.c_[xs, ys].astype(np.float64))
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) >= step:
            out.append(p)
    return np.array(out)


def _order_along(pts: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour chain starting from an extreme point (arc samples are sparse & unordered)."""
    if len(pts) < 3:
        return pts
    c = pts.mean(0)
    start = int(np.argmax(np.linalg.norm(pts - c, axis=1)))
    order = [start]
    left = set(range(len(pts))) - {start}
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda i: np.linalg.norm(pts[i] - last))
        order.append(nxt)
        left.remove(nxt)
    return pts[order]


def detect_arc(frame: np.ndarray, horizon_y: float) -> tuple[np.ndarray, np.ndarray, str]:
    """-> (centre-line samples (N,2), end points (2,2), note). Largest elongated blue component below horizon."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 50, 35), (130, 255, 255))  # slightly looser than calibrate.colour_mask (dusk footage)
    blue[: int(horizon_y) + 1] = 0
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(blue, connectivity=8)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0002 * w * h or cents[i][1] < horizon_y:
            continue
        fill = area / max(bw * bh, 1)
        if fill > 0.45:  # a blob (kit, bag), not a thin curve
            continue
        score = max(bw, bh) * (1 - fill) ** 2  # long, thin, curved
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), "no blue arc component found"
    x, y, bw, bh, area = stats[best]
    crop = (lab[y: y + bh, x: x + bw] == best).astype(np.uint8)
    cl = _component_centreline(crop)
    if len(cl) < 5:
        return np.zeros((0, 2)), np.zeros((0, 2)), "blue arc too short"
    cl = cl + np.array([x, y], dtype=np.float64)
    ends = np.array([cl[0], cl[-1]])
    return cl, ends, f"arc: {len(cl)} samples, component area {area} px"


# --------------------------------------------------------------------------
# lines
# --------------------------------------------------------------------------
def _seg_angle(s: np.ndarray) -> float:
    return math.atan2(s[3] - s[1], s[2] - s[0]) % math.pi


def _point_line_dist(p: np.ndarray, s: np.ndarray) -> float:
    a, b = s[:2], s[2:]
    return float(abs(np.cross(b - a, p - a)) / (np.linalg.norm(b - a) + 1e-9))


def merge_segments(segs: np.ndarray, ang_tol_deg: float = 2.5, dist_tol: float = 14.0) -> np.ndarray:
    merged: list[np.ndarray] = []
    for s in segs:
        ang = _seg_angle(s)
        for i, m in enumerate(merged):
            mang = _seg_angle(m)
            dang = min(abs(ang - mang), math.pi - abs(ang - mang))
            mid = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
            if dang < math.radians(ang_tol_deg) and _point_line_dist(mid, m) < dist_tol:
                pts = np.array([s[:2], s[2:], m[:2], m[2:]])
                axis = m[2:] - m[:2]
                axis /= np.linalg.norm(axis) + 1e-9
                t = pts @ axis
                merged[i] = np.concatenate([pts[np.argmin(t)], pts[np.argmax(t)]])
                break
        else:
            merged.append(s.astype(np.float64).copy())
    return np.array(merged) if merged else np.zeros((0, 4))


def detect_long_lines(mask: np.ndarray, min_len_frac: float = 0.10, max_from_vertical_deg: float = 25.0) -> np.ndarray:
    _h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60, minLineLength=int(min_len_frac * w), maxLineGap=int(0.02 * w))
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)
    # ground lines are never near-vertical in a pitch-level camera; near-vertical bars are posts/fences
    keep = [s for s in segs if abs(_seg_angle(s) - math.pi / 2) > math.radians(max_from_vertical_deg)]
    merged = merge_segments(np.array(keep)) if keep else np.zeros((0, 4))
    if len(merged):
        lens = np.linalg.norm(merged[:, 2:] - merged[:, :2], axis=1)
        merged = merged[np.argsort(-lens)]
    return merged


# --------------------------------------------------------------------------
# goal posts
# --------------------------------------------------------------------------
def detect_posts(frame: np.ndarray, goal_seg: np.ndarray, horizon_y: float) -> tuple[list[list[float]], list[list[float]], str]:
    """Two near-vertical white bars with their feet within a band around the goal line, joined by a crossbar.

    Returns (feet [S?,N?] in image-x order, post segments, note). Feet are the lower end points of the bars.
    """
    h, w = frame.shape[:2]
    white = colour_mask(frame, "white")
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=40, minLineLength=int(0.05 * h), maxLineGap=int(0.01 * h))
    if segs is None:
        return [], [], "no vertical white bars"
    segs = segs.reshape(-1, 4).astype(np.float64)
    vert = [s for s in segs if abs(_seg_angle(s) - math.pi / 2) < math.radians(20)]
    if not vert:
        return [], [], "no vertical white bars"
    vert = merge_segments(np.array(vert), ang_tol_deg=4, dist_tol=10)
    # foot = lower end; must be below the horizon and near the goal line (goals are moved around, be lenient)
    cands = []
    for s in vert:
        foot = s[2:] if s[3] > s[1] else s[:2]
        top = s[:2] if s[3] > s[1] else s[2:]
        length = np.linalg.norm(s[2:] - s[:2])
        if foot[1] < horizon_y + 0.02 * h:
            continue
        d = _point_line_dist(foot, goal_seg)
        if d > 0.20 * length + 0.06 * h:
            continue
        cands.append((s, foot, top, length, d))
    if len(cands) < 2:
        return [], [s.tolist() for s, *_ in cands], f"only {len(cands)} candidate post(s) near goal line"
    # pair: similar length, tops roughly level, spacing comparable to height, and white support along the crossbar
    best = None
    for (sa, fa, ta, la, da), (sb, fb, tb, lb, db) in itertools.combinations(cands, 2):
        if fa[0] > fb[0]:
            (sa, fa, ta, la, da), (sb, fb, tb, lb, db) = (sb, fb, tb, lb, db), (sa, fa, ta, la, da)
        gap = fb[0] - fa[0]
        if gap < 0.5 * max(la, lb) or gap > 6 * max(la, lb):
            continue
        if min(la, lb) / max(la, lb) < 0.5:
            continue
        # crossbar support: sample white pixels between the two tops
        n = 40
        xs = np.linspace(ta[0], tb[0], n)
        ys = np.linspace(ta[1], tb[1], n)
        sup = 0
        for x, y in zip(xs, ys):
            x0, y0 = round(x), round(y)
            if 0 <= x0 < w and 0 <= y0 < h and white[max(0, y0 - 6): y0 + 7, max(0, x0 - 3): x0 + 4].any():
                sup += 1
        sup /= n
        if sup < 0.6:
            continue
        score = sup * (la + lb) / (1 + (da + db) / (0.05 * h))
        if best is None or score > best[0]:
            best = (score, sa, sb, fa, fb, sup)
    if best is None:
        return [], [s.tolist() for s, *_ in cands], f"{len(cands)} vertical bars but no post pair with a crossbar"
    _, sa, sb, fa, fb, sup = best
    return [fa.tolist(), fb.tolist()], [sa.tolist(), sb.tolist()], f"goal: 2 posts, crossbar support {sup:.2f}"


# --------------------------------------------------------------------------
def detect_features(frame: np.ndarray) -> Features:
    h, w = frame.shape[:2]
    grass = grass_mask(frame)
    hz = horizon_row(grass)
    feat = Features(horizon_y=hz)
    arc, ends, note = detect_arc(frame, hz)
    feat.notes.append(note)
    lm = line_mask(frame, grass)
    lm[: int(hz)] = 0
    segs = detect_long_lines(lm)
    feat.notes.append(f"{len(segs)} long white line(s)")
    if len(arc):
        feat.arc = arc.tolist()
        feat.arc_ends = ends.tolist()
    cands: list[tuple[float, np.ndarray]] = []
    if len(segs) and len(ends) == 2:
        # the D is drawn from the goal line, so (at least) one arc end sits on it; the painted arc often fades
        # before the line, so keep every line within tolerance of an end as a candidate and let the fit decide
        tol = 0.04 * h + 0.25 * float(np.linalg.norm(ends[1] - ends[0]))
        for s in segs:
            d = min(_point_line_dist(ends[0], s), _point_line_dist(ends[1], s))
            if d < tol:
                cands.append((d, s))
        cands.sort(key=lambda c: c[0])
        uniq: list[tuple[float, np.ndarray]] = []
        for d, s in cands:  # drop pieces of a line already kept
            if all(_point_line_dist((s[:2] + s[2:]) / 2, u) > 0.03 * h for _, u in uniq):
                uniq.append((d, s))
        cands = uniq[:2]
        if not cands:
            feat.notes.append(f"no long line through an arc end (tol {tol:.0f}px)")
    elif len(segs):
        cands = [(float("nan"), segs[0])]
        feat.notes.append("no arc: taking the longest white line as goal line candidate")
    for d, goal_seg in cands:
        c = GoalLineCandidate(seg=goal_seg.tolist(), end_dist=float(d))
        c.pixels = snap_polyline_to_mask(lm, _extend(goal_seg, w, h), band=8.0, step=max(20.0, w / 100))
        for s in segs:
            if np.allclose(s, goal_seg):
                continue
            ang = min(abs(_seg_angle(s) - _seg_angle(goal_seg)), math.pi - abs(_seg_angle(s) - _seg_angle(goal_seg)))
            if _point_line_dist((s[:2] + s[2:]) / 2, goal_seg) < 0.03 * h and ang < math.radians(10):
                continue  # same line, other piece
            samples = snap_polyline_to_mask(lm, [s[:2].tolist(), s[2:].tolist()], band=6.0, step=max(15.0, w / 150))
            if len(samples) >= 4:
                c.other_lines.append(samples)
        c.posts, c.post_segs, note = detect_posts(frame, goal_seg, hz)
        c.notes.append(note)
        feat.candidates.append(c)
    if feat.candidates:
        feat.select(0)
    return feat


def _extend(seg: np.ndarray, w: int, h: int) -> list[list[float]]:
    """Extend a segment across the frame (the Hough segment is usually a fragment of the full line)."""
    a, b = seg[:2], seg[2:]
    d = b - a
    d /= np.linalg.norm(d) + 1e-9
    ts = []
    for t in (-a[0] / d[0] if abs(d[0]) > 1e-9 else None, (w - a[0]) / d[0] if abs(d[0]) > 1e-9 else None,
              -a[1] / d[1] if abs(d[1]) > 1e-9 else None, (h - a[1]) / d[1] if abs(d[1]) > 1e-9 else None):
        if t is not None:
            p = a + t * d
            if -1 <= p[0] <= w + 1 and -1 <= p[1] <= h + 1:
                ts.append(t)
    if len(ts) < 2:
        return [a.tolist(), b.tolist()]
    return [(a + min(ts) * d).tolist(), (a + max(ts) * d).tolist()]


# --------------------------------------------------------------------------
# hypotheses + scoring
# --------------------------------------------------------------------------
@dataclass
class Hypothesis:
    name: str
    entry: dict
    cand: int = 0
    cal: CameraCalibration | None = None
    score: float = -1.0
    people_scale: float = float("nan")
    notes: list[str] = field(default_factory=list)


def _subsample(pts: list[list[float]], n: int = 24) -> list[list[float]]:
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
    return [pts[i] for i in idx]


def build_hypotheses(feat: Features, end: str = "A") -> list[Hypothesis]:
    """Per goal-line candidate: line+arc, optionally + goal posts (both orderings), optionally + parallel lines."""
    arcs = [{"world": f"{end}_D", "pixels": _subsample(feat.arc)}] if len(feat.arc) >= 5 else []
    hyps = []
    for ci, c in enumerate(feat.candidates):
        if len(c.pixels) < 4:
            continue
        line = {"world": f"goal_line_{end}", "pixels": _subsample(c.pixels)}
        pars = [{"world": f"goal_line_{end}", "pixels": _subsample(pl, 12)} for pl in c.other_lines[:2]]
        post_opts: list[tuple[str, list[dict]]] = [("", [])]
        if len(c.posts) == 2:
            for tag, (s, n) in (("+postsSN", (0, 1)), ("+postsNS", (1, 0))):
                post_opts.append((tag, [{"world": f"{end}_post_S", "pixel": c.posts[s]},
                                        {"world": f"{end}_post_N", "pixel": c.posts[n]}]))
        # other long lines may be parallel to the goal line (far goal line, halfway line) or perpendicular
        # (touchlines) - we cannot tell before calibrating, so each is tried alone and the residual decides
        par_opts: list[tuple[str, list[dict]]] = [("", [])] + [(f"+par{i}", [p]) for i, p in enumerate(pars)]
        for (ptag, pts), (rtag, prs) in itertools.product(post_opts, par_opts):
            hyps.append(Hypothesis(name=f"gl{ci}+arc{ptag}{rtag}", cand=ci,
                                   entry={"points": pts, "lines": [line], "arcs": arcs, "parallels": prs}))
    return hyps


def score_hypothesis(hyp: Hypothesis, pitch: PitchModel, frame_size: tuple[int, int], boxes: np.ndarray | None) -> None:
    try:
        cal = manual_calibrate(hyp.entry, pitch, frame_size=frame_size)
    except (ValueError, RuntimeError) as e:
        hyp.notes.append(f"fit failed: {e}")
        return
    hyp.cal = cal
    w, _h = frame_size
    s_res = 1.0 / (1.0 + cal.reproj_error_px / 6.0) / (1.0 + cal.reproj_error_m / 2.0)
    z = cal.pose["height"] if cal.pose else 0.0
    f = cal.pose["focal_px"] if cal.pose else w
    s_height = 1.0 if 1.0 <= z <= 6.0 else (0.5 if 0.6 <= z <= 12.0 else 0.15)  # hand-held / tripod phone
    s_focal = 1.0 if 0.5 * w <= f <= 2.5 * w else 0.3  # phone main camera, incl. a bit of zoom
    s_people = 1.0
    if boxes is not None and len(boxes) >= 10 and cal.pose:
        scale, _n = metres_per_unit_from_people(cal, boxes, frame_size)
        hyp.people_scale = scale
        if math.isfinite(scale):
            # scale == 1 means people come out 1.75 m tall in this calibration; pitch dimensions are a guess,
            # so this is a soft check
            s_people = float(np.clip(1.0 - abs(math.log(scale)) / math.log(2.5), 0.2, 1.0))
    s_cand = 1.0 / (1.0 + 0.5 * hyp.cand)  # goal-line candidates are ordered by arc-end distance
    # richer constraint sets that still fit well beat thin exact fits
    dof_bonus = 1.0 + 0.05 * (len(hyp.entry["points"]) + len(hyp.entry["parallels"]))
    hyp.score = s_res * s_height * s_focal * s_people * s_cand * dof_bonus
    hyp.notes.append(f"err {cal.reproj_error_px:.1f}px/{cal.reproj_error_m:.2f}m h={z:.1f}m f={f / w:.2f}w "
                     f"people_scale={hyp.people_scale:.2f}")


def auto_calibrate_small_sided(frame: np.ndarray, pitch: PitchModel, boxes: np.ndarray | None = None,
                               end: str = "A") -> tuple[CameraCalibration | None, str, dict]:
    """Click-free pose fit from a detected goal line, D arc and portable goal.

    ``boxes`` (N,4) person boxes from this camera (any frames) enable the people-scale plausibility check.
    Returns (calibration | None, reason, debug) with ``debug = {"features": Features, "hypotheses": [...]}``.
    """
    h, w = frame.shape[:2]
    feat = detect_features(frame)
    hyps = build_hypotheses(feat, end)
    debug = {"features": feat, "hypotheses": hyps}
    if not hyps:
        return None, "autocalib: " + "; ".join(feat.all_notes), debug
    if not feat.arc:
        return None, "autocalib: goal line found but no D arc (need line + arc for a pose)", debug
    for hyp in hyps:
        score_hypothesis(hyp, pitch, (w, h), boxes)
    fitted = [hy for hy in hyps if hy.cal is not None]
    if not fitted:
        return None, "autocalib: every hypothesis failed to fit; " + "; ".join(feat.all_notes), debug
    best = max(fitted, key=lambda hy: hy.score)
    feat.select(best.cand)
    cal = best.cal
    assert cal is not None
    # honest confidence: pose-fit confidence capped by the hypothesis score, and by how much competing hypotheses disagree
    # 0.6 cap: the pitch dimensions are a guess and no hypothesis here has independent validation
    conf = min(0.6, best.score, max(cal.confidence, 0.5 * best.score))
    others = [hy for hy in fitted if hy is not best and hy.score > 0.5 * best.score]
    if others:
        # disagreement of projected frame-centre-bottom point between best and runner-up hypotheses (metres)
        probe = np.array([[w / 2, 0.9 * h], [0.25 * w, 0.8 * h], [0.75 * w, 0.8 * h]])
        pb = cal.project(probe)
        dis = max(float(np.median(np.linalg.norm(o.cal.project(probe) - pb, axis=1))) for o in others if o.cal is not None)
        conf *= float(np.clip(1.0 - dis / 5.0, 0.2, 1.0))
        cal.notes.append(f"competing hypotheses disagree by up to {dis:.1f}m on the near ground")
    cal.method = "auto_pose"
    cal.confidence = float(conf)
    cal.notes = [f"autocalib hypothesis {best.name}", *best.notes, *feat.all_notes, *cal.notes]
    if math.isfinite(best.people_scale):
        cal.notes.append(f"people scale check: detected people are {1.75 / best.people_scale:.2f}m tall in this fit "
                         "(pitch dimensions are a guess; ~1.75 expected)")
    return cal, "ok", debug


# --------------------------------------------------------------------------
# debug rendering
# --------------------------------------------------------------------------
def draw_features(frame: np.ndarray, feat: Features) -> np.ndarray:
    img = frame.copy()
    _h, w = img.shape[:2]
    cv2.line(img, (0, int(feat.horizon_y)), (w, int(feat.horizon_y)), (200, 200, 200), 1)
    for pl in feat.other_lines:
        for u, v in pl:
            cv2.circle(img, (int(u), int(v)), 4, (0, 200, 255), -1)
    for u, v in feat.goal_line:
        cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
    for u, v in feat.arc:
        cv2.circle(img, (int(u), int(v)), 5, (255, 0, 255), -1)
    for u, v in feat.arc_ends:
        cv2.drawMarker(img, (int(u), int(v)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
    for s in feat.post_segs:
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 255, 0), 3)
    for u, v in feat.posts:
        cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
    y = 30
    for n in feat.notes:
        cv2.putText(img, n, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
    return img


def debug_image(frame: np.ndarray, pitch: PitchModel, cal: CameraCalibration | None, debug: dict) -> np.ndarray:
    """Side by side: detected features | projected pitch model (or the reason for failure)."""
    from .viz import _pitch_grid_pixels

    left = draw_features(frame, debug["features"])
    right = frame.copy()
    h, w = right.shape[:2]
    if cal is not None:
        for seg in _pitch_grid_pixels(pitch, cal, w, h):
            cv2.polylines(right, [seg], False, (0, 255, 255), 2, cv2.LINE_AA)
        for p in cal.points:
            u, v = map(int, p["pixel"])
            cv2.drawMarker(right, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        pose = f" cam@({cal.pose['x']:.1f},{cal.pose['y']:.1f}) h={cal.pose['height']:.1f}m f={cal.pose['focal_px']:.0f}" if cal.pose else ""
        cv2.putText(right, f"{cal.method} err={cal.reproj_error_px:.1f}px/{cal.reproj_error_m:.2f}m conf={cal.confidence:.2f}{pose}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y = 30
        for n in cal.notes[:3]:
            cv2.putText(right, n, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 28
    else:
        cv2.putText(right, "no calibration", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return np.hstack([left, right])
