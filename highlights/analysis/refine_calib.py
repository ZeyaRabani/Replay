"""Joint calibration refinement: per-camera H is re-fit so that the
same player's anchor footpoint (stab-warped onto the landmark frame)
projects to the same pitch position in every camera. The reference
camera's landmarks hold it in place (weight 3), other cameras'
landmarks are weak priors (0.3), anchor cross-camera pairs weight 1.

    python -m highlights.analysis.refine_calib --project-dir P
        [--ref-angle A]

writes angles[a]["H_refined"] + a top-level "refine" summary into
multiangle/calib.json. Readers should use calib.effective_h(entry).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .calib import _dlt, apply_h, landmark_xy
from .stabilize import load_stab, warp_at

MIN_PAIRS = 8
W_REF_LM, W_OTHER_LM, W_PAIR = 3.0, 0.3, 1.0


def _anchor_frames(anchors_doc: dict, stabs: dict[int, dict],
                   offsets: list[float]) -> dict:
    """{(moment, norm_label, angle): (fx, fy)} — anchor box footpoints
    stab-warped onto each camera's calibration frame."""
    moments = {str(m.get("id")): float(m["t"])
               for m in anchors_doc.get("moments") or []}
    out: dict = {}
    for c in anchors_doc.get("clicks") or []:
        box = c.get("box")
        t = moments.get(str(c.get("moment")))
        if not box or t is None:
            continue
        a = int(c.get("angle") or 0)
        ft = t - (float(offsets[a]) if a < len(offsets) else 0.0)
        fx, fy = (float(box[0]) + float(box[2])) / 2.0, float(box[3])
        stab = stabs.get(a)
        if stab is not None:
            fx, fy = warp_at(stab, ft, fx, fy)
        out[(str(c.get("moment")),
             str(c.get("label") or "").strip().lower(), a)] = (fx, fy)
    return out


def _pairs(F: dict) -> list[tuple]:
    """Every two cameras that saw the same (moment, name)."""
    by_key: dict = {}
    for (mid, nm, a), p in F.items():
        by_key.setdefault((mid, nm), {})[a] = p
    pairs = []
    for per in by_key.values():
        ks = sorted(per)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pairs.append((ks[i], per[ks[i]], ks[j], per[ks[j]]))
    return pairs


def _lm_pts(entry: dict, lm_xy: dict) -> list[tuple]:
    """[(fx, fy, X, Y)] for the angle's saved landmarks, excluding the
    D-arc apex (its template position doesn't match real pitches)."""
    out = []
    for p in entry.get("pts") or []:
        n = str(p.get("name"))
        if n.endswith("_apex") or n not in lm_xy:
            continue
        out.append((float(p["fx"]), float(p["fy"]),
                    float(lm_xy[n][0]), float(lm_xy[n][1])))
    return out


def _lm_rms(entry: dict, lm_xy: dict) -> float:
    pts = _lm_pts(entry, lm_xy)
    H = np.asarray(entry["H"], float)
    if not pts:
        return 1e9
    e = [np.hypot(*(np.array(apply_h(H, fx, fy)) - (X, Y)))
         for fx, fy, X, Y in pts]
    return float(np.sqrt(np.mean(np.square(e))))


def _pick_ref(angles: dict, stabs: dict, dets: dict | None,
              L: float, W: float, lm_xy: dict | None = None) -> int | None:
    """Reference camera: best static-H coverage among cameras whose
    detections span >= 0.6*L in x, ties broken by the H's own landmark
    reprojection rms; else lowest landmark rms."""
    if dets:
        best = None
        for a in sorted(angles):
            H = (angles[a] or {}).get("H")
            d = dets.get(a)
            if H is None or d is None or not len(d.get("t", ())):
                continue
            keep = np.flatnonzero(np.asarray(d["conf"]) >= 0.5)
            if not len(keep):
                continue
            xs = []
            for i in keep:
                fx, fy = float(d["foot"][i][0]), float(d["foot"][i][1])
                stab = stabs.get(a)
                if stab is not None:
                    fx, fy = warp_at(stab, float(d["t"][i]), fx, fy)
                x, y = apply_h(H, fx, fy)
                if -2 <= x <= L + 2 and -2 <= y <= W + 2:
                    xs.append(x)
            frac = len(xs) / len(keep)
            span = (float(np.percentile(xs, 98) - np.percentile(xs, 2))
                    if len(xs) >= 8 else 0.0)
            if span < 0.6 * L:
                frac = 0.0
            own = _lm_rms(angles[a], lm_xy) if lm_xy else 0.0
            cand = (round(frac, 3), -own, -a)
            if best is None or cand > best[0]:
                best = (cand, a)
        if best is not None and best[0][0] > 0:
            return best[1]
    rms = {a: float(e.get("rms_m") or 1e9) for a, e in angles.items()
           if (e or {}).get("H")}
    return min(rms, key=rms.get) if rms else None


def _init_from_ref(pairs: list[tuple], Href: np.ndarray, ref: int,
                   a: int) -> np.ndarray | None:
    """DLT for camera a from its anchor points paired with the reference
    camera (targets = ref-projected pitch positions); None under 4 pairs
    or a degenerate fit. A static H fitted from mislabelled/jittered
    landmarks is too far off for the local solver to recover from."""
    src, dst = [], []
    for a1, p1, a2, p2 in pairs:
        if a1 == ref and a2 == a:
            src.append(p2); dst.append(apply_h(Href, *p1))
        elif a2 == ref and a1 == a:
            src.append(p1); dst.append(apply_h(Href, *p2))
    if len(src) < 4:
        return None
    try:
        H = _dlt(np.asarray(src, float), np.asarray(dst, float))
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
        return None
    return H


def refine(calib: dict, anchors_doc: dict, stabs: dict[int, dict],
           offsets: list[float], dets: dict[int, dict] | None = None,
           *, ref_angle: int | None = None) -> dict | None:
    """Least-squares joint refinement of every camera's H so same-name
    anchor footpoints coincide across cameras. None when <8 pairs."""
    from scipy.optimize import least_squares

    angles = {int(k): v for k, v in (calib.get("angles") or {}).items()}
    pitch = calib.get("pitch") or {}
    L = float(pitch.get("len_m") or 100.0)
    W = float(pitch.get("wid_m") or 64.0)
    lm_xy = landmark_xy(L, W, pitch)

    F = _anchor_frames(anchors_doc, stabs, offsets)
    pairs = _pairs(F)
    if len(pairs) < MIN_PAIRS:
        return None

    keys = sorted(a for a in angles if (angles[a] or {}).get("H"))
    if not keys:
        return None
    if ref_angle is None:
        ref_angle = _pick_ref(angles, stabs, dets, L, W, lm_xy)
    if ref_angle is None:
        return None

    lm = {a: _lm_pts(angles[a], lm_xy) for a in keys}
    H0 = {a: np.asarray(angles[a]["H"], dtype=float) for a in keys}
    Hinit = dict(H0)
    for a in keys:
        if a != ref_angle:
            Hd = _init_from_ref(pairs, H0[ref_angle], ref_angle, a)
            if Hd is not None:
                Hinit[a] = Hd
    x0 = np.concatenate([Hinit[a].ravel()[:8] / Hinit[a][2, 2]
                         for a in keys])

    def _H(x: np.ndarray, i: int) -> np.ndarray:
        return np.array([x[8 * i:8 * i + 3], x[8 * i + 3:8 * i + 6],
                         [x[8 * i + 6], x[8 * i + 7], 1.0]])

    idx = {a: i for i, a in enumerate(keys)}

    def resid(x: np.ndarray) -> np.ndarray:
        r = []
        for a in keys:
            Ha = _H(x, idx[a])
            w = W_REF_LM if a == ref_angle else W_OTHER_LM
            for fx, fy, X, Y in lm[a]:
                px, py = apply_h(Ha, fx, fy)
                r += [w * (px - X), w * (py - Y)]
        for a1, p1, a2, p2 in pairs:
            x1, y1 = apply_h(_H(x, idx[a1]), *p1)
            x2, y2 = apply_h(_H(x, idx[a2]), *p2)
            r += [W_PAIR * (x1 - x2), W_PAIR * (y1 - y2)]
        return np.asarray(r)

    def pair_med(hmap: dict[int, np.ndarray]) -> float:
        ds = []
        for a1, p1, a2, p2 in pairs:
            x1, y1 = apply_h(hmap[a1], *p1)
            x2, y2 = apply_h(hmap[a2], *p2)
            ds.append(float(np.hypot(x1 - x2, y1 - y2)))
        return float(np.median(ds)) if ds else 0.0

    before = pair_med({a: H0[a] / H0[a][2, 2] for a in keys})
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0,
                        x_scale="jac", max_nfev=5000)
    H1 = {a: _H(sol.x, idx[a]) for a in keys}
    after = pair_med(H1)
    lm_rms = {}
    for a in keys:
        ds = [float(np.hypot(apply_h(H1[a], fx, fy)[0] - X,
                             apply_h(H1[a], fx, fy)[1] - Y))
              for fx, fy, X, Y in lm[a]]
        lm_rms[a] = round(float(np.sqrt(np.mean(np.square(ds))))
                          if ds else 0.0, 3)
    return {"ref_angle": ref_angle, "n_pairs": len(pairs),
            "pair_median_m_before": round(before, 3),
            "pair_median_m_after": round(after, 3),
            "lm_rms_m": {str(a): v for a, v in lm_rms.items()},
            "H": {a: H1[a].tolist() for a in keys}}


MAX_LM_RMS_M = 2.0


def rejection_reason(summary: dict, calib: dict) -> str | None:
    """A refined H must still land every camera's own landmarks: inconsistent
    anchors (cameras that panned away from the calibration frame, wrong
    moments) otherwise pull the solve into a degenerate H that only fits
    the anchor pairs."""
    angles = calib.get("angles") or {}
    bad = []
    for a, rms in (summary.get("lm_rms_m") or {}).items():
        static = float((angles.get(str(a)) or {}).get("rms_m") or 0.0)
        if rms > max(MAX_LM_RMS_M, 3.0 * static):
            bad.append(f"a{a} landmarks {rms:.1f} m")
    if bad:
        return "anchors inconsistent with landmarks: " + ", ".join(bad)
    if summary["pair_median_m_after"] >= summary["pair_median_m_before"]:
        return "cross-camera agreement did not improve"
    return None


def apply_refine(project_dir: Path, players_v2: Path, log=print,
                 ref_angle: int | None = None) -> dict | None:
    """Load calib/anchors/stabs/dets, refine, persist H_refined +
    "refine" summary into multiangle/calib.json (or clear stale ones)."""
    from highlights.io import write_json_atomic

    from . import anchors as anch
    from .fuse_tracks import load_dets
    from .run import _load_json, resolve_context

    cpath = Path(project_dir) / "multiangle" / "calib.json"
    calib = _load_json(cpath) or {}
    angles = calib.get("angles") or {}
    anchor_doc = anch.load(players_v2) or {}
    ctx = resolve_context(project_dir)
    offsets = ctx["offsets"]
    n = int(ctx["n_angles"])
    stabs = {}
    dets = {}
    for a in range(n):
        s = load_stab(Path(players_v2) / f"stab_a{a}.npz")
        if s is not None:
            stabs[a] = s
        f = Path(players_v2) / f"det_a{a}.npz"
        if f.is_file():
            dets[a] = load_dets(f)
    summary = refine(calib, anchor_doc, stabs, offsets, dets,
                     ref_angle=ref_angle)
    why = (None if summary is None and not anchor_doc.get("clicks")
           else "not enough anchor pairs" if summary is None
           else rejection_reason(summary, calib))

    def _clear():
        if any((e or {}).get("H_refined") for e in angles.values()) \
                or "refine" in calib:
            for e in angles.values():
                if isinstance(e, dict):
                    e.pop("H_refined", None)
            calib.pop("refine", None)
            write_json_atomic(cpath, calib, indent=1)

    if summary is None:
        _clear()
        log(f"calib refine: {why or 'no anchors'} — skipped")
        return None
    if why:
        _clear()
        log(f"calib refine: rejected ({why}); static H kept")
        return None
    for a, H in summary["H"].items():
        if str(a) in angles:
            angles[str(a)]["H_refined"] = H
    calib["refine"] = {k: summary[k] for k in
                       ("ref_angle", "n_pairs", "pair_median_m_before",
                        "pair_median_m_after", "lm_rms_m")}
    calib["refine"]["updated_at"] = time.time()
    write_json_atomic(cpath, calib, indent=1)
    log(f"calib refine: ref=a{summary['ref_angle']} "
        f"pairs={summary['n_pairs']} median "
        f"{summary['pair_median_m_before']} -> "
        f"{summary['pair_median_m_after']} m")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.refine_calib")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--ref-angle", type=int, default=None)
    args = ap.parse_args(argv)
    s = apply_refine(args.project_dir,
                     args.project_dir / "analysis" / "players_v2",
                     ref_angle=args.ref_angle)
    print(json.dumps(s, indent=1) if s else "skipped")
    return 0 if s else 1


if __name__ == "__main__":
    sys.exit(main())
