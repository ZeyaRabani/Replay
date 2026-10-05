"""Joint homography refinement across all cameras — no anchors needed.

The landmark-only H per camera fits its own clicks (~1.5 m rms) but the
three cameras disagree by 5+ m on the same player, which defeats the
5 m cross-view merge in fuse_tracks. joint_calib starts from the
landmark-only Hs and iterates:

  1. project every detection (1 s steps, conf>=0.4, box h>=MIN_BOX_H_PX,
     stab-warped onto the landmark frame) to pitch metres;
  2. Hungarian-match each camera pair per step, keep pairs < 6 m
     (first pass 8 m while Hs are still landmark-only);
  3. simultaneously re-fit all Hs (8 params each, h33=1) with
     least_squares soft_l1: landmark residuals weight 3.0, each matched
     pair's cross-camera difference weight 1.0;

accepting only when every camera's landmark rms stays <= 2.5 m AND each
pair's median residual improved — otherwise the landmark-only H is
restored and the rejection reason reported.

    python -m highlights.analysis.joint_calib <project_dir> [--dry-run]

writes angles[a]["H"] (joint fit), angles[a]["H_landmark"] (landmark-
only) and a top-level "joint" summary into multiangle/calib.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .calib import apply_h, solve_homography
from .fuse_tracks import MIN_BOX_H_PX, load_dets
from .stabilize import load_stab, warp_at

STEP_S = 1.0
CONF_MIN = 0.4
DET_TOL_S = 0.4          # det nearest-step assignment tolerance
MATCH_M_FIRST = 8.0      # pair gate while Hs are still landmark-only
MATCH_M = 6.0
ITERS = 5
W_LM = 3.0
W_PAIR = 1.0
F_SCALE = 1.0
MAX_LM_RMS_M = 2.5


def _lm_pts(entry: dict, lm_xy: dict) -> list[tuple]:
    """[(fx, fy, X, Y)] for the angle's clicked landmarks, excluding
    the D-arc apex (its template position doesn't match real pitches —
    same convention as refine_calib)."""
    out = []
    for p in entry.get("pts") or []:
        n = str(p.get("name"))
        if n.endswith("_apex") or n not in lm_xy:
            continue
        out.append((float(p["fx"]), float(p["fy"]),
                    float(lm_xy[n][0]), float(lm_xy[n][1])))
    return out


def landmark_H(entry: dict, pitch: dict, lm_xy: dict
               ) -> np.ndarray | None:
    """Landmark-only H from the saved clicks (NOT the stored H, which
    may be an anchor-fitted matrix that violates its own landmarks)."""
    pts = [p for p in (entry.get("pts") or [])
           if not str(p.get("name")).endswith("_apex")]
    if len(pts) < 4:
        return None
    try:
        H, _rms = solve_homography(pts, float(pitch["len_m"]),
                                   float(pitch["wid_m"]), pitch)
    except (ValueError, KeyError, np.linalg.LinAlgError):
        return None
    return np.asarray(H, dtype=float)


def _det_steps(dets: dict[int, dict], stabs: dict[int, dict],
               offsets: list[float], lo: float, n_steps: int
               ) -> dict[int, list[list[tuple]]]:
    """{angle: [per-step [(fx,fy stab-warped, conf)]} — one 1 s bin per
    det row (nearest step within DET_TOL_S)."""
    out: dict[int, list[list[tuple]]] = {}
    for a, d in dets.items():
        steps: list[list[tuple]] = [[] for _ in range(n_steps)]
        off = float(offsets[a]) if a < len(offsets) else 0.0
        stab = stabs.get(a)
        keep = d["conf"] >= CONF_MIN
        if "box" in d and d["box"] is not None and len(d["box"]):
            keep = keep & ((d["box"][:, 3] - d["box"][:, 1])
                           >= MIN_BOX_H_PX)
        for i in np.flatnonzero(keep):
            tf = float(d["t"][i])
            ts = tf + off
            k = round((ts - lo) / STEP_S)
            if not (0 <= k < n_steps) or abs(ts - (lo + k * STEP_S)) \
                    > DET_TOL_S:
                continue
            fx, fy = float(d["foot"][i][0]), float(d["foot"][i][1])
            if stab is not None:
                fx, fy = warp_at(stab, tf, fx, fy)
            steps[k].append((fx, fy))
        out[a] = steps
    return out


def _match_pairs(proj: dict[int, list[list[tuple]]], keys: list[int],
                 gate_m: float) -> dict[tuple, list[tuple]]:
    """{(a,b): [(i,j,k)]} Hungarian matches between camera pairs per
    step; i/j index into the step's det list. Gate on pitch distance."""
    from scipy.optimize import linear_sum_assignment
    pairs: dict[tuple, list[tuple]] = {}
    combos = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    for a, b in combos:
        hits = []
        for k in range(len(proj[a])):
            pa, pb = proj[a][k], proj[b][k]
            if not pa or not pb:
                continue
            cost = np.full((len(pa), len(pb)), 1e6)
            for i, (x1, y1) in enumerate(pa):
                for j, (x2, y2) in enumerate(pb):
                    d = math.hypot(x1 - x2, y1 - y2)
                    if d <= gate_m:
                        cost[i, j] = d
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e6:
                    hits.append((k, int(i), int(j)))
        pairs[(a, b)] = hits
    return pairs


def _project_all(steps: dict[int, list[list[tuple]]],
                 Hs: dict[int, np.ndarray]
                 ) -> dict[int, list[list[tuple]]]:
    out = {}
    for a, per in steps.items():
        Ha = Hs[a]
        out[a] = [[apply_h(Ha, fx, fy) for fx, fy in s] for s in per]
    return out


def joint_refine(calib: dict, dets: dict[int, dict],
                 stabs: dict[int, dict], offsets: list[float], *,
                 window: tuple[float, float] | None = None,
                 iters: int = ITERS, log=print) -> dict:
    """Refine every angle's H jointly. Returns {"H": {a: 3x3 list},
    "H_landmark": {...}, "accepted": bool, "reason": str|None,
    "report": [per-iter rows], "pairs": per-pair medians, "lm_rms": ...}.
    Never raises for bad data — degenerate input yields accepted=False
    with the landmark-only Hs."""
    from scipy.optimize import least_squares

    from .calib import landmark_xy
    angles = {int(k): v for k, v in (calib.get("angles") or {}).items()}
    pitch = calib.get("pitch") or {}
    L = float(pitch.get("len_m") or 100.0)
    W = float(pitch.get("wid_m") or 64.0)
    lm_xy = landmark_xy(L, W, pitch)
    keys = sorted(a for a in angles if landmark_H(angles[a], pitch,
                                                lm_xy) is not None)
    if len(keys) < 2:
        return {"accepted": False, "reason": "fewer than 2 calibrated "
                    "cameras", "H": {}, "H_landmark": {}, "report": []}
    H_lm = {a: landmark_H(angles[a], pitch, lm_xy) for a in keys}
    lm = {a: _lm_pts(angles[a], lm_xy) for a in keys}

    if window is None:
        ts = [float(t) + (float(offsets[a]) if a < len(offsets) else 0.0)
              for a, d in dets.items() if a in H_lm for t in d["t"]]
        lo = math.floor(min(ts)) if ts else 0.0
        hi = math.ceil(max(ts)) if ts else lo + 1.0
    else:
        lo, hi = float(window[0]), float(window[1])
    n_steps = max(1, round((hi - lo) / STEP_S) + 1)
    steps = _det_steps({a: d for a, d in dets.items() if a in H_lm},
                       stabs, offsets, lo, n_steps)

    def lm_rms(hmap):
        out = {}
        for a in keys:
            ds = [float(np.hypot(apply_h(hmap[a], fx, fy)[0] - X,
                                 apply_h(hmap[a], fx, fy)[1] - Y))
                  for fx, fy, X, Y in lm[a]]
            out[a] = float(np.sqrt(np.mean(np.square(ds)))) if ds else 0.0
        return out

    def pair_meds(hmap, gate):
        proj = _project_all(steps, hmap)
        matches = _match_pairs(proj, keys, gate)
        med, inside = {}, []
        for combo, hits in matches.items():
            ds = [float(np.hypot(
                proj[combo[0]][k][i][0] - proj[combo[1]][k][j][0],
                proj[combo[0]][k][i][1] - proj[combo[1]][k][j][1]))
                for k, i, j in hits]
            med[combo] = float(np.median(ds)) if ds else math.inf
        for a in keys:
            pts = [p for s in proj[a] for p in s]
            good = [p for p in pts if math.isfinite(p[0])
                    and math.isfinite(p[1])]
            inside.append(sum(0 <= x <= L and 0 <= y <= W
                              for x, y in good) / max(1, len(good)))
        return med, matches, (float(np.mean(inside)) if inside else 0.0)

    def _proj(H: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Vectorised apply_h: (n,2) normalized frame pts -> pitch."""
        q = np.c_[P, np.ones(len(P))] @ H.T
        w = q[:, 2:3]
        out = q[:, :2] / np.where(np.abs(w) < 1e-9, np.nan, w)
        return out

    lm_src = {a: np.array([[fx, fy] for fx, fy, _X, _Y in lm[a]])
              for a in keys}
    lm_dst = {a: np.array([[X, Y] for _fx, _fy, X, Y in lm[a]])
              for a in keys}

    def _blocks(matches):
        """Flatten matched foot points into per-camera arrays so the
        whole residual is vectorised."""
        foot: dict[int, list] = {}
        pair_idx: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        offs: dict[tuple, tuple[int, int]] = {}
        seen: dict[int, int] = {}
        for (a, b), hits in matches.items():
            ia = np.array([steps[a][k][i] for k, i, j in hits],
                          dtype=float).reshape(-1, 2)
            ib = np.array([steps[b][k][j] for k, i, j in hits],
                          dtype=float).reshape(-1, 2)
            pair_idx[(a, b)] = (ia, ib)
            offs[(a, b)] = (seen.get(a, 0), seen.get(b, 0))
            for c, arr in ((a, ia), (b, ib)):
                foot.setdefault(c, []).append(arr)
            seen[a] = seen.get(a, 0) + len(ia)
            seen[b] = seen.get(b, 0) + len(ib)
        P = {a: (np.vstack(foot[a]) if foot.get(a)
                 else np.zeros((0, 2))) for a in keys}
        return P, pair_idx, offs

    def fit_translation(matches, Hcur):
        """Iteration-1 stage: per-camera 2-dof pitch translation only.
        The dominant cross-camera error is a systematic offset; solving
        a full 8-param H on 5-8 m mismatches drags the fit into a
        self-consistent but landmark-violating basin. The reference
        camera is fixed — pairs carry no absolute-position signal."""
        P, pair_idx, offs = _blocks(matches)
        fidx = {a: i for i, a in enumerate(free)}

        def resid(t):
            r = []
            for a in free:
                d = (_proj(Hcur[a], lm_src[a]) + t[2 * fidx[a]:2 * fidx[a] + 2]
                     - lm_dst[a]) * W_LM
                r.append(np.nan_to_num(d, nan=1e3, posinf=1e3,
                                       neginf=-1e3).ravel())
            for (a, b) in pair_idx:
                i0, j0 = offs[(a, b)]
                n = len(pair_idx[(a, b)][0])
                ta = (t[2 * fidx[a]:2 * fidx[a] + 2] if a in fidx
                      else np.zeros(2))
                tb = (t[2 * fidx[b]:2 * fidx[b] + 2] if b in fidx
                      else np.zeros(2))
                d = (_proj(Hcur[a], P[a][i0:i0 + n]) + ta
                     - _proj(Hcur[b], P[b][j0:j0 + n]) - tb) * W_PAIR
                r.append(np.nan_to_num(d, nan=1e3, posinf=1e3,
                                       neginf=-1e3).ravel())
            return np.concatenate(r)

        sol = least_squares(resid, np.zeros(2 * len(free)),
                            loss="soft_l1", f_scale=F_SCALE,
                            max_nfev=1000)
        out = dict(Hcur)
        for a in free:
            T = np.eye(3)
            T[0, 2], T[1, 2] = sol.x[2 * fidx[a]], sol.x[2 * fidx[a] + 1]
            out[a] = T @ Hcur[a]
        return out

    def fit_ransac(matches, Hcur, proj):
        """Per-camera RANSAC homography aligned on the reference camera
        (first key — the others' pooled matches map their projections
        onto the reference's). Mismatched Hungarian pairs are random
        outliers; correct pairs all share the one true relative
        homography, which RANSAC recovers where least_squares drifts.
        Applied in pitch space: H_a <- G_a @ H_a."""
        try:
            import cv2
        except ImportError:
            return Hcur

        def _rms_one(a, Ha):
            ds = [np.hypot(apply_h(Ha, fx, fy)[0] - X,
                           apply_h(Ha, fx, fy)[1] - Y)
                  for fx, fy, X, Y in lm[a]]
            return float(np.sqrt(np.mean(np.square(ds)))) if ds else 0.0

        out = dict(Hcur)
        for a in free:
            src, dst = [], []
            for (lo_a, hi_a), hits in matches.items():
                if a not in (lo_a, hi_a):
                    continue
                other = hi_a if a == lo_a else lo_a
                if other != ref:
                    continue
                for k, i, j in hits:
                    ia, ja = (i, j) if a == lo_a else (j, i)
                    src.append(proj[a][k][ia])
                    dst.append(proj[ref][k][ja])
            if len(src) < 10:
                continue
            S = np.asarray(src, dtype=np.float64)
            D = np.asarray(dst, dtype=np.float64)
            ok = np.isfinite(S).all(axis=1) & np.isfinite(D).all(axis=1)
            if ok.sum() < 10:
                continue
            G, _inl = cv2.findHomography(
                S[ok], D[ok], cv2.RANSAC, 1.5, maxIters=10000,
                confidence=0.999)
            if G is None or abs(G[2, 2]) <= 1e-9:
                continue
            Ha = G / G[2, 2] @ Hcur[a]
            # RANSAC ignores landmarks — revert cams it would break
            if _rms_one(a, Ha) <= max(MAX_LM_RMS_M,
                                    1.5 * _rms_one(a, Hcur[a])):
                out[a] = Ha
        return out

    def trim(matches, proj):
        """Drop matched pairs whose residual under the current Hs is
        an outlier for that camera pair (> 1.5x median, floor 1.5 m) —
        Hungarian mismatches concentrate in the upper tail and are what
        drag the full solve into a landmark-violating basin."""
        out = {}
        for combo, hits in matches.items():
            ds = np.array([np.hypot(
                proj[combo[0]][k][i][0] - proj[combo[1]][k][j][0],
                proj[combo[0]][k][i][1] - proj[combo[1]][k][j][1])
                for k, i, j in hits])
            if len(ds) < 8:
                out[combo] = hits
                continue
            cap = max(1.5, 1.5 * float(np.median(ds)))
            out[combo] = [h for h, d in zip(hits, ds) if d <= cap]
        return out

    def fit(matches, Hcur):
        """Simultaneous 8-param-per-camera fit over the FREE cameras
        (reference fixed as the gauge anchor — matched pairs carry no
        absolute-position signal, so fitting all 24 params lets the
        whole system drift into a consistent but landmark-violating
        solution). Landmark residuals weight 3.0, matched pair
        cross-camera differences weight 1.0."""
        P, pair_idx, offs = _blocks(matches)

        def _HF(x, a):
            i = fidx[a]
            return np.array([x[8 * i:8 * i + 3], x[8 * i + 3:8 * i + 6],
                             [x[8 * i + 6], x[8 * i + 7], 1.0]])

        def resid(x):
            Hs = {a: (_HF(x, a) if a in fidx else Hcur[ref])
                  for a in keys}
            projs = {a: _proj(Hs[a], P[a]) for a in keys}
            r = []
            for a in free:
                d = (_proj(Hs[a], lm_src[a]) - lm_dst[a]) * W_LM
                r.append(np.nan_to_num(d, nan=1e3, posinf=1e3,
                                       neginf=-1e3).ravel())
            for (a, b) in pair_idx:
                i0, j0 = offs[(a, b)]
                n = len(pair_idx[(a, b)][0])
                d = (projs[a][i0:i0 + n] - projs[b][j0:j0 + n]) * W_PAIR
                r.append(np.nan_to_num(d, nan=1e3, posinf=1e3,
                                       neginf=-1e3).ravel())
            return np.concatenate(r) if r else np.zeros(1)

        x0 = np.concatenate(
            [(Hcur[a] / Hcur[a][2, 2]).ravel()[:8] for a in free])
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=F_SCALE,
                            x_scale="jac", max_nfev=3000)
        return {a: (_HF(sol.x, a) if a in fidx else Hcur[a])
                for a in keys}

    def cap_landmarks(Hnew, Hprev):
        """Per camera: if the new H violates the landmark bound, blend
        back toward the previous H (bisection on the matrix) until it
        doesn't — constraints are enforced per iteration, not just at
        the final accept check."""
        def _rms_one(a, Ha):
            ds = [np.hypot(apply_h(Ha, fx, fy)[0] - X,
                           apply_h(Ha, fx, fy)[1] - Y)
                  for fx, fy, X, Y in lm[a]]
            return float(np.sqrt(np.mean(np.square(ds)))) if ds else 0.0
        out = dict(Hnew)
        for a in free:
            Ha = out[a] / out[a][2, 2]
            if _rms_one(a, Ha) <= MAX_LM_RMS_M:
                continue
            base = Hprev[a] / Hprev[a][2, 2]
            al, ah = 0.0, 1.0          # al feasible, ah infeasible
            for _ in range(25):
                am = 0.5 * (al + ah)
                Hm = base + am * (Ha - base)
                Hm = Hm / Hm[2, 2]
                if _rms_one(a, Hm) <= MAX_LM_RMS_M:
                    al = am
                else:
                    ah = am
            Hm = base + al * (Ha - base)
            out[a] = Hm / Hm[2, 2]
        return out

    report = []
    med0, matches, inside = pair_meds(H_lm, MATCH_M_FIRST)
    report.append({"iter": 0, "lm_rms": lm_rms(H_lm),
                   "pair_median": {f"{a}-{b}": round(v, 3)
                                   for (a, b), v in med0.items()},
                   "n_pairs": {f"{a}-{b}": len(h)
                               for (a, b), h in matches.items()},
                   "inside": round(inside, 3)})
    # gauge anchor: the best-connected camera stays at its landmark H
    cnt = {a: 0 for a in keys}
    for (a, b), hits in matches.items():
        cnt[a] += len(hits)
        cnt[b] += len(hits)
    ref = max(keys, key=lambda a: cnt[a])
    free = [a for a in keys if a != ref]
    fidx = {a: i for i, a in enumerate(free)}
    Hcur = dict(H_lm)
    best = (dict(H_lm), math.inf)

    def score(hmap):
        lr = lm_rms(hmap)
        md, _m, _i = pair_meds(hmap, MATCH_M)
        pm = [v for v in md.values() if math.isfinite(v)]
        return (max(pm) if pm else math.inf) \
            + 3.0 * max(lr.values())

    for it in range(1, iters + 1):
        _med, matches, inside = pair_meds(
            Hcur, MATCH_M_FIRST if it == 1 else MATCH_M)
        n_pairs = sum(len(h) for h in matches.values())
        if n_pairs < 8:
            return {"accepted": False,
                    "reason": f"only {n_pairs} matched pairs at iter "
                              f"{it}", "H": {a: H_lm[a].tolist()
                                              for a in keys},
                    "H_landmark": {a: H_lm[a].tolist() for a in keys},
                    "report": report}
        proj = _project_all(steps, Hcur)
        if it <= 2:
            Hnew = fit_translation(matches, Hcur)
        elif it <= 4:
            Hnew = fit_ransac(matches, Hcur, proj)
        else:
            Hnew = fit(trim(matches, proj), Hcur)
        Hnew = cap_landmarks(Hnew, Hcur)
        med1, _m, inside1 = pair_meds(Hnew, MATCH_M)
        report.append({"iter": it,
                       "lm_rms": {a: round(v, 3)
                                  for a, v in lm_rms(Hnew).items()},
                       "pair_median": {f"{a}-{b}": round(v, 3)
                                       for (a, b), v in med1.items()},
                       "n_pairs": {f"{a}-{b}": len(h)
                                   for (a, b), h in matches.items()},
                       "inside": round(inside1, 3)})
        s_new = score(Hnew)
        if s_new < best[1]:
            best = (dict(Hnew), s_new)
        Hcur = Hnew

    # keep the best iterate rather than the last — a late drift must not
    # win just because it came last
    if best[1] < score(Hcur):
        Hcur = best[0]
    final_lm = lm_rms(Hcur)
    med_fin, _m, inside_f = pair_meds(Hcur, MATCH_M)
    bad_lm = [a for a in keys if final_lm[a] > MAX_LM_RMS_M]
    worse = [c for c in med_fin if med_fin[c] >= med0.get(c, math.inf)]
    if bad_lm:
        reason = ("landmark rms blew up: " + ", ".join(
            f"a{a} {final_lm[a]:.1f} m" for a in bad_lm))
    elif worse:
        reason = ("pair median did not improve: " + ", ".join(
            f"{c[0]}-{c[1]} {med0[c]:.2f} -> {med_fin[c]:.2f} m"
            for c in worse))
    else:
        reason = None
    accepted = reason is None
    Hout = Hcur if accepted else H_lm
    return {"accepted": accepted, "reason": reason,
            "H": {a: Hout[a].tolist() for a in keys},
            "H_refined": {a: Hcur[a].tolist() for a in keys},
            "H_landmark": {a: H_lm[a].tolist() for a in keys},
            "lm_rms": {str(a): round(final_lm[a], 3) for a in keys},
            "pairs": {f"{a}-{b}": {"before": round(med0.get((a, b), 0.0), 3),
                                   "after": round(med_fin.get((a, b), 0.0), 3),
                                   "n": len(matches.get((a, b), []))}
                      for (a, b) in med_fin},
            "inside": round(inside_f, 3),
            "report": report}


def apply_joint(project_dir: Path, players_v2: Path, *,
                dry_run: bool = False, log=print) -> dict:
    """Load calib/dets/stabs/sync, run joint_refine, and (unless
    dry_run) write H + H_landmark + "joint" into calib.json. Stale
    H_refined/"refine" keys from the old anchor path are removed."""
    from highlights.io import write_json_atomic

    from .run import _load_json, resolve_context
    project_dir = Path(project_dir)
    players_v2 = Path(players_v2)
    cpath = project_dir / "multiangle" / "calib.json"
    calib = _load_json(cpath) or {}
    ctx = resolve_context(project_dir)
    offsets = ctx["offsets"]
    n = int(ctx["n_angles"])
    dets, stabs = {}, {}
    for a in range(n):
        f = players_v2 / f"det_a{a}.npz"
        if f.is_file():
            dets[a] = load_dets(f)
        s = load_stab(players_v2 / f"stab_a{a}.npz")
        if s is not None:
            stabs[a] = s
    res = joint_refine(calib, dets, stabs, offsets,
                       window=ctx["window"], log=log)
    for row in res.get("report") or []:
        log(f"joint iter {row['iter']}: lm_rms "
            f"{row['lm_rms']} pair_med {row['pair_median']} "
            f"n {row['n_pairs']} inside {row['inside']}")
    log(f"joint calib: {'accepted' if res['accepted'] else 'REJECTED'}"
        + (f" ({res['reason']})" if res.get("reason") else ""))
    if dry_run:
        return res
    angles = calib.setdefault("angles", {})
    for a, H in res.get("H", {}).items():
        e = angles.setdefault(str(a), {})
        e["H"] = H
        e["H_landmark"] = res["H_landmark"][a]
        e.pop("H_refined", None)
    calib.pop("refine", None)
    calib["joint"] = {
        "accepted": bool(res["accepted"]),
        "reason": res.get("reason"),
        "pairs": res.get("pairs") or {},
        "lm_rms": res.get("lm_rms") or {},
        "inside": res.get("inside"),
        "n_iters": len(res.get("report") or []) - 1,
        "updated_at": time.time(),
    }
    write_json_atomic(cpath, calib, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.joint_calib")
    ap.add_argument("project_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--players-v2", type=Path, default=None,
                    help="det/stab dir (default analysis/players_v2)")
    args = ap.parse_args(argv)
    pv2 = args.players_v2 or args.project_dir / "analysis" / "players_v2"
    res = apply_joint(args.project_dir, pv2, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("H", "H_refined", "H_landmark")},
                     indent=1, default=str))
    return 0 if res.get("accepted") else 1


if __name__ == "__main__":
    sys.exit(main())
