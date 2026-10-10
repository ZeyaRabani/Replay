"""Per-camera image stabilization: time-varying homography current ->
reference frame, estimated from background SIFT features.

Phone cameras drift/tilt during a match but calibration landmarks are
clicked on one still (ref_t = 0.3*duration). stabilize computes, per
camera and per ~1 s step, the pixel homography that maps the current
frame back onto that reference frame; fusion and the anchor resolver
compose it with the static calibration H.

    python -m highlights.analysis.stabilize --project-dir P
        [--angle A] [--force]

writes players_v2/stab_a{a}.npz {t (file s), H (n,3,3) normalized
coords, inliers, valid, step_s} + stab_a{a}.meta.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MIN_INLIERS = 25
DET_TOL_S = 0.6          # det row <-> decode frame match tolerance


def estimate_h(ref_gray: np.ndarray, cur_gray: np.ndarray,
               ref_mask: np.ndarray | None,
               cur_mask: np.ndarray | None
               ) -> tuple[np.ndarray | None, int]:
    """SIFT+RANSAC homography mapping CURRENT frame pixels -> REFERENCE
    frame pixels. (None, n_inliers) when the fit is weak or insane."""
    import cv2
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_r, des_r = sift.detectAndCompute(ref_gray, ref_mask)
    kp_c, des_c = sift.detectAndCompute(cur_gray, cur_mask)
    if (des_r is None or des_c is None or len(kp_r) < 8
            or len(kp_c) < 8):
        return None, 0
    good = [m for mm in cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            des_c, des_r, k=2)
            if len(mm) == 2 for m, n2 in [mm]
            if m.distance < 0.75 * n2.distance]
    if len(good) < 8:
        return None, 0
    cur_pts = np.float32([kp_c[m.queryIdx].pt for m in good])
    ref_pts = np.float32([kp_r[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(cur_pts, ref_pts, cv2.RANSAC, 3.0)
    n_in = int(inl.sum()) if inl is not None else 0
    if H is None or n_in < MIN_INLIERS:
        return None, n_in
    h, w = cur_gray.shape[:2]
    sc = math.sqrt(abs(float(np.linalg.det(H[:2, :2]))))
    if (abs(H[0, 2]) >= 0.35 * w or abs(H[1, 2]) >= 0.35 * h
            or abs(H[2, 0]) >= 1e-3 or abs(H[2, 1]) >= 1e-3
            or not 0.8 <= sc <= 1.25):
        return None, n_in
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H, n_in


def _probe_hw(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True)
    w, h = (int(v) for v in out.stdout.strip().split(",")[:2])
    return w, h


def _gray_reader(video: Path, lo: float, hi: float, w: int, h: int,
                 step_s: float):
    """Yield decode-gray frames at 1/step_s fps, file time lo+k*step_s."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{lo:.3f}", "-to",
           f"{hi:.3f}", "-i", str(video), "-vf",
           f"fps={1.0 / step_s:.6f},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    n_bytes = w * h
    k = 0
    while True:
        buf = proc.stdout.read(n_bytes)
        if len(buf) < n_bytes:
            break
        yield lo + k * step_s, np.frombuffer(buf, dtype=np.uint8
                                             ).reshape(h, w).copy()
        k += 1
    proc.wait()


def _grab_gray(video: Path, t: float, w: int, h: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
           "-frames:v", "1", "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if len(out.stdout) < w * h:
        raise RuntimeError(f"no frame at t={t:.1f} in {video}")
    return np.frombuffer(out.stdout[:w * h], dtype=np.uint8
                         ).reshape(h, w).copy()


def _mask(h: int, w: int, boxes: np.ndarray, sw: float,
          sh: float) -> np.ndarray:
    """255 mask with each source-pixel box (scaled to the decode grid,
    dilated 25%) zeroed so players are not used as features."""
    m = np.full((h, w), 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        bx1, bx2 = x1 * sw, x2 * sw
        by1, by2 = y1 * sh, y2 * sh
        dx, dy = (bx2 - bx1) * 0.25, (by2 - by1) * 0.25
        xa = int(max(0, bx1 - dx)); xb = int(min(w, bx2 + dx))
        ya = int(max(0, by1 - dy)); yb = int(min(h, by2 + dy))
        m[ya:yb, xa:xb] = 0
    return m


def _det_boxes(det: dict | None, t: float, nearest: bool = False
               ) -> np.ndarray | None:
    """Source-pixel boxes near file time t; None = no mask at all."""
    if det is None or not len(det["t"]):
        return None
    if nearest:
        i = int(np.abs(det["t"] - t).argmin())
        if abs(float(det["t"][i]) - t) > DET_TOL_S:
            return np.zeros((0, 4))
        idx = np.flatnonzero(det["t"] == det["t"][i])
    else:
        idx = np.flatnonzero(np.abs(det["t"] - t) <= DET_TOL_S)
    return det["box"][idx]


def _fill_invalid(H: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation of the 9 entries between the nearest valid
    neighbours; holds at the ends; identities when nothing is valid."""
    H = H.copy()
    ok = np.flatnonzero(valid)
    if not len(ok):
        H[:] = np.eye(3)
        return H
    for i in np.flatnonzero(~valid):
        lo_i = ok[ok < i].max() if (ok < i).any() else None
        hi_i = ok[ok > i].min() if (ok > i).any() else None
        if lo_i is None:
            H[i] = H[hi_i]
        elif hi_i is None:
            H[i] = H[lo_i]
        else:
            a = (i - int(lo_i)) / (int(hi_i) - int(lo_i))
            H[i] = H[lo_i] * (1 - a) + H[hi_i] * a
    return H


def compute_stab(video: Path, ref_t: float, lo: float, hi: float,
                 out: Path, *, dets_npz: Path | None = None,
                 step_s: float = 1.0, width: int = 960,
                 log=print) -> dict:
    """Decode `video` gray at 1/step_s fps over [lo,hi] (file s), fit each
    frame's H onto the ref_t frame, fill gaps, write npz + meta.json."""
    from .fuse_tracks import load_dets
    w_in, h_in = _probe_hw(video)
    w = width & ~1
    h = round(width * h_in / w_in) & ~1
    ref = _grab_gray(video, ref_t, w, h)
    det = load_dets(dets_npz) if dets_npz and Path(dets_npz).is_file() \
        else None
    sw = w / float(det["w"]) if det is not None and det.get("w") else 1.0
    sh = h / float(det["h"]) if det is not None and det.get("h") else 1.0

    def mask_at(t: float, nearest: bool = False):
        boxes = _det_boxes(det, t, nearest)
        return _mask(h, w, boxes, sw, sh) if boxes is not None else None

    ref_mask = mask_at(ref_t, nearest=True)
    ts, Hs, inls, val = [], [], [], []
    t0 = time.time()
    for k, (t, frame) in enumerate(_gray_reader(video, lo, hi, w, h,
                                                step_s)):
        Hk, n_in = estimate_h(ref, frame, ref_mask, mask_at(t))
        ts.append(t)
        Hs.append(Hk if Hk is not None else np.eye(3))
        inls.append(n_in)
        val.append(Hk is not None)
        if k % 300 == 0:
            log(f"stabilize {Path(video).name}: frame {k} "
                f"({sum(val)}/{len(val)} valid, "
                f"{time.time() - t0:.0f}s)")
    if not ts:
        raise RuntimeError(f"no frames decoded for {video}")
    t_arr = np.asarray(ts, dtype=np.float32)
    H_arr = np.asarray(Hs, dtype=np.float64)
    valid = np.asarray(val, dtype=bool)
    inliers = np.asarray(inls, dtype=np.int32)
    H_arr = _fill_invalid(H_arr, valid)
    N = np.diag([1.0 / w, 1.0 / h, 1.0])
    Ni = np.diag([float(w), float(h), 1.0])
    Hn = N @ H_arr @ Ni
    Hn = Hn / Hn[:, 2:3, 2:3]
    tmp = Path(out).with_suffix(".tmp.npz")
    np.savez(tmp, t=t_arr, H=Hn, inliers=inliers, valid=valid,
             step_s=np.float64(step_s))
    os.replace(tmp, out)
    meta = {"ref_t": ref_t, "width": w, "height": h, "step_s": step_s,
            "n": len(t_arr), "n_valid": int(valid.sum()),
            "lo": lo, "hi": hi}
    Path(out).with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=1))
    log(f"stabilize {Path(video).name}: {int(valid.sum())}/{len(ts)} "
        f"valid -> {Path(out).name}")
    return meta


def load_stab(path: Path) -> dict | None:
    try:
        z = np.load(path)
        return {"t": z["t"], "H": z["H"], "inliers": z["inliers"],
                "valid": z["valid"],
                "step_s": float(z["step_s"]) if "step_s" in z else 0.0}
    except Exception:
        return None


def warp_at(stab: dict, file_t: float, fx: float, fy: float
            ) -> tuple[float, float]:
    """Map normalized (fx,fy) at file_t through the nearest stab H;
    unchanged when no frame within 1.5*step_s."""
    t = stab.get("t")
    if t is None or not len(t):
        return fx, fy
    i = int(np.abs(t - file_t).argmin())
    step = float(stab.get("step_s") or 0.0)
    if not step and len(t) > 1:
        step = float(np.median(np.diff(t)))
    if step and abs(float(t[i]) - file_t) > 1.5 * step:
        return fx, fy
    p = stab["H"][i] @ np.array([fx, fy, 1.0])
    if abs(p[2]) < 1e-12:
        return fx, fy
    return float(p[0] / p[2]), float(p[1] / p[2])


def _stab_one(args: tuple) -> tuple[int, dict]:
    a, video, ref_t, lo, hi, out, dets = args
    return a, compute_stab(video, ref_t, lo, hi, out, dets_npz=dets,
                           log=lambda _m: None)


def run_stabilize(project_dir: Path, players_v2: Path,
                  angles: list[int] | None = None, force: bool = False,
                  log=print, workers: int | None = None) -> None:
    """Per-angle stab_a{a}.npz over the det window; ref_t = the frame
    landmarks were clicked on (probe duration_s * 0.3)."""
    from .run import _angle_dirs, _load_json, _match_ext_video, resolve_context
    ctx = resolve_context(project_dir)
    dirs = _angle_dirs(project_dir)
    n = int(ctx["n_angles"])
    jobs = []
    for a in (angles if angles is not None else range(n)):
        out = Path(players_v2) / f"stab_a{a}.npz"
        if out.exists() and not force:
            log(f"stabilize a{a}: skip (up to date)")
            continue
        v = _match_ext_video(dirs[a]) if a < len(dirs) else None
        if v is None:
            log(f"stabilize a{a}: no source video — skipped")
            continue
        probe = _load_json(project_dir / "angles" / f"a{a}" / "pipeline"
                           / "probe.json") or {}
        ref_t = float(probe.get("duration_s") or 60.0) * 0.3
        off = ctx["offsets"][a] if a < len(ctx["offsets"]) else 0.0
        dnpz = Path(players_v2) / f"det_a{a}.npz"
        lo, hi = max(0.0, ctx["window"][0] - off), \
            max(0.0, ctx["window"][1] - off)
        dets = None
        if dnpz.is_file():
            zt = np.load(dnpz)["t"]
            if len(zt):
                lo, hi = float(zt.min()), float(zt.max())
                dets = dnpz
        jobs.append((a, v, ref_t, lo, hi, out, dets))
    if not jobs:
        return
    workers = workers or int(os.environ.get("HL_ANGLE_WORKERS") or 2)
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            a, meta = _stab_one(job)
            log(f"stabilize a{a}: {meta['n_valid']}/{meta['n']} frames "
                f"valid")
        return
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for a, meta in ex.map(_stab_one, jobs):
            log(f"stabilize a{a}: {meta['n_valid']}/{meta['n']} frames "
                f"valid")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.stabilize")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--angle", type=int, action="append",
                    help="limit to this angle (repeatable)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    run_stabilize(args.project_dir,
                  args.project_dir / "analysis" / "players_v2",
                  angles=args.angle, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
