"""Automatic sync check for multi-angle cuts.

Two independent checks, both against the original camera files:

- check_map: is the camera line-up (timemap) right? Every ~60 s of the
  match, each angle's audio is cross-correlated with the reference
  angle's audio at the mapped time; reliable windows must agree to
  within MAP_TOL_S.
- check_render: is the rendered cut faithful to the line-up? For every
  segment the cut's audio is matched against the on-screen angle's own
  audio at the expected file time (catches audio drift), and sampled
  frames are matched against the source frames (catches picture jumps).

Audio matching uses an onset envelope (100 Hz) and normalised
cross-correlation; a window only counts when its peak is clearly above
the rest (z > Z_MIN), so quiet stretches never cause false alarms.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR, HOP = 4000, 40
FPS = SR // HOP            # envelope frames per second
Z_MIN = 6.0                # peak z-score for a window to count
AUDIO_TOL_S = 0.10         # rendered audio vs on-screen source
MAP_TOL_S = 0.30           # angle vs reference (line-up)
VIDEO_TOL_S = 0.10         # rendered frame vs source frame
MAX_BAD_FRAC = 0.10        # reliable windows allowed outside tolerance
MIN_RELIABLE = 5           # fewer reliable windows -> "unverified"
CANVAS = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
W, H = 192, 108


def envelope(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    x = np.diff(np.frombuffer(raw, np.int16).astype(np.float32))
    n = len(x) // HOP
    e = np.log1p((x[:n * HOP].reshape(n, HOP) ** 2).mean(1))
    return np.maximum(0, np.diff(e, prepend=e[0])).astype(np.float32)


def best_lag(w: np.ndarray, ref: np.ndarray, centre_s: float,
             search_s: float) -> tuple[float, float] | None:
    """Lag (s) of window w inside ref around centre_s, and peak z-score."""
    s = int(search_s * FPS)
    c = round(centre_s * FPS)
    lo, hi = c - s, c + s + len(w)
    if lo < 0 or hi > len(ref) or len(w) < 3 * FPS or w.std() < 1e-6:
        return None
    seg = ref[lo:hi]
    w = (w - w.mean()) / w.std()
    n = len(w)
    cs = np.concatenate([[0], np.cumsum(seg)])
    cs2 = np.concatenate([[0], np.cumsum(seg * seg)])
    m = (cs[n:] - cs[:-n]) / n
    v = np.maximum((cs2[n:] - cs2[:-n]) / n - m * m, 1e-9)
    corr = np.correlate(seg, w, "valid") / n / np.sqrt(v)
    k = int(np.argmax(corr))
    z = (corr[k] - corr.mean()) / (corr.std() + 1e-9)
    return (k - s) / FPS, float(z)


def _verdict(lags: list[float], tol: float) -> dict:
    a = np.abs(np.asarray(lags, dtype=float))
    if len(a) < MIN_RELIABLE:
        return {"ok": None, "reliable": len(a)}
    bad = float(np.mean(a > tol))
    return {"ok": bool(bad <= MAX_BAD_FRAC and np.median(a) <= tol / 2),
            "reliable": len(a),
            "median_abs_s": round(float(np.median(a)), 3),
            "p90_abs_s": round(float(np.percentile(a, 90)), 3),
            "max_abs_s": round(float(a.max()), 3),
            "bad_frac": round(bad, 3)}


def check_map(videos: list[str], tm, lo: float, hi: float,
              step_s: float = 60.0, win_s: float = 20.0,
              envs: list[np.ndarray] | None = None) -> dict:
    """Line-up check: each angle vs angle 0 every step_s of shared [lo, hi]."""
    from highlights.multiangle.syncmap import as_tm, shared_to_file
    tm = as_tm(tm)
    envs = envs or [envelope(v) for v in videos]
    out = {}
    for a in range(1, len(videos)):
        lags, bad_at = [], []
        for T in np.arange(lo + 5, hi - win_s - 5, step_s):
            fa = shared_to_file(tm, a, float(T))
            if fa is None or fa < 0 or (fa + win_s) * FPS > len(envs[a]):
                continue
            w = envs[a][int(fa * FPS):int((fa + win_s) * FPS)]
            r = best_lag(w, envs[0], float(T), 8.0)
            if r and r[1] > Z_MIN:
                lags.append(r[0])
                if abs(r[0]) > MAP_TOL_S:
                    bad_at.append([round(float(T), 1), round(r[0], 2)])
        out[str(a)] = {**_verdict(lags, MAP_TOL_S), "off_at": bad_at[:20]}
    oks = [v["ok"] for v in out.values()]
    return {"ok": not any(o is False for o in oks),
            "verified": all(o is True for o in oks), "angles": out,
            "bad_angles": [int(k) for k, v in out.items() if v["ok"] is False]}


def _gray_frames(path: str, ss: float, dur: float,
                 canvas: bool) -> np.ndarray:
    vf = (CANVAS + "," if canvas else "") + f"fps=30,scale={W}:{H},format=gray"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, ss):.3f}", "-i",
         str(path), "-t", f"{dur:.3f}", "-vf", vf, "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def check_render(out_path: str, videos: list[str], tm,
                 segments: list[dict], union_lo: float,
                 n_frames: int = 12,
                 envs: list[np.ndarray] | None = None) -> dict:
    """Rendered-cut check. segments: director segments in output time
    (t_start/t_end/angle), plain-speed only (replays are skipped)."""
    from highlights.multiangle.syncmap import as_tm, shared_to_file
    tm = as_tm(tm)
    envs = envs or [envelope(v) for v in videos]
    oute = envelope(out_path)
    plain = [s for s in segments
             if float(s.get("speed", 1.0)) == 1.0 and not s.get("overlay")
             and "t_src_start" not in s]
    lags, bad_at = [], []
    for s in plain:
        a = int(s["angle"])
        t0, t1 = float(s["t_start"]) + 0.6, float(s["t_end"]) - 0.6
        if t1 - t0 < 3:
            continue
        fa = shared_to_file(tm, a, t0 + union_lo)
        if fa is None:
            continue
        r = best_lag(oute[int(t0 * FPS):int(t1 * FPS)], envs[a], fa, 3.0)
        if r and r[1] > Z_MIN:
            lags.append(r[0])
            if abs(r[0]) > AUDIO_TOL_S:
                bad_at.append([round(t0, 1), a, round(r[0], 2)])
    audio = {**_verdict(lags, AUDIO_TOL_S), "off_at": bad_at[:20]}
    # picture: sampled segments, output frame vs source frames around the
    # expected file time; ambiguous (static) frames don't count against
    long_ = [s for s in plain if float(s["t_end"]) - float(s["t_start"]) >= 3]
    vrows, vbad = [], []
    for j in (np.linspace(0, len(long_) - 1, min(n_frames, len(long_)))
              .astype(int) if long_ else []):
        s = long_[int(j)]
        a = int(s["angle"])
        t = float(s["t_start"]) + 1.5
        fe = shared_to_file(tm, a, t + union_lo)
        if fe is None:
            continue
        try:
            o = _gray_frames(out_path, t, 1 / 30, False)[0]
            span = 3.0
            src = _gray_frames(videos[a], fe - span, 2 * span, True)
        except (subprocess.CalledProcessError, IndexError, ValueError):
            continue
        mad = np.abs(src - o).mean((1, 2))
        k = int(np.argmin(mad))
        base = max(0.0, fe - span)
        lag = base + k / 30 - fe
        ke = round((fe - base) * 30)
        near = mad[max(0, ke - 3):ke + 4]
        ok = abs(lag) <= VIDEO_TOL_S or (
            len(near) and near.min() <= mad[k] + 0.5)
        vrows.append(round(lag, 3))
        if not ok:
            vbad.append([round(t, 1), a, round(lag, 2)])
    video = {"checked": len(vrows), "bad": vbad,
             "ok": None if len(vrows) < 3 else len(vbad) == 0}
    ok = audio["ok"] is not False and video["ok"] is not False
    return {"ok": bool(ok),
            "verified": audio["ok"] is True and video["ok"] is True,
            "audio": audio, "video": video}


def check_project(project_dir: str, cut: str | None = None) -> dict:
    """Audit an existing project's active (or given) cut."""
    from highlights.multiangle.syncmap import as_tm
    p = Path(project_dir)
    sync = json.loads((p / "multiangle" / "sync.json").read_text())
    tm = as_tm(sync.get("timemap") or sync["offsets"])
    cuts = sorted((p / "multiangle" / "cuts").glob("*/"))
    cdir = next((c for c in cuts if cut and c.name == cut), cuts[-1])
    meta = json.loads((cdir / "meta.json").read_text())
    d = json.loads((cdir / "director.json").read_text())
    lo, hi = meta["range"]
    videos = [str(v) for v in sorted(p.glob("angles/a*/match.mp4"))]
    t = time.time()
    envs = [envelope(v) for v in videos]
    res = {"cut": cdir.name,
           "map": check_map(videos, tm, lo, hi, envs=envs),
           "render": check_render(str(cdir / "match.mp4"), videos, tm,
                                  d["segments"], lo, envs=envs)}
    res["ok"] = res["map"]["ok"] and res["render"]["ok"]
    res["elapsed_s"] = round(time.time() - t, 1)
    return res


if __name__ == "__main__":
    print(json.dumps(check_project(sys.argv[1],
                                   sys.argv[2] if len(sys.argv) > 2 else None),
                     indent=1))
