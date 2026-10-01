"""Manual zones, version 2: per-keyframe zone sets.

zones.json v2 = {"version": 2, "angles": [ [ {"t": <file seconds>,
"zones": [poly, ...]}, ... ] per angle ]} — keyframes sorted by t; each
keyframe's zones apply from its t until the next keyframe's t (a camera
that gets moved mid-match can be re-zoned on a later still).

Legacy format {"angles": [[poly, ...] per angle], "ref_t": [t per
angle]} loads as one keyframe at ref_t[i] (or 0.3 * duration).
"""

from __future__ import annotations

import numpy as np


def _poly_ok(p) -> bool:
    return isinstance(p, list) and len(p) >= 3


def normalize_zones(zd: dict, durations: list[float]) -> list[list[dict]]:
    """Return per-angle keyframe lists [{"t": float, "zones": [poly..]}],
    sorted by t. Accepts both v2 (angle entries are keyframe dicts) and
    legacy (angle entries are polygon lists) shapes."""
    angles = (zd or {}).get("angles") or []
    ref_ts = (zd or {}).get("ref_t") or []
    out: list[list[dict]] = []
    for i, entry in enumerate(angles):
        kfs: list[dict] = []
        if entry and isinstance(entry[0], dict):
            # v2
            for kf in entry:
                t = float(kf.get("t") or 0.0)
                zs = [p for p in (kf.get("zones") or []) if _poly_ok(p)]
                kfs.append({"t": t, "zones": zs})
        elif entry:
            # legacy: a flat list of polygons -> single keyframe
            rt = (ref_ts[i] if i < len(ref_ts) and ref_ts[i] is not None
                  else None)
            dur = durations[i] if i < len(durations) else 0.0
            t = float(rt) if rt is not None else (float(dur) * 0.3
                                                  if dur else 0.0)
            kfs.append({"t": t, "zones": [p for p in entry if _poly_ok(p)]})
        kfs.sort(key=lambda k: k["t"])
        out.append(kfs)
    return out


def zones_at(kfs: list[dict], file_t: float) -> list:
    """Zones of the last keyframe with t <= file_t; the first keyframe's
    when file_t precedes all of them; [] when there are none."""
    if not kfs:
        return []
    chosen = kfs[0]
    for kf in kfs:
        if kf["t"] <= file_t:
            chosen = kf
        else:
            break
    return chosen.get("zones") or []


def kf_index(kfs: list[dict], T: int, lo: float, off: float,
             dur: float) -> np.ndarray:
    """int array [T]: which keyframe of this angle is in effect at each
    output second, using the file-time mapping ft = clip(t + lo - off).
    Times before the first keyframe map to keyframe 0 (same as zones_at)."""
    if not kfs:
        return np.zeros(T, dtype=int)
    times = np.asarray([k["t"] for k in kfs], dtype=float)
    ft = np.clip(np.arange(T) + lo - off, 0.0, max(0.0, dur))
    idx = np.searchsorted(times, ft, side="right") - 1
    return np.clip(idx, 0, len(kfs) - 1).astype(int)
