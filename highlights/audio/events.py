"""Turn per-second audio features (+ optional PANNs tags, whistle segments) into events.

Output follows the shared hl3 schema (see README). Times are seconds from the
start of the video file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

PANNS_EXCITE = ["Cheering", "Applause", "Crowd", "Shout", "Yell", "Children shouting", "Clapping", "Whoop", "Screaming"]


def load_table(p: Path):
    d = json.loads(p.read_text())
    M = np.array(d["rows"], dtype=float)
    return d, {c: M[:, i] for i, c in enumerate(d["columns"])}


def half_time_gap(t: np.ndarray, rms_db: np.ndarray, onset_density: np.ndarray, min_len_s: float = 120):
    """Longest stretch of low activity (level and onset density both in the bottom quartile,
    smoothed over 60 s). Returns (t_start, t_end) or None."""
    k = 60
    lvl = uniform_filter1d(rms_db, k, mode="nearest")
    ons = uniform_filter1d(onset_density, k, mode="nearest")
    low = (lvl < np.percentile(lvl, 25)) & (ons < np.percentile(ons, 25))
    best = None
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if (j - i) >= min_len_s and (best is None or (j - i) > (best[1] - best[0])):
                best = (i, j)
            i = j
        else:
            i += 1
    if best is None:
        return None
    return float(t[best[0]]), float(t[best[1] - 1])


def main(features: Path, out: Path, panns: Path | None, whistles: Path | None,
         z_thr: float = 1.5, min_sep_s: float = 20.0, max_events: int = 40, match_start_s: float = 1020.0):
    meta, F = load_table(features)
    t = F["t"]
    step = meta["step_s"]
    dur = meta["video_duration_s"]

    # --- excitement score: sustained speech-band elevation vs 5-min baseline
    z = F["z300_speech_s5"]
    zr = F["z300_rms"]
    tags = None
    if panns and panns.exists():
        pm, P = load_table(panns)
        # align PANNs (window centres) to feature grid by nearest time
        tp = P["t"]
        idx = np.clip(np.searchsorted(tp, t), 0, len(tp) - 1)
        tags = {c: P[c][idx] for c in pm["columns"] if c != "t"}
        excite_tag = np.max(np.stack([tags[c] for c in PANNS_EXCITE]), axis=0)
        excite_tag_s = uniform_filter1d(excite_tag, round(5 / step), mode="nearest")
    else:
        excite_tag_s = np.zeros_like(z)

    # combined score: z-score (dominant) + tagger evidence (bonus)
    score = z + 2.0 * excite_tag_s

    wsegs = []
    if whistles and whistles.exists():
        wsegs = json.loads(whistles.read_text())["whistles"]

    peaks, props = find_peaks(score, height=z_thr, distance=round(min_sep_s / step))
    order = np.argsort(props["peak_heights"])[::-1][:max_events]
    events = []
    for pi in order:
        i = int(peaks[pi])
        # supporting window: contiguous region around peak where z > 0.5*peak (min 0.8)
        thr = max(0.8, 0.5 * z[i])
        a = i
        while a > 0 and z[a - 1] > thr:
            a -= 1
        b = i
        while b < len(z) - 1 and z[b + 1] > thr:
            b += 1
        t_start, t_end = float(t[a]), float(t[b] + step)
        duration = t_end - t_start
        near_w = [s for s in wsegs if abs((s["t_start"] + s["t_end"]) / 2 - t[i]) <= 5]
        sig = {
            "z300_speech_s5_peak": round(float(z[i]), 2),
            "z300_rms_peak": round(float(zr[i]), 2),
            "rms_db_peak": round(float(F["rms_db"][a:b + 1].max()), 1),
            "duration_s": round(duration, 1),
            "onset_density_mean": round(float(F["onset_density"][a:b + 1].mean()), 2),
            "whistle_within_5s": bool(near_w),
            "whistle_times": [round((s["t_start"] + s["t_end"]) / 2, 1) for s in near_w],
        }
        if tags is not None:
            for c in ["Cheering", "Applause", "Crowd", "Shout", "Speech", "Whistle", "Wind"]:
                sig[f"panns_{c.lower()}_max"] = round(float(tags[c][a:b + 1].max()), 3)
        # confidence: saturating function of z, duration, tagger support
        conf = 1 - np.exp(-(z[i] / 2.5)) * 1.0
        conf = 0.6 * conf + 0.25 * min(1.0, duration / 8.0) + 0.15 * float(excite_tag_s[i])
        if near_w:
            conf = min(1.0, conf + 0.05)
        notes = (f"sustained speech-band level {z[i]:.1f} sd above 5-min baseline for {duration:.0f}s"
                 + ("; referee whistle within 5 s" if near_w else "")
                 + (f"; PANNs cheer/shout p={excite_tag_s[i]:.2f}" if tags is not None else ""))
        ev = {
            "type": "excitement", "t": float(t[i]), "t_start": t_start, "t_end": t_end,
            "confidence": round(float(np.clip(conf, 0, 1)), 3), "signals": sig, "notes": notes,
        }
        if t[i] < match_start_s:
            ev["type"] = "other"
            ev["signals"]["kind"] = "warmup"
            ev["notes"] = "WARM-UP (before kick-off) audio peak: " + notes
        events.append(ev)

    # --- whistle events
    for s in wsegs:
        conf = (min(1.0, s["duration_s"] / 0.6) * 0.5
                + min(1.0, s["tonality"] / 60) * 0.3
                + (0.2 if s["peak_hz_std"] < 300 else 0.05))
        tm = round((s["t_start"] + s["t_end"]) / 2, 2)
        events.append({
            "type": "other", "t": tm, "t_start": s["t_start"], "t_end": s["t_end"],
            "confidence": round(float(conf), 3),
            "signals": {"kind": "whistle", "warmup": bool(tm < match_start_s),
                        **{k: v for k, v in s.items() if k not in ("t_start", "t_end")}},
            "notes": f"referee whistle {s['duration_s']}s @ {s['peak_hz']:.0f} Hz"
                     + (" (warm-up period)" if tm < match_start_s else ""),
        })

    events.sort(key=lambda e: e["t"])
    gap = half_time_gap(t, F["rms_db"], F["onset_density"])
    top_notes = ("audio track: grassroots match, tiny crowd; excitement = sustained speech-band "
                 "elevation vs rolling 300 s baseline. ")
    if gap:
        top_notes += f"Longest low-activity stretch (half-time candidate): {gap[0]:.0f}-{gap[1]:.0f}s."
    else:
        top_notes += "No >=2 min low-activity stretch found (half-time likely cut from the video)."
    result = {"source": "audio", "video_duration_s": dur, "events": events,
              "notes": top_notes, "half_time_gap": gap, "match_start_s": match_start_s}
    out.write_text(json.dumps(result, indent=1))
    print(out, len(events), "events;", sum(e["type"] == "excitement" for e in events), "excitement")
    print(top_notes)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--panns", type=Path)
    ap.add_argument("--whistles", type=Path)
    ap.add_argument("--z-thr", type=float, default=1.5)
    ap.add_argument("--match-start", type=float, default=1020.0,
                    help="peaks before this are tagged type=other/kind=warmup")
    a = ap.parse_args()
    main(a.features, a.out, a.panns, a.whistles, a.z_thr, match_start_s=a.match_start)
