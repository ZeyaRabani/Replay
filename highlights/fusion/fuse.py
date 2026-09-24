#!/usr/bin/env python3
"""Fuse reviewer labels + pipeline signals into the canonical candidate list.

Inputs (all under highlights/):
  review{1..4}/outputs/*.json   visual review events (labels source of truth)
  fusion/outputs/features_1s.parquet
  fusion/outputs/peaks_{rule,learned}.json
  fusion/outputs/net_occupancy_1s.json
  tracking/outputs/events.json
  audio/outputs/events.json

Writes fusion/outputs/candidates.json and app/outputs/candidates.json
(the latter keeps the app schema: top-level `events` + per-event
t_start/t_end/status fields alongside the fusion fields).
"""
import json
import os
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HL = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs", "candidates.json")
APP_OUT = os.path.join(HL, "app", "outputs", "candidates.json")

MATCH_LO, MATCH_HI = 1050, 4990
VIDEO_DUR = 5337.153
GOAL_T = 2736.0            # user-confirmed goal (see REPORT.md)
GOAL_NOTES = ("Direct visual re-check: the ball is visible in the near net "
              "and the keeper enters the net to retrieve it.")
REJECT_NOTES = ("Rejected as a goal: tracking's attack-to-centre heuristic "
                "is too noisy and direct visual review found no goal.")
PIPE_ONLY_NOTES = ("Held-out learned fusion peak without an independently "
                   "observed event.")

REVIEW_FILES = [
    ("review1", "review1/outputs/review1_events.json"),
    ("review2", "review2/outputs/events.json"),
    ("review3", "review3/outputs/review3_events.json"),
    ("review4", "review4/outputs/review4_events.json"),
]
TYPE_PRIO = {"goal": 3, "shot": 2, "chance": 1}
FEAT_COLS = ["z300_rms", "whistle", "motion_goal_roi", "net_disturbance",
             "n_near_box", "rush_near_3s", "p_ball_out_of_play", "p_goal",
             "p_shots_on_target", "p_shots_off_target"]
AUDIO_BONUS = ["z300_rms"]
TRK_BONUS = ["n_near_box", "rush_near_3s", "motion_goal_roi",
             "net_disturbance"]


def hhmmss(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def nearest(peaks, t, max_d=6.0):
    best = None
    for p in peaks:
        d = abs(p["t"] - t)
        if d <= max_d and (best is None or d < abs(best["t"] - t)):
            best = p
    return best


def feat_at(df, t, col):
    i = round(t)
    if i in df.index:
        v = df.at[i, col]
        return None if pd.isna(v) else float(v)
    return None


def main():
    # ---- inputs ----
    visual = []
    for src, rel in REVIEW_FILES:
        with open(os.path.join(HL, rel)) as f:
            for ev in json.load(f)["events"]:
                if ev.get("type") in TYPE_PRIO and \
                        MATCH_LO <= ev.get("t", -1) <= MATCH_HI:
                    visual.append({"t": float(ev["t"]), "type": ev["type"],
                                   "conf": float(ev.get("confidence", 0)),
                                   "notes": ev.get("notes", ""),
                                   "t_start": float(ev.get("t_start", ev["t"])),
                                   "t_end": float(ev.get("t_end", ev["t"])),
                                   "source": src})
    visual.sort(key=lambda e: e["t"])

    # dedupe clusters <=6 s apart: type = highest priority, t/notes from
    # the highest-confidence member, all sources kept
    clusters = []
    for ev in visual:
        if clusters and ev["t"] - clusters[-1][-1]["t"] <= 6.0:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])
    cands = []
    for cl in clusters:
        best = max(cl, key=lambda e: e["conf"])
        cands.append({
            "t": best["t"],
            "type": max(cl, key=lambda e: TYPE_PRIO[e["type"]])["type"],
            "visual_conf": best["conf"],
            "notes": best["notes"],
            "t_start": min(e["t_start"] for e in cl),
            "t_end": max(e["t_end"] for e in cl),
            "sources": sorted({e["source"] for e in cl}),
        })

    # authored correction: 2736 is a confirmed goal
    for c in cands:
        if abs(c["t"] - GOAL_T) <= 2.0:
            c.update(t=GOAL_T, type="goal", visual_conf=0.99,
                     notes=GOAL_NOTES, net_retrieval=True)

    df = pd.read_parquet(os.path.join(
        HERE, "outputs", "features_1s.parquet")).set_index("t")
    inm = df[(df.index >= MATCH_LO) & (df.index <= MATCH_HI)]
    p90 = {c: float(inm[c].quantile(0.90)) for c in FEAT_COLS}
    whistle_one = float(inm["whistle"].max())  # z-scored 1

    with open(os.path.join(HERE, "outputs", "peaks_rule.json")) as f:
        rule_peaks = json.load(f)
    with open(os.path.join(HERE, "outputs", "peaks_learned.json")) as f:
        learn_peaks = json.load(f)
    with open(os.path.join(HL, "tracking", "outputs", "events.json")) as f:
        trk_events = json.load(f)["events"]
    with open(os.path.join(HL, "audio", "outputs", "events.json")) as f:
        audio_events = json.load(f)["events"]

    def audio_near(t):
        return [{"kind": e.get("signals", {}).get("kind"), "t": e["t"],
                 "confidence": e.get("confidence")}
                for e in audio_events if abs(e["t"] - t) <= 3.0]

    def feat_signals(t):
        fv = {c: feat_at(df, t, c) for c in FEAT_COLS}
        hot = [c for c in FEAT_COLS
               if fv[c] is not None and
               ((c == "whistle" and abs(fv[c] - whistle_one) < 1e-9)
                or (c != "whistle" and fv[c] >= p90[c]))]
        return fv, hot

    def build_signals(t, sources=None, vconf=None, notes=None,
                      lp=None, rp=None, net_retrieval=False,
                      trk=None):
        fv, hot = feat_signals(t)
        sig = {
            "visual_review": ({"sources": sources, "confidence": vconf,
                               "notes": notes} if sources else None),
            "learned_fusion": ({"score": lp["score"], "rank": lp["rank"],
                                "top_signals": lp["top_signals"]}
                               if lp else None),
            "rule_fusion": ({"score": rp["score"], "rank": rp["rank"],
                             "top_signals": rp["top_signals"]}
                            if rp else None),
            "net_retrieval": bool(net_retrieval),
            "tracking_event": trk,
            "audio_events": audio_near(t),
            "features_at_t": fv,
        }
        contrib = []
        if sources:
            contrib.append("visual")
        if lp:
            contrib.append("learned_fusion")
        if rp:
            contrib.append("rule_fusion")
        contrib += hot
        if net_retrieval:
            contrib.append("net_retrieval")
        return sig, contrib

    # ---- visual candidates ----
    recs = []
    for c in cands:
        lp = nearest(learn_peaks, c["t"])
        rp = nearest(rule_peaks, c["t"])
        fv, _ = feat_signals(c["t"])
        audio_hit = ((fv["z300_rms"] or -9e9) >= p90["z300_rms"]
                     or (fv["whistle"] is not None
                         and abs(fv["whistle"] - whistle_one) < 1e-9))
        trk_hit = any((fv[k] or -9e9) >= p90[k] for k in TRK_BONUS)
        is_goal = abs(c["t"] - GOAL_T) <= 1e-9 and c["type"] == "goal"
        conf = 0.70 * c["visual_conf"]
        if lp:
            conf += 0.12
        if rp:
            conf += 0.08
        if audio_hit:
            conf += 0.05
        if trk_hit:
            conf += 0.05
        conf = min(conf, 0.98)
        if is_goal:
            conf = 0.99
        xv = "confirmed" if (lp or rp or is_goal) else "visual_only"
        sig, contrib = build_signals(
            c["t"], sources=c["sources"], vconf=c["visual_conf"],
            notes=c["notes"], lp=lp, rp=rp,
            net_retrieval=c.get("net_retrieval", False))
        recs.append({"type": c["type"], "t": c["t"], "confidence": conf,
                     "signals": sig, "contributing_signals": contrib,
                     "cross_validation": xv, "notes": c["notes"],
                     "t_start": c["t_start"], "t_end": c["t_end"],
                     "selected": bool(is_goal or c["visual_conf"] >= 0.60)})

    vis_ts = [r["t"] for r in recs]

    # ---- pipeline-only: top-20 learned peaks >12 s from visual ----
    for p in learn_peaks[:20]:
        if any(abs(p["t"] - vt) <= 12.0 for vt in vis_ts):
            continue
        lp, rp = p, nearest(rule_peaks, p["t"])
        conf = min(0.49, 0.25 + 0.24 * p["score"])
        sig, contrib = build_signals(p["t"], lp=lp, rp=rp)
        recs.append({"type": "chance", "t": p["t"], "confidence": conf,
                     "signals": sig, "contributing_signals": contrib,
                     "cross_validation": "pipeline_only",
                     "notes": PIPE_ONLY_NOTES,
                     "t_start": p["t"] - 3, "t_end": p["t"] + 3,
                     "selected": False})

    # ---- rejected tracking goals ----
    cand_ts = [r["t"] for r in recs]
    for e in trk_events:
        if e.get("type") != "goal":
            continue
        if abs(e["t"] - GOAL_T) <= 12.0:
            continue
        if any(abs(e["t"] - ct) <= 12.0 for ct in cand_ts):
            continue
        sig, contrib = build_signals(
            e["t"], trk={"t": e["t"], "type": e["type"],
                         "confidence": e.get("confidence"),
                         "signals": e.get("signals"),
                         "notes": e.get("notes")})
        recs.append({"type": "goal", "t": float(e["t"]),
                     "confidence": min(float(e.get("confidence", 0)), 0.25),
                     "signals": sig, "contributing_signals": contrib,
                     "cross_validation": "rejected", "notes": REJECT_NOTES,
                     "t_start": float(e.get("t_start", e["t"] - 3)),
                     "t_end": float(e.get("t_end", e["t"] + 3)),
                     "selected": False})

    # ---- finalize ----
    recs.sort(key=lambda r: -r["confidence"])
    for i, r in enumerate(recs):
        pad = 5.0 if r["type"] == "goal" else 3.0
        r["id"] = f"event_{i + 1:03d}"
        r["rank"] = i + 1
        r["timestamp"] = hhmmss(r["t"])
        r["clip_start"] = max(0.0, r["t"] - pad)
        r["clip_end"] = min(VIDEO_DUR, r["t"] + pad)
        r["status"] = ("rejected" if r["cross_validation"] == "rejected"
                       else "confirmed" if r["cross_validation"] == "confirmed"
                       else "pending")
        r.setdefault("t_start", r["clip_start"])
        r.setdefault("t_end", r["clip_end"])
        # field order
        ordered = {k: r[k] for k in
                   ["id", "rank", "type", "t", "timestamp", "confidence",
                    "signals", "contributing_signals", "cross_validation",
                    "status", "notes", "clip_start", "clip_end", "selected",
                    "t_start", "t_end"]}
        recs[i] = ordered

    summary = {
        "n_candidates": len(recs),
        "by_type": dict(Counter(r["type"] for r in recs)),
        "by_cross_validation": dict(Counter(r["cross_validation"]
                                            for r in recs)),
        "n_selected": sum(1 for r in recs if r["selected"]),
    }
    doc = {
        "source": "fusion",
        "source_video": "match.mp4",
        "local_video_filename": "match.mp4",
        "video_duration_s": VIDEO_DUR,
        "match_window": [MATCH_LO, MATCH_HI],
        "summary": summary,
        "candidates": recs,
        "events": recs,  # app schema alias
    }
    for path in (OUT, APP_OUT):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {path}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
