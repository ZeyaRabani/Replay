"""Fuse per-angle candidate events onto the shared timeline.

Each angle's pipeline/candidates.json events are mapped to T (t + offset_i),
then greedily clustered within ±4 s. Two or more angles agreeing upgrades a
candidate to cross_validation="confirmed"; single-angle stays "pipeline_only";
type disagreement inside a cluster flags signals.disputed.
"""

from __future__ import annotations

import json
from pathlib import Path

TYPE_PRIO = {"goal": 5, "shot": 4, "chance": 3, "excitement": 2, "other": 1}
CLUSTER_WIN = 4.0
PRE, POST = 3.0, 5.0


def fuse_events(events_by_angle: list[list[dict]], offsets: list[float],
                labels: list[str]) -> dict:
    """events_by_angle[i]: candidate events (t in angle file time).
    Returns a candidates.json-shaped dict on the shared timeline."""
    items = []
    for i, evs in enumerate(events_by_angle):
        for e in evs:
            items.append({**e, "_angle": i, "_T": float(e["t"]) + offsets[i]})
    items.sort(key=lambda x: x["_T"])

    clusters: list[list[dict]] = []
    for it in items:
        if clusters and it["_T"] - clusters[-1][-1]["_T"] <= CLUSTER_WIN:
            clusters[-1].append(it)
        else:
            clusters.append([it])

    out_events = []
    for c in clusters:
        weights = [max(1e-3, float(e.get("confidence", 0.5))) for e in c]
        t = sum(e["_T"] * w for e, w in zip(c, weights)) / sum(weights)
        angles = sorted({e["_angle"] for e in c})
        types = {e.get("type", "other") for e in c}
        etype = max(types, key=lambda x: TYPE_PRIO.get(x, 0))
        conf = 1.0
        for e in c:
            conf *= 1 - float(e.get("confidence", 0.5))
        conf = min(0.95, 1 - conf) * (1.0 if len(angles) >= 2 else 0.85)
        cross = len(angles) >= 2
        disputed = len(types) > 1
        out_events.append({
            "t": round(t, 1),
            "t_start": max(0.0, round(t - PRE, 1)),
            "t_end": round(t + POST, 1),
            "type": etype,
            "confidence": round(conf, 3),
            "signals": {
                "angles": angles,
                "angle_conf": {i: round(float(e.get("confidence", 0.5)), 3)
                               for i, e in zip([e["_angle"] for e in c], c)},
                **({"disputed": True, "types": sorted(types)} if disputed else {}),
            },
            "notes": (f"cross-confirmed by angles {','.join(map(str, angles))}"
                      if cross else
                      f"single-angle (angle {angles[0]}: {labels[angles[0]]})"),
            "cross_validation": "confirmed" if cross else "pipeline_only",
            "status": "pending",
        })
    out_events.sort(key=lambda e: -e["confidence"])
    for k, e in enumerate(out_events, 1):
        e["id"] = f"event_{k:03d}"
        e["rank"] = k
    return {"source": "multiangle", "events": out_events, "candidates": out_events}


def to_output_time(events: list[dict], lo: float) -> list[dict]:
    """Shared T -> output (rendered-video) time: out_t = T - lo."""
    out = []
    for e in events:
        e = dict(e)
        for k in ("t", "t_start", "t_end", "clip_start", "clip_end"):
            if k in e and e[k] is not None:
                e[k] = round(float(e[k]) - lo, 1)
        out.append(e)
    return out


def fuse_candidates(candidates_files: list[str | Path], offsets: list[float],
                    labels: list[str], out_path: str | Path) -> dict:
    events_by_angle = []
    for f in candidates_files:
        d = json.loads(Path(f).read_text())
        events_by_angle.append(d.get("events") or d.get("candidates") or [])
    out = fuse_events(events_by_angle, offsets, labels)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    from highlights.io import write_json_atomic
    write_json_atomic(out_path, out, indent=1)
    return out
