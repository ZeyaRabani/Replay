#!/usr/bin/env python3
"""Build the fused event-label list from the four reviewer JSONs.

Keeps events of type goal/shot/chance whose t falls inside the match
window [1050, 4990]. Writes highlights/fusion/outputs/labels.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HL = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs", "labels.json")

MATCH_LO, MATCH_HI = 1050, 4990
TYPES = {"goal", "shot", "chance"}

FILES = [
    ("review1", os.path.join(HL, "review1", "outputs", "review1_events.json")),
    ("review2", os.path.join(HL, "review2", "outputs", "events.json")),
    ("review3", os.path.join(HL, "review3", "outputs", "review3_events.json")),
    ("review4", os.path.join(HL, "review4", "outputs", "review4_events.json")),
]


def main():
    labels = []
    for src, path in FILES:
        with open(path) as f:
            events = json.load(f)["events"]
        n = 0
        for ev in events:
            if ev.get("type") in TYPES and MATCH_LO <= ev.get("t", -1) <= MATCH_HI:
                labels.append({"t": float(ev["t"]), "type": ev["type"],
                               "confidence": float(ev.get("confidence", 0.0)),
                               "source": src})
                n += 1
        print(f"{src}: {n} events kept (of {len(events)})", flush=True)

    labels.sort(key=lambda e: e["t"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"wrote {OUT}: {len(labels)} labels", flush=True)


if __name__ == "__main__":
    main()
