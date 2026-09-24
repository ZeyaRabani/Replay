"""Plot energy z-score over the full match with detected peaks + whistles marked."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(features: Path, events: Path, out: Path, panns: Path | None = None):
    d = json.loads(features.read_text())
    M = np.array(d["rows"])
    c = d["columns"]
    t = M[:, c.index("t")] / 60
    ev = json.loads(events.read_text())["events"]
    exc = [e for e in ev if e["type"] == "excitement"]
    wh = [e for e in ev if e["type"] == "other" and e["signals"].get("kind") == "whistle"]
    warm = [e for e in ev if e["type"] == "other" and e["signals"].get("kind") == "warmup"]

    n = 3 if panns else 2
    _fig, ax = plt.subplots(n, 1, figsize=(20, 3.2 * n), sharex=True)
    ax[0].plot(t, M[:, c.index("rms_db")], lw=0.4, color="0.4", label="RMS dB")
    ax[0].plot(t, M[:, c.index("speech_db")] - 10 * np.log10(1024), lw=0.4, color="tab:orange", alpha=0.8, label="speech band 300-3k dB")
    ax[0].set_ylabel("level (dB)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Audio track - full match")

    z = M[:, c.index("z300_speech_s5")]
    ax[1].plot(t, M[:, c.index("z300_rms")], lw=0.3, color="0.6", label="z300 rms (1 s)")
    ax[1].plot(t, z, lw=0.8, color="tab:blue", label="z300 speech band, 5 s smoothed")
    for i, e in enumerate(sorted(exc, key=lambda e: -e["confidence"])):
        ax[1].plot(e["t"] / 60, e["signals"]["z300_speech_s5_peak"], "v", color="tab:red", ms=7 if i < 8 else 4)
        if i < 8:
            ax[1].annotate(f"{int(e['t'] // 60)}:{int(e['t'] % 60):02d}", (e["t"] / 60, e["signals"]["z300_speech_s5_peak"]),
                           textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="tab:red")
    for e in warm:
        ax[1].plot(e["t"] / 60, e["signals"]["z300_speech_s5_peak"], "v", color="tab:gray", ms=5)
    for w in wh:
        ax[1].axvline(w["t"] / 60, color="tab:green", lw=0.4, alpha=0.5)
    ax[1].axhline(1.5, color="k", lw=0.5, ls="--")
    ax[1].set_ylabel("z-score")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].text(0.005, 0.92, "green lines = whistles; red = excitement peaks (large = top 8); grey = warm-up peaks", transform=ax[1].transAxes, fontsize=8)

    if panns:
        pd_ = json.loads(panns.read_text())
        PM = np.array(pd_["rows"])
        pc = pd_["columns"]
        tp = PM[:, 0] / 60
        for name, col in [("Cheering", "tab:red"), ("Applause", "tab:purple"), ("Shout", "tab:orange"),
                          ("Whistle", "tab:green"), ("Speech", "0.5"), ("Wind", "tab:cyan")]:
            ax[2].plot(tp, PM[:, pc.index(name)], lw=0.5, color=col, label=name, alpha=0.9)
        ax[2].set_ylabel("PANNs CNN14 prob")
        ax[2].legend(loc="upper right", fontsize=8, ncol=6)
    ax[-1].set_xlabel("video time (min)")
    ax[-1].set_xlim(0, t[-1])
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("events", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--panns", type=Path)
    a = ap.parse_args()
    main(a.features, a.events, a.out, a.panns)
