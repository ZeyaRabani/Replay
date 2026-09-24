"""Run PANNs CNN14 (AudioSet tagger, CPU) over the match in sliding windows.

Windows of `win` seconds with hop `hop` seconds. For each window we keep the
probabilities of a handful of football-relevant AudioSet classes. Weights
(~300 MB) are downloaded once by panns_inference to ~/panns_data.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from panns_inference import AudioTagging, labels

CLASSES = ["Cheering", "Applause", "Crowd", "Whistle", "Whistling", "Shout", "Yell",
           "Children shouting", "Speech", "Wind", "Wind noise (microphone)", "Music",
           "Clapping", "Chant", "Whoop", "Screaming", "Laughter"]


def run(wav: Path, out: Path, win: float = 2.0, hop: float = 1.0, batch: int = 32) -> None:
    torch.set_num_threads(8)
    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 32000:  # PANNs expects 32 kHz
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=32000)
        sr = 32000
    at = AudioTagging(checkpoint_path=None, device="cpu")
    idx = [labels.index(c) for c in CLASSES]
    n = int(win * sr)
    h = int(hop * sr)
    starts = np.arange(0, len(y) - n, h)
    rows = []
    t0 = time.time()
    for b in range(0, len(starts), batch):
        seg = np.stack([y[s:s + n] for s in starts[b:b + batch]])
        clip, _ = at.inference(seg)
        for s, p in zip(starts[b:b + batch], clip):
            rows.append([round(float(s / sr + win / 2), 2)] + [round(float(p[i]), 4) for i in idx])
        if b % (batch * 20) == 0:
            print(f"{b}/{len(starts)} windows, {time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "audio_panns_cnn14", "step_s": hop, "window_s": win,
                               "columns": ["t"] + CLASSES, "rows": rows}))
    print(f"done {len(rows)} windows in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    a = ap.parse_args()
    run(a.wav, a.out, a.win, a.hop)
