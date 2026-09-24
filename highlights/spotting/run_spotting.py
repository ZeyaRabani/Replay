#!/usr/bin/env python3
"""Run pretrained E2E-Spot (SoccerNet-v2, 17 classes) on a match video, CPU only.

Pipeline:
  1. ffmpeg -> 398x224 JPEG frames at 2 fps (cached in --work_dir/frames)
  2. sliding 100-frame clips (50% overlap, optional h-flip TTA), softmax averaged
  3. raw per-frame class probabilities -> scores.npz
  4. per-class NMS -> events.json (shared schema), features_1s.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from e2e_model import SOCCERNET_CLASSES, load_model
from PIL import Image
from tqdm import tqdm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FRAME_W, FRAME_H = 398, 224
SAMPLE_FPS = 2.0
PAD_LEN = 5

CLASS_TO_TYPE = {
    "Goal": "goal",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Penalty": "chance",
    "Corner": "chance",
    "Direct free-kick": "chance",
    "Indirect free-kick": "chance",
    # On this fixed-camera footage this class empirically fires on near-goal action (see README), not on the ball leaving play.
    "Ball out of play": "excitement",
}
# supporting window (seconds before / after the spotted frame) per event type
TYPE_WINDOW = {"goal": (10.0, 8.0), "shot": (6.0, 4.0), "chance": (4.0, 6.0), "excitement": (8.0, 4.0), "other": (3.0, 3.0)}


def read_json(path: Path) -> dict:
    with open(path) as fp:
        return json.load(fp)


def write_json(obj, path: Path, indent: int | None = None) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=indent)


def video_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], text=True
    )
    return float(out.strip())


def extract_frames(video: str, frame_dir: Path) -> int:
    done = frame_dir / "DONE"
    if done.exists():
        return len(list(frame_dir.glob("*.jpg")))
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video,
        "-vf", f"fps={SAMPLE_FPS},scale={FRAME_W}:{FRAME_H}", "-q:v", "2",
        str(frame_dir / "%06d.jpg"),
    ]
    subprocess.check_call(cmd)
    n = len(list(frame_dir.glob("*.jpg")))
    done.write_text(str(n))
    return n


def load_all_frames(frame_dir: Path, n: int) -> torch.Tensor:
    cache = frame_dir / "frames_u8.npy"
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape[0] == n:
            return torch.from_numpy(np.ascontiguousarray(arr))
    arr = np.empty((n, FRAME_H, FRAME_W, 3), np.uint8)
    for i in tqdm(range(n), desc="load frames"):
        # ffmpeg numbers frames from 1
        arr[i] = np.asarray(Image.open(frame_dir / f"{i + 1:06d}.jpg").convert("RGB"))
    np.save(cache, arr)
    return torch.from_numpy(arr)


def normalize(clip_u8: torch.Tensor) -> torch.Tensor:
    x = clip_u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def run_inference(model, frames_u8: torch.Tensor, clip_len: int, overlap: int, flip_tta: bool, batch_size: int):
    n = frames_u8.shape[0]
    num_classes = model._pred_fine._fc_out._fc_out.out_features
    scores = np.zeros((n, num_classes), np.float32)
    support = np.zeros(n, np.int32)
    step = clip_len - overlap
    starts = list(range(-PAD_LEN, max(1, n - overlap), step))

    def make_clip(start: int) -> torch.Tensor:
        s, e = max(start, 0), min(start + clip_len, n)
        x = normalize(frames_u8[s:e])
        pad_front, pad_back = s - start, (start + clip_len) - e
        if pad_front or pad_back:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, pad_front, pad_back))
        return x

    with torch.inference_mode():
        for bi in tqdm(range(0, len(starts), batch_size), desc="inference"):
            batch_starts = starts[bi : bi + batch_size]
            x = torch.stack([make_clip(s) for s in batch_starts])
            prob = torch.softmax(model(x), dim=2)
            if flip_tta:
                prob = 0.5 * (prob + torch.softmax(model(x.flip(-1)), dim=2))
            prob = prob.numpy()
            for k, start in enumerate(batch_starts):
                p = prob[k]
                if start < 0:
                    p, start = p[-start:], 0
                end = min(start + p.shape[0], n)
                scores[start:end] += p[: end - start]
                support[start:end] += 1
    assert support.min() > 0
    return scores / support[:, None]


def nms_1d(score: np.ndarray, window_frames: int, threshold: float) -> list[int]:
    keep = []
    s = score.copy()
    while True:
        i = int(np.argmax(s))
        if s[i] < threshold:
            break
        keep.append(i)
        lo, hi = max(0, i - window_frames), min(len(s), i + window_frames + 1)
        s[lo:hi] = -1.0
    return sorted(keep)


def build_events(probs: np.ndarray, classes: list[str], threshold: float, nms_s: float, duration: float,
                 warmup_end: float, topk: int | None = None, only_classes: set[str] | None = None) -> list[dict]:
    """Per-class NMS peaks >= threshold. If topk is set, instead keep the top-k peaks per class regardless of threshold."""
    events = []
    win = round(nms_s * SAMPLE_FPS)
    for ci, cname in enumerate(classes):
        if only_classes and cname not in only_classes:
            continue
        col = ci + 1  # column 0 is background
        peaks = nms_1d(probs[:, col], win, 0.0 if topk else threshold)
        if topk:
            peaks = sorted(sorted(peaks, key=lambda i: -probs[i, col])[:topk])
        for fi in peaks:
            t = fi / SAMPLE_FPS
            etype = CLASS_TO_TYPE.get(cname, "other")
            before, after = TYPE_WINDOW[etype]
            warmup = t < warmup_end
            if warmup:
                etype = "other"
            top3_idx = np.argsort(-probs[fi, 1:])[:3]
            top3 = [[classes[j], round(float(probs[fi, j + 1]), 4)] for j in top3_idx]
            score = float(probs[fi, col])
            events.append(
                {
                    "type": etype,
                    "t": round(t, 2),
                    "t_start": round(max(0.0, t - before), 2),
                    "t_end": round(min(duration, t + after), 2),
                    "confidence": round(score, 4),
                    "signals": {
                        "class": cname,
                        "score": round(score, 4),
                        "top3": top3,
                        "background": round(float(probs[fi, 0]), 4),
                        "kind": "warmup" if warmup else "match",
                    },
                    "notes": f"E2E-Spot {cname} p={score:.2f} at {t:.1f}s (2fps frame {fi})",
                }
            )
    events.sort(key=lambda e: (-e["confidence"], e["t"]))
    return events


def build_features_1s(probs: np.ndarray, classes: list[str], duration: float) -> dict:
    n_sec = int(np.ceil(duration))
    cols = ["t"] + [f"p_{c.lower().replace(' ', '_').replace('->', '_to_')}" for c in classes] + ["p_foreground_max"]
    rows = []
    for s in range(n_sec):
        lo, hi = int(s * SAMPLE_FPS), int((s + 1) * SAMPLE_FPS)
        seg = probs[lo:hi]
        if seg.shape[0] == 0:
            continue
        mx = seg.max(axis=0)
        row = [float(s)] + [round(float(v), 4) for v in mx[1:]] + [round(float(mx[1:].max()), 4)]
        rows.append(row)
    return {"source": "spotting", "step_s": 1.0, "columns": cols, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model_dir", default=str(Path(__file__).parent / "third_party/e2e-spot-models/soccer_challenge_rny008gsm_gru_rgb"))
    ap.add_argument("--work_dir", default=str(Path.home() / "match/spotting_work"))
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "outputs"))
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--nms_s", type=float, default=10.0)
    ap.add_argument("--no_flip_tta", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--max_seconds", type=float, default=None, help="only process the first N seconds (throughput test)")
    ap.add_argument("--topk_per_class", type=int, default=10)
    ap.add_argument("--reuse_scores", action="store_true", help="skip inference and re-postprocess out_dir/scores.npz")
    ap.add_argument("--warmup_end", type=float, default=0.0,
                    help="events before this time are typed 'other' with signals.kind='warmup'")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    work, out = Path(args.work_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = read_json(Path(args.model_dir) / "config.json")
    ckpt = sorted(Path(args.model_dir).glob("checkpoint_*.pt"))[-1]

    t0 = time.time()
    duration = video_duration(args.video)
    n = extract_frames(args.video, work / "frames")
    t_frames = time.time() - t0
    frames = load_all_frames(work / "frames", n)
    if args.max_seconds:
        frames = frames[: int(args.max_seconds * SAMPLE_FPS)]
        duration = min(duration, args.max_seconds)
    print(f"{frames.shape[0]} frames @ {SAMPLE_FPS} fps (extract {t_frames:.0f}s), duration {duration:.1f}s")

    t1 = time.time()
    if args.reuse_scores and (out / "scores.npz").exists():
        probs = np.load(out / "scores.npz")["probs"]
        assert probs.shape[0] == frames.shape[0]
    else:
        model = load_model(str(ckpt), cfg)
        probs = run_inference(model, frames, cfg["clip_len"], cfg["clip_len"] // 2, not args.no_flip_tta, args.batch_size)
    t_inf = time.time() - t1
    print(f"inference {t_inf:.0f}s ({frames.shape[0] / t_inf:.1f} frames/s, {duration / t_inf:.1f}x realtime)")

    classes = SOCCERNET_CLASSES
    np.savez_compressed(
        out / "scores.npz", t=np.arange(probs.shape[0]) / SAMPLE_FPS, probs=probs,
        classes=np.array(["background"] + classes),
    )
    events = build_events(probs, classes, args.threshold, args.nms_s, duration, args.warmup_end)
    write_json(
        {"source": "spotting", "video_duration_s": duration, "model": ckpt.parent.name,
         "sample_fps": SAMPLE_FPS, "nms_s": args.nms_s, "threshold": args.threshold, "events": events},
        out / "events.json", indent=1,
    )
    write_json(build_features_1s(probs, classes, duration), out / "features_1s.json")
    # Top-k peaks for the highlight-relevant classes even when far below threshold (domain-gap fallback).
    topk_events = build_events(probs, classes, 0.0, args.nms_s, duration, args.warmup_end, topk=args.topk_per_class,
                               only_classes=set(CLASS_TO_TYPE))
    write_json(
        {"source": "spotting_topk", "video_duration_s": duration, "model": ckpt.parent.name,
         "note": f"top-{args.topk_per_class} NMS peaks per class for {sorted(CLASS_TO_TYPE)}; confidences are raw and may be tiny",
         "events": topk_events},
        out / "events_topk.json", indent=1,
    )

    fg = probs[:, 1:]
    match_mask = (np.arange(probs.shape[0]) / SAMPLE_FPS) >= args.warmup_end

    def baseline(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {}
        sub = fg[mask]
        return {
            c: {"p50": round(float(np.percentile(sub[:, i], 50)), 5), "p90": round(float(np.percentile(sub[:, i], 90)), 5),
                "p99": round(float(np.percentile(sub[:, i], 99)), 5), "p99.9": round(float(np.percentile(sub[:, i], 99.9)), 5),
                "max": round(float(sub[:, i].max()), 4),
                "max_over_p99": round(float(sub[:, i].max() / max(np.percentile(sub[:, i], 99), 1e-6)), 1)}
            for i, c in enumerate(classes)
        }

    summary = {
        "warmup_end_s": args.warmup_end,
        "baseline_match_period": baseline(match_mask),
        "baseline_warmup_period": baseline(~match_mask),
        "runtime_s": {"frame_extraction": round(t_frames, 1), "inference": round(t_inf, 1), "total": round(time.time() - t0, 1)},
        "frames": int(probs.shape[0]),
        "events_ge_0.1": int(sum(e["confidence"] >= 0.1 for e in events)),
        "events_ge_0.3": int(sum(e["confidence"] >= 0.3 for e in events)),
        "events_ge_0.5": int(sum(e["confidence"] >= 0.5 for e in events)),
        "per_class": {
            c: {"max": round(float(fg[:, i].max()), 4), "mean": round(float(fg[:, i].mean()), 5),
                "n_ge_0.3": int(sum(e["confidence"] >= 0.3 and e["signals"]["class"] == c for e in events)),
                "n_ge_0.5": int(sum(e["confidence"] >= 0.5 and e["signals"]["class"] == c for e in events))}
            for i, c in enumerate(classes)
        },
        "frac_frames_background_argmax": round(float((probs.argmax(1) == 0).mean()), 4),
    }
    write_json(summary, out / "summary.json", indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
