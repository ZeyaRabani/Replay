#!/usr/bin/env python3
"""ballfind stage: per-angle per-file-second ball detections with the
fine-tuned ball_v2 student model, cached incrementally in
angles/aN/track/ball_v2.json for load_director_inputs (BALL_SOURCE).

Detection config is identical to the yardstick dets_full_S1280v2 run:
imgsz=1280, conf=0.10, classes=[1] (ball), 1 fps ffmpeg extraction to
1280-wide jpegs, boxes scaled back to native pixels.

Model path: env REPLAY_BALL_MODEL, else /data/shared/models/ball_v2.pt,
else highlights/multiangle/models/ball_v2.pt (gitignored). No model ->
log + skip (not an error)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from highlights.io import write_json_atomic
from highlights.pipeline.probe import probe as ffprobe

IMGSZ = 1280
CONF = 0.10
CHUNK = 300          # max file seconds per ffmpeg extraction run
# static-ball filter, identical to yardstick exp3.load
RAD = 18             # native px centre distance
WIN = 60             # +- file seconds
REP = 5              # repeats elsewhere -> static


def _model_path() -> Path | None:
    env = os.environ.get("REPLAY_BALL_MODEL")
    cands = ([Path(env)] if env else []) + [
        Path("/data/shared/models/ball_v2.pt"),
        Path(__file__).parent / "models" / "ball_v2.pt",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def needed_seconds(ctx) -> dict[int, set[int]]:
    """File seconds the director needs per angle, same lo/hi and
    shared->file mapping as load_director_inputs."""
    from highlights.multiangle.syncmap import shared_to_file, timemap_from_sync
    sync = json.loads((ctx.pipe / "sync.json").read_text())
    from highlights.multiangle.run import apply_match_window_src
    apply_match_window_src(ctx, sync)
    tm = timemap_from_sync(
        sync, getattr(ctx, "durations", None)
        or [ctx.duration(i) for i in range(len(ctx.angles))]
        if ctx.angles else None)
    lo, hi = ctx.union(sync)
    T = int(np.ceil(hi - lo))
    out = {}
    for i in range(len(ctx.angles)):
        dur = ctx.duration(i)
        need = set()
        for k in range(T):
            ft = shared_to_file(tm, i, k + lo)
            if ft is not None and 0 <= ft <= dur - 1:
                need.add(round(ft))
        out[i] = need
    return out


def ball_rows(dets_doc: dict, thr: float = 0.25) -> dict[int, tuple]:
    """exp3.load exactly: keep dets >= thr, drop static boxes (>=REP
    other frames within WIN s have a >=thr box within RAD px), per
    second keep the max-score box -> (score, h/H, cx/W, cy/H)."""
    W, H = dets_doc["native_wh"]
    dets = {int(t): [b for b in v if b[4] >= thr]
            for t, v in dets_doc["dets"].items()}
    out = {}
    for t, bs in dets.items():
        keep = []
        for b in bs:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            rep = 0
            for u in range(t - WIN, t + WIN + 1):
                if u == t or u not in dets:
                    continue
                rep += any(((o[0] + o[2]) / 2 - cx) ** 2
                           + ((o[1] + o[3]) / 2 - cy) ** 2 <= RAD * RAD
                           for o in dets[u])
            if rep < REP:
                keep.append(b)
        if keep:
            b = max(keep, key=lambda x: x[4])
            out[t] = (b[4], (b[3] - b[1]) / H,
                      (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H)
    return out


def _extract(video: Path, S: int, E: int, tmp: Path) -> None:
    """1 fps jpegs of file seconds [S,E] at 1280 wide; frame n (1-based)
    is file second S+n-1. Identical to SPEC_full extraction."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(S), "-i", str(video),
         "-t", str(E - S + 1), "-vf", f"fps=1,scale={IMGSZ}:-2",
         "-q:v", "2", str(tmp / "%05d.jpg")],
        check=True)


def _runs(seconds: set[int], chunk: int = CHUNK):
    """Contiguous [S,E] runs covering `seconds`, split to <= chunk."""
    runs = []
    for t in sorted(seconds):
        if runs and t == runs[-1][1] + 1 and runs[-1][1] - runs[-1][0] + 1 < chunk:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return runs


def _load_model(path: Path):
    import torch
    torch.set_num_threads(4)
    from ultralytics import YOLO
    return YOLO(str(path))


def run_ballfind(ctx) -> None:
    mp = _model_path()
    if mp is None:
        ctx.log("ballfind: no model, skipping")
        return
    need = needed_seconds(ctx)
    model = None
    for i, a in enumerate(ctx.angles):
        video = ctx.angle_video(i)
        if video is None:
            ctx.log(f"ballfind: a{i} no video, skipping")
            continue
        out_p = a["dir"] / "track" / "ball_v2.json"
        doc = {}
        if out_p.exists():
            try:
                doc = json.loads(out_p.read_text()) or {}
            except (OSError, ValueError):
                doc = {}
        dets = {int(k): v for k, v in (doc.get("dets") or {}).items()}
        missing = need[i] - dets.keys()
        if not missing:
            ctx.log(f"ballfind: a{i} cached ({len(dets)} s)")
            continue
        pr = ffprobe(str(video))
        W, H = int(pr["width"]), int(pr["height"])
        scale = W / float(IMGSZ)
        doc = {"model": "ball_v2", "imgsz": IMGSZ, "native_wh": [W, H],
               "dets": {str(t): v for t, v in sorted(dets.items())}}
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if model is None:
            model = _load_model(mp)
        ctx.log(f"ballfind: a{i} {len(missing)} s to detect "
                f"({len(dets)} cached)")
        done = 0
        for S, E in _runs(missing):
            tmp = Path(tempfile.mkdtemp(prefix="ballfind_",
                                        dir=a["dir"] / "track"))
            try:
                _extract(video, S, E, tmp)
                for f in sorted(tmp.glob("*.jpg")):
                    t = S + int(f.stem) - 1
                    r = model.predict(str(f), imgsz=IMGSZ, conf=CONF,
                                      classes=[1], verbose=False)[0]
                    boxes = []
                    for b in r.boxes:
                        x0, y0, x1, y1 = (round(float(v) * scale, 1)
                                          for v in b.xyxy[0])
                        boxes.append([x0, y0, x1, y1,
                                      round(float(b.conf[0]), 3)])
                    boxes.sort(key=lambda x: -x[4])
                    doc["dets"][str(t)] = boxes
                    done += 1
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            write_json_atomic(out_p, doc)   # checkpoint each run
            ctx.status.update(
                stage_progress=(i + done / max(1, len(missing)))
                / len(ctx.angles),
                message=f"ballfind a{i} {done}/{len(missing)} s")
        ctx.log(f"ballfind: a{i} done, {len(doc['dets'])} s total")


def main(argv: list[str] | None = None) -> int:
    """python -m highlights.multiangle.ballfind <project_dir> —
    precompute ball_v2.json on a local mirror."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from highlights.multiangle.run import Ctx, _load_angles, _NoopStatus
    project_dir = Path((argv or sys.argv[1:])[0])
    ctx = Ctx(project_dir=project_dir, pipe=project_dir / "multiangle",
              status=_NoopStatus(),
              angles=_load_angles(project_dir, None, create_dirs=False),
              read_only=False)
    run_ballfind(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
