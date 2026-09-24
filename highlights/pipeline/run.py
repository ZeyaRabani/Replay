"""End-to-end highlight pipeline (Contract 2).

    python -m highlights.pipeline.run --project-dir DIR \
        (--youtube-url URL | --video FILE) [--stages a,b] [--cookies F] [--force]

Stages (in order): download, probe, audio, motion, features, score,
candidates, stats. Each stage is skipped when all of its outputs exist and
--force is not given. Progress is written to <project>/pipeline/status.json
and the log to <project>/pipeline/log.txt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from highlights.pipeline.errors import PipelineError
from highlights.pipeline.probe import probe as ffprobe
from highlights.pipeline.status import StatusWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


@dataclass
class Ctx:
    project_dir: Path
    pipe: Path
    status: StatusWriter
    youtube_url: str | None = None
    video_arg: Path | None = None
    cookies: str | None = None
    force: bool = False
    video_path: Path | None = None
    duration: float = 0.0
    log_fh: object = None
    match_window: tuple = (0.0, 0.0)
    halves: list = field(default_factory=list)
    mw_warning: str | None = None

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_fh:
            self.log_fh.write(line + "\n")
            self.log_fh.flush()


def _canonical_video(project_dir: Path) -> Path | None:
    for ext in VIDEO_EXTS:
        p = project_dir / f"match{ext}"
        if p.exists():
            return p
    return None


def _link_or_copy(src: Path, dst: Path) -> Path:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _resolve_video(ctx: Ctx) -> None:
    """Ensure ctx.video_path points at the canonical <project>/match.<ext>."""
    existing = _canonical_video(ctx.project_dir)
    if ctx.video_arg is not None:
        src = ctx.video_arg.resolve()
        if existing and existing.resolve() == src:
            ctx.video_path = existing
            return
        ext = src.suffix.lower() if src.suffix.lower() in VIDEO_EXTS else ".mp4"
        ctx.video_path = _link_or_copy(src, ctx.project_dir / f"match{ext}")
        ctx.log(f"local video {src} -> {ctx.video_path}")
        return
    if existing:
        ctx.video_path = existing
        return
    src_dir = ctx.project_dir / "source"
    if src_dir.is_dir():
        vids = [p for p in src_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        if len(vids) == 1:
            ctx.video_path = _link_or_copy(vids[0], ctx.project_dir / f"match{vids[0].suffix.lower()}")
            ctx.log(f"source video {vids[0]} -> {ctx.video_path}")
            return
    if ctx.youtube_url:
        ctx.video_path = ctx.project_dir / "match.mp4"
        return
    raise PipelineError("no video: pass --video, --youtube-url, drop one file "
                        "in source/, or provide an existing match.mp4")


# ------------------------------ stages ------------------------------------

def stage_download(ctx: Ctx) -> None:
    if ctx.video_arg is not None or _canonical_video(ctx.project_dir):
        ctx.log("download: canonical video already present, nothing to do")
        return
    if not ctx.youtube_url:
        raise PipelineError("download stage needs --youtube-url")
    from highlights.pipeline.download import download
    ctx.video_path = download(ctx.youtube_url, ctx.project_dir,
                              status=ctx.status, cookies=ctx.cookies, log=ctx.log)


def stage_probe(ctx: Ctx) -> None:
    info = ffprobe(ctx.video_path)
    ctx.duration = float(info["duration_s"])
    out = ctx.pipe / "probe.json"
    out.write_text(json.dumps(info, indent=1))
    ctx.status.update(video=info, video_path=str(ctx.video_path))
    ctx.log(f"probe: {ctx.duration:.1f}s {info['width']}x{info['height']} @{info['fps']:.2f}fps")


def stage_audio(ctx: Ctx) -> None:
    from highlights.audio.extract_audio import extract
    from highlights.audio.features import compute, detect_whistles
    wav = ctx.pipe / "audio" / "audio.wav"
    ctx.log("audio: extracting wav")
    extract(ctx.video_path, wav)
    ctx.status.update(stage_progress=0.3, message="audio features")
    feats = compute(wav, 1.0)
    feat_out = ctx.pipe / "audio" / "features_1s.json"
    feat_out.write_text(json.dumps(feats))
    ctx.status.update(stage_progress=0.8, message="whistle detection")
    segs = detect_whistles(wav)
    (ctx.pipe / "audio" / "whistles.json").write_text(
        json.dumps({"source": "audio", "whistles": segs}, indent=1))
    ctx.log(f"audio: {len(feats['rows'])} rows, {len(segs)} whistle segments")


def stage_motion(ctx: Ctx) -> None:
    out = ctx.pipe / "motion" / "features_1s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "highlights.motion.motion",
           "--video", str(ctx.video_path), "--out-json", str(out)]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    assert proc.stderr is not None
    for line in proc.stderr:
        line = line.strip()
        m = re.search(r"~(\d+)s", line)
        if m and ctx.duration > 0:
            ctx.status.update(stage_progress=min(1.0, int(m.group(1)) / ctx.duration),
                              message=f"motion @{m.group(1)}s")
        elif line:
            ctx.log(f"motion: {line}")
    proc.wait()
    if proc.returncode != 0:
        raise PipelineError(f"motion extractor exited {proc.returncode}")


def stage_features(ctx: Ctx) -> None:
    from highlights.pipeline.features import build_features, detect_match_window
    audio_json = ctx.pipe / "audio" / "features_1s.json"
    motion_json = ctx.pipe / "motion" / "features_1s.json"
    whistles_json = ctx.pipe / "audio" / "whistles.json"
    lo, hi, halves, warning = detect_match_window(audio_json, whistles_json, ctx.duration)
    ctx.match_window, ctx.halves, ctx.mw_warning = (lo, hi), halves, warning
    if warning:
        ctx.log(f"features: {warning}")
        ctx.status.update(message=warning, force=True)
    else:
        ctx.log(f"features: match window {lo:.0f}-{hi:.0f}s, {len(halves)} halves")
    df = build_features(audio_json, motion_json, whistles_json, ctx.duration, (lo, hi))
    df.to_parquet(ctx.pipe / "features_1s.parquet", index=False)
    (ctx.pipe / "match_window.json").write_text(json.dumps(
        {"match_window": [lo, hi], "halves": halves, "warning": warning}, indent=1))
    ctx.log(f"features: {df.shape[0]} rows x {df.shape[1]} cols")


def stage_score(ctx: Ctx) -> None:
    from highlights.pipeline.score import load_model, score_frame
    df = pd.read_parquet(ctx.pipe / "features_1s.parquet")
    model = load_model()
    in_match = df["in_match"].to_numpy(dtype=bool) if "in_match" in df.columns \
        else np.ones(len(df), dtype=bool)
    out = score_frame(df, model, in_match)
    out.to_parquet(ctx.pipe / "scores.parquet", index=False)
    ctx.log(f"score: {len(out)} rows, max learned {out['learned'].max():.3f}")


def stage_candidates(ctx: Ctx) -> None:
    from highlights.pipeline.candidates import make_candidates
    df = pd.read_parquet(ctx.pipe / "features_1s.parquet")
    sc = pd.read_parquet(ctx.pipe / "scores.parquet")
    in_match = df["in_match"].to_numpy(dtype=bool) if "in_match" in df.columns \
        else np.ones(len(df), dtype=bool)
    res = make_candidates(df, sc["learned"].to_numpy(), sc["rule"].to_numpy(),
                          in_match, ctx.duration or float(sc["t"].max() + 1))
    (ctx.pipe / "candidates.json").write_text(json.dumps(res, indent=1))
    ctx.log(f"candidates: {len(res['events'])} events")


def stage_stats(ctx: Ctx) -> None:
    from highlights.pipeline.score import MODEL_PATH, load_model
    from highlights.pipeline.stats import compute_stats
    df = pd.read_parquet(ctx.pipe / "features_1s.parquet")
    cand = json.loads((ctx.pipe / "candidates.json").read_text())
    mw = json.loads((ctx.pipe / "match_window.json").read_text()) \
        if (ctx.pipe / "match_window.json").exists() else {}
    whistles = []
    wj = ctx.pipe / "audio" / "whistles.json"
    if wj.exists():
        whistles = [(s["t_start"] + s["t_end"]) / 2
                    for s in json.loads(wj.read_text()).get("whistles", [])]
    try:
        meta = load_model() if MODEL_PATH.exists() else {}
    except Exception:
        meta = {}
    stats = compute_stats(
        df, cand.get("events", []), ctx.duration or cand.get("video_duration_s", 0.0),
        mw.get("match_window"), mw.get("halves"), whistles,
        pipeline={"model": meta.get("version", "audio_motion_lr v1"),
                  "auroc_reference": meta.get("auroc_heldout"),
                  "notes": mw.get("warning") or ""})
    (ctx.pipe / "stats.json").write_text(json.dumps(stats, indent=1))
    ctx.log("stats: written")


@dataclass
class Stage:
    weight: float
    fn: object
    outputs: object  # Ctx -> list[Path]


STAGES: dict[str, Stage] = {
    "download": Stage(0.35, stage_download, lambda c: [c.project_dir / "match.mp4",
                                                       c.project_dir / "match.mkv"]),
    "probe": Stage(0.02, stage_probe, lambda c: [c.pipe / "probe.json"]),
    "audio": Stage(0.10, stage_audio, lambda c: [c.pipe / "audio" / "audio.wav",
                                                 c.pipe / "audio" / "features_1s.json",
                                                 c.pipe / "audio" / "whistles.json"]),
    "motion": Stage(0.35, stage_motion, lambda c: [c.pipe / "motion" / "features_1s.json"]),
    "features": Stage(0.05, stage_features, lambda c: [c.pipe / "features_1s.parquet",
                                                       c.pipe / "match_window.json"]),
    "score": Stage(0.05, stage_score, lambda c: [c.pipe / "scores.parquet"]),
    "candidates": Stage(0.03, stage_candidates, lambda c: [c.pipe / "candidates.json"]),
    "stats": Stage(0.05, stage_stats, lambda c: [c.pipe / "stats.json"]),
}

# download stage produces match.<ext>; outputs check uses glob instead
def _stage_done(ctx: Ctx, name: str) -> bool:
    if name == "download":
        return _canonical_video(ctx.project_dir) is not None
    outs = STAGES[name].outputs(ctx)
    return all(Path(o).exists() for o in outs)


class _Heartbeat:
    """Rewrite status.json every 2 s while a stage runs."""

    def __init__(self, status: StatusWriter):
        self.status = status
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(2.0):
            self.status.update(force=True)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=3)


def _load_duration(ctx: Ctx) -> float:
    if ctx.duration:
        return ctx.duration
    pj = ctx.pipe / "probe.json"
    if pj.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            ctx.duration = float(json.loads(pj.read_text()).get("duration_s", 0.0))
    if not ctx.duration and ctx.video_path and Path(ctx.video_path).exists():
        with contextlib.suppress(Exception):
            ctx.duration = float(ffprobe(ctx.video_path)["duration_s"])
    return ctx.duration


def run_stages(ctx: Ctx, names: list[str]) -> None:
    base = 0.0
    for name in names:
        stage = STAGES[name]
        if not ctx.force and _stage_done(ctx, name):
            ctx.log(f"{name}: skip (outputs exist)")
            if name == "probe":
                _load_duration(ctx)
            base += stage.weight
            ctx.status.update(progress=base, message=f"{name} skipped")
            continue
        ctx.status.update(state="running", stage=name, stage_progress=0.0,
                          progress=base, message=f"{name} running", force=True)
        ctx.log(f"{name}: start")
        with _Heartbeat(ctx.status):
            stage.fn(ctx)
        if name == "probe":
            _load_duration(ctx)
        base += stage.weight
        ctx.status.update(stage_progress=1.0, progress=base,
                          message=f"{name} done", force=True)
        ctx.log(f"{name}: done")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.pipeline.run")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--youtube-url")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--stages", default=",".join(STAGES),
                    help="comma-separated stage subset, in pipeline order")
    ap.add_argument("--cookies")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    project_dir = args.project_dir
    pipe = project_dir / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(pipe / "status.json")
    ctx = Ctx(project_dir=project_dir, pipe=pipe, status=status,
              youtube_url=args.youtube_url, video_arg=args.video,
              cookies=args.cookies, force=args.force)

    names = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        print(f"unknown stages: {unknown}; valid: {list(STAGES)}", file=sys.stderr)
        return 2
    names = [n for n in STAGES if n in names]  # pipeline order

    def on_sigterm(signum, frame):
        status.update(state="failed", error="cancelled",
                      finished_at=time.time(), force=True)
        sys.exit(1)
    signal.signal(signal.SIGTERM, on_sigterm)

    with open(pipe / "log.txt", "a") as log_fh:
        ctx.log_fh = log_fh
        return _run(ctx, names)


def _run(ctx: Ctx, names: list[str]) -> int:
    status = ctx.status
    try:
        status.update(state="running", stage=None, force=True)
        _resolve_video(ctx)
        ctx.status.update(video_path=str(ctx.video_path))
        _load_duration(ctx)
        run_stages(ctx, names)
    except PipelineError as e:
        ctx.log(f"FAILED: {e}")
        status.update(state="failed", error=str(e),
                      finished_at=time.time(), force=True)
        return 1
    except Exception as e:
        ctx.log(f"FAILED ({type(e).__name__}): {e}")
        status.update(state="failed", error=f"{type(e).__name__}: {e}",
                      finished_at=time.time(), force=True)
        return 1

    warning = ctx.mw_warning
    if warning is None:
        mwj = ctx.pipe / "match_window.json"
        if mwj.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                warning = json.loads(mwj.read_text()).get("warning")
    message = "done" + (f" ({warning})" if warning else "")
    status.update(state="done", stage="done", progress=1.0, stage_progress=1.0,
                  message=message, finished_at=time.time(), force=True)
    ctx.log("pipeline done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
