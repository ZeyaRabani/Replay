"""Multi-angle pipeline runner — mirrors highlights.pipeline.run.

Layout (per spec): <project>/angles/aN/ are Option-1 project dirs (each gets
its own pipeline/ outputs via `python -m highlights.pipeline.run`), and
<project>/multiangle/ holds this stage's status/log + sync.json,
director.json, fused_candidates.json, score.json. The director cut lands at
<project>/match.mp4 with <project>/pipeline/ populated so the existing
review UI works unchanged.

Stages: download, angles, sync, track, director, render, fuse, stats.

    python -m highlights.multiangle.run --project-dir DIR [--stages ...]
        [--offsets 0,12.5,-3.2] [--cookies F] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from highlights.io import write_json_atomic, write_parquet_atomic
from highlights.pipeline.errors import PipelineError
from highlights.pipeline.joblock import job_slot, workdir_for
from highlights.pipeline.probe import probe as ffprobe
from highlights.pipeline.run import _stdout_is
from highlights.pipeline.status import StatusWriter, load_status

ANGLE_STAGES = "probe,audio,motion,features,score,candidates"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


@dataclass
class Ctx:
    project_dir: Path
    pipe: Path                       # <project>/multiangle
    status: StatusWriter
    angles: list[dict] = field(default_factory=list)   # [{"label","url"|None,"dir"}]
    offsets: list[float] | None = None                  # --offsets manual
    cookies: str | None = None
    force: bool = False
    style: str = "normal"
    durations: list[float] = field(default_factory=list)
    coverage: dict | None = None
    log_fh: object = None

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_fh is not None and not _stdout_is(self.log_fh):
            self.log_fh.write(line + "\n")
            self.log_fh.flush()

    def duration(self, i: int) -> float:
        """Angle i's video duration: stage_sync fills ctx.durations, but a
        re-cut (director,render,fuse) skips it — fall back to the per-angle
        pipeline/probe.json and cache the result."""
        while len(self.durations) <= i:
            self.durations.append(0.0)
        if self.durations[i] <= 0:
            try:
                pr = json.loads(
                    (self.angles[i]["dir"] / "pipeline" / "probe.json")
                    .read_text())
                self.durations[i] = float(pr.get("duration_s") or 0.0)
            except Exception:
                pass
        return self.durations[i]

    def union(self, sync: dict) -> tuple[float, float]:
        """Effective [lo, hi) shared-T range for director/render/fuse:
        multiangle/cut_range.json {lo, hi} (absolute shared-T seconds)
        clipped to sync's coverage union; else the full union."""
        lo, hi = sync["coverage"]["union"]
        try:
            cr = json.loads((self.pipe / "cut_range.json").read_text())
            lo2 = max(lo, float(cr["lo"]))
            hi2 = min(hi, float(cr["hi"]))
            if hi2 > lo2:
                return lo2, hi2
        except Exception:
            pass
        return float(lo), float(hi)

    def angle_video(self, i: int) -> Path | None:
        d = self.angles[i]["dir"]
        for ext in VIDEO_EXTS:
            p = d / f"match{ext}"
            if p.exists():
                return p
        return None


def _load_angles(project_dir: Path, angles_json: str | None) -> list[dict]:
    """Angle list from --angles-json, else project.json source_info, else dirs."""
    if angles_json:
        spec = json.loads(Path(angles_json).read_text())
    else:
        pj = project_dir / "project.json"
        spec = {}
        if pj.exists():
            data = json.loads(pj.read_text())
            spec = data.get("source") or data.get("source_info") or {}
        if not spec.get("angles"):
            dirs = sorted(project_dir.joinpath("angles").glob("a*"))
            if not dirs:
                raise PipelineError("no angles: pass --angles-json or create angles/aN dirs")
            spec = {"angles": [{"label": d.name, "url": None} for d in dirs]}
    out = []
    for i, a in enumerate(spec["angles"]):
        d = project_dir / "angles" / f"a{i}"
        d.mkdir(parents=True, exist_ok=True)
        out.append({"label": a.get("label") or f"angle {i}", "url": a.get("url"), "dir": d})
    return out


# ------------------------------ stages ------------------------------------

def stage_download(ctx: Ctx) -> None:
    from highlights.pipeline.download import download
    for i, a in enumerate(ctx.angles):
        if ctx.angle_video(i) is not None:
            ctx.log(f"download: a{i} already has a file")
            continue
        if not a.get("url"):
            raise PipelineError(f"angle {i}: no file and no url")
        ctx.log(f"download: angle {i}/{len(ctx.angles)-1} {a['url']}")
        download(a["url"], a["dir"], status=ctx.status, cookies=ctx.cookies, log=ctx.log)


def stage_angles(ctx: Ctx) -> None:
    """Run each angle's Option-1 pipeline concurrently (ffmpeg+numpy,
    ~1.5 cores each). Progress = mean of sub-progress; first nonzero
    returncode terminates the rest and fails the stage."""
    n = len(ctx.angles)
    running = []          # (i, proc, sub_status)
    for i, a in enumerate(ctx.angles):
        vid = ctx.angle_video(i)
        if vid is None:
            raise PipelineError(f"angle {i}: no video file after download")
        sub_status = a["dir"] / "pipeline" / "status.json"
        done_marker = a["dir"] / "pipeline" / "candidates.json"
        if done_marker.exists() and not ctx.force:
            ctx.log(f"angles: a{i} skipped (candidates exist)")
            continue
        ctx.log(f"angles: running Option-1 pipeline on a{i} ({a['label']})")
        proc = subprocess.Popen(
            [sys.executable, "-m", "highlights.pipeline.run",
             "--project-dir", str(a["dir"]), "--video", str(vid),
             "--stages", ANGLE_STAGES, "--no-job-lock"],
            stdout=ctx.log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
        running.append((i, proc, sub_status))

    pending = list(running)
    while pending:
        msgs = []
        still = []
        for i, proc, sub_status in pending:
            st = load_status(sub_status) or {}
            rc = proc.poll()
            if rc is None:
                msgs.append(f"a{i}: {st.get('message', 'running')}")
                still.append((i, proc, sub_status))
            elif rc == 0:
                ctx.log(f"angles: a{i} done")
            else:
                for _, q, _ in pending:
                    if q is not proc and q.poll() is None:
                        q.terminate()
                tail = str(st.get("error", ""))
                raise PipelineError(
                    f"angle {i} pipeline failed ({rc}) {tail}")
        pending = still
        # progress = mean over angles (skipped + finished count as 1.0)
        subs = [1.0] * (n - len(running)) + \
            [float((load_status(ss) or {}).get("progress", 0.0))
             for _, _, ss in pending]
        ctx.status.update(progress=min(1.0, sum(subs) / n),
                          message=" · ".join(msgs) or "angles done")
        if pending:
            time.sleep(2)


def stage_sync(ctx: Ctx) -> dict:
    from highlights.multiangle.sync import write_sync
    wavs = [a["dir"] / "pipeline" / "audio" / "audio.wav" for a in ctx.angles]
    durations = []
    for i, a in enumerate(ctx.angles):
        pj = a["dir"] / "pipeline" / "probe.json"
        durations.append(float(json.loads(pj.read_text())["duration_s"]))
    ctx.durations = durations
    out = write_sync(wavs, durations, ctx.pipe / "sync.json",
                     manual_offsets=ctx.offsets)
    ctx.coverage = out["coverage"]
    ctx.log(f"sync: offsets {out['offsets']} method={out['method']} "
            f"needs_manual={out['needs_manual']}")
    ctx.log(f"sync: {out.get('confidence_note', '')}")
    if out["needs_manual"]:
        ctx.status.update(state="needs_input",
                          message=f"Sync confidence low for angle(s) "
                                  f"{out['needs_manual']} — enter offsets",
                          finished_at=time.time(), force=True)
        raise SystemExit(0)  # clean stop, not failed
    return out


def stage_track(ctx: Ctx) -> None:
    """Run trackfeat as a subprocess per angle, HL_TRACK_WORKERS (2) at
    a time. OMP/TORCH threads capped at 2 so two workers fit the box.
    Each child writes <out>.progress {t, frames} every 30 frames; the
    stage's stage_progress is the mean of per-angle fractions."""
    workers = int(os.environ.get("HL_TRACK_WORKERS", "2"))
    model = os.environ.get("HL_MA_MODEL",
                           str(Path(__file__).parent / "models" / "yolov8n.pt"))
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TORCH_NUM_THREADS": "2"}
    n = len(ctx.angles)
    pending = []          # (i, vid, out, prog_file)
    n_done = 0
    for i, a in enumerate(ctx.angles):
        vid = ctx.angle_video(i)
        out = a["dir"] / "track" / "features_1s.json"
        if out.exists() and not ctx.force:
            ctx.log(f"track: a{i} skip")
            n_done += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        prog = out.with_suffix(".progress")
        prog.unlink(missing_ok=True)
        pending.append((i, vid, out, prog))

    def _frac(i: int, prog: Path) -> float:
        try:
            t = float(json.loads(prog.read_text()).get("t", 0.0))
        except Exception:
            t = 0.0
        d = ctx.duration(i)
        return min(1.0, t / d) if d else 0.0

    running = []          # (i, proc, prog)
    while pending or running:
        while pending and len(running) < workers:
            i, vid, out, prog = pending.pop(0)
            ctx.log(f"track: angle {i+1}/{n} {ctx.angles[i]['label']}")
            running.append((i, subprocess.Popen(
                [sys.executable, "-m", "highlights.multiangle.trackfeat",
                 "--video", str(vid), "--out", str(out), "--model", model,
                 "--imgsz", "960", "--fps", "1",
                 "--progress-file", str(prog)],
                stdout=ctx.log_fh or subprocess.DEVNULL,
                stderr=subprocess.STDOUT, env=env), prog))
        still = []
        for i, proc, prog in running:
            rc = proc.poll()
            if rc is None:
                still.append((i, proc, prog))
            elif rc == 0:
                n_done += 1
                ctx.log(f"track: a{i} done")
            else:
                for _, q, _ in running:
                    if q is not proc and q.poll() is None:
                        q.terminate()
                raise PipelineError(f"track angle {i} failed ({rc})")
        running = still
        subs = [1.0] * n_done + [_frac(i, prog) for i, _, prog in running]
        frac = min(1.0, sum(subs) / n) if n else 1.0
        ctx.status.update(
            stage_progress=frac,
            message=f"tracking {len(running) or 1}/{n} angles · {frac*100:.0f}%")
        if pending or running:
            time.sleep(2)


def _load_track_rows(angle_dir: Path) -> dict:
    """Per-second track arrays keyed by file-time second."""
    f = angle_dir / "track" / "features_1s.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    cols = d["columns"]
    rows = d["rows"]
    out = {}
    for c in cols:
        vals = [r[cols.index(c)] for r in rows]
        if c == "players_xy":
            out[c] = [json.loads(v) if isinstance(v, str) else v
                      for v in vals]
        else:
            out[c] = np.asarray(vals, dtype=float)
    return out


def _load_motion(angle_dir: Path) -> dict[int, float]:
    f = angle_dir / "pipeline" / "motion" / "features_1s.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    cols = d["columns"]
    ti, mi = cols.index("t"), cols.index("motion_total")
    return {int(r[ti]): float(r[mi]) for r in d["rows"]}


def _np_json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def stage_director(ctx: Ctx) -> dict:
    from highlights.multiangle.director import cut_director

    sync = json.loads((ctx.pipe / "sync.json").read_text())
    offsets = sync["offsets"]
    lo, hi = ctx.union(sync)
    T = int(np.ceil(hi - lo))
    from highlights.multiangle.director import EVENT_POST, EVENT_PRE, EVENT_TYPES

    tracks, motion, avail = [], [], np.zeros((len(ctx.angles), T), dtype=bool)
    for i, a in enumerate(ctx.angles):
        tr = _load_track_rows(a["dir"])
        mo = _load_motion(a["dir"])
        dur = ctx.duration(i)
        off = offsets[i]
        t_idx = np.arange(T)
        ft = t_idx + lo - off                      # angle file time at T second
        ok = (ft >= 0) & (ft <= dur - 1)
        avail[i] = ok
        fsec = np.clip(np.round(ft), 0, 1 << 30).astype(int)
        def _row(col, tr=tr, ok=ok, fsec=fsec):
            src = tr.get(col)
            if src is None or len(src) == 0:
                return np.zeros(T)
            return np.where(ok, src[np.clip(fsec, 0, len(src) - 1)], 0.0)
        event = np.zeros(T)
        ev_file = a["dir"] / "pipeline" / "candidates.json"
        if ev_file.exists():
            evs = json.loads(ev_file.read_text())
            for e in (evs.get("events") or evs.get("candidates") or []):
                if str(e.get("type")) not in EVENT_TYPES:
                    continue
                conf = float(e.get("confidence", 0.5))
                ti = float(e.get("t", 0.0)) + off - lo   # shared-T -> output idx
                for k in range(int(ti - EVENT_PRE), int(ti + EVENT_POST) + 1):
                    if 0 <= k < T and ok[k]:
                        event[k] = max(event[k], conf)
        n_ev = int((event > 0).sum())
        tracks.append({"ball_conf": _row("ball_conf"), "ball_size": _row("ball_size"),
                       "ball_x": _row("ball_x"), "ball_y": _row("ball_y"),
                       "cluster": _row("cluster_score"), "event": event})
        pxy_src = tr.get("players_xy")
        if pxy_src is not None and len(pxy_src):
            tracks[-1]["players_xy"] = [
                (pxy_src[min(fsec[k], len(pxy_src) - 1)] or []) if ok[k] else []
                for k in range(T)]
        motion.append(np.array([mo.get(int(s), 0.0) for s in fsec]))
        ctx.log(f"director: angle {i} event channel {n_ev} s")
    zones, zone_ok = None, None
    zf = ctx.pipe / "zones.json"
    if zf.exists():
        try:
            zd = json.loads(zf.read_text()) or {}
            zones = zd.get("angles")
            ref_ts = zd.get("ref_t") or []
            if zones:
                ctx.log("director: zones on angles "
                        f"{[i for i, z in enumerate(zones) if z]}")
                zone_ok = np.ones((len(ctx.angles), T), dtype=bool)
                suspended = [0.0] * len(ctx.angles)
                for i, a in enumerate(ctx.angles):
                    if not zones[i]:
                        continue
                    vid = ctx.angle_video(i)
                    if vid is None:
                        continue
                    off = offsets[i]
                    dur = ctx.duration(i)
                    ref_t = (ref_ts[i] if i < len(ref_ts)
                             and ref_ts[i] is not None else dur * 0.3)
                    ctx.status.update(
                        stage_progress=(i + 0.5) / len(ctx.angles),
                        message=f"director: viewcheck angle {i}")
                    try:
                        ok = _zone_view_ok(ctx, a["dir"], vid, ref_t)
                    except Exception as e:
                        ctx.log(f"director: viewcheck a{i} failed ({e})")
                        continue
                    times, okarr = ok
                    zone_ok[i] = _map_view_ok(times, okarr, T, lo, off, dur)
                    suspended[i] = float(
                        (~zone_ok[i] & avail[i]).sum() / max(1, avail[i].sum()))
                if any(s > 0 for s in suspended):
                    ctx.log("director: zones suspended "
                            f"{[round(s, 3) for s in suspended]} "
                            "(view differs from reference)")
        except Exception as e:
            ctx.log(f"director: ignoring bad zones.json ({e})")
            zones, zone_ok = None, None
    out = cut_director(tracks, avail, motion, ctx.style,
                       zones=zones, zone_ok=zone_ok)
    if zones:
        out["zone_suspended_share"] = [round(s, 4) for s in suspended]
    write_json_atomic(ctx.pipe / "director.json", out, indent=1,
                      default=_np_json)
    ctx.log(f"director: {out['n_cuts']} cuts, ratios {out['ratios']}")
    return out


def _map_view_ok(times: np.ndarray, okarr: np.ndarray, T: int,
                 lo: float, off: float, dur: float) -> np.ndarray:
    """Nearest-sample map of viewcheck ok flags (angle file time) onto the
    shared output timeline of length T."""
    t_idx = np.arange(T)
    ft = np.clip(t_idx + lo - off, 0, max(0, dur))
    idx = np.clip(np.searchsorted(times, ft), 0, len(okarr) - 1)
    return np.asarray(okarr[idx], dtype=bool)


def _zone_view_ok(ctx: Ctx, angle_dir: Path, video: Path,
                  ref_t: float) -> tuple[np.ndarray, np.ndarray]:
    """view_ok with a per-angle cache keyed on ref_t."""
    from highlights.multiangle.viewcheck import view_ok
    cache = angle_dir / "track" / f"viewcheck_{ref_t:.0f}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return (np.asarray(d["times"], dtype=float),
                np.asarray(d["ok"], dtype=bool))
    times, ok = view_ok(video, ref_t)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache, {"times": times.tolist(), "ok": ok.tolist()})
    return times, ok


def stage_render(ctx: Ctx) -> None:
    from highlights.multiangle.render import render
    sync = json.loads((ctx.pipe / "sync.json").read_text())
    director = json.loads((ctx.pipe / "director.json").read_text())
    videos = [str(ctx.angle_video(i)) for i in range(len(ctx.angles))]
    lo, hi = ctx.union(sync)
    out = render(videos, sync["offsets"], director["segments"], lo, hi,
                 ctx.pipe, ctx.project_dir / "match.mp4", videos[0],
                 durations=list(ctx.durations), log=ctx.log)
    # register the cut as the project video for the Option-1 UI
    pipe1 = ctx.project_dir / "pipeline"
    pipe1.mkdir(exist_ok=True)
    info = ffprobe(out)
    write_json_atomic(pipe1 / "probe.json", info, indent=1)
    ctx.status.update(video=info, video_path=str(out))
    ctx.log(f"render: wrote {out} ({info['width']}x{info['height']})")


def stage_fuse(ctx: Ctx) -> dict:
    from highlights.multiangle.fuse import fuse_candidates, to_output_time
    sync = json.loads((ctx.pipe / "sync.json").read_text())
    files = [a["dir"] / "pipeline" / "candidates.json" for a in ctx.angles]
    labels = [a["label"] for a in ctx.angles]
    out = fuse_candidates(files, sync["offsets"], labels,
                          ctx.pipe / "fused_candidates.json")
    lo, hi = ctx.union(sync)
    # fused events are on shared T; the UI plays the rendered video whose
    # time axis is output time (0 = union start) -> shift everything by -lo
    for key in ("events", "candidates"):
        if key in out:
            out[key] = to_output_time(out[key], lo)
    write_json_atomic(ctx.pipe / "fused_candidates.json", out, indent=1)
    pdir = ctx.project_dir / "pipeline"
    pdir.mkdir(exist_ok=True)
    write_json_atomic(pdir / "candidates.json", out, indent=1)
    # a0's match window + features, shifted to output time; with a
    # cut_range the rendered video IS the match — window covers it all
    dur_out = hi - lo
    ranged = (ctx.pipe / "cut_range.json").exists()
    shift = sync["offsets"][0] - lo
    mw_src = ctx.angles[0]["dir"] / "pipeline" / "match_window.json"
    if mw_src.exists():
        mw = json.loads(mw_src.read_text())

        def _clamp(v: float) -> float:
            return min(max(float(v), 0.0), dur_out)

        def _halves() -> list:
            kept = []
            for h in mw.get("halves", []) or []:
                s = _clamp(h.get("start", 0) + shift)
                e = _clamp(h.get("end", 0) + shift)
                if e > s:
                    kept.append({**h, "start": s, "end": e})
            return kept

        if ranged:
            mw["match_window"] = [0.0, dur_out]
        elif isinstance(mw.get("match_window"), list):
            mw["match_window"] = [_clamp(v + shift)
                                  for v in mw["match_window"]]
        if "halves" in mw:
            mw["halves"] = _halves()
        write_json_atomic(pdir / "match_window.json", mw, indent=1)
    fsrc = ctx.angles[0]["dir"] / "pipeline" / "features_1s.parquet"
    if fsrc.exists():
        df = pd.read_parquet(fsrc)
        df["t"] = df["t"] + shift
        df = df[(df["t"] >= 0.0) & (df["t"] <= hi - lo)]
        write_parquet_atomic(df, pdir / "features_1s.parquet")
    ctx.log(f"fuse: {len(out['events'])} fused events")
    return out


def stage_stats(ctx: Ctx) -> None:
    from highlights.pipeline.stats import compute_stats
    pdir = ctx.project_dir / "pipeline"
    feats = pd.read_parquet(pdir / "features_1s.parquet")
    cands = json.loads((ctx.pipe / "fused_candidates.json").read_text())["events"]
    mw = json.loads((pdir / "match_window.json").read_text()) \
        if (pdir / "match_window.json").exists() else {}
    # the rendered video covers the effective cut range, not angle 0's file
    sync_dur = json.loads((ctx.pipe / "sync.json").read_text())
    lo, hi = ctx.union(sync_dur)
    dur = hi - lo
    stats = compute_stats(feats, cands, dur, tuple(mw.get("match_window", ())),
                          mw.get("halves"), None)
    director = json.loads((ctx.pipe / "director.json").read_text())
    sync = json.loads((ctx.pipe / "sync.json").read_text())
    n_cross = sum(1 for e in cands if e.get("cross_validation") == "confirmed")
    n_single = len(cands) - n_cross
    n_disp = sum(1 for e in cands if (e.get("signals") or {}).get("disputed"))
    stats["multiangle"] = {
        "sync": {"method": sync["method"], "offsets": sync["offsets"],
                 "needs_manual": sync["needs_manual"],
                 "triangle_residual_s": sync.get("triangle_residual_s")},
        "director": {"ratios": director["ratios"], "n_cuts": director["n_cuts"],
                     "angle_share": director["angle_share"]},
        "confirmation": {"cross": n_cross, "single": n_single, "disputed": n_disp},
        "score": {"home": {"label": "home", "goals": 0},
                  "away": {"label": "away", "goals": 0},
                  "basis": "confirmed goals with team set"},
    }
    write_json_atomic(pdir / "stats.json", stats, indent=1)
    ctx.log("stats: written")


# ------------------------------ driver ------------------------------------

@dataclass
class Stage:
    weight: float
    fn: object
    outputs: object


STAGES: dict[str, Stage] = {
    "download": Stage(0.15, stage_download,
                      lambda c: [c.angle_video(i) or a["dir"] / "match.missing"
                                 for i, a in enumerate(c.angles)]),
    "angles": Stage(0.30, stage_angles,
                    lambda c: [a["dir"] / "pipeline" / "candidates.json" for a in c.angles]),
    "sync": Stage(0.05, stage_sync, lambda c: [c.pipe / "sync.json"]),
    "track": Stage(0.20, stage_track,
                   lambda c: [a["dir"] / "track" / "features_1s.json" for a in c.angles]),
    "director": Stage(0.02, stage_director, lambda c: [c.pipe / "director.json"]),
    "render": Stage(0.20, stage_render, lambda c: [c.project_dir / "match.mp4"]),
    "fuse": Stage(0.03, stage_fuse,
                  lambda c: [c.pipe / "fused_candidates.json",
                             c.project_dir / "pipeline" / "candidates.json"]),
    "stats": Stage(0.05, stage_stats,
                   lambda c: [c.project_dir / "pipeline" / "stats.json"]),
}


def _stage_done(ctx: Ctx, name: str) -> bool:
    outs = STAGES[name].outputs(ctx)
    return all(Path(o).exists() if not isinstance(o, Path) else o.exists() for o in outs)


class _Heartbeat:
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


def run_stages(ctx: Ctx, names: list[str]) -> None:
    base = 0.0
    for name in names:
        stage = STAGES[name]
        if not ctx.force and _stage_done(ctx, name):
            ctx.log(f"{name}: skip (outputs exist)")
            base += stage.weight
            ctx.status.update(progress=base, message=f"{name} skipped")
            continue
        ctx.status.update(state="running", stage=name, stage_progress=0.0,
                          progress=base, message=f"{name} running", force=True)
        ctx.log(f"{name}: start")
        with _Heartbeat(ctx.status):
            stage.fn(ctx)
        base += stage.weight
        ctx.status.update(stage_progress=1.0, progress=base,
                          message=f"{name} done", force=True)
        ctx.log(f"{name}: done")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.multiangle.run")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--angles-json", help="JSON file with {\"angles\": [{url,label}]}")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--offsets", help="comma list, len==n angles, first must be 0")
    ap.add_argument("--cookies")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--style", default="normal",
                    choices=("normal", "fast"))
    args = ap.parse_args(argv)

    project_dir = args.project_dir
    pipe = project_dir / "multiangle"
    pipe.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(pipe / "status.json")
    angles = _load_angles(project_dir, args.angles_json)
    offsets = None
    if args.offsets:
        offsets = [float(x) for x in args.offsets.split(",")]
        if len(offsets) != len(angles) or offsets[0] != 0:
            print("--offsets must have len == n angles, first == 0", file=sys.stderr)
            return 2
    ctx = Ctx(project_dir=project_dir, pipe=pipe, status=status, angles=angles,
              offsets=offsets, cookies=args.cookies, force=args.force,
              style=args.style)

    names = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        print(f"unknown stages: {unknown}; valid: {list(STAGES)}", file=sys.stderr)
        return 2

    def on_sigterm(signum, frame):
        status.update(state="failed", error="cancelled",
                      finished_at=time.time(), force=True)
        sys.exit(1)
    import signal
    signal.signal(signal.SIGTERM, on_sigterm)

    with open(pipe / "log.txt", "a") as log_fh:
        ctx.log_fh = log_fh
        try:
            status.update(state="running", force=True)
            with job_slot(workdir_for(project_dir), status=status,
                          log=ctx.log):
                run_stages(ctx, names)
            if "render" in names:
                from highlights.multiangle.cuts import snapshot_cut
                cr = None
                try:
                    crd = json.loads((pipe / "cut_range.json").read_text())
                    cr = [float(crd["lo"]), float(crd["hi"])]
                except Exception:
                    pass
                try:
                    meta = snapshot_cut(project_dir, ctx.style, cut_range=cr)
                    if meta:
                        ctx.log(f"cut snapshot: {meta['id']} ({meta['label']})")
                except Exception as e:
                    ctx.log(f"cut snapshot failed ({e})")
        except SystemExit:
            raise
        except PipelineError as e:
            ctx.log(f"FAILED: {e}")
            status.update(state="failed", error=str(e),
                          finished_at=time.time(), force=True)
            return 1
        except Exception as e:
            ctx.log(f"FAILED ({type(e).__name__}): {e}")
            status.update(state="failed", error=str(e),
                          finished_at=time.time(), force=True)
            return 1
        status.update(state="done", stage="done", progress=1.0,
                      stage_progress=1.0, message="done",
                      finished_at=time.time(), force=True)
        ctx.log("multiangle done")
        return 0


if __name__ == "__main__":
    sys.exit(main())
