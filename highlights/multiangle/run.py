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
import contextlib
import json
import os
import re
import shutil
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
    goal_aware: bool = True
    read_only: bool = False
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
        if self.durations[i] <= 0:
            try:
                features = json.loads(
                    (self.angles[i]["dir"] / "track" / "features_1s.json")
                    .read_text())
                rows = features.get("rows") or []
                columns = features.get("columns") or []
                t_index = columns.index("t") if "t" in columns else 0
                if rows:
                    self.durations[i] = float(rows[-1][t_index]) + 1.0
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


def _load_angles(project_dir: Path, angles_json: str | None,
                 create_dirs: bool = True) -> list[dict]:
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
        if create_dirs:
            d.mkdir(parents=True, exist_ok=True)
        out.append({"label": a.get("label") or f"angle {i}", "url": a.get("url"), "dir": d})
    return out


# ------------------------------ stages ------------------------------------

_DEFAULT_LABEL = re.compile(r"^\s*angle\s+\d+\s*$", re.IGNORECASE)
_LABEL_LOCK = threading.Lock()


def _label_from_title(ctx: Ctx, i: int) -> None:
    """Name angle i after its YouTube title when the label is still the
    default 'Angle N'. Persists into project.json's source block —
    best-effort, failures are logged not raised."""
    a = ctx.angles[i]
    if not _DEFAULT_LABEL.match(a.get("label") or ""):
        return
    if not a.get("url"):
        return
    try:
        from highlights.pipeline.download import video_title
        title = video_title(a["url"], cookies=ctx.cookies, log=ctx.log)
        if not title:
            return
        pj = ctx.project_dir / "project.json"
        with _LABEL_LOCK:      # parallel download threads share the file
            data = json.loads(pj.read_text())
            spec = data.get("source") or data.get("source_info") or {}
            ang = spec.get("angles") or []
            if (i < len(ang)
                    and _DEFAULT_LABEL.match(ang[i].get("label") or "")):
                ang[i]["label"] = title
                write_json_atomic(pj, data, indent=2)
        a["label"] = title
        ctx.log(f"download: a{i} named '{title[:60]}'")
    except Exception as e:
        ctx.log(f"download: a{i} title lookup failed ({e})")


def stage_download(ctx: Ctx) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from highlights.pipeline.download import download
    todo = []
    for i, a in enumerate(ctx.angles):
        if ctx.angle_video(i) is not None:
            ctx.log(f"download: a{i} already has a file")
            continue
        if not a.get("url"):
            raise PipelineError(f"angle {i}: no file and no url")
        todo.append(i)
    if not todo:
        return
    workers = int(os.environ.get("HL_DOWNLOAD_WORKERS", "2"))

    def _dl(i: int) -> None:
        a = ctx.angles[i]
        ctx.log(f"download: angle {i}/{len(ctx.angles)-1} {a['url']}")
        download(a["url"], a["dir"], status=ctx.status,
                 cookies=ctx.cookies, log=ctx.log)
        _label_from_title(ctx, i)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_dl, todo))


def stage_angles(ctx: Ctx) -> None:
    """Run each angle's Option-1 pipeline concurrently (ffmpeg+numpy,
    ~1.5 cores each). Progress = mean of sub-progress; first nonzero
    returncode terminates the rest and fails the stage."""
    n = len(ctx.angles)
    queue = []            # angle indexes still to run
    for i, a in enumerate(ctx.angles):
        vid = ctx.angle_video(i)
        if vid is None:
            raise PipelineError(f"angle {i}: no video file after download")
        done_marker = a["dir"] / "pipeline" / "candidates.json"
        if done_marker.exists() and not ctx.force:
            ctx.log(f"angles: a{i} skipped (candidates exist)")
            continue
        queue.append(i)

    # HL_ANGLE_WORKERS caps concurrent angle pipelines (RAM is the binding
    # constraint — ~8 GB peak each on long matches); 0/absent = all at once.
    try:
        workers = int(os.environ.get("HL_ANGLE_WORKERS", "0"))
    except ValueError:
        workers = 0
    if workers <= 0:
        workers = len(queue) or 1

    n_running = 0

    def launch(i: int):
        a = ctx.angles[i]
        ctx.log(f"angles: running Option-1 pipeline on a{i} ({a['label']})")
        proc = subprocess.Popen(
            [sys.executable, "-m", "highlights.pipeline.run",
             "--project-dir", str(a["dir"]), "--video", str(ctx.angle_video(i)),
             "--stages", ANGLE_STAGES, "--no-job-lock"],
            stdout=ctx.log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
        return (i, proc, a["dir"] / "pipeline" / "status.json")

    pending = []
    while queue or pending:
        while queue and len(pending) < workers:
            pending.append(launch(queue.pop(0)))
            n_running += 1
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
        subs = [1.0] * (n - len(queue) - len(pending)) + \
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
    if out["method"].startswith("xcorr+"):
        ctx.log(f"sync: {out.get('confidence_note', '')}")
    apply_match_window_src(ctx, out)
    if out["needs_manual"]:
        ctx.status.update(state="needs_input",
                          message=f"Sync confidence low for angle(s) "
                                  f"{out['needs_manual']} — enter offsets",
                          finished_at=time.time(), force=True)
        raise SystemExit(0)  # clean stop, not failed
    return out


TRACK_PAD = 30.0


def apply_match_window_src(ctx: Ctx, sync: dict) -> None:
    """Apply multiangle/match_window_src.json {angle, start, end} (file
    seconds of that angle) to cut_range.json in shared-T
    (shared-T = file_t + offsets[angle]). Idempotent; called after sync
    and at the start of stages that consume cut_range."""
    try:
        src = json.loads((ctx.pipe / "match_window_src.json").read_text())
    except Exception:
        return
    try:
        read_only = bool(getattr(ctx, "read_only", False))
        ang = src.get("angle")
        if ang is None:
            # "measured on the longest video" — resolve only once every
            # angle has a known duration (an undownloaded angle would
            # resolve to the wrong index); leave angle null until then
            durs = [ctx.duration(i) for i in range(len(ctx.angles))]
            if not durs or any(not d or d <= 0 for d in durs):
                ctx.log("match window: waiting for all angle durations "
                        "before resolving the longest")
                return
            ang = int(max(range(len(durs)), key=lambda i: durs[i]))
            if not read_only:
                with contextlib.suppress(Exception):
                    write_json_atomic(
                        ctx.pipe / "match_window_src.json",
                        {**src, "angle": ang}, indent=1)
        ang = int(ang)
        s, e = float(src["start"]), float(src["end"])
        from highlights.multiangle.syncmap import file_to_shared, timemap_from_sync
        tm = timemap_from_sync(sync, getattr(ctx, "durations", None) or [ctx.duration(i) for i in range(len(ctx.angles))] if ctx.angles else None)
        lo, hi = (max(0.0, file_to_shared(tm, ang, s)),
                  file_to_shared(tm, ang, e))
        cr_path = ctx.pipe / "cut_range.json"
        cur = json.loads(cr_path.read_text()) if cr_path.exists() else None
        if cur != {"lo": lo, "hi": hi} and not read_only:
            write_json_atomic(cr_path, {"lo": lo, "hi": hi}, indent=1)
            ctx.log(f"match window: a{ang} {s:.0f}-{e:.0f} -> "
                    f"shared-T {lo:.0f}-{hi:.0f}")
    except Exception as exc:
        ctx.log(f"match window: could not apply src ({exc})")


def angle_track_window(lo: float, hi: float, offset,
                       duration: float, pad: float = TRACK_PAD
                       ) -> tuple[float, float]:
    """Shared-T window [lo, hi] -> angle file-time range, padded and
    clamped to [0, duration]. `offset` may be a legacy scalar or a
    segment list / timemap row (piecewise map via syncmap)."""
    if isinstance(offset, (int, float)):
        f0, f1 = lo - offset, hi - offset
    else:
        from highlights.multiangle.syncmap import file_range_for_shared
        f0, f1 = file_range_for_shared(offset, 0, lo, hi)
    lo_f = max(0.0, f0 - pad)
    hi_f = min(max(0.0, duration), f1 + pad)
    return lo_f, max(lo_f, hi_f)


def stage_track(ctx: Ctx) -> None:
    """Run trackfeat as a subprocess per angle, HL_TRACK_WORKERS (2) at
    a time. OMP/TORCH threads capped at 2 so two workers fit the box.
    Each child writes <out>.progress {t, frames} every 30 frames; the
    stage's stage_progress is the mean of per-angle fractions.
    When multiangle/cut_range.json exists (shared-T == angle-0 file time,
    offsets[0]==0), each angle only tracks its overlap with the match
    window, padded by TRACK_PAD."""
    workers = int(os.environ.get("HL_TRACK_WORKERS", "2"))
    model = os.environ.get("HL_MA_MODEL",
                           str(Path(__file__).parent / "models" / "yolov8n.pt"))
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TORCH_NUM_THREADS": "2"}
    n = len(ctx.angles)
    # match window in shared-T (cut_range.json), applied per angle;
    # match_window_src.json (file-time on a chosen angle) is converted
    # with the current sync offsets first
    sync = None
    try:
        sync = json.loads((ctx.pipe / "sync.json").read_text())
    except Exception:
        sync = None
    if sync is not None:
        apply_match_window_src(ctx, sync)
    win = None
    try:
        cr = json.loads((ctx.pipe / "cut_range.json").read_text())
        w_lo, w_hi = float(cr["lo"]), float(cr["hi"])
        if w_hi > w_lo >= 0:
            win = (w_lo, w_hi)
    except Exception:
        win = None
    from highlights.multiangle.syncmap import timemap_from_sync
    tm = None
    if win is not None and sync is not None:
        try:
            tm = timemap_from_sync(sync, getattr(ctx, "durations", None) or [ctx.duration(i) for i in range(len(ctx.angles))] if ctx.angles else None)
        except Exception:
            win = None
    elif win is not None:
        win = None      # no sync yet — track the full videos
    pending = []          # (i, vid, out, prog_file, lo_f, hi_f)
    n_done = 0
    for i, a in enumerate(ctx.angles):
        vid = ctx.angle_video(i)
        out = a["dir"] / "track" / "features_1s.json"
        dur = ctx.duration(i)
        if win is None:
            lo_f, hi_f = 0.0, dur
        else:
            lo_f, hi_f = angle_track_window(
                win[0], win[1], tm[i] if tm is not None else 0.0, dur)
        if out.exists():
            # reuse the track when its recorded window covers what this
            # sync needs (piecewise sync must not re-pay the full track
            # on an unchanged window). Coverage is judged against the
            # UNPADDED range — the pad is pure slack for new tracks, so
            # a recording short only inside the pad is still sufficient.
            need_lo, need_hi = lo_f, hi_f
            if win is not None:
                need_lo, need_hi = angle_track_window(
                    win[0], win[1], tm[i] if tm is not None else 0.0,
                    dur, pad=0.0)
            try:
                meta = (json.loads(out.read_text()).get("meta") or {})
                rec_lo = float(meta.get("start_s") or 0.0)
                rec_hi = meta.get("end_s")
                covered = (rec_lo <= need_lo + 1.0
                           and (rec_hi is None
                                or float(rec_hi) >= need_hi - 1.0))
            except Exception:
                covered = True    # can't tell; keep the existing file
            if covered:
                ctx.log(f"track: a{i} up to date")
                n_done += 1
                continue
            ctx.log(f"track: a{i} re-tracking — saved window "
                    f"{rec_lo:.0f}-{rec_hi} < needed "
                    f"{need_lo:.0f}-{need_hi:.0f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        prog = out.with_suffix(".progress")
        prog.unlink(missing_ok=True)
        ctx.log(f"track: a{i} window {lo_f:.0f}-{hi_f:.0f} s "
                f"(of {dur:.0f})")
        pending.append((i, vid, out, prog, lo_f, hi_f))

    # 480p analysis proxies for the pending angles (parallel; falls back
    # to the original when neither download nor transcode works)
    proxies: dict[int, tuple[str, float, str]] = {}
    if os.environ.get("HL_TRACK_PROXY", "1") != "0" and pending:
        from concurrent.futures import ThreadPoolExecutor

        from highlights.multiangle.proxy import ensure_analysis_proxy

        def _mk(item):
            i, vid, _out, _prog, lo_f, hi_f = item
            w = (lo_f, hi_f) if win is not None else None
            pi = ensure_analysis_proxy(
                ctx.angles[i]["dir"], vid,
                url=ctx.angles[i].get("url"), cookies=ctx.cookies,
                window=w, log=ctx.log)
            return i, pi

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, pi in ex.map(_mk, pending):
                proxies[i] = (str(pi.path), pi.offset, pi.src)

    def _frac(i: int, prog: Path, lo_f: float, hi_f: float) -> float:
        try:
            t = float(json.loads(prog.read_text()).get("t", 0.0))
        except Exception:
            t = 0.0
        span = hi_f - lo_f
        if span <= 0:
            span = ctx.duration(i) or 1.0
            lo_f = 0.0
        return min(1.0, max(0.0, (t - lo_f) / span))

    def _spawn(i: int, vid, out, prog, lo_f: float, hi_f: float):
        ctx.log(f"track: angle {i+1}/{n} {ctx.angles[i]['label']}")
        cmd = [sys.executable, "-m", "highlights.multiangle.trackfeat",
               "--video", str(vid), "--out", str(out), "--model", model,
               "--imgsz", "960", "--fps", "1",
               "--progress-file", str(prog)]
        if win is not None:
            cmd += ["--start-s", f"{lo_f:.3f}", "--end-s", f"{hi_f:.3f}"]
        pp, off, src = proxies.get(i, (None, 0.0, "original"))
        if pp:
            cmd += ["--proxy", pp, "--proxy-offset", f"{off:.3f}"]
        cmd += ["--analysis-src", src]
        return subprocess.Popen(
            cmd,
            stdout=ctx.log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT, env=env)

    running = []          # (i, vid, out, proc, prog, lo_f, hi_f)
    retried = set()       # angles already retried once after a non-zero exit
    while pending or running:
        while pending and len(running) < workers:
            i, vid, out, prog, lo_f, hi_f = pending.pop(0)
            running.append((i, vid, out,
                            _spawn(i, vid, out, prog, lo_f, hi_f),
                            prog, lo_f, hi_f))
        still = []
        for i, vid, out, proc, prog, lo_f, hi_f in running:
            rc = proc.poll()
            if rc is None:
                still.append((i, vid, out, proc, prog, lo_f, hi_f))
            elif rc == 0:
                n_done += 1
                ctx.log(f"track: a{i} done")
            elif i not in retried:
                # transient child failure (e.g. empty ffprobe output under
                # load) — retry this angle once before failing the stage
                retried.add(i)
                prog.unlink(missing_ok=True)
                ctx.log(f"track: a{i} failed ({rc}), retrying once")
                still.append((i, vid, out,
                              _spawn(i, vid, out, prog, lo_f, hi_f),
                              prog, lo_f, hi_f))
            else:
                for _, _, _, q, _, _, _ in running:
                    if q is not proc and q.poll() is None:
                        q.terminate()
                raise PipelineError(f"track angle {i} failed ({rc})")
        running = still
        subs = [1.0] * n_done + [_frac(i, prog, lo_f, hi_f)
                                 for i, _, _, _, prog, lo_f, hi_f in running]
        frac = min(1.0, sum(subs) / n) if n else 1.0
        ctx.status.update(
            stage_progress=frac,
            message=f"tracking {len(running) or 1}/{n} angles · {frac*100:.0f}%")
        if pending or running:
            time.sleep(2)


def _load_track_rows(angle_dir: Path) -> dict:
    """Per-second track arrays indexed by FILE second (t rounded to int):
    rows may start at t>0 when tracking was windowed, so arrays are
    densified to length max(t)+1 with zeros (or [] for players_xy)."""
    f = angle_dir / "track" / "features_1s.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    cols = d["columns"]
    rows = d["rows"]
    if not rows:
        return {}
    ti = cols.index("t")
    n = round(float(rows[-1][ti])) + 1
    out = {}
    for c in cols:
        idx = cols.index(c)
        arr = [[] for _ in range(n)] if c == "players_xy" else np.zeros(n)
        for r in rows:
            v = r[idx]
            if c == "players_xy" and isinstance(v, str):
                v = json.loads(v)
            arr[round(float(r[ti]))] = v
        out[c] = arr
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


def load_director_inputs(ctx: Ctx) -> dict:
    """Build every array stage_director feeds cut_director, so a recut
    and the you-direct learner replay identical inputs.

    Returns {tracks, avail, motion, zones, zone_ok, zone_kf, lo, hi, T}:
    tracks[i] has ball_conf/ball_size/ball_x/ball_y/cluster/event (and
    players_xy when the track file carries it); avail/motion on the
    shared [lo,hi) timeline; zones None when unusable/absent."""
    from highlights.multiangle.director import EVENT_POST, EVENT_PRE, EVENT_TYPES

    sync = json.loads((ctx.pipe / "sync.json").read_text())
    apply_match_window_src(ctx, sync)
    from highlights.multiangle.syncmap import file_to_shared, shared_to_file, timemap_from_sync
    tm = timemap_from_sync(sync, getattr(ctx, "durations", None) or [ctx.duration(i) for i in range(len(ctx.angles))] if ctx.angles else None)
    offsets = sync["offsets"]
    lo, hi = ctx.union(sync)
    T = int(np.ceil(hi - lo))

    tracks, motion, avail = [], [], np.zeros((len(ctx.angles), T), dtype=bool)
    for i, a in enumerate(ctx.angles):
        tr = _load_track_rows(a["dir"])
        mo = _load_motion(a["dir"])
        dur = ctx.duration(i)
        t_idx = np.arange(T)
        # angle file time at each shared second; NaN in a map gap
        ft = np.array(
            [x if (x := shared_to_file(tm, i, t + lo)) is not None
             else np.nan for t in t_idx], dtype=float)
        ok = ~np.isnan(ft) & (ft >= 0) & (ft <= dur - 1)
        avail[i] = ok
        fsec = np.clip(np.round(np.nan_to_num(ft, nan=0.0)),
                       0, 1 << 30).astype(int)
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
                ti = file_to_shared(tm, i, float(e.get("t", 0.0))) - lo
                for k in range(int(ti - EVENT_PRE), int(ti + EVENT_POST) + 1):
                    if 0 <= k < T and ok[k]:
                        event[k] = max(event[k], conf)
        n_ev = int((event > 0).sum())
        tracks.append({"ball_conf": _row("ball_conf"), "ball_size": _row("ball_size"),
                       "ball_x": _row("ball_x"), "ball_y": _row("ball_y"),
                       "players_cx": _row("players_cx"),
                       "players_cy": _row("players_cy"),
                       "cluster": _row("cluster_score"), "event": event})
        pxy_src = tr.get("players_xy")
        if pxy_src is not None and len(pxy_src):
            tracks[-1]["players_xy"] = [
                (pxy_src[min(fsec[k], len(pxy_src) - 1)] or []) if ok[k] else []
                for k in range(T)]
        motion.append(np.array([mo.get(int(s), 0.0) for s in fsec]))
        ctx.log(f"director: angle {i} event channel {n_ev} s")

    zones, zone_ok, zone_kf, zone_source = None, None, None, None
    zd = None
    zl = ctx.pipe / "zones_learned.json"
    if zl.exists():
        try:
            _d = json.loads(zl.read_text()) or {}
            if _d.get("active", True):
                zd, zone_source = _d, "learned"
        except (OSError, ValueError) as e:
            ctx.log(f"director: ignoring bad zones_learned.json ({e})")
    if zd is None and (ctx.pipe / "zones.json").exists():
        try:
            zd = json.loads((ctx.pipe / "zones.json").read_text()) or {}
            zone_source = "drawn"
        except (OSError, ValueError):
            zd = None
    if zd is not None:
        zones, zone_ok, zone_kf, suspended = _load_zone_inputs(
            ctx, zd, avail, T, lo, tm)
        if zones is None:
            zone_source = None
    else:
        suspended = [0.0] * len(ctx.angles)
    return {"tracks": tracks, "avail": avail, "motion": motion,
            "zones": zones, "zone_ok": zone_ok, "zone_kf": zone_kf,
            "zone_source": zone_source,
            "lo": lo, "hi": hi, "T": T, "suspended": suspended,
            "offsets": offsets, "timemap": tm,
            "durations": [ctx.duration(i) for i in range(len(ctx.angles))]}


def _load_zone_inputs(ctx: Ctx, zd: dict, avail: np.ndarray, T: int,
                      lo: float, tm
                      ) -> tuple[list | None, np.ndarray | None,
                                 list | None, list[float]]:
    """Normalise + viewcheck a zones doc (drawn or learned) exactly like
    the inline block used to. `tm` is a piecewise timemap or a legacy
    flat offsets list. Returns (zones, zone_ok, zone_kf, suspended);
    zones None when the doc has no usable polygons."""
    zones, zone_ok, zone_kf = None, None, None
    suspended = [0.0] * len(ctx.angles)
    try:
        from highlights.multiangle.zones import kf_index_ft, normalize_zones
        zones = normalize_zones(
            zd, [ctx.duration(i) for i in range(len(ctx.angles))])
        has = [i for i, kfs in enumerate(zones)
               if any(k.get("zones") for k in kfs)]
        if not (zones and has):
            return None, None, None, suspended
        ctx.log("director: zones on angles/keyframes "
                f"{[(i, [k for k, kf in enumerate(zones[i]) if kf.get('zones')]) for i in has]}")
        zone_kf = [np.zeros(T, dtype=int)
                   for _ in range(len(ctx.angles))]
        zone_ok = np.ones((len(ctx.angles), T), dtype=bool)
        from highlights.multiangle.syncmap import shared_to_file
        for i, a in enumerate(ctx.angles):
            dur = ctx.duration(i)
            # per-second file time via the map; gaps clamp to an edge so
            # zone keyframes stay attached to the nearest covered second
            ft = np.array(
                [shared_to_file(tm, i, t + lo, clamp=True)
                 for t in range(T)], dtype=float)
            zone_kf[i] = kf_index_ft(zones[i], ft)
            if i not in has:
                continue
            vid = ctx.angle_video(i)
            if vid is None:
                continue
            ok_by_k: dict[int, np.ndarray] = {}
            for k, kf in enumerate(zones[i]):
                if not (kf.get("zones") or []):
                    continue  # no polys -> no hits anyway
                ctx.status.update(
                    stage_progress=(i + 0.5) / len(ctx.angles),
                    message=f"director: viewcheck angle {i} "
                            f"keyframe {k}")
                try:
                    times, okarr = _zone_view_ok(
                        ctx, a["dir"], vid, float(kf["t"]))
                except Exception as e:
                    ctx.log(f"director: viewcheck a{i} k{k} "
                            f"failed ({e})")
                    continue
                ok_by_k[k] = _map_view_ok(times, okarr, ft, dur)
            if ok_by_k:
                row = np.ones(T, dtype=bool)
                for k, okk in ok_by_k.items():
                    m = zone_kf[i] == k
                    row[m] = okk[m]
                zone_ok[i] = row
                suspended[i] = float(
                    (~zone_ok[i] & avail[i]).sum()
                    / max(1, avail[i].sum()))
        if any(s > 0 for s in suspended):
            ctx.log("director: zones suspended "
                    f"{[round(s, 3) for s in suspended]} "
                    "(view differs from reference)")
        return zones, zone_ok, zone_kf, suspended
    except Exception as e:
        ctx.log(f"director: ignoring bad zones doc ({e})")
        return None, None, None, suspended


def director_style_overrides(ctx: Ctx) -> dict | None:
    """Learned knobs for this project: multiangle/director_params.json
    {"style_overrides": {...}} — validated against the Style fields."""
    try:
        ov = json.loads((ctx.pipe / "director_params.json").read_text())
        ov = (ov or {}).get("style_overrides") or None
        if ov:
            from highlights.multiangle.director import resolve_style
            resolve_style(ctx.style, ov)        # raises on bad keys
        return ov
    except FileNotFoundError:
        return None
    except ValueError as e:
        ctx.log(f"director: ignoring bad director_params.json ({e})")
        return None


def director_prefs(ctx: Ctx) -> dict | None:
    """Learned per-match prefs: multiangle/director_params.json
    {"prefs": {...}} — passed to cut_director as-is."""
    try:
        pr = json.loads((ctx.pipe / "director_params.json").read_text())
        pr = (pr or {}).get("prefs") or None
        return pr if isinstance(pr, dict) else None
    except (OSError, ValueError):
        return None


def confirmed_events(ctx: Ctx, old_director: dict | None = None) -> list[dict]:
    from highlights.multiangle.goal_aware import SHOT_FAMILY
    from highlights.multiangle.timemap import from_output_time_with_replays

    try:
        project = json.loads((ctx.project_dir / "project.json").read_text())
    except (OSError, ValueError):
        project = {}
    candidates = project.get("candidates") or []
    if candidates:
        replays = (project.get("meta") or {}).get("replays_applied") or []
    else:
        candidate_file = ctx.project_dir / "pipeline" / "candidates.json"
        try:
            doc = json.loads(candidate_file.read_text())
        except (OSError, ValueError):
            doc = {}
        candidates = doc.get("candidates") or doc.get("events") or []
        if old_director is None:
            try:
                old_director = json.loads(
                    (ctx.pipe / "director.json").read_text())
            except (OSError, ValueError):
                old_director = {}
        replays = old_director.get("replays") or []

    events = []
    for candidate in candidates:
        if (candidate.get("status") != "confirmed"
                or candidate.get("type") not in SHOT_FAMILY):
            continue
        try:
            t = from_output_time_with_replays(float(candidate["t"]), replays)
        except (KeyError, TypeError, ValueError):
            continue
        events.append({"id": str(candidate["id"]),
                       "type": str(candidate["type"]), "t": float(t)})
    return sorted(events, key=lambda event: event["t"])


def stage_director(ctx: Ctx) -> dict:
    from highlights.multiangle.director import cut_director
    from highlights.multiangle.goal_aware import load_calib, plan_goal_aware
    from highlights.multiangle.timemap import output_playlist

    old_director = {}
    with contextlib.suppress(OSError, ValueError):
        old_director = json.loads((ctx.pipe / "director.json").read_text())
    inp = load_director_inputs(ctx)
    zones, zone_ok, zone_kf = inp["zones"], inp["zone_ok"], inp["zone_kf"]
    plan = (plan_goal_aware(
        confirmed_events(ctx, old_director), inp["tracks"], inp["avail"],
        load_calib(ctx.pipe))
            if ctx.goal_aware else {"windows": [], "replays": [], "events": []})
    out = cut_director(inp["tracks"], inp["avail"], inp["motion"],
                       ctx.style,
                       zones=zones, zone_ok=zone_ok, zone_kf=zone_kf,
                       style_overrides=director_style_overrides(ctx),
                       prefs=director_prefs(ctx),
                       goal_windows=plan["windows"] or None)
    if ctx.goal_aware:
        out["goal_aware"] = {"enabled": True, "events": plan["events"]}
        for event in plan["events"]:
            ctx.log(f"goal-aware: {event['id']} t={event['t']:.1f} "
                    f"method={event['method']} hold={event['hold_angle']} "
                    f"replay={event['replay_angle']}")
        if plan["windows"]:
            out["replays"] = plan["replays"]
            out["segments_out"] = output_playlist(
                out["segments"], plan["replays"])
            out["duration_live"] = float(out["segments"][-1]["t_end"])
            out["duration_out"] = float(out["segments_out"][-1]["t_end"])
    out["zone_source"] = inp.get("zone_source")
    if zones:
        out["zone_suspended_share"] = [round(s, 4)
                                       for s in inp["suspended"]]
        out["zone_keyframes"] = [len(kfs) for kfs in zones]
    write_json_atomic(ctx.pipe / "director.json", out, indent=1,
                      default=_np_json)
    ctx.log(f"director: {out['n_cuts']} cuts, ratios {out['ratios']}")
    return out


def _map_view_ok(times: np.ndarray, okarr: np.ndarray,
                 ft: np.ndarray, dur: float) -> np.ndarray:
    """Nearest-sample map of viewcheck ok flags (angle file time) onto
    the shared output timeline via the per-second ft array."""
    ft = np.clip(ft, 0, max(0, dur))
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
    if not ctx.read_only:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(cache, {"times": times.tolist(), "ok": ok.tolist()})
    return times, ok


def stage_render(ctx: Ctx) -> None:
    from highlights.multiangle.render import render
    from highlights.multiangle.syncmap import timemap_from_sync
    sync = json.loads((ctx.pipe / "sync.json").read_text())
    director = json.loads((ctx.pipe / "director.json").read_text())
    videos = [str(ctx.angle_video(i)) for i in range(len(ctx.angles))]
    lo, hi = ctx.union(sync)
    tm = timemap_from_sync(sync, getattr(ctx, "durations", None) or [ctx.duration(i) for i in range(len(ctx.angles))] if ctx.angles else None)
    out = render(videos, tm,
                 director.get("segments_out") or director["segments"], lo, hi,
                 ctx.pipe, ctx.project_dir / "match.mp4", videos[0],
                 durations=list(getattr(ctx, "durations", None) or [ctx.duration(i) for i in range(len(ctx.angles))]), log=ctx.log)
    # register the cut as the project video for the Option-1 UI
    pipe1 = ctx.project_dir / "pipeline"
    pipe1.mkdir(exist_ok=True)
    info = ffprobe(out)
    write_json_atomic(pipe1 / "probe.json", info, indent=1)
    ctx.status.update(video=info, video_path=str(out))
    ctx.log(f"render: wrote {out} ({info['width']}x{info['height']})")


def stage_fuse(ctx: Ctx) -> dict:
    from highlights.multiangle.fuse import (
        drop_outside_window,
        fuse_candidates,
        to_output_time,
    )
    from highlights.multiangle.syncmap import timemap_from_sync
    from highlights.multiangle.timemap import events_to_output, to_output_time_with_replays
    sync = json.loads((ctx.pipe / "sync.json").read_text())
    tm = timemap_from_sync(sync, getattr(ctx, "durations", None) or [ctx.duration(i) for i in range(len(ctx.angles))] if ctx.angles else None)
    director = {}
    with contextlib.suppress(OSError, ValueError):
        director = json.loads((ctx.pipe / "director.json").read_text())
    replays = director.get("replays") or []
    files = [a["dir"] / "pipeline" / "candidates.json" for a in ctx.angles]
    labels = [a["label"] for a in ctx.angles]
    out = fuse_candidates(files, tm, labels,
                          ctx.pipe / "fused_candidates.json")
    lo, hi = ctx.union(sync)
    dur_live = hi - lo
    # bound the window to the actual rendered file — the container can be a
    # few s shorter than the declared union
    match = ctx.project_dir / "match.mp4"
    if match.is_file():
        dur_live = min(dur_live, float(ffprobe(match)["duration_s"]))
    # fused events are on shared T; the UI plays the rendered video whose
    # time axis is output time (0 = union start) -> shift everything by -lo,
    # then drop events the render doesn't cover (before/after a cut_range)
    for key in ("events", "candidates"):
        if key in out:
            shifted = to_output_time(out[key], lo)
            kept = drop_outside_window(shifted, dur_live)
            if len(kept) != len(shifted):
                ctx.log(f"fuse: dropped {len(shifted) - len(kept)} events "
                        "outside the rendered window")
            out[key] = kept
    write_json_atomic(ctx.pipe / "fused_candidates.json", out, indent=1)
    output = dict(out)
    for key in ("events", "candidates"):
        if key in output:
            output[key] = events_to_output(output[key], replays)
    pdir = ctx.project_dir / "pipeline"
    pdir.mkdir(exist_ok=True)
    write_json_atomic(pdir / "candidates.json", output, indent=1)
    # a0's match window + features, shifted to output time; with a
    # cut_range the rendered video IS the match — window covers it all
    dur_out = dur_live + sum(
        float(replay["t_out_end"]) - float(replay["t_out_start"])
        for replay in replays)
    ranged = (ctx.pipe / "cut_range.json").exists()
    shift = sync["offsets"][0] - lo
    mw_src = ctx.angles[0]["dir"] / "pipeline" / "match_window.json"
    if mw_src.exists():
        mw = json.loads(mw_src.read_text())

        def _clamp(v: float) -> float:
            return min(max(float(v), 0.0), dur_live)

        def _halves() -> list:
            kept = []
            for h in mw.get("halves", []) or []:
                s = _clamp(h.get("start", 0) + shift)
                e = _clamp(h.get("end", 0) + shift)
                if e > s:
                    kept.append({
                        **h,
                        "start": to_output_time_with_replays(s, replays),
                        "end": to_output_time_with_replays(e, replays),
                    })
            return kept

        if ranged:
            mw["match_window"] = [0.0, dur_out]
        elif isinstance(mw.get("match_window"), list):
            mw["match_window"] = [
                to_output_time_with_replays(_clamp(v + shift), replays)
                                  for v in mw["match_window"]]
        if "halves" in mw:
            mw["halves"] = _halves()
        write_json_atomic(pdir / "match_window.json", mw, indent=1)
    fsrc = ctx.angles[0]["dir"] / "pipeline" / "features_1s.parquet"
    if fsrc.exists():
        df = pd.read_parquet(fsrc)
        df["t"] = df["t"] + shift
        df = df[(df["t"] >= 0.0) & (df["t"] <= dur_live)]
        df["t"] = df["t"].map(
            lambda t: to_output_time_with_replays(float(t), replays))
        write_parquet_atomic(df, pdir / "features_1s.parquet")
    ctx.log(f"fuse: {len(out['events'])} fused events")
    return out


def stage_stats(ctx: Ctx) -> None:
    from highlights.multiangle.timemap import events_to_output
    from highlights.pipeline.stats import compute_stats
    pdir = ctx.project_dir / "pipeline"
    feats = pd.read_parquet(pdir / "features_1s.parquet")
    director = json.loads((ctx.pipe / "director.json").read_text())
    replays = director.get("replays") or []
    live_cands = json.loads(
        (ctx.pipe / "fused_candidates.json").read_text())["events"]
    cands = events_to_output(live_cands, replays)
    mw = json.loads((pdir / "match_window.json").read_text()) \
        if (pdir / "match_window.json").exists() else {}
    # the rendered video covers the effective cut range, not angle 0's file
    sync_dur = json.loads((ctx.pipe / "sync.json").read_text())
    lo, hi = ctx.union(sync_dur)
    dur = float(director.get("duration_out") or hi - lo)
    stats = compute_stats(feats, cands, dur, tuple(mw.get("match_window", ())),
                          mw.get("halves"), None)
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


def stage_scoreboard(ctx: Ctx) -> None:
    """Burn scoreboard + clock into match.mp4. Reads
    multiangle/scoreboard.json (written by POST /multiangle/scoreboard):
    {home:{label,hex}, away:{label,hex}, goals:[{t,team}], kickoff} and
    replay intervals from director.json — all times are OUTPUT time."""
    from highlights.multiangle.scoreboard import apply_scoreboard, find_bold_font, scoreboard_filter
    spec_path = ctx.pipe / "scoreboard.json"
    if not spec_path.exists():
        ctx.log("scoreboard: skip (no spec requested)")
        return
    spec = json.loads(spec_path.read_text())
    replays = []
    director_path = ctx.pipe / "director.json"
    if director_path.is_file():
        replays = json.loads(director_path.read_text()).get("replays") or []
    match = ctx.project_dir / "match.mp4"
    if not match.is_file():
        raise PipelineError("no match.mp4 to overlay")
    font = find_bold_font()
    if not font:
        raise PipelineError("no usable font for the scoreboard")
    kickoff = float(spec.get("kickoff") or 0.0)
    goals = spec.get("goals") or []
    dur = 0.0
    with contextlib.suppress(Exception):
        dur = float(ffprobe(match).get("duration_s") or 0.0)
    vf = scoreboard_filter(
        goals,
        (spec.get("home") or {}).get("label") or "Home",
        (spec.get("away") or {}).get("label") or "Away",
        (spec.get("home") or {}).get("hex"),
        (spec.get("away") or {}).get("hex"),
        kickoff, font, dur=dur, replays=replays)
    apply_scoreboard(
        match, match, vf,
        progress_cb=lambda f: ctx.status.update(
            stage_progress=f,
            message=f"scoreboard {f * 100:.0f}%"),
        log=ctx.log)
    pipe1 = ctx.project_dir / "pipeline"
    pipe1.mkdir(exist_ok=True)
    info = ffprobe(match)
    write_json_atomic(pipe1 / "probe.json", info, indent=1)
    ctx.status.update(video=info, video_path=str(match))
    cut_from = None
    try:
        cr = json.loads((ctx.pipe / "cut_range.json").read_text())
        cut_from = [float(cr["lo"]), float(cr["hi"])]
    except Exception:
        pass
    write_json_atomic(ctx.pipe / "scoreboard_applied.json",
                      {"cut_from": cut_from, "goals": goals,
                       "kickoff": kickoff, "replays": replays,
                       "created_at": time.time()}, indent=1)
    ctx.log(f"scoreboard: burned {len(goals)} goals, kickoff {kickoff:.1f}s"
            f" (dur {dur:.0f}s)")


# ------------------------------ driver ------------------------------------

def cleanup_caches(ctx: Ctx) -> int:
    """Delete the regenerable render caches (per-angle mezzanines and the
    extracted segment cache) after a successful run — ~tens of GB per
    match. Kept when HL_KEEP_CACHES=1 (e.g. debugging a re-cut). Sources,
    cuts, track features and viewcheck caches are never touched.
    Returns bytes freed."""
    if os.environ.get("HL_KEEP_CACHES") == "1":
        ctx.log("cleanup: HL_KEEP_CACHES=1 — caches kept")
        return 0
    freed = 0
    for d in (ctx.pipe / "mezz", ctx.pipe / "segs"):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file():
                freed += f.stat().st_size
        shutil.rmtree(d)
    ctx.log(f"cleanup: freed {freed / 1e9:.1f} GB of render caches")
    return freed


class _NoopStatus:
    def update(self, **_kwargs) -> None:
        return None


def dry_run_goal_aware(project_dir: Path) -> dict:
    """Compare the existing and goal-aware cuts without writing project files."""
    project_dir = Path(project_dir)
    pipe = project_dir / "multiangle"
    project = json.loads((project_dir / "project.json").read_text())
    meta = project.get("meta") or {}
    ctx = Ctx(
        project_dir=project_dir,
        pipe=pipe,
        status=_NoopStatus(),
        angles=_load_angles(project_dir, None, create_dirs=False),
        style=meta.get("cut_style", "fast"),
        goal_aware=True,
        read_only=True,
    )
    ctx.log = lambda _message: None
    inputs = load_director_inputs(ctx)
    style_overrides = director_style_overrides(ctx)
    prefs = director_prefs(ctx)
    from highlights.multiangle.director import cut_director
    from highlights.multiangle.goal_aware import load_calib, plan_goal_aware
    from highlights.multiangle.timemap import output_playlist

    common = {
        "zones": inputs["zones"],
        "zone_ok": inputs["zone_ok"],
        "zone_kf": inputs["zone_kf"],
        "style_overrides": style_overrides,
        "prefs": prefs,
    }
    old = cut_director(inputs["tracks"], inputs["avail"], inputs["motion"],
                       ctx.style, **common)
    plan = plan_goal_aware(
        confirmed_events(ctx), inputs["tracks"], inputs["avail"],
        load_calib(pipe))
    new = cut_director(
        inputs["tracks"], inputs["avail"], inputs["motion"], ctx.style,
        goal_windows=plan["windows"] or None, **common)
    old_duration = float(old["segments"][-1]["t_end"])
    output_segments = (output_playlist(new["segments"], plan["replays"])
                       if plan["windows"] else new["segments"])
    new_duration = (float(output_segments[-1]["t_end"])
                    if plan["windows"] else old_duration)
    return {
        "events": plan["events"],
        "replays": plan["replays"],
        "duration_live": old_duration,
        "duration_out": new_duration,
    }


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
    "scoreboard": Stage(0.2, stage_scoreboard,
                        lambda c: [c.pipe / "scoreboard_applied.json"]),
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
    ap.add_argument("--stages",
                    default=",".join(n for n in STAGES if n != "scoreboard"))
    ap.add_argument("--offsets", help="comma list, len==n angles, first must be 0")
    ap.add_argument("--cookies")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--style", default="fast",
                    choices=("normal", "fast"))
    ap.add_argument("--goal-aware", choices=("on", "off"), default="on")
    ap.add_argument("--dry-run-goal-aware", action="store_true")
    args = ap.parse_args(argv)

    project_dir = args.project_dir
    if args.dry_run_goal_aware:
        print(json.dumps(dry_run_goal_aware(project_dir), indent=1))
        return 0
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
              style=args.style, goal_aware=args.goal_aware == "on")

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
            if "render" in names or "scoreboard" in names:
                from highlights.multiangle.cuts import snapshot_cut
                sb = (pipe / "scoreboard_applied.json").is_file()
                cr = None
                try:
                    crd = json.loads((pipe / "cut_range.json").read_text())
                    cr = [float(crd["lo"]), float(crd["hi"])]
                except Exception:
                    pass
                try:
                    meta = snapshot_cut(
                        project_dir, ctx.style, cut_range=cr,
                        label_suffix=" + scoreboard" if sb else "",
                        extra_meta={"scoreboard": True} if sb else None)
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
        try:
            cleanup_caches(ctx)
        except Exception as e:
            ctx.log(f"cleanup: failed ({e})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
