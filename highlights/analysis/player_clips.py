"""Per-player highlight reel for one whole-match identity.

    python -m highlights.analysis.player_clips --project-dir P --identity A3

Cuts the ACTIVE cut (`<project>/match.mp4`, output time) around:
  * the identity's fastest runs: top N_SPRINTS smoothed-speed peaks
    (sprints first, then the fastest remaining runs), PRE_S before /
    POST_S after the peak;
  * confirmed goal / shot candidates whose scorer is this identity, or
    where the identity was within BALL_NEAR_M of the fused ball.
Tracks / ball live on the shared timeline; output t = shared t - lo where
lo is the active cut's start (same as the radar / players_run mapping).
Overlapping windows are merged, then ffmpeg.render_reel cuts + concats.
Writes analysis/players_v2/reels/<iid>/{status.json,log.txt,reel.json,
reel.mp4,clips/}.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from highlights.app.backend import ffmpeg as fx
from highlights.pipeline.status import StatusWriter

from .fuse_tracks import SPRINT_MS, STEP
from .identity import smoothed_steps

PRE_S = 4.0
POST_S = 3.0
N_SPRINTS = 8
MIN_RUN_MS = 4.0          # speed peaks below this aren't reel-worthy
BALL_NEAR_M = 5.0
BALL_DT_S = 1.5           # ball sample must be this close to the event
POS_DT_S = 1.0            # identity position must be this close too
EVENT_TYPES = ("goal", "shot")
PRIORITY = {"goal": 3, "shot": 2, "sprint": 1, "run": 0}


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def reels_dir(project_dir: Path, iid: str) -> Path:
    return Path(project_dir) / "analysis" / "players_v2" / "reels" / iid


# ---------- timeline ----------

def output_offset(project_dir: Path, tracks_doc: dict | None = None) -> float:
    """Shared-timeline second at output t=0 of the active cut's match.mp4:
    the active cut's range start, else the sync union clipped by
    cut_range.json (analysis.run.resolve_context), else the tracks' t0."""
    ma = Path(project_dir) / "multiangle"
    cid = (_read(ma / "cuts" / "active.json") or {}).get("id")
    if cid:
        rng = (_read(ma / "cuts" / str(cid) / "meta.json") or {}).get("range")
        if rng:
            return float(rng[0])
    sync = _read(ma / "sync.json") or {}
    un = (sync.get("coverage") or {}).get("union")
    cr = _read(ma / "cut_range.json") or {}
    if un:
        lo, hi = float(un[0]), float(un[1])
        if cr.get("lo") is not None and cr.get("hi") is not None:
            lo2, hi2 = max(lo, float(cr["lo"])), min(hi, float(cr["hi"]))
            if hi2 > lo2:
                lo = lo2
        return lo
    if cr.get("lo") is not None:
        return float(cr["lo"])
    return float((tracks_doc or {}).get("t0") or 0.0)


# ---------- identity trajectory ----------

def identity_path(ident: dict, tracks: list[dict]) -> tuple[float, list]:
    """(start, STEP-spaced xy) over the identity's member tracks;
    simultaneous cross-camera duplicates are averaged."""
    ids = set(ident.get("track_ids") or [])
    members = [t for t in tracks if int(t["id"]) in ids]
    if not members:
        return 0.0, []
    start = min(float(t["start"]) for t in members)
    end = max(float(t["end"]) for t in members)
    n = round((end - start) / STEP) + 1
    acc = np.zeros((n, 2))
    cnt = np.zeros(n)
    for t in members:
        k0 = round((float(t["start"]) - start) / STEP)
        for j, p in enumerate(t.get("xy") or []):
            if p is None or p[0] is None or not 0 <= k0 + j < n:
                continue
            acc[k0 + j] += (float(p[0]), float(p[1]))
            cnt[k0 + j] += 1
    xy = [[float(acc[k, 0] / cnt[k]), float(acc[k, 1] / cnt[k])]
          if cnt[k] else [None, None] for k in range(n)]
    return start, xy


def pos_at(start: float, xy: list, t: float,
           tol: float = POS_DT_S) -> tuple[float, float] | None:
    if not xy:
        return None
    k = round((t - start) / STEP)
    r = round(tol / STEP)
    for dk in sorted(range(-r, r + 1), key=abs):
        j = k + dk
        if 0 <= j < len(xy) and xy[j][0] is not None:
            return float(xy[j][0]), float(xy[j][1])
    return None


def fastest_runs(start: float, xy: list, n: int = N_SPRINTS,
                 sep_s: float = PRE_S + POST_S) -> list[dict]:
    """Top-n smoothed speed peaks >= MIN_RUN_MS, at least sep_s apart;
    peaks >= SPRINT_MS are 'sprint', the rest 'run'."""
    if len(xy) < 2:
        return []
    _sm, d, ok = smoothed_steps(xy)
    sp = np.where(ok, d / STEP, 0.0)
    picked: list[dict] = []
    for k in np.argsort(-sp, kind="stable"):
        v = float(sp[k])
        if v < MIN_RUN_MS or len(picked) >= n:
            break
        t = start + (int(k) + 1) * STEP
        if any(abs(t - q["t"]) < sep_s for q in picked):
            continue
        picked.append({"t": round(t, 2), "speed_ms": round(v, 2),
                       "type": "sprint" if v >= SPRINT_MS else "run"})
    return picked


# ---------- events ----------

def load_candidates(project_dir: Path) -> list[dict]:
    """Reviewed candidates (project.json) else the pipeline's; t is
    output time."""
    pj = _read(Path(project_dir) / "project.json") or {}
    cands = pj.get("candidates") or []
    if not cands:
        doc = _read(Path(project_dir) / "pipeline" / "candidates.json") or {}
        cands = doc.get("candidates") or doc.get("events") or []
    return [c for c in cands if isinstance(c, dict) and "t" in c]


def scorer_matches(cid: str, ident: dict, roster: dict) -> bool:
    """Roster scorer is this identity: by id, or a roster player whose
    name equals the identity's name (roster tracklets are v1 ids)."""
    pid = (roster.get("scorers") or {}).get(str(cid))
    if not pid:
        return False
    if pid == ident["id"]:
        return True
    name = (ident.get("name") or "").strip().casefold()
    return bool(name) and any(
        pl.get("id") == pid
        and (pl.get("name") or "").strip().casefold() == name
        for pl in roster.get("players") or [])


def ball_at(ball: list, t: float, tol: float = BALL_DT_S
            ) -> tuple[float, float] | None:
    best = None
    for s in ball or []:
        dt = abs(float(s[0]) - t)
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, float(s[1]), float(s[2]))
    return (best[1], best[2]) if best else None


def select_items(ident: dict, tracks: list[dict], *, ball: list,
                 candidates: list[dict], roster: dict, lo: float,
                 duration: float) -> list[dict]:
    """render_reel items (output time), merged where windows overlap."""
    start, xy = identity_path(ident, tracks)
    items: list[dict] = []

    def add(kind: str, t_out: float, s: float, e: float, **extra) -> None:
        s, e = max(0.0, s), min(duration, e) if duration > 0 else e
        if e - s < 1.0:
            return
        items.append({"type": kind, "t": round(t_out, 2),
                      "clip_start": round(s, 2), "clip_end": round(e, 2),
                      **extra})

    for r in fastest_runs(start, xy):
        t_out = r["t"] - lo
        add(r["type"], t_out, t_out - PRE_S, t_out + POST_S,
            speed_ms=r["speed_ms"])

    for c in candidates:
        if str(c.get("type")) not in EVENT_TYPES:
            continue
        if str(c.get("status")) != "confirmed":
            continue
        t_out = float(c["t"])
        reason = None
        if scorer_matches(str(c.get("id")), ident, roster):
            reason = "scorer"
        else:
            me = pos_at(start, xy, t_out + lo)
            b = ball_at(ball, t_out + lo)
            if me and b and float(np.hypot(me[0] - b[0], me[1] - b[1])
                                  ) <= BALL_NEAR_M:
                reason = "near_ball"
        if reason is None:
            continue
        s = float(c.get("clip_start", c.get("t_start", t_out - PRE_S)))
        e = float(c.get("clip_end", c.get("t_end", t_out + POST_S)))
        add(str(c["type"]), t_out, min(s, t_out - PRE_S),
            max(e, t_out + POST_S), candidate_id=str(c.get("id")),
            reason=reason)
    return merge_items(items)


def merge_items(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for it in sorted(items, key=lambda i: i["clip_start"]):
        if out and it["clip_start"] <= out[-1]["clip_end"]:
            cur = out[-1]
            cur["clip_end"] = max(cur["clip_end"], it["clip_end"])
            cur["parts"].append({k: v for k, v in it.items()
                                 if k not in ("clip_start", "clip_end")})
            if PRIORITY.get(it["type"], 0) > PRIORITY.get(cur["type"], 0):
                cur["type"], cur["t"] = it["type"], it["t"]
            continue
        out.append({**it, "parts": [{k: v for k, v in it.items()
                                     if k not in ("clip_start", "clip_end")}]})
    for i, it in enumerate(out):
        it["id"] = f"{i + 1:02d}"
        it["name"] = f"{i + 1:02d}_{it['type']}_{it['t']:07.1f}.mp4"
    return out


# ---------- reel ----------

def build_reel(project_dir: Path, iid: str, *,
               progress_cb: Callable[[float, str], None] | None = None,
               log=print) -> dict:
    project_dir = Path(project_dir)
    v2 = project_dir / "analysis" / "players_v2"
    idoc = _read(v2 / "identities.json")
    if not idoc:
        raise FileNotFoundError("no identities.json — link identities first")
    ident = next((i for i in idoc.get("identities") or []
                  if i["id"] == iid), None)
    if ident is None:
        raise KeyError(f"unknown identity {iid}")
    tdoc = _read(v2 / "tracks.json") or {}
    src = project_dir / "match.mp4"
    if not src.is_file():
        raise FileNotFoundError("no active cut match.mp4")
    duration = float(fx.probe(src)["duration_s"])
    lo = output_offset(project_dir, tdoc)
    roster = _read(project_dir / "analysis" / "players" / "roster.json") or {}
    items = select_items(ident, tdoc.get("tracks") or [],
                         ball=tdoc.get("ball") or [],
                         candidates=load_candidates(project_dir),
                         roster=roster, lo=lo, duration=duration)
    if not items:
        raise ValueError(f"{iid}: no sprints or events inside the cut")
    log(f"{iid}: {len(items)} clips "
        f"({sum(len(i['parts']) for i in items)} moments), lo={lo:.1f}")
    out_dir = reels_dir(project_dir, iid)
    clips = out_dir / "clips"
    if clips.is_dir():
        for f in clips.glob("*.mp4"):
            f.unlink()
    res = fx.render_reel(src, items, out_dir, progress_cb=progress_cb)
    manifest = {"identity": iid, "name": ident.get("name"),
                "team": ident.get("team"), "lo_shared": lo,
                "items": items, "reel_s": round(res["reel_s"], 2),
                "created_at": time.time()}
    (out_dir / "reel.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.player_clips")
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--identity", required=True)
    args = ap.parse_args(argv)
    out_dir = reels_dir(args.project_dir, args.identity)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(out_dir / "status.json", min_interval=0.5)

    def on_sigterm(signum, frame):
        status.update(state="failed", error="cancelled",
                      finished_at=time.time(), force=True)
        sys.exit(1)
    signal.signal(signal.SIGTERM, on_sigterm)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    status.update(state="running", stage="reel", message="selecting clips",
                  pid=os.getpid(), started_at=time.time(), force=True)
    try:
        m = build_reel(
            args.project_dir, args.identity, log=log,
            progress_cb=lambda f, msg: status.update(
                progress=round(f, 3), stage_progress=round(f, 3),
                message=msg))
    except Exception as e:
        log(f"FAILED ({type(e).__name__}): {e}")
        status.update(state="failed", error=str(e), message="failed",
                      finished_at=time.time(), force=True)
        return 1
    status.update(state="done", progress=1.0, message="done",
                  n_clips=len(m["items"]), reel_s=m["reel_s"],
                  finished_at=time.time(), force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
