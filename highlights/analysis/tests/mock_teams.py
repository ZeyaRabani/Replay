"""Synthesise a plausible analysis/teams.json from an angle's features.

Used by tests (and UI screenshots) when no video exists to run the real
teams pass on: players are split deterministically into two pseudo-teams
around players_cx ± players_spread, and the ball comes from features.

    python -m highlights.analysis.tests.mock_teams <project_dir> [--ref N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from highlights.analysis.run import _angle_dirs, _angle_duration, _load_json
from highlights.analysis.teams import COLS
from highlights.io import write_json_atomic

TEAMS = {"A": {"name": "orange", "hsv": [12.0, 240.0, 240.0], "hex": "#f08c00"},
         "B": {"name": "white", "hsv": [0.0, 20.0, 235.0], "hex": "#ebebeb"}}


def _context(project_dir: Path) -> dict:
    """offsets + clipped window + ref angle, tolerating missing video."""
    project_dir = Path(project_dir)
    sync = _load_json(project_dir / "multiangle" / "sync.json") or {}
    offsets = [float(o) for o in sync.get("offsets") or [0.0]]
    ulo, uhi = (sync.get("coverage") or {}).get("union") or (0.0, 0.0)
    lo, hi = float(ulo), float(uhi)
    cr = _load_json(project_dir / "multiangle" / "cut_range.json")
    if cr and cr.get("hi") is not None and cr.get("lo") is not None:
        lo2, hi2 = max(lo, float(cr["lo"])), min(hi, float(cr["hi"]))
        if hi2 > lo2:
            lo, hi = lo2, hi2
    dirs = _angle_dirs(project_dir)
    durs = [_angle_duration(d) for d in dirs]
    ref = int(max(range(len(durs)), key=lambda i: durs[i])) if dirs else 0
    return {"offsets": offsets, "window": (lo, hi), "ref": ref,
            "dirs": dirs, "project_dir": project_dir}


def make_mock_teams(project_dir: Path, ref_angle: int | None = None,
                    seed: int = 7) -> dict:
    ctx = _context(project_dir)
    ref = ref_angle if ref_angle is not None else ctx["ref"]
    offsets = ctx["offsets"]
    lo, hi = ctx["window"]
    off = offsets[ref] if ref < len(offsets) else 0.0
    feat = _load_json(ctx["dirs"][ref] / "track" / "features_1s.json") or {}
    cols = feat.get("columns") or []
    idx = {c: i for i, c in enumerate(cols)}

    rng = np.random.default_rng(seed)
    rows = []
    for r in feat.get("rows") or []:
        tf = float(r[idx.get("t", 0)])
        ts = tf + off
        if not (lo <= ts <= hi):
            continue
        n = int(float(r[idx.get("n_players", 1)] or 0))
        cx = float(r[idx.get("players_cx", 6)] or 0.5)
        spread = float(r[idx.get("players_spread", 8)] or 0.1)
        nA = max(0, min(n, round(n * 0.55)))
        nB = n - nA
        m = 0.10 if ts < (lo + hi) / 2 else -0.10   # swap ends at half time
        ax = [round(float(x), 3) for x in
              np.clip(rng.normal(cx - m, max(spread, 0.02) / 2, nA), 0.02, 0.98)]
        bx = [round(float(x), 3) for x in
              np.clip(rng.normal(cx + m, max(spread, 0.02) / 2, nB), 0.02, 0.98)]
        rows.append([int(ts),
                     round(float(r[idx.get("ball_x", 4)] or 0.0), 3),
                     round(float(r[idx.get("ball_y", 5)] or 0.0), 3),
                     round(float(r[idx.get("ball_conf", 2)] or 0.0), 3),
                     ax, bx, nA, nB])
    return {
        "fps": 0.5, "columns": COLS, "rows": rows,
        "teams": TEAMS, "team_confidence": 0.72,
        "n_descriptors": len(rows) * 6, "n_outliers": len(rows),
        "model": "mock_teams", "imgsz": 960, "video": None,
        "window_file": [lo - off, hi - off],
        "window_shared": [lo, hi],
        "shared_offset": off,
        "generated_at": round(time.time(), 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.tests.mock_teams")
    ap.add_argument("project_dir", type=Path)
    ap.add_argument("--ref", type=int, default=None)
    args = ap.parse_args(argv)
    out = args.project_dir / "analysis" / "teams.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = make_mock_teams(args.project_dir, args.ref)
    write_json_atomic(out, d, indent=0)
    print(f"mock teams: {len(d['rows'])} rows -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
