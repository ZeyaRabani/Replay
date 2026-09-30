"""Whole-match player identities from fused players-v2 tracks.

Links tracks.json fragments into at most `team_size` persistent
identities per team (plus an unassigned bucket for referee, spectators,
leftover duplicates) and aggregates per-identity stats.

Formulation — min-cost flow, one network per team (k disjoint paths):

    S -> in_i  (enter cost)      in_i -> out_i (-coverage reward, cap 1)
    out_i -> in_j (link cost)    out_i -> T    (0)

Each unit of flow is one identity: a time-ordered chain of tracks. With
at most `team_size` units, no more than `team_size` identities of a
team are ever on the pitch together, and a chain can never contain two
tracks overlapping >= MIN_OVERLAP_S (links only go forward in time), so
both cannot-link rules hold by construction. Link cost combines the gap
time, the displacement against a plausible speed (<= VMAX_MS, hard
gate), the crop-appearance distance and a goalkeeper-role mismatch.
Solved exactly with successive shortest paths (vectorised
Bellman-Ford; no networkx dependency), stopping early once another
identity would not pay for itself.

    python -m highlights.analysis.identity <players_v2_dir> [--team-size 11]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from highlights.io import write_json_atomic

from .fuse_tracks import SPRINT_MS, STEP
from .groups import (
    FEAT_DIM,
    MAX_PER_TEAM,
    MIN_OVERLAP_S,
    _fingerprint_one,
    tracklet_fingerprint,
)
from .groups_v2 import merge_duplicates

VMAX_MS = 8.0            # hard gate: implied speed across a link
MAX_GAP_S = 90.0         # longest gap a single link may bridge
REACH_M = 1.5            # displacement tolerated at zero gap
REACH_MS = 2.5           # + typical mean speed while unobserved
C_LINK = 0.0             # fixed cost per link (s of coverage)
W_GAP = 0.2              # per second of gap
W_DIST = 5.0             # per (d / reach)^2
W_APP = 12.0             # per unit cosine distance of crop fingerprints
W_ROLE = 0.0             # goalkeeper <-> outfield mismatch (off: hurt
                         # held-out re-linking on 28/8, see PR)
C_ENTER = 4.0            # starting a new identity
MIN_TRACK_S = 4.0        # shorter fragments stay unassigned
GK_BAND_M = 9.0          # "near own goal" band for the role prior
GK_FRAC = 0.8            # fraction of points in the band to be gk-like
GLITCH_MS = 10.0         # step speeds above this are projection jumps
SPRINT_MIN_S = 1.0       # a sprint episode lasts at least this long
SMOOTH_STEPS = 3         # moving-average window for distance / speed
N_CROPS = 6


# ---------- per-track features ----------

def _valid(xy: list) -> np.ndarray:
    """[n, 2] float array with NaN for missing steps."""
    return np.array([[np.nan, np.nan] if p is None or p[0] is None
                     else [float(p[0]), float(p[1])] for p in xy],
                    dtype=float).reshape(-1, 2)


def _endpoints(t: dict) -> dict:
    """First/last valid position + velocity over the last/first ~2 s."""
    a = _valid(t["xy"])
    ok = np.where(~np.isnan(a[:, 0]))[0]
    if not len(ok):
        return {"p0": None, "p1": None, "v0": np.zeros(2), "v1": np.zeros(2)}
    i0, i1 = int(ok[0]), int(ok[-1])
    k = max(1, round(2.0 / STEP))

    def vel(ia: int, ib: int) -> np.ndarray:
        if ib <= ia or np.isnan(a[ia, 0]) or np.isnan(a[ib, 0]):
            return np.zeros(2)
        v = (a[ib] - a[ia]) / ((ib - ia) * STEP)
        s = float(np.hypot(*v))
        return v * (VMAX_MS / s) if s > VMAX_MS else v

    j1 = ok[ok >= i1 - k][0]
    j0 = ok[ok <= i0 + k][-1]
    return {"p0": a[i0], "p1": a[i1], "v0": vel(i0, int(j0)),
            "v1": vel(int(j1), i1),
            "s0": float(t["start"]) + i0 * STEP,
            "s1": float(t["start"]) + i1 * STEP}


def _gk_side(t: dict, pitch_len: float) -> int:
    """-1 / +1 when the track stays within GK_BAND_M of the x=0 / x=L
    goal line, else 0."""
    a = _valid(t["xy"])
    x = a[~np.isnan(a[:, 0]), 0]
    if len(x) < 8:
        return 0
    if np.mean(x <= GK_BAND_M) >= GK_FRAC:
        return -1
    if np.mean(x >= pitch_len - GK_BAND_M) >= GK_FRAC:
        return 1
    return 0


def smoothed_steps(xy: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(smoothed positions, per-step distance, step-is-valid) for a
    STEP-spaced xy list; see motion_stats."""
    a = _valid(xy)
    sm = a.copy()
    h = SMOOTH_STEPS // 2
    for i in range(len(a)):
        w = a[max(0, i - h):i + h + 1]
        w = w[~np.isnan(w[:, 0])]
        if len(w) and not np.isnan(a[i, 0]):
            sm[i] = w.mean(axis=0)
    d = np.hypot(*(sm[1:] - sm[:-1]).T)
    ok = ~np.isnan(d) & (d / STEP <= GLITCH_MS)
    return sm, d, ok


def motion_stats(xy: list) -> dict:
    """Distance / speed / sprints from a STEP-spaced xy list.

    Positions are smoothed with a SMOOTH_STEPS moving average (projection
    jitter otherwise inflates distance); steps implying > GLITCH_MS are
    projection jumps and are ignored. A sprint is a run of smoothed speed
    >= SPRINT_MS lasting >= SPRINT_MIN_S; its peak is reported."""
    out = {"distance_m": 0.0, "top_speed_ms": 0.0, "sprints": 0,
           "sprint_events": [], "moving_s": 0.0}
    if len(xy) < 2:
        return out
    sm, d, ok = smoothed_steps(xy)
    sp = d / STEP
    out["distance_m"] = float(d[ok].sum())
    out["moving_s"] = float(ok.sum() * STEP)
    if ok.any():
        out["top_speed_ms"] = float(sp[ok].max())
    fast = ok & (sp >= SPRINT_MS)
    i = 0
    min_steps = max(1, round(SPRINT_MIN_S / STEP))
    while i < len(fast):
        if not fast[i]:
            i += 1
            continue
        j = i
        while j < len(fast) and fast[j]:
            j += 1
        if j - i >= min_steps:
            k = i + int(np.argmax(sp[i:j]))
            out["sprint_events"].append(
                {"step": k, "speed_ms": round(float(sp[k]), 2),
                 "dur_s": round((j - i) * STEP, 2),
                 "xy": [round(float(sm[k + 1, 0]), 2),
                        round(float(sm[k + 1, 1]), 2)]})
        i = j
    out["sprints"] = len(out["sprint_events"])
    return out


# ---------- link costs ----------

def link_cost(ea: dict, eb: dict, *, app_d: float = 0.0,
              role_a: int = 0, role_b: int = 0) -> float | None:
    """Cost of "track b continues track a", None when implausible.

    ea/eb are _endpoints() dicts (+ 'end' / 'start' track times)."""
    if ea["p1"] is None or eb["p0"] is None:
        return None
    gap = float(eb["s0"]) - float(ea["s1"])
    if gap < -MIN_OVERLAP_S + 1e-6 or gap > MAX_GAP_S:
        return None
    g = max(gap, 0.0)
    pred = ea["p1"] + ea["v1"] * min(g, 1.5)
    d_raw = float(np.hypot(*(eb["p0"] - ea["p1"])))
    d = min(d_raw, float(np.hypot(*(eb["p0"] - pred))))
    if d > REACH_M + VMAX_MS * g:
        return None
    reach = REACH_M + REACH_MS * g
    cost = C_LINK + W_GAP * g + W_DIST * (d / reach) ** 2 + W_APP * app_d
    if (role_a != 0) != (role_b != 0) or (role_a and role_b
                                          and role_a != role_b):
        cost += W_ROLE
    return float(cost)


# ---------- min-cost flow (successive shortest paths) ----------

def _min_cost_paths(n: int, reward: np.ndarray, enter: np.ndarray,
                    links: list[tuple[int, int, float]], k: int
                    ) -> list[list[int]]:
    """Up to k vertex-disjoint S->T chains maximising total reward minus
    link / enter costs. Nodes 0..n-1 are tracks (in=2i, out=2i+1),
    S=2n, T=2n+1. Returns chains as lists of track indices."""
    S, T = 2 * n, 2 * n + 1
    us, vs, cs = [], [], []
    for i in range(n):
        us += [S, 2 * i, 2 * i + 1]
        vs += [2 * i, 2 * i + 1, T]
        cs += [float(enter[i]), -float(reward[i]), 0.0]
    for a, b, c in links:
        us.append(2 * a + 1)
        vs.append(2 * b)
        cs.append(c)
    m = len(us)
    # forward edge e at 2e, residual reverse at 2e+1
    U = np.empty(2 * m, dtype=np.int64)
    V = np.empty(2 * m, dtype=np.int64)
    C = np.empty(2 * m)
    cap = np.zeros(2 * m, dtype=np.int64)
    U[0::2], V[0::2], C[0::2] = us, vs, cs
    U[1::2], V[1::2], C[1::2] = vs, us, -np.asarray(cs)
    cap[0::2] = 1
    cap[0::2][np.asarray(us) == S] = 1
    n_nodes = 2 * n + 2
    for _ in range(k):
        dist = np.full(n_nodes, np.inf)
        dist[S] = 0.0
        pred = np.full(n_nodes, -1, dtype=np.int64)
        live = np.where(cap > 0)[0]
        for _it in range(n_nodes):
            du = dist[U[live]]
            fin = np.isfinite(du)
            e = live[fin]
            cand = du[fin] + C[e]
            better = cand < dist[V[e]] - 1e-9
            if not better.any():
                break
            e, cand = e[better], cand[better]
            order = np.argsort(-cand, kind="stable")
            dist[V[e[order]]] = cand[order]      # last write = minimum
            pred[V[e[order]]] = e[order]
        if not np.isfinite(dist[T]) or dist[T] >= 0:
            break
        v = T
        while v != S:
            e = pred[v]
            cap[e] -= 1
            cap[e ^ 1] += 1
            v = U[e]
    # decode: follow saturated forward link edges from S
    nxt: dict[int, int] = {}
    starts = []
    for e in range(0, 2 * m, 2):
        if cap[e] != 0:
            continue
        a, b = int(U[e]), int(V[e])
        if a == S:
            starts.append(b // 2)
        elif a % 2 == 1 and b != T and b % 2 == 0 and b < S:
            nxt[a // 2] = b // 2
    chains = []
    for s in starts:
        ch = [s]
        while ch[-1] in nxt:
            ch.append(nxt[ch[-1]])
        chains.append(ch)
    return chains


# ---------- main entry ----------

def _union_s(intervals: list[tuple[float, float]]) -> float:
    tot, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in sorted(intervals):
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                tot += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        tot += cur_hi - cur_lo
    return tot


def link_identities(tracks: list[dict], *, team_size: int = MAX_PER_TEAM,
                    feats: dict[int, np.ndarray] | None = None,
                    pitch_len: float = 100.0,
                    window: tuple[float, float] | None = None,
                    merge: bool = True) -> dict:
    """Core linker (no I/O). tracks: tracks.json entries. feats: track id
    -> L2-normalised appearance vector. Returns the identities doc
    (without crops paths resolved / names)."""
    feats = feats or {}
    units = merge_duplicates(tracks) if merge else [
        dict(t, member_ids=[int(t["id"])]) for t in tracks]
    by_id = {int(t["id"]): t for t in tracks}
    if window is None:
        window = (min((float(t["start"]) for t in tracks), default=0.0),
                  max((float(t["end"]) for t in tracks), default=0.0))
    match_s = max(1e-6, window[1] - window[0])

    def unit_feat(u: dict) -> np.ndarray | None:
        vs = [feats[m] for m in u["member_ids"] if m in feats
              and np.linalg.norm(feats[m]) > 0]
        if not vs:
            return None
        f = np.mean(vs, axis=0)
        nrm = float(np.linalg.norm(f))
        return f / nrm if nrm > 0 else None

    identities: list[dict] = []
    unassigned: list[int] = []
    per_team_q: dict[str, dict] = {}
    for team in ("A", "B"):
        pool = [u for u in units if u.get("team") == team
                and float(u["end"]) - float(u["start"]) >= MIN_TRACK_S]
        pool.sort(key=lambda u: (float(u["start"]), int(u["id"])))
        ends = [dict(_endpoints(u), start=float(u["start"]),
                     end=float(u["end"])) for u in pool]
        roles = [_gk_side(u, pitch_len) for u in pool]
        uf = [unit_feat(u) for u in pool]
        dur = np.array([float(u["end"]) - float(u["start"]) for u in pool])
        n = len(pool)
        links = []
        starts = np.array([e["s0"] if e["p0"] is not None else np.inf
                           for e in ends])
        for i in range(n):
            if ends[i]["p1"] is None:
                continue
            s1 = ends[i]["s1"]
            lo_j = np.searchsorted(starts, s1 - MIN_OVERLAP_S, "left")
            for j in range(max(lo_j - 20, 0), n):
                if j == i or ends[j]["p0"] is None:
                    continue
                if starts[j] > s1 + MAX_GAP_S:
                    break
                if ends[j]["end"] <= ends[i]["end"]:
                    continue
                app = 0.0
                if uf[i] is not None and uf[j] is not None:
                    app = float(np.clip(1.0 - uf[i] @ uf[j], 0.0, 1.0))
                c = link_cost(ends[i], ends[j], app_d=app,
                              role_a=roles[i], role_b=roles[j])
                if c is not None:
                    links.append((i, j, c))
        enter = np.full(n, C_ENTER)
        chains = _min_cost_paths(n, dur, enter, links, team_size) if n else []
        link_cost_of = {(a, b): c for a, b, c in links}
        in_chain = {i for ch in chains for i in ch}
        unassigned += [m for i, u in enumerate(pool) if i not in in_chain
                       for m in u["member_ids"]]
        chain_links = []
        for ch in chains:
            members = [pool[i] for i in ch]
            ident = _aggregate(members, by_id, match_s)
            ident["team"] = team
            ident["role"] = ("gk" if sum(dur[i] for i in ch if roles[i])
                             > 0.5 * sum(dur[i] for i in ch) else "outfield")
            ident["n_links"] = len(ch) - 1
            ident["link_cost_mean"] = round(float(np.mean(
                [link_cost_of[(a, b)] for a, b in itertools.pairwise(ch)]
            )) if len(ch) > 1 else 0.0, 2)
            chain_links += [link_cost_of[(a, b)] for a, b in itertools.pairwise(ch)]
            vs = [uf[i] for i in ch if uf[i] is not None]
            if len(vs) > 1:
                M = np.stack(vs)
                sim = M @ M.T
                iu = np.triu_indices(len(vs), 1)
                ident["cohesion"] = round(float(sim[iu].mean()), 4)
            else:
                ident["cohesion"] = 1.0
            ident["_feat"] = (np.mean(vs, axis=0) if vs else None)
            identities.append(ident)
        team_s = float(dur.sum())
        got_s = float(sum(dur[i] for i in in_chain))
        per_team_q[team] = {
            "n_identities": len(chains),
            "n_units": n, "n_links_considered": len(links),
            "track_s": round(team_s, 1),
            "assigned_track_s": round(got_s, 1),
            "assigned_frac": round(got_s / team_s, 4) if team_s else 0.0,
            "link_cost_mean": round(float(np.mean(chain_links)), 2)
            if chain_links else 0.0,
        }
    # team None and too-short fragments
    placed = {m for i in identities for m in i["track_ids"]} | set(unassigned)
    unassigned += [int(t["id"]) for t in tracks if int(t["id"]) not in placed]

    order = {"A": 0, "B": 1}
    identities.sort(key=lambda i: (order[i["team"]], -i["coverage_s"]))
    counters: dict[str, int] = {}
    for ident in identities:
        counters[ident["team"]] = counters.get(ident["team"], 0) + 1
        ident["id"] = f"{ident['team']}{counters[ident['team']]}"
    quality = _quality(identities, tracks, unassigned, window, team_size,
                       per_team_q)
    return {"identities": identities,
            "unassigned_track_ids": sorted(set(unassigned)),
            "quality": quality}


def _aggregate(members: list[dict], by_id: dict, match_s: float) -> dict:
    """Stats over a chain of (merged) units on one continuous timeline."""
    start = min(float(u["start"]) for u in members)
    end = max(float(u["end"]) for u in members)
    n = round((end - start) / STEP) + 1
    xy: list = [[None, None] for _ in range(n)]
    for u in members:
        k0 = round((float(u["start"]) - start) / STEP)
        for j, p in enumerate(u.get("xy") or []):
            if p is not None and p[0] is not None and 0 <= k0 + j < n:
                xy[k0 + j] = p
    ms = motion_stats(xy)
    events = sorted(ms["sprint_events"], key=lambda e: -e["speed_ms"])
    track_ids = sorted(m for u in members for m in u["member_ids"])
    cov = _union_s([(float(u["start"]), float(u["end"])) for u in members])
    return {
        "track_ids": track_ids,
        "unit_ids": [int(u["id"]) for u in members],
        "first_s": round(start, 3), "last_s": round(end, 3),
        "coverage_s": round(cov, 1),
        "coverage_pct": round(100.0 * cov / match_s, 1),
        "distance_m": round(ms["distance_m"], 1),
        "distance_raw_m": round(sum(float(by_id[m].get("dist_m") or 0.0)
                                    for m in track_ids if m in by_id), 1),
        "sprints": ms["sprints"],
        "top_speed_ms": round(ms["top_speed_ms"], 2),
        "sprint_events": [
            {"t": round(start + (e["step"] + 1) * STEP, 2),
             "speed_ms": e["speed_ms"], "dur_s": e["dur_s"], "xy": e["xy"]}
            for e in events[:20]],
        "name": None,
    }


def _quality(identities, tracks, unassigned, window, team_size,
             per_team) -> dict:
    lo, hi = window
    n = max(1, round((hi - lo) / STEP) + 1)
    by_id = {int(t["id"]): t for t in tracks}
    max_conc = {}
    violations = 0
    for team in ("A", "B"):
        conc = np.zeros(n, dtype=int)
        for ident in (i for i in identities if i["team"] == team):
            active = np.zeros(n, dtype=bool)
            ivs = []
            for u in ident["unit_ids"]:
                t = by_id.get(u)
                if t is None:
                    continue
                ivs.append((float(t["start"]), float(t["end"])))
            for tid in ident["track_ids"]:
                t = by_id[tid]
                a = max(0, round((float(t["start"]) - lo) / STEP))
                b = min(n, round((float(t["end"]) - lo) / STEP) + 1)
                active[a:b] = True
            conc += active
            ivs.sort()
            violations += sum(1 for (a0, a1), (b0, _b1) in itertools.pairwise(ivs)
                              if a1 - b0 >= MIN_OVERLAP_S)
        max_conc[team] = int(conc.max()) if n else 0
    un_s = sum(float(by_id[t]["end"]) - float(by_id[t]["start"])
               for t in unassigned if t in by_id)
    all_s = sum(float(t["end"]) - float(t["start"]) for t in tracks)
    return {
        "team_size": team_size,
        "match_s": round(hi - lo, 1),
        "n_tracks": len(tracks),
        "n_identities": len(identities),
        "n_unassigned": len(set(unassigned)),
        "unassigned_s": round(un_s, 1),
        "unassigned_frac_s": round(un_s / all_s, 4) if all_s else 0.0,
        "max_concurrent": max_conc,
        "overlap_violations": violations,
        "mean_coverage_pct": round(float(np.mean(
            [i["coverage_pct"] for i in identities])), 1)
        if identities else 0.0,
        "per_team": per_team,
    }


def _pick_crops(ident: dict, units_by_id: dict, feats: dict,
                crops_dir: Path) -> list[str]:
    """Up to N_CROPS crops from the longest member units, preferring the
    crops whose fingerprint is closest to the identity's mean."""
    mean = ident.get("_feat")
    cand = []
    for uid in ident["unit_ids"]:
        u = units_by_id.get(uid)
        if u is None:
            continue
        for c in u.get("crops") or []:
            cand.append((float(u["end"]) - float(u["start"]), c))
    cand.sort(key=lambda x: -x[0])
    cand = cand[:60]
    scored = []
    for dur, c in cand:
        path = crops_dir / c
        img = cv2.imread(str(path)) if path.is_file() else None
        if img is None or img.shape[0] < 24:
            continue
        f = _fingerprint_one(img)
        nrm = float(np.linalg.norm(f))
        sim = float((f / nrm) @ mean) if mean is not None and nrm else 0.0
        scored.append((sim + 0.001 * min(dur, 300.0) / 300.0, c))
    scored.sort(key=lambda x: -x[0])
    out: list[str] = []
    for _s, c in scored:
        stem = c.rsplit("_", 1)[0]
        if sum(1 for o in out if o.rsplit("_", 1)[0] == stem) >= 2:
            continue
        out.append(c)
        if len(out) >= N_CROPS:
            break
    return out


def build_identities(v2_dir: Path, *, team_size: int = MAX_PER_TEAM,
                     pitch_len: float | None = None,
                     window: tuple[float, float] | None = None,
                     log=print) -> dict:
    """tracks.json (+ crops/) -> identities.json; re-attaches names from
    names.json. Offline: needs only the saved tracks + crops."""
    v2_dir = Path(v2_dir)
    doc = json.loads((v2_dir / "tracks.json").read_text())
    tracks = doc.get("tracks") or []
    crops_dir = v2_dir / "crops"
    if pitch_len is None:
        pitch_len = _pitch_len_guess(v2_dir, tracks)
    if window is None:
        hist = (doc.get("summary") or {}).get("visible_hist") or []
        t0 = float(doc.get("t0") or 0.0)
        if hist:
            window = (t0, t0 + STEP * (len(hist) - 1))
    feats = {}
    for t in tracks:
        f = tracklet_fingerprint([crops_dir / c for c in t.get("crops") or []])
        if f.shape == (FEAT_DIM,) and np.linalg.norm(f) > 0:
            feats[int(t["id"])] = f
    out = link_identities(tracks, team_size=team_size, feats=feats,
                          pitch_len=pitch_len, window=window)
    units_by_id = {int(u["id"]): u for u in merge_duplicates(tracks)}
    for ident in out["identities"]:
        ident["crops"] = _pick_crops(ident, units_by_id, feats, crops_dir)
        ident.pop("_feat", None)
    names_path = v2_dir / "names.json"
    names = _load(names_path)
    if names:
        durs = {int(t["id"]): float(t["end"]) - float(t["start"])
                for t in tracks}
        ndoc = names_doc_for(out["identities"], names, durs)
        for ident in out["identities"]:
            ident["name"] = (ndoc["names"].get(ident["id"]) or {}
                             ).get("name")
        write_json_atomic(names_path, ndoc, indent=1)
    out["generated_at"] = time.time()
    out["pitch_len_m"] = pitch_len
    out["window"] = list(window) if window else None
    write_json_atomic(v2_dir / "identities.json", out, indent=1)
    q = out["quality"]
    log(f"identities: {q['n_identities']} "
        f"(A {q['per_team'].get('A', {}).get('n_identities', 0)}, "
        f"B {q['per_team'].get('B', {}).get('n_identities', 0)}), "
        f"{q['n_unassigned']} tracks unassigned, "
        f"mean coverage {q['mean_coverage_pct']}%")
    return out


def _load(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _pitch_len_guess(v2_dir: Path, tracks: list[dict]) -> float:
    """calib.json pitch length when v2_dir sits in a project, else the
    track extent."""
    cal = _load(v2_dir.parent.parent / "multiangle" / "calib.json") or {}
    L = (cal.get("pitch") or {}).get("len_m")
    if L:
        return float(L)
    xs = [p[0] for t in tracks for p in t.get("xy") or []
          if p and p[0] is not None]
    return float(max(xs)) if xs else 100.0


# ---------- names ----------

def _track_seconds(ident: dict, durs: dict[int, float] | None) -> dict:
    return {int(t): (durs or {}).get(int(t), 1.0) for t in ident["track_ids"]}


def reattach_names(identities: list[dict], names: dict,
                   durs: dict[int, float] | None = None) -> dict[str, str]:
    """names.json {"names": {iid: {"name", "track_ids"}}} -> {new iid:
    name}: each saved name goes to the identity sharing the most track
    ids with it (weighted by durs when given); one name per identity,
    resolved by a best-overlap-first greedy."""
    saved = (names or {}).get("names") or {}
    pairs = []
    for key, ent in saved.items():
        nm = (ent or {}).get("name")
        if not nm:
            continue
        tids = {int(t) for t in ent.get("track_ids") or []}
        for ident in identities:
            w = _track_seconds(ident, durs)
            ov = sum(w[t] for t in tids if t in w)
            if ov > 0:
                pairs.append((ov, key, ident["id"], nm))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    used_keys, used_ids = set(), set()
    out = {}
    for _ov, key, iid, nm in pairs:
        if key in used_keys or iid in used_ids:
            continue
        used_keys.add(key)
        used_ids.add(iid)
        out[iid] = nm
    return out


def names_doc_for(identities: list[dict], names: dict | None,
                  durs: dict[int, float] | None = None) -> dict:
    """Rewrite names.json against the current identity ids. A name whose
    tracks no longer overlap any identity is kept under a "~"-prefixed
    key so a later re-run can still re-attach it."""
    saved = (names or {}).get("names") or {}
    mapping = reattach_names(identities, names or {}, durs)
    by_id = {i["id"]: i for i in identities}
    out: dict[str, dict] = {}
    for iid, nm in mapping.items():
        out[iid] = {"name": nm, "track_ids": by_id[iid]["track_ids"]}
    placed = set(mapping.values())
    for key, ent in saved.items():
        nm = (ent or {}).get("name")
        if nm and nm not in placed:
            k = key if key.startswith("~") else f"~{key}"
            while k in out:
                k += "~"
            out[k] = ent
    return {"names": out, "updated_at": time.time()}


def set_name(v2_dir: Path, iid: str, name: str | None) -> dict:
    """Persist one identity's name (None/"" clears it); updates both
    names.json and identities.json. Returns the updated identity."""
    v2_dir = Path(v2_dir)
    doc = _load(v2_dir / "identities.json")
    if not doc:
        raise FileNotFoundError("identities.json")
    ident = next((i for i in doc.get("identities") or []
                  if i["id"] == iid), None)
    if ident is None:
        raise KeyError(iid)
    name = (name or "").strip() or None
    names = _load(v2_dir / "names.json") or {"names": {}}
    entries = dict(names.get("names") or {})
    tids = set(ident["track_ids"])
    # an orphaned ("~") name whose tracks overlap this identity is
    # superseded by the explicit edit
    for key in list(entries):
        ent_t = set((entries[key] or {}).get("track_ids") or [])
        if key == iid or (key.startswith("~") and ent_t & tids):
            entries.pop(key)
    if name:
        entries[iid] = {"name": name, "track_ids": ident["track_ids"]}
    write_json_atomic(v2_dir / "names.json",
                      {"names": entries, "updated_at": time.time()},
                      indent=1)
    ident["name"] = name
    write_json_atomic(v2_dir / "identities.json", doc, indent=1)
    return ident


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="highlights.analysis.identity")
    ap.add_argument("players_v2_dir", type=Path)
    ap.add_argument("--team-size", type=int, default=MAX_PER_TEAM)
    ap.add_argument("--pitch-len", type=float, default=None)
    args = ap.parse_args(argv)
    if not (args.players_v2_dir / "tracks.json").is_file():
        print(f"no tracks.json in {args.players_v2_dir}", file=sys.stderr)
        return 1
    t = time.time()
    build_identities(args.players_v2_dir, team_size=args.team_size,
                     pitch_len=args.pitch_len)
    print(f"done in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
