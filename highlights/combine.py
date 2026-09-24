"""Align per-bin signals and fuse into ranked highlight candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Config


@dataclass
class Candidate:
    id: str
    rank: int
    type: str              # "goal" | "chance"
    confidence: float
    goal: str | None       # "A" | "B" | None (unknown)
    t_event: float
    start: float
    end: float
    signals: dict[str, float] = field(default_factory=dict)
    clip: str | None = None
    anchor: str = "attack"   # "attack" | "audio"

    def to_dict(self) -> dict:
        return asdict(self)


def _win(sig: np.ndarray, b0: int, b1: int) -> float:
    b0, b1 = max(0, b0), min(len(sig), b1)
    return float(sig[b0:b1].max()) if b1 > b0 else 0.0


def play_gate(n_players: np.ndarray, bin_s: float, min_players: float) -> np.ndarray:
    """Per-bin bool: rolling 60 s median of n_players >= min_players (real play, not warm-up)."""
    w = max(1, int(60.0 / bin_s))
    med = np.array([np.median(n_players[max(0, i - w // 2):i + w // 2 + 1])
                    for i in range(len(n_players))])
    return med >= min_players


def active_windows(gate: np.ndarray, bin_s: float, t_offset: float) -> list[list[float]]:
    """Merge active gate bins into [[t0, t1], ...] absolute-second windows."""
    out = []
    start = None
    for i, on in enumerate(np.append(gate, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append([round(start * bin_s + t_offset, 1), round(i * bin_s + t_offset, 1)])
            start = None
    return out


def _audio_episodes(audio: np.ndarray, gate: np.ndarray, bin_s: float,
                    merge_gap_s: float) -> list[tuple[int, int]]:
    """Episodes of audio>0.5 inside the play gate, gaps <= merge_gap_s merged -> (b0, b1) inclusive."""
    eps = []
    hi = np.where(audio > 0.5)[0]
    max_gap = int(merge_gap_s / bin_s)
    for b in hi:
        if not gate[b]:
            continue
        if eps and b - eps[-1][1] - 1 <= max_gap:
            eps[-1] = (eps[-1][0], b)
        else:
            eps.append((b, b))
    return eps


def combine(signals: dict[str, np.ndarray], bin_s: float, cfg: Config, duration_s: float,
            t_offset: float = 0.0) -> tuple[list[Candidate], list[list[float]]]:
    n_bins = max((len(v) for v in signals.values()), default=0)
    n_players = signals.get("n_players", np.full(n_bins, float(cfg.min_players_active)))
    gate = play_gate(n_players, bin_s, cfg.min_players_active)
    windows = active_windows(gate, bin_s, t_offset)
    audio = signals.get("audio", np.zeros(n_bins))
    cluster = signals.get("cluster", np.zeros(n_bins))
    restart = signals.get("restart", np.zeros(n_bins))
    cands: list[Candidate] = []

    # audio anchors: one per loud episode
    for b0, b1 in _audio_episodes(audio, gate, bin_s, 10.0):
        dur = (b1 - b0 + 1) * bin_s
        audio_score = min(1.0, 0.5 + dur / 8.0)
        att_win = (int(b0 - 10.0 / bin_s), int(b0 + 20.0 / bin_s))
        sup_win = (int(b0 - 5.0 / bin_s), int(b0 + 25.0 / bin_s))
        atk, cl_g, lo_v = {}, {}, {}
        for g in ("A", "B"):
            attack_g = np.maximum(signals.get(f"ball_attack_{g}", 0.0),
                                  0.7 * signals.get(f"attack_{g}", 0.0))
            atk[g] = _win(attack_g, *att_win)
            cl_g[g] = max(_win(signals.get(f"cluster_{g}", np.zeros(n_bins)), *sup_win),
                          0.6 * _win(cluster, *sup_win))
            lo_v[g] = _win(signals.get(f"ball_lost_{g}", np.zeros(n_bins)), *sup_win)
        g = max(atk, key=atk.get) if atk["A"] != atk["B"] else ("A" if atk["A"] > 0 else None)
        attack_v = max(atk.values())
        cl = cl_g[g] if g else max(cl_g.values())
        lo = lo_v[g] if g else max(lo_v.values())
        re = _win(restart, *sup_win)
        conf = float(np.clip(cfg.w_audio * audio_score + cfg.w_attack * attack_v
                             + cfg.w_cluster * cl + cfg.w_lost * lo + cfg.w_restart * re, 0, 1))
        typ = "goal" if dur >= 4.0 and (cl >= 0.5 or lo >= 0.5 or attack_v >= 0.3) else "chance"
        t_event = b0 * bin_s - cfg.audio_lead_s + t_offset  # crowd reacts after the event
        roll = cfg.roll_goal_s if typ == "goal" else cfg.roll_chance_s
        end = min(t_event + roll, duration_s)
        cands.append(Candidate("", 0, typ, round(conf, 3), g, round(t_event, 2),
                               round(max(0.0, t_event - roll), 2), round(end, 2),
                               {"attack": round(attack_v, 3), "audio": round(audio_score, 3),
                                "cluster": round(cl, 3), "restart": round(re, 3),
                                "ball_lost": round(lo, 3), "audio_dur_s": round(dur, 1)},
                               anchor="audio"))

    for g in ("A", "B"):
        if f"ball_attack_{g}" not in signals:
            continue
        attack = np.maximum(signals.get(f"ball_attack_{g}", 0.0),
                            0.7 * signals.get(f"attack_{g}", 0.0))
        cluster_g = signals.get(f"cluster_{g}", np.zeros_like(attack))
        lost = signals.get(f"ball_lost_{g}", np.zeros_like(attack))

        # anchors: local maxima above threshold
        for b in range(min(len(attack), len(gate))):
            if attack[b] <= cfg.attack_anchor or not gate[b]:
                continue
            if n_players[b] < 4:
                continue
            # isolated single-bin spikes are tracker artifacts; require a neighbour > anchor,
            # relaxed when a loud audio bin sits within +-5 s
            aw = int(5.0 / bin_s)
            audio_near = _win(audio, b - aw, b + aw) > 0.5
            if not audio_near and not ((b > 0 and attack[b - 1] > cfg.attack_anchor)
                                       or (b + 1 < len(attack) and attack[b + 1] > cfg.attack_anchor)):
                continue
            lo, hi = max(0, b - 1), min(len(attack), b + 2)
            if attack[b] < attack[lo:hi].max() or (b > 0 and attack[b] == attack[b - 1]):
                pass
            if attack[b] != attack[lo:hi].max():
                continue
            if b > 0 and attack[b - 1] == attack[b]:
                continue  # keep left edge of a plateau
            w2 = int(20.0 / bin_s)
            a = _win(audio, b, b + w2)
            cl = _win(cluster, b, b + w2)
            cl_near = _win(cluster_g, b, b + w2)
            r_lo, r_hi = int(5.0 / bin_s), int(90.0 / bin_s)
            re = _win(restart, b + r_lo, b + r_hi)
            lo_w = int(1.0 / bin_s)
            lo_v = _win(lost, b - lo_w, b + int(3.0 / bin_s))
            cluster_score = max(cl_near, 0.6 * cl)
            conf = float(np.clip(cfg.w_attack * attack[b] + cfg.w_audio * a + cfg.w_cluster * cluster_score
                                 + cfg.w_restart * re + cfg.w_lost * lo_v, 0, 1))
            typ = "goal" if (cl > 0 or re > 0.5 or lo_v > 0) and a > cfg.audio_goal_min else "chance"
            t_event = b * bin_s + t_offset
            roll = cfg.roll_goal_s if typ == "goal" else cfg.roll_chance_s
            end = min(t_event + roll, duration_s)
            cands.append(Candidate("", 0, typ, round(conf, 3), g, round(t_event, 2),
                                   round(max(0.0, t_event - roll), 2), round(end, 2),
                                   {"attack": round(float(attack[b]), 3), "audio": round(a, 3),
                                    "cluster": round(cluster_score, 3), "restart": round(re, 3),
                                    "ball_lost": round(lo_v, 3)}))
    # merge closer than merge_gap_s (keep max confidence, union signals)
    cands.sort(key=lambda c: -c.confidence)
    merged: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.t_event):
        if merged and c.t_event - merged[-1].t_event < cfg.merge_gap_s:
            prev = merged[-1]
            keep = c if c.confidence > prev.confidence else prev
            other = prev if keep is c else c
            keep.signals = {k: max(keep.signals.get(k, 0.0), other.signals.get(k, 0.0))
                            for k in keep.signals.keys() | other.signals.keys()}
            keep.start = min(prev.start, c.start)
            keep.end = max(prev.end, c.end)
            keep.t_event = min(prev.t_event, c.t_event)
            merged[-1] = keep
        else:
            merged.append(c)
    merged.sort(key=lambda c: -c.confidence)
    for rank, c in enumerate(merged, start=1):
        c.rank = rank
        c.id = f"c{rank:02d}"
    return merged, windows
