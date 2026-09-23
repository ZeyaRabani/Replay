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
    goal: str              # "A" | "B"
    t_event: float
    start: float
    end: float
    signals: dict[str, float] = field(default_factory=dict)
    clip: str | None = None

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


def combine(signals: dict[str, np.ndarray], bin_s: float, cfg: Config, duration_s: float,
            t_offset: float = 0.0) -> tuple[list[Candidate], list[list[float]]]:
    n_bins = max((len(v) for v in signals.values()), default=0)
    n_players = signals.get("n_players", np.full(n_bins, float(cfg.min_players_active)))
    gate = play_gate(n_players, bin_s, cfg.min_players_active)
    windows = active_windows(gate, bin_s, t_offset)
    cands: list[Candidate] = []
    for g in ("A", "B"):
        if f"ball_attack_{g}" not in signals:
            continue
        attack = np.maximum(signals.get(f"ball_attack_{g}", 0.0),
                            0.7 * signals.get(f"attack_{g}", 0.0))
        audio = signals.get("audio", np.zeros_like(attack))
        cluster = signals.get("cluster", np.zeros_like(attack))
        cluster_g = signals.get(f"cluster_{g}", np.zeros_like(attack))
        restart = signals.get("restart", np.zeros_like(attack))
        lost = signals.get(f"ball_lost_{g}", np.zeros_like(attack))

        # anchors: local maxima above threshold
        for b in range(min(len(attack), len(gate))):
            if attack[b] <= cfg.attack_anchor or not gate[b]:
                continue
            if n_players[b] < 4:
                continue
            # isolated single-bin spikes are tracker artifacts; require a neighbour > anchor
            if not ((b > 0 and attack[b - 1] > cfg.attack_anchor)
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
            # extend end to cover audio/cluster peak within the window
            tail = np.zeros_like(audio)
            tail[b:b + w2] = np.maximum(audio[b:b + w2], cluster[b:b + w2])
            peaks = np.where(tail > 0.3)[0]
            end_extra = max(0.0, (peaks.max() - b) * bin_s + 3.0) if len(peaks) else 0.0
            end = min(t_event + cfg.post_roll_s + end_extra, t_event + cfg.post_roll_s + 25.0, duration_s)
            cands.append(Candidate("", 0, typ, round(conf, 3), g, round(t_event, 2),
                                   round(max(0.0, t_event - cfg.pre_roll_s), 2), round(end, 2),
                                   {"attack": round(float(attack[b]), 3), "audio": round(a, 3),
                                    "cluster": round(cluster_score, 3), "restart": round(re, 3),
                                    "ball_lost": round(lo_v, 3)}))
    # merge closer than merge_gap_s (keep max confidence)
    cands.sort(key=lambda c: -c.confidence)
    merged: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.t_event):
        if merged and c.t_event - merged[-1].t_event < cfg.merge_gap_s:
            prev = merged[-1]
            keep = c if c.confidence > prev.confidence else prev
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
