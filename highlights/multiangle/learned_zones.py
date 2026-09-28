"""Learned zones from you-direct sessions.

For each camera we learn, from the seconds the user chose camera k,
WHERE in camera i's picture the ball / player feet were — and turn
"cells where the user preferred camera i" into v2 zones.json polygons.
"""
from __future__ import annotations

import math

import numpy as np

GRID_W, GRID_H = 8, 5          # cells over the normalised frame [0,1]x[0,1]
MIN_VOTES = 2                  # seconds of evidence a cell needs
MIN_SHARE = 0.60               # share of votes for camera i in camera i's cell
BALL_CONF = 0.2                # ZONE_BALL_OK
DENS_W = 0.5                   # weight of a density vote relative to a ball vote


def _cell(fx: float, fy: float) -> tuple[int, int]:
    return (min(GRID_H - 1, max(0, math.floor(fy * GRID_H))),
            min(GRID_W - 1, max(0, math.floor(fx * GRID_W))))


def _rects(cells: set[tuple[int, int]]) -> list[list[list[float]]]:
    """Merge selected cells into rectangles: horizontal runs per row,
    then vertically adjacent rects with identical x-ranges."""
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    rects = []  # (c0, c1, r0, r1)
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        start = prev = cols[0]
        for c in cols[1:] + [None]:
            if c is None or c != prev + 1:
                rects.append([start, prev, r, r])
                if c is not None:
                    start = c
            if c is not None:
                prev = c
    # merge vertically adjacent identical x-ranges
    merged: list[list[int]] = []
    for rc in sorted(rects, key=lambda x: (x[0], x[2])):
        if (merged and merged[-1][0] == rc[0] and merged[-1][1] == rc[1]
                and merged[-1][3] == rc[2] - 1):
            merged[-1][3] = rc[3]
        else:
            merged.append(list(rc))
    polys = []
    for c0, c1, r0, r1 in merged:
        x0, x1 = c0 / GRID_W, (c1 + 1) / GRID_W
        y0, y1 = r0 / GRID_H, (r1 + 1) / GRID_H
        polys.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return polys


def learn_zones(sessions: list[dict], tracks: list[dict], avail: np.ndarray,
                range_lo: float, offsets: list[float],
                durations: list[float]) -> dict | None:
    """Returns zones.json v2 {"version": 2, "learned": True,
    "angles": [[{"t": file_t, "zones": [poly..]}, ...] per angle],
    "cells": [[GRID_H x GRID_W share-or-None] per angle], "n_votes": [...]}
    or None when no camera gets any cell."""
    n_angles, T = avail.shape
    # votes[i][(r,c)] = {user_angle: weight}
    votes: list[dict[tuple[int, int], dict[int, float]]] = [
        {} for _ in range(n_angles)]

    for s in sessions:
        choices = {int(c["t"]): int(c["angle"]) for c in s["choices"]}
        last = None
        for t in range(int(s["t_start"]), int(s["t_end"])):
            if t in choices:
                last = choices[t]
            u = last
            if u is None:
                continue
            idx = int(t - range_lo)
            if not (0 <= idx < T):
                continue
            for i in range(n_angles):
                if not avail[i, idx]:
                    continue
                tr = tracks[i]
                bc = float(np.asarray(tr["ball_conf"])[idx])
                bx = float(np.asarray(tr.get("ball_x", 0))[idx])
                by = float(np.asarray(tr.get("ball_y", 0))[idx])
                if bc >= BALL_CONF and bx > 0 and by > 0:
                    cell = _cell(bx, by)
                    votes[i].setdefault(cell, {})
                    votes[i][cell][u] = votes[i][cell].get(u, 0.0) + 1.0
                feet = None
                if "players_xy" in tr:
                    feet = tr["players_xy"][idx]
                if feet:
                    feet = [f for f in feet if f]
                    if feet:
                        w = DENS_W / len(feet)
                        for f in feet:
                            cell = _cell(float(f[0]), float(f[1]))
                            votes[i].setdefault(cell, {})
                            votes[i][cell][u] = (
                                votes[i][cell].get(u, 0.0) + w)

    sel: list[set[tuple[int, int]]] = []
    n_votes: list[int] = []
    cells_out: list[list] = []
    for i in range(n_angles):
        mine = set()
        grid = [[None] * GRID_W for _ in range(GRID_H)]
        tot_i = 0
        for cell, tally in votes[i].items():
            tot = sum(tally.values())
            tot_i += 1
            share = tally.get(i, 0.0) / tot if tot else 0.0
            grid[cell[0]][cell[1]] = round(share, 3)
            if tot >= MIN_VOTES and share >= MIN_SHARE:
                mine.add(cell)
        sel.append(mine)
        n_votes.append(tot_i)
        cells_out.append(grid)

    if not any(sel):
        return None

    angles = []
    for i in range(n_angles):
        polys = _rects(sel[i])
        kfs = [{"t": float(min(max(s["t_start"] - offsets[i], 0.0),
                               durations[i])),
                "zones": polys} for s in sessions]
        kfs.sort(key=lambda k: k["t"])
        angles.append(kfs)

    return {"version": 2, "learned": True, "angles": angles,
            "cells": cells_out, "n_votes": n_votes}
