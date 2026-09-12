"""Parametric pitch model + named landmarks.

World frame (metres): x runs along the pitch from goal A (x=0) to goal B
(x=length); y runs across the pitch from the "S" touchline (y=0) to the "N"
touchline (y=width). Goals are centred on y = width/2.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

Point = tuple[float, float]


@dataclass
class PitchModel:
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32
    # Semicircular "D" in front of each goal (small-sided pitches). 0 disables.
    d_radius: float = 0.0
    # Standard 11-a-side box markings. 0 disables.
    penalty_depth: float = 16.5
    penalty_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    centre_circle_radius: float = 9.15

    @classmethod
    def standard(cls) -> PitchModel:
        return cls()

    @classmethod
    def small_sided(cls, length: float = 40.0, width: float = 30.0, goal_width: float = 3.66,
                    d_radius: float = 6.0) -> PitchModel:
        return cls(length=length, width=width, goal_width=goal_width, d_radius=d_radius,
                   penalty_depth=0, penalty_width=0, goal_area_depth=0, goal_area_width=0,
                   centre_circle_radius=0)

    @classmethod
    def load(cls, path: Path) -> PitchModel:
        data = json.loads(path.read_text())
        preset = data.pop("preset", None)
        base = cls.small_sided() if preset == "small_sided" else cls()
        return cls(**{**asdict(base), **data})

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- landmarks -----------------------------------------------------
    def landmarks(self) -> dict[str, Point]:
        L, W, cy, g = self.length, self.width, self.width / 2, self.goal_width / 2
        pts: dict[str, Point] = {
            "corner_A_S": (0, 0), "corner_A_N": (0, W), "corner_B_S": (L, 0), "corner_B_N": (L, W),
            "half_S": (L / 2, 0), "half_N": (L / 2, W), "centre": (L / 2, cy),
            "A_post_S": (0, cy - g), "A_post_N": (0, cy + g),
            "B_post_S": (L, cy - g), "B_post_N": (L, cy + g),
        }
        if self.d_radius > 0:
            r = self.d_radius
            pts.update({
                "A_D_S": (0, cy - r), "A_D_N": (0, cy + r), "A_D_apex": (r, cy),
                "A_D_45S": (r * math.cos(math.pi / 4), cy - r * math.sin(math.pi / 4)),
                "A_D_45N": (r * math.cos(math.pi / 4), cy + r * math.sin(math.pi / 4)),
                "B_D_S": (L, cy - r), "B_D_N": (L, cy + r), "B_D_apex": (L - r, cy),
                "B_D_45S": (L - r * math.cos(math.pi / 4), cy - r * math.sin(math.pi / 4)),
                "B_D_45N": (L - r * math.cos(math.pi / 4), cy + r * math.sin(math.pi / 4)),
            })
        if self.penalty_depth > 0:
            pd, pw = self.penalty_depth, self.penalty_width / 2
            pts.update({
                "A_box_S_line": (0, cy - pw), "A_box_N_line": (0, cy + pw),
                "A_box_S": (pd, cy - pw), "A_box_N": (pd, cy + pw),
                "B_box_S_line": (L, cy - pw), "B_box_N_line": (L, cy + pw),
                "B_box_S": (L - pd, cy - pw), "B_box_N": (L - pd, cy + pw),
            })
        if self.goal_area_depth > 0:
            gd, gw = self.goal_area_depth, self.goal_area_width / 2
            pts.update({
                "A_6yd_S_line": (0, cy - gw), "A_6yd_N_line": (0, cy + gw),
                "A_6yd_S": (gd, cy - gw), "A_6yd_N": (gd, cy + gw),
                "B_6yd_S_line": (L, cy - gw), "B_6yd_N_line": (L, cy + gw),
                "B_6yd_S": (L - gd, cy - gw), "B_6yd_N": (L - gd, cy + gw),
            })
        return pts

    def resolve(self, world) -> Point:
        """Accept a landmark name or an explicit [x, y] pair."""
        if isinstance(world, str):
            try:
                return self.landmarks()[world]
            except KeyError:
                raise KeyError(f"unknown landmark {world!r}; known: {sorted(self.landmarks())}") from None
        x, y = world
        return float(x), float(y)

    def contains(self, x: float, y: float, margin: float = 2.0) -> bool:
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin

    # ---- drawing helpers (for debug visualisation) ---------------------
    def segments(self) -> list[tuple[Point, Point]]:
        L, W, cy = self.length, self.width, self.width / 2
        segs = [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L / 2, 0), (L / 2, W))]
        for depth, half in ((self.penalty_depth, self.penalty_width / 2), (self.goal_area_depth, self.goal_area_width / 2)):
            if depth > 0:
                segs += [((0, cy - half), (depth, cy - half)), ((depth, cy - half), (depth, cy + half)),
                         ((depth, cy + half), (0, cy + half)),
                         ((L, cy - half), (L - depth, cy - half)), ((L - depth, cy - half), (L - depth, cy + half)),
                         ((L - depth, cy + half), (L, cy + half))]
        return segs

    def arcs(self) -> list[tuple[Point, float, float, float]]:
        """(centre, radius, start_deg, end_deg)"""
        L, cy = self.length, self.width / 2
        out = []
        if self.d_radius > 0:
            out += [((0, cy), self.d_radius, -90, 90), ((L, cy), self.d_radius, 90, 270)]
        if self.centre_circle_radius > 0:
            out.append(((L / 2, cy), self.centre_circle_radius, 0, 360))
        return out
