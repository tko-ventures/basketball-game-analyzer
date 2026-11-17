#!/usr/bin/env python3
"""Write the FIBA court geometry the labeller's court page draws: tools/labeler/static/fiba_court.json.

    uv run --extra demo scripts/court_geometry.py

Vertices (33 landmarks, centimetres, Skalski's order) and edges come from
``sports.basketball.CourtConfiguration(League.FIBA)``. The polylines add what the edges do not
cover: both three-point arcs, the free-throw circles and the centre circle, sampled every 5 degrees.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from sports.basketball import CourtConfiguration, League

OUT = Path(__file__).resolve().parents[1] / "tools/labeler/static/fiba_court.json"


def arc(cx, cy, r, a0, a1, step=5.0):
    n = max(2, int(abs(a1 - a0) / step) + 1)
    return [
        [
            round(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / (n - 1))), 1),
            round(cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / (n - 1))), 1),
        ]
        for i in range(n)
    ]


def main():
    c = CourtConfiguration(league=League.FIBA)
    v = [list(p) for p in c.vertices]
    L, W = c.court_length, c.court_width
    bx = c.vertices[c.left_basket_index][0]
    r3 = c.three_point_arc_radius
    side = c.sideline_to_three_point_line
    half = math.degrees(math.asin((W / 2 - side) / r3))  # arc meets the straight part
    ft = c.vertices[10][0]
    poly = [
        arc(bx, W / 2, r3, -half, half),
        arc(L - bx, W / 2, r3, 180 - half, 180 + half),
        arc(ft, W / 2, c.center_circle_radius, -90, 90),
        arc(L - ft, W / 2, c.center_circle_radius, 90, 270),
        arc(L / 2, W / 2, c.center_circle_radius, 0, 360),
    ]
    doc = {
        "units": "cm",
        "length": L,
        "width": W,
        "vertices": v,
        "labels": list(c.labels),
        "edges": [list(e) for e in c.edges],
        "polylines": poly,
    }
    OUT.write_text(json.dumps(doc) + "\n")
    print(f"{len(v)} vertices, {len(c.edges)} edges, {len(poly)} polylines -> {OUT}")


if __name__ == "__main__":
    main()
