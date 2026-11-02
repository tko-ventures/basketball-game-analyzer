#!/usr/bin/env python3
"""Visual check of the court mapping: FIBA lines drawn back into broadcast frames through the inverse
of the cached image-to-court homography (scripts/demo_web.py cache). Lines on the painted lines mean
the mapping is right; offset, mirrored or tilted lines show where it is not.

    uv run --extra demo scripts/qa_court_overlay.py OSp445e0mXU 10 --n 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bga.review.sheets import contact_sheet  # noqa: E402
from scripts.demo_video import FrameReader, clip_files  # noqa: E402


def main(argv=None):
    from sports.basketball import CourtConfiguration, League, draw_court

    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id")
    ap.add_argument("shot", type=int)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=Path("runs/demo/cache"))
    a = ap.parse_args(argv)
    labels = json.loads(clip_files(a.clip_id)[0].read_text())
    c = json.loads((a.cache / f"{a.clip_id}_s{a.shot}_step2_web.json").read_text())
    H = {int(f): np.array(h) for f, h in c["H"].items() if h}
    scale, pad = 0.2, 0
    lines = draw_court(
        CourtConfiguration(league=League.FIBA),
        scale=scale,
        padding=pad,
        line_thickness=2,
        background_color=__import__("supervision").Color.BLACK,
    )
    court_px_to_cm = np.array([[1 / scale, 0, -pad / scale], [0, 1 / scale, -pad / scale], [0, 0, 1]])
    frames = sorted(H)
    picks = [frames[round(i * (len(frames) - 1) / (a.n - 1))] for i in range(a.n)]
    read, items = FrameReader(labels["clip"]), []
    for f in picks:
        img = read(f)
        warp = np.linalg.inv(H[f]) @ court_px_to_cm
        proj = cv2.warpPerspective(lines, warp, (img.shape[1], img.shape[0]))
        mask = proj.max(axis=2) > 128
        img[mask] = (255, 255, 0)
        items.append((img, f"{a.clip_id} f{f}"))
    out = Path("runs/demo") / f"qa_court_{a.clip_id}_s{a.shot}.jpg"
    cv2.imwrite(str(out), contact_sheet(items, cols=2, tile_wh=(800, 450)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
