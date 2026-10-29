#!/usr/bin/env python3
"""Contact sheets of a clip's shots for camera tagging: the middle frame of every shot, captioned
with shot index, frame range and duration.

    uv run scripts/shot_sheets.py data/labels/<clip>.json [--per-sheet 24] [--min-s 0]

Writes runs/<clip-stem>/review/shots_NN.jpg and prints their paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bga.review.sheets import contact_sheet  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--min-s", type=float, default=0.0, help="skip shots shorter than this")
    a = ap.parse_args(argv)
    d = json.loads(a.labels.read_text())
    fps = d["fps"]
    cap = cv2.VideoCapture(d["clip"])
    items = []
    for i, s in enumerate(d["shots"]):
        dur = (s["end"] - s["start"] + 1) / fps
        if dur < a.min_s:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, (s["start"] + s["end"]) // 2)
        ok, img = cap.read()
        if ok:
            items.append((img, f"s{i} {s['start']}-{s['end']} {dur:.1f}s"))
    cap.release()
    out = Path("runs") / a.labels.stem / "review"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(items), a.per_sheet):
        p = out / f"shots_{k // a.per_sheet:02d}.jpg"
        cv2.imwrite(
            str(p), contact_sheet(items[k : k + a.per_sheet], cols=a.cols), [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        print(p)


if __name__ == "__main__":
    main()
