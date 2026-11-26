#!/usr/bin/env python3
"""Review sheets for the round-4 hard frames: the implementer's first pass on people, ball and holder.

    uv run scripts/r4_sheets.py [--round 4] [--per-sheet 4]

Reads runs/autolabel/r<N>/draft.json and writes runs/autolabel/r<N>/sheets/sheet_NN.jpg. Each frame is one tile:

  - top: a 960x540 window of the frame at full resolution, centred on the draft's ball (else the most crowded rim,
    else the frame centre). Every kept people box is drawn with its index in the draft (white player, magenta
    player-in-possession, red shot classes, yellow referee); the kept ball is ``B``, the three best weaker ball
    candidates ``c0``..``c2`` (thin, cyan); rims green.
  - with ``--proposals``: only the frames the first pass found people missing in, listed in
    runs/autolabel/r<N>/reads/*.tsv; kept people are thin grey without labels and the draft's weaker people boxes
    (score >= 0.15, overlapping no kept person or each other at IoU 0.3) are green ``p0``, ``p1``..; no strip.
    Written to runs/autolabel/r<N>/proposals/sheet_NN.jpg.
  - bottom: a strip of five 300x300 crops at -0.2, -0.1, 0, +0.1, +0.2 s, centred where the ball is in this frame,
    so a holder is judged with its neighbouring frames (possession calls need context).

A labelling aid, not a pipeline layer: it only decides what the reviewer looks at.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from autolabel import proposals

PERSON = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee"}
COLOUR = {
    "player": (255, 255, 255),
    "player-in-possession": (255, 0, 255),
    "player-jump-shot": (0, 0, 255),
    "player-layup-dunk": (0, 0, 255),
    "player-shot-block": (0, 0, 255),
    "referee": (0, 220, 255),
}
WIN_W, WIN_H, STRIP, CAP = 960, 540, 192, 26


def label(img, text, x, y, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = int(max(0, x)), int(max(th + 2, y))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def centre(f):
    balls = [b for b in f["boxes"] if b[5] == "ball"]
    if balls:
        b = max(balls, key=lambda b: b[4])
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, True
    best, n = None, -1
    for r in (b for b in f["boxes"] if b[5] == "rim"):
        w, cx = r[2] - r[0], (r[0] + r[2]) / 2
        k = sum(abs((q[0] + q[2]) / 2 - cx) < 4 * w and q[3] > r[3] for q in f["boxes"] if q[5] in PERSON)
        if k > n:
            best, n = (cx, r[3] + 150), k
    return (*(best or (960, 540)), False)


def draw(img, f, x0, y0):
    """Boxes on a copy of the frame; labels are kept inside the window whose top-left corner is (x0, y0)."""
    out = img.copy()
    for b in f["boxes"]:
        if b[5] == "rim":
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 1)
    cands = sorted((c for c in f["candidates"] if c[5] == "ball"), key=lambda c: -c[4])[:3]
    for j, c in enumerate(cands):
        cv2.rectangle(out, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 255, 0), 1)
        label(out, f"c{j}", c[2] + 2, c[3] + 14, (255, 255, 0))
    for i, b in enumerate(f["boxes"]):
        if b[5] in PERSON:
            col = COLOUR[b[5]]
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
            label(out, str(i), max(b[0], x0), max(b[1], y0 + 22), col)
    for b in f["boxes"]:
        if b[5] == "ball":
            cv2.rectangle(out, (int(b[0]) - 2, int(b[1]) - 2), (int(b[2]) + 2, int(b[3]) + 2), (0, 140, 255), 2)
            label(out, "B", b[2] + 3, b[1], (0, 140, 255))
    return out


def corner(shape, cx, cy, w, h):
    H, W = shape[:2]
    return int(min(max(0, cx - w / 2), W - w)), int(min(max(0, cy - h / 2), H - h))


def window(img, cx, cy, w, h):
    x0, y0 = corner(img.shape, cx, cy, w, h)
    return img[y0 : y0 + h, x0 : x0 + w]


def tile(cap, fps, f, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
    img = cap.read()[1]
    cx, cy, has_ball = centre(f)
    top = window(draw(img, f, *corner(img.shape, cx, cy, WIN_W, WIN_H)), cx, cy, WIN_W, WIN_H)
    strip = []
    for dt in (-0.2, -0.1, 0.0, 0.1, 0.2):
        k = f["frame"] + round(dt * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = cap.read()
        crop = window(im if ok else img, cx, cy, 300, 300).copy()
        crop = cv2.resize(crop, (STRIP, STRIP))
        label(crop, f"{dt:+.1f}s", 2, 18, (255, 255, 255) if dt else (0, 140, 255))
        strip.append(crop)
    strip = np.hstack(strip)
    cap_band = np.zeros((CAP, WIN_W, 3), np.uint8)
    ball = "ball" if has_ball else "NO BALL BOX"
    text = f"#{idx} {f['clip'][9:20]} s{f['shot']} f{f['frame']} [{f['bucket']}] {ball}"
    cv2.putText(cap_band, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    pad = np.zeros((STRIP, WIN_W - strip.shape[1], 3), np.uint8)
    return np.vstack([cap_band, top, np.hstack([strip, pad])])


def proposal_tile(cap, f, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
    img = cap.read()[1]
    cx, cy, _ = centre(f)
    x0, y0 = corner(img.shape, cx, cy, WIN_W, WIN_H)
    out = img.copy()
    for b in f["boxes"]:
        if b[5] in PERSON:
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (150, 150, 150), 1)
    for j, c in enumerate(proposals(f)):
        cv2.rectangle(out, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
        label(out, f"p{j}", max(c[0], x0), max(c[1], y0 + 22), (0, 255, 0))
    top = out[y0 : y0 + WIN_H, x0 : x0 + WIN_W]
    band = np.zeros((CAP, WIN_W, 3), np.uint8)
    text = f"#{idx} {f['clip'][9:20]} s{f['shot']} f{f['frame']} proposals: {len(proposals(f))}"
    cv2.putText(band, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([band, top])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--per-sheet", type=int, default=4)
    ap.add_argument("--proposals", action="store_true")
    a = ap.parse_args(argv)
    out = Path(f"runs/autolabel/r{a.round}")
    frames = json.loads((out / "draft.json").read_text())["frames"]
    if a.proposals:
        _proposal_sheets(out, frames, a.per_sheet)
        return
    fps_of = {k: v["fps"] for k, v in json.loads(Path("runs/autolabel/r1/central.json").read_text()).items()}
    (out / "sheets").mkdir(exist_ok=True)
    caps, tiles = {}, []
    for i, f in enumerate(frames):
        cap = caps.setdefault(f["video"], cv2.VideoCapture(f["video"]))
        tiles.append(tile(cap, fps_of[f["clip"]], f, i))
    blank = np.zeros_like(tiles[0])
    n = 0
    for s in range(0, len(tiles), a.per_sheet):
        group = tiles[s : s + a.per_sheet]
        group += [blank] * (a.per_sheet - len(group))
        rows = [np.hstack(group[j : j + 2]) for j in range(0, len(group), 2)]
        cv2.imwrite(str(out / "sheets" / f"sheet_{n:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        n += 1
    print(f"{n} sheets of {a.per_sheet} frames -> {out / 'sheets'}")


def _proposal_sheets(out, frames, per_sheet):
    import csv

    reads = {
        int(r["idx"]): r
        for p in sorted((out / "reads").glob("*.tsv"))
        for r in csv.DictReader(p.open(), delimiter="\t")
    }
    want = [i for i, r in sorted(reads.items()) if int(r["missing"]) > 0 or r["holder"] == "nobox"]
    (out / "proposals").mkdir(exist_ok=True)
    caps, tiles = {}, []
    for i in want:
        f = frames[i]
        tiles.append(proposal_tile(caps.setdefault(f["video"], cv2.VideoCapture(f["video"])), f, i))
    n = 0
    for s in range(0, len(tiles), per_sheet):
        group = tiles[s : s + per_sheet] + [np.zeros_like(tiles[0])] * (per_sheet - len(tiles[s : s + per_sheet]))
        rows = [np.hstack(group[j : j + 2]) for j in range(0, len(group), 2)]
        cv2.imwrite(str(out / "proposals" / f"sheet_{n:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        n += 1
    print(f"{len(want)} frames, {n} proposal sheets -> {out / 'proposals'}")


if __name__ == "__main__":
    main()
