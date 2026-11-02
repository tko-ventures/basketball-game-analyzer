#!/usr/bin/env python3
"""Filmstrip contact sheets for the first review pass: one row per track of a shot, 8 crops spread
over the track's life, captioned with track id, team, frame span and box source.

    uv run scripts/track_sheets.py data/labels/<clip>.json --shot 10 [--tracks-per-sheet 16]

Writes runs/<clip>/review/tracks_s<shot>_NN.jpg. Crops come from the label file's current boxes, so
rerunning after review_ops.py shows the corrected tracks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bga.review.sheets import contact_sheet  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--per-track", type=int, default=8)
    ap.add_argument("--tracks-per-sheet", type=int, default=16)
    a = ap.parse_args(argv)
    d = json.loads(a.labels.read_text())
    s = d["shots"][a.shot]
    tr = defaultdict(list)
    for k, f in d["frames"].items():
        if s["start"] <= int(k) <= s["end"]:
            for b in f["boxes"]:
                tr[str(b["tid"])].append((int(k), b))
    cap, cache = cv2.VideoCapture(d["clip"]), {}

    def frame(i):
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            cache[i] = cap.read()[1]
        return cache[i]

    rows = sorted(tr.items(), key=lambda kv: (min(f for f, _ in kv[1]), kv[0]))
    out = Path("runs") / a.labels.stem / "review"
    out.mkdir(parents=True, exist_ok=True)
    for n in range(0, len(rows), a.tracks_per_sheet):
        items = []
        for tid, obs in rows[n : n + a.tracks_per_sheet]:
            obs.sort(key=lambda x: x[0])
            k = a.per_track
            picks = [obs[round(i * (len(obs) - 1) / (k - 1))] for i in range(k)] if len(obs) >= k else obs
            ident = d["identities"].get(tid, {})
            for j in range(k):
                if j >= len(picks):
                    items.append((frame(obs[0][0])[:2, :2] * 0, ""))
                    continue
                fi, b = picks[j]
                x1, y1, x2, y2 = (int(b[c]) for c in ("x1", "y1", "x2", "y2"))
                w, h = x2 - x1, y2 - y1
                crop = frame(fi)[max(0, y1 - h // 8) : y2 + h // 8, max(0, x1 - w // 4) : x2 + w // 4]
                if j == 0:
                    team = ident.get("team") or "?"
                    num = f"#{ident['number']}" if ident.get("number") else ""
                    cap_txt = f"{tid} {team}{num} {obs[0][0]}-{obs[-1][0]} n{len(obs)}"
                else:
                    cap_txt = f"f{fi}"
                items.append((crop, cap_txt))
        p = out / f"tracks_s{a.shot}_{n // a.tracks_per_sheet:02d}.jpg"
        cv2.imwrite(
            str(p),
            contact_sheet(items, cols=a.per_track, tile_wh=(150, 190), caption_h=16),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        print(p, min(len(rows) - n, a.tracks_per_sheet), "tracks")


if __name__ == "__main__":
    main()
