#!/usr/bin/env python3
"""Team audit sheets for gold track teams: one row per gold track of a shot, rows grouped by gold team.

    uv run scripts/team_gold_sheets.py [--gold data/gold/v0] [--per-track 7] [--tracks-per-sheet 18]

Crops come from the gold MOTChallenge boxes, not from any classifier, so the reader judges each kit by eye. Within a
sheet, home tracks come first, then away, then referees, so a track in the wrong group stands out. Each row's first
caption gives the track id, its gold team and number, and its box count; ``V1!`` marks a track where box gold v1 gives
another team on some box. Only boxes with full visibility are cropped.

Writes runs/team_audit/<sequence>_NN.jpg and prints one line per sheet.

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.review.sheets import contact_sheet

TEAM_ORDER = {"home": 0, "away": 1, "ref": 2}


def v1_disagreements(boxes_json: Path, ids: dict) -> set:
    """(sequence, tid) pairs where a box gold v1 box carries a team other than the track's gold team."""
    out = set()
    for f in json.loads(boxes_json.read_text())["frames"]:
        seq = f"{f['youtube_id']}_s{f['shot']}"
        for b in f["boxes"]:
            key = (seq, str(b.get("tid")))
            if key in ids and b.get("team") and ids[key]["team"] != b["team"]:
                out.add(key)
    return out


def load_seq(seq_dir: Path):
    """Gold boxes per track (source frame, xywh), identities by tid, and the clip name of one sequence."""
    frames = [int(x) for x in (seq_dir / "frames.txt").read_text().split()]
    clip = next(
        ln.split("=", 1)[1] for ln in (seq_dir / "seqinfo.ini").read_text().splitlines() if ln.startswith("clip=")
    )
    ids = {r["tid"]: r for r in csv.DictReader((seq_dir / "identities.csv").open())}
    tracks = defaultdict(list)
    for row in csv.reader((seq_dir / "gt" / "gt.txt").open()):
        k, tid, x, y, w, h, vis = int(row[0]), row[1], *map(float, row[2:6]), float(row[8])
        if vis >= 1.0:
            tracks[tid].append((frames[k - 1], x, y, w, h))
    return tracks, ids, clip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("data/gold/v0"))
    ap.add_argument("--boxes-v1", type=Path, default=Path("data/gold/v1/boxes.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("runs/team_audit"))
    ap.add_argument("--per-track", type=int, default=7)
    ap.add_argument("--tracks-per-sheet", type=int, default=18)
    a = ap.parse_args(argv)
    seq_dirs = sorted(p for p in a.gold.iterdir() if (p / "gt" / "gt.txt").exists())
    all_ids = {}
    for p in seq_dirs:
        for tid, r in load_seq(p)[1].items():
            all_ids[(p.name, tid)] = r
    flagged = v1_disagreements(a.boxes_v1, all_ids)
    a.out.mkdir(parents=True, exist_ok=True)
    for p in seq_dirs:
        tracks, ids, clip = load_seq(p)
        cap, cache = cv2.VideoCapture(str(a.raw / f"{clip}.mp4")), {}

        def frame(i, cap=cap, cache=cache):
            if i not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                cache[i] = cap.read()[1]
            return cache[i]

        rows = sorted(
            (t for t in tracks if t in ids),
            key=lambda t: (TEAM_ORDER.get(ids[t]["team"], 3), int(t)),
        )
        k = a.per_track
        for n in range(0, len(rows), a.tracks_per_sheet):
            items = []
            for tid in rows[n : n + a.tracks_per_sheet]:
                obs = sorted(tracks[tid])
                picks = [obs[round(i * (len(obs) - 1) / (k - 1))] for i in range(k)] if len(obs) >= k else obs
                ident = ids[tid]
                for j in range(k):
                    if j >= len(picks):
                        items.append((np.zeros((2, 2, 3), np.uint8), ""))
                        continue
                    fi, x, y, w, h = picks[j]
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    crop = frame(fi)[max(0, y1) : y2, max(0, x1) : x2]
                    if j == 0:
                        num = f"#{ident['number']}" if ident.get("number") else ""
                        mark = " V1!" if (p.name, tid) in flagged else ""
                        cap_txt = f"{tid} {ident['team'].upper()}{num} n{len(obs)}{mark}"
                    else:
                        cap_txt = f"{ident['team'][0].upper()} f{fi}"
                    items.append((crop, cap_txt))
            out = a.out / f"{p.name}_{n // a.tracks_per_sheet:02d}.jpg"
            sheet = contact_sheet(items, cols=k, tile_wh=(130, 210), caption_h=16)
            cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(out, min(len(rows) - n, a.tracks_per_sheet), "tracks")


if __name__ == "__main__":
    main()
