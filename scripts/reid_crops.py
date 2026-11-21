#!/usr/bin/env python3
"""Identity crops for the PLK appearance encoder: a crop manifest from a crops config's linked tracklets, and filmstrip
sheets to check each identity's purity by eye.

    uv run scripts/reid_crops.py manifest --config configs/crops-reid-r1.yaml --out data/labels/appearance/crops_r1.json
    uv run scripts/reid_crops.py sheets data/labels/appearance/crops_r1.json [--sample 96] [--seed 0]
    uv run scripts/reid_crops.py drop data/labels/appearance/crops_r1.json <identity> [...]

One linked track of one shot is one identity, with its majority team and the share of its boxes on that team. A
crop is a non-referee row of the link output with a detected box (det_index >= 0) scoring at least --min-score and at
least --min-h pixels tall; an identity needs --min-boxes such rows, and keeps --per-identity of them spread over its
life. The manifest stores boxes only (frame and xyxy per
crop); the pictures are cut from the clips at training time and never committed. Gold games are refused.

Sheets go to runs/reid_crops/<manifest stem>/sheet_NN.jpg, one row of 8 crops per identity captioned with its index
in the sample; sheet_NN.txt maps the indices to identities. `drop` marks identities
seen to be mixed (a tracker swap) as dropped, with the date, so a training set leaves them out.

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bga.config import load_config
from bga.pipeline import git_commit, load_output, sequences
from bga.review.sheets import contact_sheet
from bga.stage import FrameReader
from bga.train.leakage import assert_no_gold


def spread(items, k):
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def identities_of(seq, link, teams, a):
    by_id = {}
    for f in seq.frames:
        for r in link["frames"].get(str(f), []):
            tid, x1, y1, x2, y2, score, cls, det = r
            if cls == "referee" or det < 0 or score < a.min_score or y2 - y1 < a.min_h:
                continue
            team = teams.get(str(f), [])[det] if det < len(teams.get(str(f), [])) else ""
            by_id.setdefault(tid, []).append((f, [round(v, 1) for v in (x1, y1, x2, y2)], team))
    out = []
    for tid, rows in sorted(by_id.items()):
        if len(rows) < a.min_boxes:
            continue
        votes = Counter(t for _, _, t in rows if t)
        team = votes.most_common(1)
        out.append({
            "id": f"{seq.name}_t{tid}",
            "clip": seq.clip,
            "video": seq.video,
            "shot": seq.shot,
            "team": team[0][0] if team else "",
            "team_share": round(team[0][1] / sum(votes.values()), 3) if team else 0.0,
            "boxes": len(rows),
            "span": [rows[0][0], rows[-1][0]],
            "crops": [[f, *b] for f, b, _ in spread(rows, a.per_identity)],
        })
    return out


def manifest(a):
    cfg = load_config(a.config)
    seqs = sequences(cfg)
    assert_no_gold({s.clip for s in seqs})
    ids = []
    for seq in seqs:
        link = load_output(cfg, Path("runs"), seq, "link")
        teams = load_output(cfg, Path("runs"), seq, "team")["frames"]
        ids += identities_of(seq, link, teams, a)
    out = {
        "version": 1,
        "config": cfg.name,
        "config_hash": cfg.config_hash(),
        "commit": git_commit(),
        "made": dt.date.today().isoformat(),
        "rule": {k: getattr(a, k) for k in ("min_score", "min_h", "min_boxes", "per_identity")},
        "clips": sorted({s.clip for s in seqs}),
        "sequences": len(seqs),
        "identities": ids,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.out}: {len(ids)} identities, {sum(len(i['crops']) for i in ids)} crops from {len(seqs)} shots")


def sheets(a):
    m = json.loads(a.manifest.read_text())
    ids = [i for i in m["identities"] if not i.get("dropped")]
    pick = sorted(random.Random(a.seed).sample(ids, min(a.sample, len(ids))), key=lambda i: i["id"])
    out = Path("runs/reid_crops") / a.manifest.stem
    out.mkdir(parents=True, exist_ok=True)
    readers = {}
    for n in range(0, len(pick), a.per_sheet):
        items, index = [], []
        for k, ident in enumerate(pick[n : n + a.per_sheet], start=n):
            read = readers.setdefault(ident["video"], FrameReader(ident["video"]))
            index.append(f"{k} {ident['id']} {ident['team']} n{ident['boxes']}")
            for j, (f, x1, y1, x2, y2) in enumerate(spread(ident["crops"], 8)):
                crop = read(f)[int(y1) : int(y2), int(x1) : int(x2)]
                items.append((crop, f"#{k} {ident['team']}" if j == 0 else f"f{f}"))
            items += [(crop[:2, :2] * 0, "")] * (8 - min(8, len(ident["crops"])))
        p = out / f"sheet_{n // a.per_sheet:02d}.jpg"
        p.with_suffix(".txt").write_text("\n".join(index) + "\n")
        cv2.imwrite(
            str(p), contact_sheet(items, cols=8, tile_wh=(120, 220), caption_h=16), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        print(p)
    for r in readers.values():
        r.close()


def drop(a):
    m = json.loads(a.manifest.read_text())
    today = dt.date.today().isoformat()
    known = {i["id"] for i in m["identities"]}
    missing = sorted(set(a.ids) - known)
    if missing:
        raise SystemExit(f"unknown identities: {missing}")
    for i in m["identities"]:
        if i["id"] in a.ids:
            i["dropped"] = f"mixed, by eye {today}"
    a.manifest.write_text(json.dumps(m, indent=1) + "\n")
    print(f"dropped {len(a.ids)}; {sum(not i.get('dropped') for i in m['identities'])} identities left")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--config", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--min-score", type=float, default=0.5)
    m.add_argument("--min-h", type=int, default=64)
    m.add_argument("--min-boxes", type=int, default=10)
    m.add_argument("--per-identity", type=int, default=24)
    s = sub.add_parser("sheets")
    s.add_argument("manifest", type=Path)
    s.add_argument("--sample", type=int, default=96)
    s.add_argument("--per-sheet", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("drop")
    d.add_argument("manifest", type=Path)
    d.add_argument("ids", nargs="+")
    a = ap.parse_args(argv)
    {"manifest": manifest, "sheets": sheets, "drop": drop}[a.cmd](a)


if __name__ == "__main__":
    main()
