#!/usr/bin/env python3
"""Gold number crops (plan section 8): the number boxes of box gold v1, each labelled with the digits it shows.

    uv run scripts/number_gold.py prefill     # data/labels/numbers/gold_crops.json from data/gold/v1/boxes.json
    uv run scripts/number_gold.py negatives   # adds the detector's number boxes that match no gold box
    uv run scripts/number_gold.py sheets      # blind contact sheets: crop ids only, no track number shown
    uv run scripts/number_gold.py apply data/labels/review/number_gold.ops.txt
    uv run scripts/number_gold.py compare     # reads against the number of the gold track holding the box
    uv run scripts/number_gold.py spotcheck   # a seeded sample of 30 for Tomasz's check at /numbers
    uv run scripts/number_gold.py export      # data/gold/v1/numbers.json

The boxes are Tomasz's hand-drawn `number` boxes (box gold v1), so the crop geometry depends on no model. Each crop is
matched to the gold person boxes of its frame that hold it (intersection over the number box's area at least 0.9);
the gold v0 track of each holder gives `hints` (track id, team, number). Hints are kept off the reading sheets so the
read is not anchored on them; `compare` then lists every crop whose read differs from its holder's number for a
second look. `negatives` adds the PLK detector's `number` boxes on the same frames that overlap no gold box, at the
score the numbers layer reads from (0.3): sponsor logos, badges and parts of numbers, the crops a reader must refuse.

Labels: the number as digits (``"7"``, ``"23"``, ``"00"``) when every digit is legible; ``illegible`` for a jersey
number whose digits cannot all be read (blur, fold, cut off, turned away); ``not_number`` for a box that holds no
jersey number. Tomasz's verdicts from the labeller's /numbers page sit in each crop's ``review``; `export` takes his
label where he changed one. Ops file lines: ``<crop_id> <label> [note...]``; ``#`` starts a comment.

"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.numbers.base import roster_numbers
from bga.review.sheets import contact_sheet

BOXES = Path("data/gold/v1/boxes.json")
GOLD_V0 = Path("data/gold/v0")
CROPS = Path("data/labels/numbers/gold_crops.json")
EXPORT = Path("data/gold/v1/numbers.json")
RAW = Path("data/raw")
OUT = Path("runs/number_gold")
PEOPLE = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee"}
SPECIAL = {"illegible", "not_number"}


def ios(a, b):
    """Share of box a's area inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def load_identities(gold: Path):
    ids, held = {}, {}
    for s in json.loads((gold / "manifest.json").read_text())["sequences"]:
        held[s["name"]] = s["held_out"]
        for r in csv.DictReader((gold / s["name"] / "identities.csv").open()):
            ids[(s["name"], int(r["tid"]))] = r
    return ids, held


def holder_hints(xyxy, people, seq, ids):
    """Track id, team and gold number of each person box holding a number box, smallest box first."""
    holders = sorted(
        (p for p in people if ios(xyxy, p["xyxy"]) >= 0.9),
        key=lambda p: (p["xyxy"][2] - p["xyxy"][0]) * (p["xyxy"][3] - p["xyxy"][1]),
    )
    return [
        {"tid": p["tid"], "team": p["team"], "number": ids.get((seq, p["tid"]), {}).get("number") or None}
        for p in holders
    ]


def new_crop(cid, f, xyxy, hints, held, source):
    return {
        "id": cid,
        "source": source,
        "sequence": f"{f['youtube_id']}_s{f['shot']}",
        "youtube_id": f["youtube_id"],
        "clip": f["clip"],
        "shot": f["shot"],
        "frame": f["frame"],
        "held_out": held,
        "xyxy": xyxy,
        "hints": hints,
        "label": None,
        "note": "",
    }


def people_of(f):
    return [b for b in f["boxes"] if b["cls"] in PEOPLE and b.get("tid") is not None]


def prefill(a):
    if a.crops.exists() and not a.force:
        sys.exit(f"{a.crops} exists; pass --force to rebuild it (labels would be lost)")
    ids, held = load_identities(a.gold)
    crops = []
    for f in json.loads(a.boxes.read_text())["frames"]:
        seq = f"{f['youtube_id']}_s{f['shot']}"
        for b in sorted((b for b in f["boxes"] if b["cls"] == "number"), key=lambda b: (b["xyxy"][0], b["xyxy"][1])):
            hints = holder_hints(b["xyxy"], people_of(f), seq, ids)
            crops.append(new_crop(f"n{len(crops) + 1:03d}", f, b["xyxy"], hints, held[seq], "box_gold_v1"))
    a.crops.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": "v1",
        "source": str(a.boxes),
        "labels": "digits when all are legible, else illegible or not_number",
        "crops": crops,
    }
    a.crops.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(crops)} crops -> {a.crops}", Counter(len(c["hints"]) for c in crops))


def iou(a, b):
    i = ios(a, b) * (a[2] - a[0]) * (a[3] - a[1])
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def negatives(a):
    """Add the detector's number boxes that match no gold number box (IoU below 0.3) on the box-gold frames."""
    doc = json.loads(a.crops.read_text())
    if any(c["source"] != "box_gold_v1" for c in doc["crops"]):
        sys.exit("negatives already added")
    ids, held = load_identities(a.gold)
    added = []
    for f in json.loads(a.boxes.read_text())["frames"]:
        seq = f"{f['youtube_id']}_s{f['shot']}"
        out = Path("runs") / f["clip"] / "detect" / a.detect / f"{seq}.json"
        dets = json.loads(out.read_text())["frames"][str(f["frame"])]
        gold = [b["xyxy"] for b in f["boxes"] if b["cls"] == "number"]
        for x1, y1, x2, y2, score, cls in dets:
            box = [round(v, 1) for v in (x1, y1, x2, y2)]
            if cls != "number" or score < a.min_score or max((iou(box, g) for g in gold), default=0) >= 0.3:
                continue
            c = new_crop(f"d{len(added) + 1:03d}", f, box, holder_hints(box, people_of(f), seq, ids), held[seq], "")
            c["source"], c["score"] = f"detector {a.detect} {a.weights}", round(score, 3)
            added.append(c)
    doc["crops"] += added
    doc["negatives"] = f"detector number boxes scoring at least {a.min_score} with IoU below 0.3 to every gold box"
    a.crops.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(added)} detector crops added")


def tiles(frame, xyxy, size):
    """A tight tile (number box plus 40% margin, upscaled) and a context tile (5x the box) of one crop."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    out = []
    for m in (0.4, 2.0):
        cx1, cy1 = int(max(0, x1 - m * bw)), int(max(0, y1 - m * bh))
        cx2, cy2 = int(min(w, x2 + m * bw)), int(min(h, y2 + m * bh))
        crop = frame[cy1:cy2, cx1:cx2]
        s = size / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        if m > 1:
            p1 = (int((x1 - cx1) * s), int((y1 - cy1) * s))
            p2 = (int((x2 - cx1) * s), int((y2 - cy1) * s))
            cv2.rectangle(crop, p1, p2, (0, 255, 255), 1)
        out.append(crop)
    return out


class Frames:
    """Decoded source frames, one open video per clip."""

    def __init__(self, raw):
        self.raw, self.caps, self.cache = raw, {}, {}

    def get(self, clip, i):
        if (clip, i) not in self.cache:
            cap = self.caps.setdefault(clip, cv2.VideoCapture(str(self.raw / f"{clip}.mp4")))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                sys.exit(f"cannot read frame {i} of {clip}")
            self.cache[(clip, i)] = img
        return self.cache[(clip, i)]


def render(crops, frames, path, caption, per_row=5, size=200):
    items = []
    for c in crops:
        tight, ctx = tiles(frames.get(c["clip"], c["frame"]), c["xyxy"], size)
        items += [(tight, caption(c)), (ctx, "")]
    while len(items) % (2 * per_row):
        items.append((np.zeros((2, 2, 3), np.uint8), ""))
    sheet = contact_sheet(items, cols=2 * per_row, tile_wh=(size, size), caption_h=20)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(path, len(crops), "crops")


def sheets(a):
    crops = json.loads(a.crops.read_text())["crops"]
    if a.ids:
        want = set(a.ids.split(","))
        crops = [c for c in crops if c["id"] in want]
    elif not a.all:
        crops = [c for c in crops if c["label"] is None]
    a.out.mkdir(parents=True, exist_ok=True)
    frames = Frames(a.raw)
    for n in range(0, len(crops), a.per_sheet):
        part = crops[n : n + a.per_sheet]
        render(part, frames, a.out / f"{a.prefix}_{n // a.per_sheet:02d}.jpg", lambda c: c["id"])


def parse_ops(path):
    ops = []
    for ln in path.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        cid, label, *note = ln.split()
        if label not in SPECIAL and not re.fullmatch(r"\d{1,2}", label):
            sys.exit(f"bad label {label!r} for {cid}")
        ops.append((cid, label, " ".join(note)))
    return ops


def apply(a):
    doc = json.loads(a.crops.read_text())
    by_id = {c["id"]: c for c in doc["crops"]}
    n = 0
    for cid, label, note in parse_ops(a.ops):
        if cid not in by_id:
            sys.exit(f"unknown crop {cid}")
        by_id[cid]["label"], by_id[cid]["note"] = label, note
        n += 1
    a.crops.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{n} labels applied; {sum(c['label'] is None for c in doc['crops'])} crops unlabelled")


def compare(a):
    crops = json.loads(a.crops.read_text())["crops"]
    tally, flagged = Counter(), []
    for c in crops:
        nums = {h["number"] for h in c["hints"] if h["number"]}
        lab = c["label"]
        if lab is None:
            kind = "unlabelled"
        elif lab in SPECIAL:
            kind = lab
        elif lab in nums:
            kind = "agrees" if len(c["hints"]) == 1 else "agrees (one of several holders)"
        elif not nums:
            kind = "read, holder has no gold number"
        else:
            kind = "differs"
        tally[kind] += 1
        if kind in ("differs", "read, holder has no gold number"):
            flagged.append(c)
            hints = ", ".join(f"{h['tid']} {h['team']} #{h['number']}" for h in c["hints"]) or "no holder"
            print(f"{c['id']} {c['sequence']} f{c['frame']} read {lab}; holders: {hints}")
    print(dict(tally))
    if a.sheet and flagged:
        a.out.mkdir(parents=True, exist_ok=True)
        render(flagged, Frames(a.raw), a.out / "compare.jpg", lambda c: f"{c['id']} read {c['label']}")


def spotcheck(a):
    doc = json.loads(a.crops.read_text())
    crops = [c for c in doc["crops"] if c["label"] is not None]
    pick = sorted(random.Random(a.seed).sample(crops, min(a.n, len(crops))), key=lambda c: c["id"])
    doc["spotcheck"] = {"seed": a.seed, "n": len(pick), "ids": [c["id"] for c in pick]}
    a.crops.write_text(json.dumps(doc, indent=1) + "\n")
    a.out.mkdir(parents=True, exist_ok=True)
    frames = Frames(a.raw)
    for n in range(0, len(pick), 15):
        render(pick[n : n + 15], frames, a.out / f"spotcheck_{n // 15:02d}.jpg", lambda c: f"{c['id']}: {c['label']}")
    print(" ".join(c["id"] for c in pick))


def crop_team(c, rosters):
    """The side wearing the number: the smallest holder's team, unless only the other side's roster has the read."""
    team = c["hints"][0]["team"] if c["hints"] else ""
    if c["label"] in SPECIAL:
        return team
    on = [s for s in ("home", "away") if c["label"] in rosters[s]]
    return on[0] if len(on) == 1 else team


def resolved(c):
    """The crop with the reviewer's label in place of the implementer's where the review changed it."""
    review = c.get("review") or {}
    return {**c, "label": review.get("label", c["label"]) if review.get("verdict") == "fix" else c["label"]}


def export(a):
    doc = json.loads(a.crops.read_text())
    doc["crops"] = [resolved(c) for c in doc["crops"]]
    missing = [c["id"] for c in doc["crops"] if c["label"] is None]
    if missing:
        sys.exit(f"{len(missing)} crops unlabelled, e.g. {missing[:5]}")
    rosters = {}
    for c in doc["crops"]:
        if c["clip"] not in rosters:
            meta = json.loads((Path("data/labels") / f"{c['clip']}.json").read_text())["meta"]
            rosters[c["clip"]] = {s: roster_numbers(meta, s) for s in ("home", "away")}
    out = {
        "version": "v1",
        "source": str(a.crops),
        "labels": doc["labels"],
        "crops": [
            {k: c[k] for k in ("id", "source", "sequence", "clip", "frame", "held_out", "xyxy", "label")}
            | {"team": crop_team(c, rosters[c["clip"]]), "review": (c.get("review") or {}).get("verdict")}
            for c in doc["crops"]
        ],
    }
    moved = sum(
        o["team"] != (c["hints"][0]["team"] if c["hints"] else "")
        for o, c in zip(out["crops"], doc["crops"], strict=True)
    )
    a.export.write_text(json.dumps(out, indent=1) + "\n")
    kinds = Counter("digits" if c["label"] not in SPECIAL else c["label"] for c in out["crops"])
    held = Counter("held-out" if c["held_out"] else "dev" for c in out["crops"])
    reviews = dict(Counter(c["review"] for c in out["crops"] if c["review"]))
    print(
        f"{len(out['crops'])} crops -> {a.export}", dict(kinds), dict(held), f"team from the roster on {moved}", reviews
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", type=Path, default=CROPS)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefill")
    p.add_argument("--boxes", type=Path, default=BOXES)
    p.add_argument("--gold", type=Path, default=GOLD_V0)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("negatives")
    p.add_argument("--boxes", type=Path, default=BOXES)
    p.add_argument("--gold", type=Path, default=GOLD_V0)
    p.add_argument("--detect", default="rfdetr_torch@d76aab0d")
    p.add_argument("--weights", default="detect/rfdetr_torch/plk-r3-poss@c62744fe")
    p.add_argument("--min-score", type=float, default=0.3)
    p = sub.add_parser("sheets")
    p.add_argument("--per-sheet", type=int, default=20)
    p.add_argument("--prefix", default="read")
    p.add_argument("--ids", default="")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("ops", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("--sheet", action="store_true")
    p = sub.add_parser("spotcheck")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("export")
    p.add_argument("--export", type=Path, default=EXPORT)
    a = ap.parse_args(argv)
    {
        "prefill": prefill,
        "negatives": negatives,
        "sheets": sheets,
        "apply": apply,
        "compare": compare,
        "spotcheck": spotcheck,
    }.get(a.cmd, export)(a)


if __name__ == "__main__":
    main()
