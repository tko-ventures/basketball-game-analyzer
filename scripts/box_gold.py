#!/usr/bin/env python3
"""Box gold (plan section 8): about 100 frames drawn evenly from the gold shots, every person boxed.

    uv run scripts/box_gold.py select [--n 100]       # writes data/labels/box_gold.json (frame list)
    uv run scripts/box_gold.py sheets                 # runs/box_gold/sheet_NN.jpg for the review
    uv run scripts/box_gold.py export --version v0    # data/gold/<version>/boxes.json

A frame's boxes are the track-gold boxes at that frame, plus detector candidates the reviewer adds,
minus track boxes the reviewer drops, both listed in data/labels/review/box_gold.ops.txt:
    add <youtube_id> <frame> c<k> <home|away|ref|other>   candidate k on that frame's sheet tile
    drop <youtube_id> <frame> <tid>                        a track box that is not a person on court
Candidates are the prefill detector's person boxes at confidence >= 0.1 that overlap no gold box
(IoU < 0.3), so people the tracks miss can be added without drawing. Box geometry stays the
prefill's (detector, or SAM mask box): tightening by hand is gold v1 work, so v0 supports AP50 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.review.sheets import contact_sheet  # noqa: E402

GOLD = Path("data/labels/gold.json")
SEL = Path("data/labels/box_gold.json")
OPS = Path("data/labels/review/box_gold.ops.txt")
PREFILL = "basketball-player-detection-3-ycjdo_v4_c0.1_lb90_ar1"
PERSON = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
}
TEAM_BGR = {"home": (240, 240, 240), "away": (60, 200, 60), "ref": (0, 200, 255), "other": (200, 0, 200)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def labels_for(clip: str) -> dict:
    return json.loads((Path("data/labels") / f"{clip}.json").read_text())


def select(n: int) -> None:
    gold = json.loads(GOLD.read_text())
    shots = [(g, s) for g in gold["games"] for s in g["shots"]]
    total = sum(s["end"] - s["start"] for _, s in shots)
    frames, acc = [], 0.0
    for g, s in shots:
        labelled = sorted(int(k) for k in labels_for(g["clip"])["frames"] if s["start"] <= int(k) <= s["end"])
        acc += n * (s["end"] - s["start"]) / total
        k = round(acc) - len(frames)
        step = len(labelled) / k
        for i in range(k):  # evenly spaced, centred in each 1/k of the shot
            f = labelled[int((i + 0.5) * step)]
            frames.append({"youtube_id": g["youtube_id"], "clip": g["clip"], "shot": s["shot"], "frame": f})
    SEL.write_text(json.dumps({"note": "Box-gold frames, plan section 8", "frames": frames}, indent=1) + "\n")
    print(f"{len(frames)} frames over {len(shots)} shots -> {SEL}")


def candidates(clip: str, frame: int, gold_boxes: list) -> list[dict]:
    path = Path("runs") / clip / "prefill" / PREFILL / "detections.jsonl"
    for line in path.open():
        rec = json.loads(line)
        if rec["frame_idx"] == frame:
            dets = [d for d in rec["detections"] if d["class"] in PERSON]
            gb = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in gold_boxes]
            dets = [d for d in dets if all(iou(d["xyxy"], g) < 0.3 for g in gb)]
            return sorted(dets, key=lambda d: -d["score"])
    return []


def frame_boxes(sel: dict) -> list[dict]:
    lab = labels_for(sel["clip"])
    ids = lab["identities"]
    return [
        {**b, "team": ids.get(str(b["tid"]), {}).get("team", "")}
        for b in lab["frames"].get(str(sel["frame"]), {}).get("boxes", [])
    ]


def sheets() -> None:
    out = Path("runs/box_gold")
    out.mkdir(parents=True, exist_ok=True)
    items, caps = [], {}
    for sel in json.loads(SEL.read_text())["frames"]:
        clip = sel["clip"]
        if clip not in caps:
            caps[clip] = cv2.VideoCapture(
                json.loads((Path("data/labels") / f"{clip}.json").read_text())["clip"]
            )
        cap = caps[clip]
        cap.set(cv2.CAP_PROP_POS_FRAMES, sel["frame"])
        img = cap.read()[1]
        boxes = frame_boxes(sel)
        for b in boxes:
            col = TEAM_BGR.get(b["team"], (255, 0, 255))
            cv2.rectangle(img, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), col, 2)
            cv2.putText(img, str(b["tid"]), (int(b["x1"]), int(b["y1"]) - 6), 0, 1.0, col, 2)
        for k, d in enumerate(candidates(clip, sel["frame"], boxes)):
            x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"c{k} {d['score']:.2f}", (x1, y2 + 30), 0, 1.1, (0, 0, 255), 3)
        items.append((img, f"{sel['youtube_id']} s{sel['shot']} f{sel['frame']}  ({len(boxes)} boxes)"))
    for n in range(0, len(items), 4):
        p = out / f"sheet_{n // 4:02d}.jpg"
        cv2.imwrite(
            str(p),
            contact_sheet(items[n : n + 4], cols=2, tile_wh=(960, 540)),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
    print(f"{len(items)} frames -> {out}/sheet_*.jpg")


def export(version: str) -> None:
    adds, drops = {}, set()
    for raw in OPS.read_text().splitlines() if OPS.exists() else []:
        line = raw.split("  #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        op, yid, frame, ref, *rest = line.split()
        if op == "add":
            adds.setdefault((yid, int(frame)), []).append(
                (int(ref.lstrip("c")), rest[0] if rest else "other")
            )
        elif op == "drop":
            drops.add((yid, int(frame), int(ref)))
        else:
            raise ValueError(raw)
    frames = []
    for sel in json.loads(SEL.read_text())["frames"]:
        yid, f = sel["youtube_id"], sel["frame"]
        boxes = [
            {"xyxy": [b["x1"], b["y1"], b["x2"], b["y2"]], "team": b["team"], "tid": b["tid"], "src": "track"}
            for b in frame_boxes(sel)
            if (yid, f, int(b["tid"])) not in drops
        ]
        cands = candidates(sel["clip"], f, frame_boxes(sel))
        for k, team in adds.get((yid, f), []):
            boxes.append({"xyxy": cands[k]["xyxy"], "team": team, "tid": None, "src": "detector-added"})
        frames.append({**sel, "boxes": boxes})
    out = Path("data/gold") / version
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": version,
        "geometry": "prefill (detector or SAM mask box); AP50 only in v0",
        "frames": frames,
    }
    (out / "boxes.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(frames)} frames, {sum(len(f['boxes']) for f in frames)} boxes -> {out / 'boxes.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "sheets", "export"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--version", default="v0")
    a = ap.parse_args(argv)
    {"select": lambda: select(a.n), "sheets": sheets, "export": lambda: export(a.version)}[a.cmd]()


if __name__ == "__main__":
    main()
