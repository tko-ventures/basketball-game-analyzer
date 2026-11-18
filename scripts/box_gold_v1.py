#!/usr/bin/env python3
"""Box gold v1: tight, 10-class boxes on the 100 box-gold frames, prefilled and flagged for Tomasz's review.

    uv run scripts/box_gold_v1.py prefill            # runs/box_gold/v1/sam_input.json and, with SAM results, the queue
    uv run --extra remote modal run src/bga/remote/sam2_boxmask.py
    uv run scripts/box_gold_v1.py prefill            # again: merges SAM into data/labels/box_gold_v1.json
    uv run scripts/box_gold_v1.py crops              # runs/box_gold/v1/crops/*.jpg for the implementer's first pass
    uv run scripts/box_gold_v1.py sheets             # runs/box_gold/v1/sheets/*.jpg, whole frames with numbered boxes
    uv run scripts/box_gold_v1.py apply              # applies data/labels/review/box_gold_v1.ops.txt to the queue
    uv run scripts/box_gold_v1.py candidates         # recompute candidates (incl. missed people) on frames not done
    uv run scripts/box_gold_v1.py export             # data/gold/v1/boxes.json from the frames Tomasz marked done

The queue (data/labels/box_gold_v1.json) is what the labeller's /boxes page edits. Each frame starts from:
- the v0 person boxes (track gold plus reviewer-added detector boxes), keeping their track id and team;
- the class of the best-matching prefill detection when it is an action class (possession, jump shot, layup or dunk,
  shot block), else player, or referee for referee tracks;
- the prefill detector's ball, ball-in-basket, rim and number boxes at score >= KEEP; lower ones down to 0.1 are
  candidates the page can show and adopt with one click, as are its player and referee boxes that overlap no person
  box (IoU < PERSON_FREE), for people the tracks missed.

Geometry: where SAM 2.1's mask box agrees with the prefill box (IoU >= SAM_AGREE) the mask box is used, being the
tighter of the two; otherwise the prefill box stays, with the mask box offered as the alternative and a hint (not a
flag: on the 2026-11-18 check SAM mostly cut off arms, feet and took in rim brackets where the two disagreed). A box
whose class, existence or geometry needs a decision carries a flag with the reason, and the page sorts flagged boxes
first. Ops file lines (applied in order, only to frames still "todo"):
    cls  <youtube_id> <frame> <box id> <class>       change a box's class
    drop <youtube_id> <frame> <box id>               remove a box that is not an object of the schema
    adopt <youtube_id> <frame> c<k> <class>          promote candidate k to a box
    flag <youtube_id> <frame> <box id|-> <note...>   add a note for the reviewer, to a box or ("-") the frame
    noflag <youtube_id> <frame> <box id> [words]     clear a box's flags after checking it, or only those starting with words

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.review.sheets import contact_sheet
from bga.types import DETECTION_CLASSES

SEL = Path("data/labels/box_gold.json")
V0 = Path("data/gold/v0/boxes.json")
QUEUE = Path("data/labels/box_gold_v1.json")
OPS = Path("data/labels/review/box_gold_v1.ops.txt")
OUT = Path("runs/box_gold/v1")
SAM_IN, SAM_OUT = OUT / "sam_input.json", OUT / "sam_boxes.json"
PREFILL = "basketball-player-detection-3-ycjdo_v4_c0.1_lb90_ar1"
PERSON = {c for c in DETECTION_CLASSES if c.startswith("player")} | {"referee"}
ACTION = {"player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}
OBJECTS = {"ball", "ball-in-basket", "rim", "number"}
KEEP = 0.3  # objects at or above this score become boxes; 0.1 to KEEP are candidates
SURE = 0.6  # objects below this score are flagged
ACTION_MIN = 0.3
SAM_AGREE = 0.85
NMS_IOU = 0.5
PERSON_FREE = 0.3  # a person detection overlapping no person box below this IoU is a candidate
SAM_CLASSES = PERSON | {"ball", "ball-in-basket", "rim"}  # SAM does not separate digits from the shirt
CLS_BGR = {
    "player": (80, 220, 80),
    "player-in-possession": (0, 140, 255),
    "player-jump-shot": (255, 120, 0),
    "player-layup-dunk": (255, 0, 160),
    "player-shot-block": (160, 0, 255),
    "referee": (255, 255, 0),
    "ball": (0, 100, 255),
    "ball-in-basket": (0, 0, 255),
    "rim": (0, 255, 255),
    "number": (255, 0, 255),
}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def key(yid: str, frame: int) -> str:
    return f"{yid}_{frame}"


def detections(clip: str, frames: set[int]) -> dict[int, list[dict]]:
    out = {}
    for line in (Path("runs") / clip / "prefill" / PREFILL / "detections.jsonl").open():
        rec = json.loads(line)
        if rec["frame_idx"] in frames:
            out[rec["frame_idx"]] = rec["detections"]
    return out


def video_of(clip: str) -> str:
    return json.loads((Path("data/labels") / f"{clip}.json").read_text())["clip"]


def draft_frame(sel: dict, v0: dict, dets: list[dict]) -> dict:
    boxes, cands = [], []
    people = [d for d in dets if d["class"] in PERSON]
    for b in v0["boxes"]:
        flags = []
        best = max(people, key=lambda d: iou(d["xyxy"], b["xyxy"]), default=None)
        cls = "referee" if b["team"] == "ref" else "player"
        if cls == "player" and best and iou(best["xyxy"], b["xyxy"]) >= 0.5:
            if best["class"] in ACTION and best["score"] >= ACTION_MIN:
                cls = best["class"]
                flags.append(f"action from detector ({best['score']:.2f})")
            elif best["class"] == "referee":
                flags.append("detector says referee")
        if b["src"] == "detector-added":
            flags.append("added in v0 review")
        boxes.append({
            "xyxy": [round(v, 1) for v in b["xyxy"]],
            "cls": cls,
            "team": b["team"],
            "tid": b["tid"],
            "src": "v0",
            "flags": flags,
            "hints": [],
            "alt": {},
        })
    kept: list[dict] = []  # per-class NMS: the prefill ran with IoU 0.9, so one ball can carry three boxes
    for d in sorted((d for d in dets if d["class"] in OBJECTS), key=lambda d: -d["score"]):
        if any(k["class"] == d["class"] and iou(k["xyxy"], d["xyxy"]) >= NMS_IOU for k in kept):
            continue
        kept.append(d)
        rec = {"xyxy": [round(v, 1) for v in d["xyxy"]], "cls": d["class"], "score": round(d["score"], 3)}
        if d["score"] >= KEEP:
            flags = [] if d["score"] >= SURE else [f"low score ({d['score']:.2f})"]
            boxes.append({**rec, "team": "", "tid": None, "src": "det", "flags": flags, "hints": [], "alt": {}})
        else:
            cands.append(rec)
    taken = [b["xyxy"] for b in boxes if b["cls"] in PERSON]
    for d in sorted(people, key=lambda d: -d["score"]):  # people the tracks missed, one click to adopt
        if all(iou(d["xyxy"], x) < PERSON_FREE for x in taken):
            taken.append(d["xyxy"])
            cls = "referee" if d["class"] == "referee" else "player"
            cands.append({"xyxy": [round(v, 1) for v in d["xyxy"]], "cls": cls, "score": round(d["score"], 3)})
    for i, b in enumerate(boxes):
        b["id"] = i
    return {
        "youtube_id": sel["youtube_id"],
        "clip": sel["clip"],
        "video": video_of(sel["clip"]),
        "shot": sel["shot"],
        "frame": sel["frame"],
        "status": "todo",
        "note": "",
        "boxes": boxes,
        "candidates": cands,
    }


def merge_sam(fr: dict, sam: dict) -> None:
    by_box = {tuple(p): m for p, m in zip(sam["boxes"], sam["masks"], strict=True)}
    for b in fr["boxes"]:
        if b["cls"] not in SAM_CLASSES:
            continue
        if tuple(b["xyxy"]) not in by_box:
            raise SystemExit(f"no SAM mask for {fr['youtube_id']} {fr['frame']} {b['xyxy']}: rerun sam2_boxmask")
        m = by_box[tuple(b["xyxy"])]
        if m is None:
            b["hints"].append("SAM found no mask")
            continue
        sb = [float(v) for v in m[:4]]
        b["alt"]["sam"] = sb
        if iou(sb, b["xyxy"]) >= SAM_AGREE:
            b["alt"]["prefill"] = b["xyxy"]
            b["xyxy"], b["src"] = sb, b["src"] + "+sam"
        else:
            b["hints"].append(f"SAM box differs (IoU {iou(sb, b['xyxy']):.2f}): A swaps")


def flag_overlaps(fr: dict) -> None:
    people = [b for b in fr["boxes"] if b["cls"] in PERSON]
    for i, a in enumerate(people):
        for b in people[i + 1 :]:
            if iou(a["xyxy"], b["xyxy"]) >= 0.3:
                for x, y in ((a, b), (b, a)):
                    note = f"overlaps box {y['id']}"
                    if note not in x["flags"]:
                        x["flags"].append(note)


def prefill(force: bool) -> None:
    sels = json.loads(SEL.read_text())["frames"]
    v0 = {key(f["youtube_id"], f["frame"]): f for f in json.loads(V0.read_text())["frames"]}
    by_clip: dict[str, set[int]] = {}
    for s in sels:
        by_clip.setdefault(s["clip"], set()).add(s["frame"])
    dets = {c: detections(c, fr) for c, fr in by_clip.items()}
    frames = [draft_frame(s, v0[key(s["youtube_id"], s["frame"])], dets[s["clip"]][s["frame"]]) for s in sels]
    OUT.mkdir(parents=True, exist_ok=True)
    SAM_IN.write_text(
        json.dumps({
            "frames": [
                {
                    "key": key(f["youtube_id"], f["frame"]),
                    "video": f["video"],
                    "frame": f["frame"],
                    "boxes": [b["xyxy"] for b in f["boxes"] if b["cls"] in SAM_CLASSES],
                }
                for f in frames
            ]
        })
    )
    n = sum(len(f["boxes"]) for f in frames)
    print(f"{len(frames)} frames, {n} boxes -> {SAM_IN}")
    if not SAM_OUT.exists():
        print(f"no {SAM_OUT} yet: run src/bga/remote/sam2_boxmask.py on Modal, then prefill again")
        return
    if QUEUE.exists() and not force:
        done = sum(f["status"] == "done" for f in json.loads(QUEUE.read_text())["frames"])
        raise SystemExit(f"{QUEUE} exists ({done} frames done); pass --force to rebuild it from scratch")
    sam = json.loads(SAM_OUT.read_text())
    for f in frames:
        merge_sam(f, sam["frames"][key(f["youtube_id"], f["frame"])])
        flag_overlaps(f)
    doc = {
        "version": "v1",
        "note": "Box gold v1 queue; edited in the labeller at /boxes. Conventions in docs/gold.md, 'Box gold v1'.",
        "classes": list(DETECTION_CLASSES),
        "sam": {k: sam[k] for k in ("model", "sam2_sha")},
        "frames": frames,
        "meta": {"review_seconds": {}},
    }
    QUEUE.write_text(json.dumps(doc, indent=1) + "\n")
    flagged = sum(bool(b["flags"]) for f in frames for b in f["boxes"])
    print(f"{QUEUE}: {n} boxes, {flagged} flagged, {sum(len(f['candidates']) for f in frames)} candidates")


def sheets() -> None:
    doc = json.loads(QUEUE.read_text())
    out = OUT / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    caps = {}
    for n, fr in enumerate(doc["frames"]):
        cap = caps.setdefault(fr["video"], cv2.VideoCapture(fr["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame"])
        img = cap.read()[1]
        for k, c in enumerate(fr["candidates"]):
            x1, y1, x2, y2 = (int(v) for v in c["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(img, f"c{k}:{c['cls'][:4]}", (x1, y2 + 14), 0, 0.45, (200, 200, 200), 1)
        for b in fr["boxes"]:
            x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
            col = CLS_BGR[b["cls"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if b["flags"] else 1)
            tag = f"{b['id']}{'*' if b['flags'] else ''}"
            cv2.putText(img, tag, (x1, max(12, y1 - 4)), 0, 0.55, col, 2)
        head = f"{n:02d} {fr['youtube_id']} s{fr['shot']} f{fr['frame']}"
        # whole frame, then the two halves at full resolution, so small objects stay legible
        tiles = [(img, head), (img[:, :960], head + " left"), (img[:, 960:], head + " right")]
        cv2.imwrite(
            str(out / f"{n:02d}.jpg"),
            contact_sheet(tiles, cols=1, tile_wh=(1920, 1080)),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
    print(f"{len(doc['frames'])} sheets -> {out}")


def crop_tile(img, b: dict, label: str, ctx: float, tile: int = 300):
    x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
    p = int(max(x2 - x1, y2 - y1) * ctx) + 10
    X1, Y1 = max(0, x1 - p), max(0, y1 - p)
    c = img[Y1 : y2 + p, X1 : x2 + p].copy()
    s = tile / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    cv2.rectangle(
        c,
        (int((x1 - X1) * s), int((y1 - Y1) * s)),
        (int((x2 - X1) * s), int((y2 - Y1) * s)),
        CLS_BGR.get(b["cls"], (255, 255, 255)),
        2,
    )
    return c, label


def crops() -> None:
    """Crop sheets for the implementer's first pass: every flagged object, action box and candidate."""
    doc = json.loads(QUEUE.read_text())
    out = OUT / "crops"
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {"objects": [], "people": [], "candidates": []}
    caps = {}
    for n, fr in enumerate(doc["frames"]):
        cap = caps.setdefault(fr["video"], cv2.VideoCapture(fr["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame"])
        img = cap.read()[1]
        for b in fr["boxes"]:
            want = [f for f in b["flags"] if not f.startswith("overlaps")]
            if not want:
                continue
            g = "objects" if b["cls"] in OBJECTS else "people"
            ctx = 1.5 if g == "objects" else 0.5
            groups[g].append(crop_tile(img, b, f"{n}:{b['id']} {b['cls'][:13]} {b.get('score') or ''}", ctx))
        for k, c in enumerate(fr["candidates"]):
            groups["candidates"].append(crop_tile(img, c, f"{n}:c{k} {c['cls'][:13]} {c['score']}", 1.5))
    for g, items in groups.items():
        for s in range(0, len(items), 30):
            cv2.imwrite(
                str(out / f"{g}_{s // 30:02d}.jpg"),
                contact_sheet(items[s : s + 30], cols=6, tile_wh=(300, 300)),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
        print(f"{g}: {len(items)} crops")


def apply_ops() -> None:
    doc = json.loads(QUEUE.read_text())
    frames = {key(f["youtube_id"], f["frame"]): f for f in doc["frames"]}
    n = 0
    for raw in OPS.read_text().splitlines() if OPS.exists() else []:
        line = raw.split("  #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        op, yid, frame, ref, *rest = line.split()
        fr = frames[key(yid, int(frame))]
        if fr["status"] != "todo":
            continue
        box = next((b for b in fr["boxes"] if str(b["id"]) == ref), None)
        if op == "cls":
            if rest[0] not in DETECTION_CLASSES:
                raise ValueError(raw)
            box["cls"] = rest[0]
            box["flags"] = [f for f in box["flags"] if not f.startswith("action from")]
        elif op == "drop":
            fr["boxes"].remove(box)
        elif op == "adopt":
            c = fr["candidates"][int(ref.lstrip("c"))]
            nid = max((b["id"] for b in fr["boxes"]), default=-1) + 1
            fr["boxes"].append({
                "xyxy": c["xyxy"],
                "cls": rest[0],
                "team": "",
                "tid": None,
                "src": "det-candidate",
                "flags": [],
                "hints": [f"adopted candidate ({c['score']:.2f})"],
                "alt": {},
                "id": nid,
            })
        elif op == "flag":
            note = "implementer: " + " ".join(rest)
            if ref == "-":
                fr["note"] = (fr["note"] + "; " if fr["note"] else "") + note
            elif note not in box["flags"]:
                box["flags"].append(note)
        elif op == "noflag":  # all flags, or only those starting with the given words
            box["flags"] = [f for f in box["flags"] if rest and not f.startswith(" ".join(rest))]
        else:
            raise ValueError(raw)
        n += 1
    QUEUE.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{n} ops applied -> {QUEUE}")


def refresh_candidates() -> None:
    """Recompute the candidates of every frame not done, leaving boxes untouched (reload the /boxes page after)."""
    doc = json.loads(QUEUE.read_text())
    sels = json.loads(SEL.read_text())["frames"]
    v0 = {key(f["youtube_id"], f["frame"]): f for f in json.loads(V0.read_text())["frames"]}
    by_clip: dict[str, set[int]] = {}
    for s in sels:
        by_clip.setdefault(s["clip"], set()).add(s["frame"])
    dets = {c: detections(c, fr) for c, fr in by_clip.items()}
    n = 0
    for fr in doc["frames"]:
        if fr["status"] == "done":
            continue
        sel = {k: fr[k] for k in ("youtube_id", "clip", "shot", "frame")}
        cands = draft_frame(sel, v0[key(fr["youtube_id"], fr["frame"])], dets[fr["clip"]][fr["frame"]])["candidates"]

        def free(c, boxes=fr["boxes"]):
            person = c["cls"] in PERSON
            return all(
                iou(c["xyxy"], b["xyxy"]) < (PERSON_FREE if person else NMS_IOU)
                for b in boxes
                if (b["cls"] in PERSON) == person and (person or b["cls"] == c["cls"])
            )

        fr["candidates"] = [c for c in cands if free(c)]
        n += len(fr["candidates"])
    QUEUE.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{n} candidates over the frames not done -> {QUEUE}")


def export(version: str) -> None:
    doc = json.loads(QUEUE.read_text())
    done = [f for f in doc["frames"] if f["status"] == "done"]
    frames = [
        {k: f[k] for k in ("youtube_id", "clip", "shot", "frame")}
        | {"boxes": [{"xyxy": b["xyxy"], "cls": b["cls"], "team": b["team"], "tid": b["tid"]} for b in f["boxes"]]}
        for f in done
    ]
    out = Path("data/gold") / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "boxes.json").write_text(
        json.dumps(
            {
                "version": version,
                "geometry": "reviewed by hand (labeller /boxes); visible extent, conventions in docs/gold.md",
                "classes": doc["classes"],
                "frames": frames,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(frames)} of {len(doc['frames'])} frames done -> {out / 'boxes.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefill", "sheets", "crops", "apply", "candidates", "export"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--version", default="v1")
    a = ap.parse_args(argv)
    {
        "prefill": lambda: prefill(a.force),
        "sheets": sheets,
        "crops": crops,
        "apply": apply_ops,
        "candidates": refresh_candidates,
        "export": lambda: export(a.version),
    }[a.cmd]()


if __name__ == "__main__":
    main()
