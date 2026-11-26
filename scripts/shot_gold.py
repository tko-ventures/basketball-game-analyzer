#!/usr/bin/env python3
"""Shot gold: pick the shots, cache detector boxes for the labeller, draw first-pass sheets, apply the first pass.

    uv run scripts/shot_gold.py select      # data/labels/shot_gold.json (refuses once anything is labelled)
    uv run bga run --config configs/shot-gold-prefill.yaml --layer shots --layer detect
    uv run scripts/shot_gold.py prefill     # data/labels/.cache/shot_gold/<sequence>.json, for /shotgold
    uv run scripts/shot_gold.py sheets      # runs/_scratch/shot_gold/{holders,events}/sheet_NNN.jpg and items.json
    uv run scripts/shot_gold.py apply       # data/labels/review/shot_gold_{holders,events}.tsv into the queue
    uv run scripts/shot_gold.py export      # confirmed answers -> data/gold/v1/shot_gold.json
    uv run scripts/shot_gold.py status

The logic of the working file lives in bga.eval.shot_gold (shared with the labeller's /shotgold page); see that
module and docs/gold.md, "Shot gold". The sheets are the implementer's first pass: every frame is read with its
neighbouring frames, and every shot attempt the detector hints at is read around the rim and around the release.

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from bga import pipeline
from bga.ball.base import bbox_iou
from bga.config import load_config
from bga.eval.shot_gold import (
    EXPORT_PATH,
    QUEUE_PATH,
    STRIDE,
    Status,
    build_queue,
    export_shot_gold,
    find_shot,
    load_queue,
    progress,
    save_queue,
    set_frame_label,
    set_shot_events,
)
from bga.stage import FrameReader

CONFIG = Path("configs/shot-gold-prefill.yaml")
BOXES_DIR = Path("data/labels/.cache/shot_gold")
SCRATCH = Path("runs/_scratch/shot_gold")
HOLDERS_TSV = Path("data/labels/review/shot_gold_holders.tsv")
EVENTS_TSV = Path("data/labels/review/shot_gold_events.tsv")
PEOPLE = ("player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block")
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cmd_select(args):
    if QUEUE_PATH.exists() and not args.force:
        p = progress(load_queue())
        if p["frames_done"] or p["shots_done"]:
            raise SystemExit(f"{QUEUE_PATH} already holds answers; --force would drop them")
    queue = build_queue(args.per_dev, args.per_held_out)
    save_queue(queue)
    print(f"{QUEUE_PATH}: {progress(queue)}")


def dedupe_people(rows):
    """Keep one box per person: the detector can box one player under two classes."""
    kept = []
    for r in sorted((r for r in rows if r[5] in PEOPLE and r[4] >= 0.2), key=lambda r: -r[4]):
        twin = next((k for k in kept if bbox_iou(k[:4], r[:4]) >= 0.8), None)
        if twin is None:
            kept.append(list(r[:6]))
        elif r[5] == "player-in-possession" and r[4] >= 0.3:
            twin[5] = r[5]
    return kept


def cmd_prefill(args):
    cfg = load_config(CONFIG)
    BOXES_DIR.mkdir(parents=True, exist_ok=True)
    for seq in pipeline.sequences(cfg):
        frames = pipeline.load_output(cfg, Path("runs"), seq, "detect")["frames"]
        out = {}
        for f, rows in frames.items():
            balls = sorted((r[:5] for r in rows if r[5] == "ball" and r[4] >= 0.1), key=lambda r: -r[4])[:3]
            rims = [r[:5] for r in rows if r[5] == "rim" and r[4] >= 0.3]
            basket = [r[:5] for r in rows if r[5] == "ball-in-basket" and r[4] >= 0.3]
            out[f] = {"people": dedupe_people(rows), "balls": balls, "rims": rims, "basket": basket}
        (BOXES_DIR / f"{seq.name}.json").write_text(json.dumps({"frames": out}), encoding="utf-8")
    print(f"{BOXES_DIR}: boxes of {len(pipeline.sequences(cfg))} sequences")


def boxes_of(sequence):
    return json.loads((BOXES_DIR / f"{sequence}.json").read_text(encoding="utf-8"))["frames"]


def video_of(shot):
    return json.loads((Path("data/labels") / f"{shot['clip']}.json").read_text(encoding="utf-8"))


def region(cx, cy, w, h, width, height):
    x = int(min(max(cx - w / 2, 0), width - w))
    y = int(min(max(cy - h / 2, 0), height - h))
    return x, y, int(w), int(h)


def crop(img, reg, size):
    x, y, w, h = reg
    return cv2.resize(img[y : y + h, x : x + w], size, interpolation=cv2.INTER_AREA if size[0] < w else cv2.INTER_CUBIC)


def text(img, s, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw_people(cell, people, reg, scale, pick=None):
    x0, y0, _, _ = reg
    for i, p in enumerate(people):
        a = (int((p[0] - x0) * scale), int((p[1] - y0) * scale))
        b = (int((p[2] - x0) * scale), int((p[3] - y0) * scale))
        color = (80, 220, 80) if pick == i else (255, 255, 255)
        cv2.rectangle(cell, a, b, color, 2 if pick == i else 1)
        text(cell, str(i), (a[0] + 2, max(a[1] - 4, 12)), 0.55, (0, 255, 255))


def kits(shot):
    meta = video_of(shot).get("meta", {})
    side = {s: (meta.get(s) or {}) for s in ("home", "away")}
    return {s: f"{side[s].get('team', s)} ({side[s].get('kit', '?')} kit)" for s in side}


def holder_items(queue):
    for shot in queue["shots"]:
        boxes = boxes_of(shot["sequence"])
        for fr in shot["frames"]:
            b = boxes.get(str(fr["frame"]), {"people": [], "balls": []})
            balls = [x for x in b["balls"] if x[4] >= 0.15][:2]
            poss = next((p for p in b["people"] if p[5] == "player-in-possession"), None)
            if balls:
                centre = ((balls[0][0] + balls[0][2]) / 2, (balls[0][1] + balls[0][3]) / 2)
            elif poss:
                centre = ((poss[0] + poss[2]) / 2, (poss[1] + poss[3]) / 2)
            else:
                centre = None
            yield {
                "id": f"{shot['sequence']}:{fr['frame']}",
                "sequence": shot["sequence"],
                "frame": fr["frame"],
                "people": b["people"],
                "balls": balls,
                "centre": centre,
            }


def holder_row(read, shot, item, width, height):
    """One frame: the labelled frame large with numbered boxes, and the frames 0.2 and 0.4 s either side."""
    reg = region(*item["centre"], 640, 400, width, height) if item["centre"] else (0, 0, width, height)
    scale = 768 / reg[2]
    big = crop(read(item["frame"]), reg, (768, 480))
    draw_people(big, item["people"], reg, scale)
    for letter, b in zip("ab", item["balls"], strict=False):
        c = (int(((b[0] + b[2]) / 2 - reg[0]) * scale), int(((b[1] + b[3]) / 2 - reg[1]) * scale))
        cv2.circle(big, c, int(max(b[2] - b[0], b[3] - b[1]) * scale / 2 + 9), (0, 145, 255), 1, cv2.LINE_AA)
        text(big, letter, (c[0] + 14, c[1] - 12), 0.6, (0, 145, 255))
    text(big, "LABEL THIS FRAME", (8, 20), 0.55, (0, 255, 255))
    step = round(0.2 * shot["fps"] / STRIDE) * STRIDE
    cells = []
    for k in (-2, -1, 1, 2):
        f = min(max(item["frame"] + k * step, shot["start"]), shot["end"])
        cell = crop(read(f), reg, (384, 240))
        text(cell, f"{k * 0.2:+.1f} s", (6, 18), 0.5)
        cells.append(cell)
    side = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    row = np.hstack([big, side])
    cap = np.zeros((26, row.shape[1], 3), np.uint8)
    k = kits(shot)
    text(cap, f"ID {item['id']}   home = {k['home']}   away = {k['away']}", (6, 18), 0.5, (0, 255, 255))
    return np.vstack([cap, row])


def rim_candidates(shot, boxes):
    """Frames where the detector's ball comes near a rim, grouped into moments at least 1 s apart."""
    hits = []
    for f in sorted(int(k) for k in boxes):
        b = boxes[str(f)]
        if not b["rims"]:
            continue
        ball = next((x for x in b["balls"] if x[4] >= 0.2), None)
        best = None
        for rim in b["rims"]:
            rc, rw = ((rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2), max(rim[2] - rim[0], 1.0)
            if ball is not None:
                d = np.hypot((ball[0] + ball[2]) / 2 - rc[0], (ball[1] + ball[3]) / 2 - rc[1]) / rw
                if d < 2.0 and (best is None or d < best[0]):
                    best = (float(d), rc, rw)
            if any(bbox_iou(k[:4], rim[:4]) > 0 for k in b["basket"]) and best is None:
                best = (0.0, rc, rw)
        if best:
            hits.append((f, *best))
    groups, cur = [], []
    for h in hits:
        if cur and h[0] - cur[-1][0] > shot["fps"]:
            groups.append(cur)
            cur = []
        cur.append(h)
    if cur:
        groups.append(cur)
    return [min(g, key=lambda h: h[1]) for g in groups]


def release_guess(shot, boxes, rim_frame):
    """Walk back from the rim frame to the last frame where the ball sits in a player's box."""
    f = rim_frame - STRIDE
    while f >= max(shot["start"], rim_frame - int(2.5 * shot["fps"])):
        b = boxes.get(str(f))
        ball = next((x for x in b["balls"] if x[4] >= 0.2), None) if b else None
        if ball is not None:
            cx, cy = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
            inside = [i for i, p in enumerate(b["people"]) if p[0] - 8 <= cx <= p[2] + 8 and p[1] - 8 <= cy <= p[3] + 8]
            if inside and rim_frame - f >= 2 * STRIDE:
                poss = [i for i in inside if b["people"][i][5] != "player"]
                return min(f + STRIDE, rim_frame), f, (poss or inside)[0]
        f -= STRIDE
    return max(shot["start"], rim_frame - 45), None, None


def event_items(queue):
    for shot in queue["shots"]:
        boxes = boxes_of(shot["sequence"])
        for n, (rim_frame, _, rc, rw) in enumerate(rim_candidates(shot, boxes)):
            release, held_frame, held_idx = release_guess(shot, boxes, rim_frame)
            shooter = None
            if held_idx is not None:
                prev = boxes[str(held_frame)]["people"][held_idx]
                now = boxes.get(str(release), {"people": []})["people"]
                ious = [bbox_iou(prev[:4], p[:4]) for p in now]
                shooter = int(np.argmax(ious)) if ious and max(ious) > 0.3 else None
            yield {
                "id": f"{shot['sequence']}#{n}",
                "sequence": shot["sequence"],
                "rim_frame": rim_frame,
                "rim_centre": list(rc),
                "rim_width": rw,
                "release_frame": release,
                "shooter": shooter,
                "people": boxes.get(str(release), {"people": []})["people"],
            }


def event_sheet(read, shot, item, width, height):
    """One shot attempt: the rim over 0.9 s, the release over 0.4 s, and the whole release frame."""
    clamp = lambda f: min(max(f, shot["start"]), shot["end"])  # one-line clamp used three times below
    rc, rim = item["rim_centre"], item["rim_frame"]
    reg = region(rc[0], rc[1] + 40, 360, 300, width, height)
    cells = []
    for off in (-15, -10, -5, 0, 5, 10, 20, 30):
        cell = crop(read(clamp(rim + off)), reg, (192, 160))
        text(cell, f"f{clamp(rim + off)}", (4, 16), 0.45, (0, 255, 255) if off == 0 else (255, 255, 255))
        cells.append(cell)
    rows = [np.hstack(cells)]
    rel, ppl = item["release_frame"], item["people"]
    if item["shooter"] is not None:
        p = ppl[item["shooter"]]
        centre = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2 - 40)
    else:
        centre = (rc[0], rc[1] + 200)
    reg2 = region(*centre, 560, 420, width, height)
    cells = []
    for off in (-10, -5, 0, 5, 10):
        cell = crop(read(clamp(rel + off)), reg2, (307, 230))
        text(cell, f"f{clamp(rel + off)}", (4, 16), 0.45, (0, 255, 255) if off == 0 else (255, 255, 255))
        cells.append(cell)
    strip = np.hstack(cells)
    rows.append(cv2.copyMakeBorder(strip, 0, 0, 0, 1536 - strip.shape[1], cv2.BORDER_CONSTANT))
    whole = cv2.resize(read(rel), (1536, 864), interpolation=cv2.INTER_AREA)
    draw_people(whole, ppl, (0, 0, width, height), 1536 / width, pick=item["shooter"])
    text(whole, f"whole frame f{rel} (proposed release); green box = proposed shooter", (8, 22), 0.6, (0, 255, 255))
    rows.append(whole)
    cap = np.zeros((26, 1536, 3), np.uint8)
    k = kits(shot)
    text(cap, f"ID {item['id']}   home = {k['home']}   away = {k['away']}   rim f{rim}", (6, 18), 0.5, (0, 255, 255))
    return np.vstack([cap, *rows])


def cmd_sheets(args):
    queue = load_queue()
    for kind in ("holders", "events"):
        (SCRATCH / kind).mkdir(parents=True, exist_ok=True)
    readers, sizes = {}, {}

    def reader(shot):
        if shot["sequence"] not in readers:
            labels = video_of(shot)
            readers[shot["sequence"]] = FrameReader(labels["clip"])
            sizes[shot["sequence"]] = (int(labels["width"]), int(labels["height"]))
        return readers[shot["sequence"]], *sizes[shot["sequence"]]

    items = list(holder_items(queue))
    for n in range(0, len(items), 4):
        rows = []
        for it in items[n : n + 4]:
            shot = find_shot(queue, it["sequence"])
            read, w, h = reader(shot)
            rows.append(holder_row(read, shot, it, w, h))
            it["sheet"] = f"sheet_{n // 4:03d}.jpg"
        cv2.imwrite(
            str(SCRATCH / "holders" / f"sheet_{n // 4:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
    (SCRATCH / "holders" / "items.json").write_text(json.dumps(items), encoding="utf-8")
    events = list(event_items(queue))
    for n, it in enumerate(events):
        shot = find_shot(queue, it["sequence"])
        read, w, h = reader(shot)
        it["sheet"] = f"sheet_{n:03d}.jpg"
        cv2.imwrite(
            str(SCRATCH / "events" / it["sheet"]), event_sheet(read, shot, it, w, h), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
    (SCRATCH / "events" / "items.json").write_text(json.dumps(events), encoding="utf-8")
    for r in readers.values():
        r.close()
    print(f"{SCRATCH}: {len(items)} frames on {(len(items) + 3) // 4} holder sheets, {len(events)} event sheets")


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t") if r.get("id") and not r["id"].startswith("#")]


def side_of(shot, kit):
    meta = video_of(shot).get("meta", {})
    return next((s for s in ("home", "away") if (meta.get(s) or {}).get("kit") == kit), None)


def apply_holders(queue):
    items = {it["id"]: it for it in json.loads((SCRATCH / "holders" / "items.json").read_text(encoding="utf-8"))}
    settled = flagged = 0
    for r in read_tsv(HOLDERS_TSV):
        it, shot = items[r["id"]], find_shot(queue, r["id"].split(":")[0])
        fr = next(f for f in shot["frames"] if f["frame"] == it["frame"])
        if fr["status"] == Status.DONE:
            continue
        ball = it["balls"]["ab".index(r["ball"])][:4] if r["ball"] in ("a", "b") else None
        team = side_of(shot, r["kit"])
        holder = None
        if r["holder"].isdigit() and team:
            holder = {"xyxy": it["people"][int(r["holder"])][:4], "team": team}
        clear = r["sure"] == "yes" and r["ball"] != "other" and (r["holder"] == "none" or holder is not None)
        body = {"sequence": shot["sequence"], "frame": it["frame"], "ball": ball, "holder": holder, "note": r["note"]}
        set_frame_label(queue, body, Status.FIRST_PASS if clear else Status.FLAGGED)
        unread = not clear and ball is None and holder is None
        fr["proposal"] = None if unread else {"ball": fr["ball"], "holder": fr["holder"], "note": r["note"]}
        if not clear:
            fr["ball"], fr["holder"] = None, None
        settled, flagged = settled + clear, flagged + (not clear)
    return settled, flagged


def apply_events(queue):
    items = {it["id"]: it for it in json.loads((SCRATCH / "events" / "items.json").read_text(encoding="utf-8"))}
    by_shot, notes = {}, {}
    for r in read_tsv(EVENTS_TSV):
        it, shot = items[r["id"]], find_shot(queue, r["id"].split("#")[0])
        by_shot.setdefault(shot["sequence"], [])
        if r["note"]:
            notes.setdefault(shot["sequence"], []).append(r["note"])
        if r["is_shot"] != "yes":
            continue
        rim = int(r["rim_frame"]) if r["rim_frame"].isdigit() else it["rim_frame"]
        release = min(int(r["release_frame"]) if r["release_frame"].isdigit() else it["release_frame"], rim)
        team = side_of(shot, r["kit"])
        shooter = None
        if r["shooter"].isdigit() and team and release == it["release_frame"] and r["release_frame"] != "off":
            shooter = {"xyxy": it["people"][int(r["shooter"])][:4], "team": team}
        by_shot[shot["sequence"]].append({
            "release_frame": release,
            "rim_frame": rim,
            "outcome": r["outcome"] if r["outcome"] in ("made", "missed") else "missed",
            "points": 3 if r["points"] == "3" else 2,
            "shooter": shooter,
            "note": "",
        })
    done = 0
    for shot in queue["shots"]:
        if shot["events_status"] == Status.DONE:
            continue
        set_shot_events(
            queue, {"sequence": shot["sequence"], "events": by_shot.get(shot["sequence"], [])}, Status.FIRST_PASS
        )
        shot["events_note"] = " ".join(notes.get(shot["sequence"], []))
        done += 1
    return done


def cmd_apply(args):
    queue = load_queue()
    if HOLDERS_TSV.exists():
        settled, flagged = apply_holders(queue)
        print(f"holders: {settled} settled by the first pass, {flagged} flagged for Tomasz")
    if EVENTS_TSV.exists():
        print(f"events: proposals written for {apply_events(queue)} shots")
    save_queue(queue)
    print(f"{QUEUE_PATH}: {progress(queue)}")


def cmd_export(args):
    doc = export_shot_gold()
    print(
        f"{EXPORT_PATH}: {len(doc['shots'])} shots, {len(doc['verified_sequences'])} verified, "
        f"{len(doc['events'])} attempts, {len(doc['frames'])} frames"
    )


def cmd_status(args):
    queue = load_queue()
    print(progress(queue))
    print("attempts:", sum(len(s["events"]) for s in queue["shots"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--per-dev", type=int, default=6)
    sel.add_argument("--per-held-out", type=int, default=8)
    sel.add_argument("--force", action="store_true")
    sel.set_defaults(fn=cmd_select)
    for name, fn in (
        ("prefill", cmd_prefill),
        ("sheets", cmd_sheets),
        ("apply", cmd_apply),
        ("export", cmd_export),
        ("status", cmd_status),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
