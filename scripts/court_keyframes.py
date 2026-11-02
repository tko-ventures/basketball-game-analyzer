#!/usr/bin/env python3
"""Choose court keyframes for Tomasz's session and render the sheets the implementer flags them on.

    uv run scripts/court_keyframes.py pick      # runs/court_keyframes/candidates.json + sheet_NN.jpg
    uv run scripts/court_keyframes.py queue     # data/labels/court/queue.json from the reviewed flags

pick: per hall, central shots are those where the NBA court model has >= 4 confident points on most
sampled frames (or, for gold games, the gold shots and their camera tags). Gold games give 6 frames
per hall for evaluation, spread over different shots; other clips give up to --train-per-hall
training keyframes from different shots, each in the middle of a central stretch. Every candidate is
drawn with the NBA model's FIBA lines on a sheet (6 per sheet, numbered).

queue: reads runs/court_keyframes/review.txt, one line per candidate number:
    <n> ok        lines sit on the painted lines: accepted as is, not queued (training only)
    <n> fix <why> queued for Tomasz with the reason shown on the page
    <n> drop      not usable (close-up, replay, no floor)
Gold candidates are always queued: evaluation frames need a human check even when they look right.
A candidate without an NBA mapping starts from the hall's best mapping, or a generic broadcast view.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.review.sheets import contact_sheet  # noqa: E402

GEO = json.loads((Path(__file__).resolve().parents[1] / "tools/labeler/static/fiba_court.json").read_text())
OUT = Path("runs/court_keyframes")
QUEUE = Path("data/labels/court/queue.json")
# Clips whose home side plays elsewhere: hall names by YouTube id, the rest by home team.
HALL_BY_ID = {"Tnuscrm9RyU": "Superpuchar venue (2026)"}
HALL_ALIAS = {
    "MKS": "Dąbrowa Górnicza",
    "MKS DG": "Dąbrowa Górnicza",
    "MKS Dąbrowa Górnicza": "Dąbrowa Górnicza",
}


def hall_of(stem: str, labels: dict) -> str:
    yid = stem[9:20]
    if yid in HALL_BY_ID:
        return HALL_BY_ID[yid]
    home = (labels.get("meta", {}).get("home") or {}).get("team", "?")
    return HALL_ALIAS.get(home, home)


def draw_court(img: np.ndarray, H, color=(255, 0, 255), width=3) -> np.ndarray:
    if H is None:
        return img
    G = np.linalg.inv(np.asarray(H))

    def P(X, Y):
        p = G @ [X, Y, 1.0]
        return (int(p[0] / p[2]), int(p[1] / p[2])) if p[2] > 1e-9 else None

    V = GEO["vertices"]
    lines = [
        [(V[a][0] + (V[b][0] - V[a][0]) * t / 20, V[a][1] + (V[b][1] - V[a][1]) * t / 20) for t in range(21)]
        for a, b in GEO["edges"]
    ] + GEO["polylines"]
    for pl in lines:
        pts = [p for p in (P(*q) for q in pl) if p and abs(p[0]) < 20000 and abs(p[1]) < 20000]
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, width)
    return img


def pick(train_per_hall: int, seed: int = 7) -> None:
    gold = json.loads(Path("data/labels/gold.json").read_text())
    gold_ids = {g["youtube_id"]: g for g in gold["games"]}
    rng = np.random.default_rng(seed)
    cands = []
    for path in sorted(Path("data/labels").glob("2*.json")):
        stem, labels = path.stem, json.loads(path.read_text())
        pre_p = next((Path("runs") / stem / "court_prefill").glob("*.jsonl"), None)
        if pre_p is None:
            continue
        pre = [json.loads(line) for line in pre_p.open()]
        yid, hall = stem[9:20], hall_of(stem, labels)
        by_shot = defaultdict(list)
        for r in pre:
            by_shot[r["shot"]].append(r)
        if yid in gold_ids:
            shots = [s["shot"] for s in gold_ids[yid]["shots"]]
            others = [
                i for i, s in enumerate(labels["shots"]) if s.get("camera") == "central" and i not in shots
            ]
            shots += list(rng.permutation(others))[: max(0, 6 - len(shots))]
            want, purpose = 6, "gold"
        else:
            long = {si: rows for si, rows in by_shot.items() if len(rows) >= 6}
            score = {si: np.mean([r["n_conf"] for r in rows]) for si, rows in long.items()}
            shots = [si for si, rows in long.items() if np.mean([r["n_conf"] >= 4 for r in rows]) >= 0.6]
            if not shots:  # the NBA model barely sees this hall: take its most court-like long shots
                shots = sorted(score, key=lambda si: -score[si])[: 2 * train_per_hall]
            shots = list(rng.permutation(shots))[:train_per_hall]
            want, purpose = len(shots), "train"
        for si in shots[:want]:
            rows = by_shot.get(si) or []
            if not rows:
                continue
            r = min(rows, key=lambda r: abs(r["frame"] - np.median([x["frame"] for x in rows])))
            cands.append(
                {
                    "clip": stem,
                    "youtube_id": yid,
                    "hall": hall,
                    "shot": int(si),
                    "frame": r["frame"],
                    "video": labels["clip"],
                    "purpose": purpose,
                    "prefill_H": r["H"],
                    "n_conf": r["n_conf"],
                }
            )
    # Training keyframes: at most train_per_hall per hall over all of its clips.
    keep, per_hall = [], defaultdict(int)
    for c in cands:
        if c["purpose"] == "train":
            if per_hall[c["hall"]] >= train_per_hall:
                continue
            per_hall[c["hall"]] += 1
        keep.append(c)
    for n, c in enumerate(keep):
        c["n"] = n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(json.dumps(keep, indent=1))
    caps, items = {}, []
    for c in keep:
        cap = caps.setdefault(c["video"], cv2.VideoCapture(c["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame"])
        img = draw_court(cap.read()[1], c["prefill_H"])
        tag = "" if c["prefill_H"] else "  NO MAPPING"
        items.append(
            (
                img,
                f"#{c['n']} {c['purpose']} {c['hall'][:22]} {c['youtube_id']} "
                f"s{c['shot']} f{c['frame']}{tag}",
            )
        )
    for k in range(0, len(items), 6):
        cv2.imwrite(
            str(OUT / f"sheet_{k // 6:02d}.jpg"),
            contact_sheet(items[k : k + 6], cols=2, tile_wh=(800, 450), caption_h=26),
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
    halls = defaultdict(lambda: [0, 0])
    for c in keep:
        halls[c["hall"]][c["purpose"] == "train"] += 1
    print(f"{len(keep)} candidates, {len(halls)} halls -> {OUT}/sheet_*.jpg")
    for h, (g, t) in sorted(halls.items()):
        print(f"  {h:30s} gold {g}  train {t}")


def queue() -> None:
    cands = {c["n"]: c for c in json.loads((OUT / "candidates.json").read_text())}
    review = {}
    for line in (OUT / "review.txt").read_text().splitlines():
        line = line.split("  #", 1)[0].strip()
        if line and not line.startswith("#"):
            n, verdict, *why = line.split()
            review[int(n.lstrip("#"))] = (verdict, " ".join(why))
    hall_H = {}
    for c in cands.values():  # best available starting mapping per hall
        if c["prefill_H"] and review.get(c["n"], ("",))[0] == "ok":
            hall_H.setdefault(c["hall"], c["prefill_H"])
    generic = next(
        (c["prefill_H"] for c in cands.values() if c["prefill_H"] and review.get(c["n"], ("",))[0] == "ok"),
        None,
    )
    q, accepted = [], []
    for n, c in sorted(cands.items()):
        verdict, why = review.get(n, ("fix", "not reviewed"))
        if verdict == "drop":
            continue
        if verdict == "ok" and c["purpose"] == "train":
            accepted.append({k: c[k] for k in ("clip", "frame", "prefill_H")})
            continue
        start = c["prefill_H"] or hall_H.get(c["hall"]) or generic
        flags = [why] if why else []
        if not c["prefill_H"]:
            flags.append("no model mapping: starting from another frame's court, drag 4+ points")
        if verdict == "ok":
            flags.append("looks right: confirm or nudge, then S")
        q.append(
            {
                **{k: c[k] for k in ("clip", "youtube_id", "hall", "shot", "frame", "video", "purpose")},
                "prefill_H": start,
                "flags": flags,
            }
        )
    q.sort(key=lambda k: (k["purpose"] != "gold", k["hall"], k["clip"], k["frame"]))
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(
        json.dumps(
            {"note": "Court keyframes for Tomasz (decision 2026-11-02)", "keyframes": q},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    (QUEUE.parent / "accepted_by_implementer.json").write_text(json.dumps(accepted, indent=1) + "\n")
    print(
        f"queued {len(q)} keyframes ({sum(k['purpose'] == 'gold' for k in q)} gold), "
        f"{len(accepted)} accepted as is"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "queue"])
    ap.add_argument("--train-per-hall", type=int, default=2)
    a = ap.parse_args(argv)
    pick(a.train_per_hall) if a.cmd == "pick" else queue()


if __name__ == "__main__":
    main()
