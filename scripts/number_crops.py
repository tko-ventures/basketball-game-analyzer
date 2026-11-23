#!/usr/bin/env python3
"""Training crops for the number readers: number boxes on linked tracklets of non-gold games, labelled per tracklet.

    uv run scripts/number_crops.py manifest --config configs/crops-reid-r1.yaml --out data/labels/numbers/train_r1.json
    uv run scripts/number_crops.py sample data/labels/numbers/train_r1.json --n 300 --seed 0
    uv run scripts/number_crops.py sheets data/labels/numbers/train_r1.json
    uv run scripts/number_crops.py apply data/labels/numbers/train_r1.json data/labels/review/number_train_r1.ops.txt
    uv run scripts/number_crops.py check data/labels/numbers/train_r1.json
    uv run --extra parseq scripts/number_crops.py prefill data/labels/numbers/train_r2.json \
        --config configs/numbers-parseq-r1.yaml
    uv run scripts/number_crops.py accept data/labels/numbers/train_r2.json --min-share 0.8 --min-reads 6

Plan section 7, "Number values": the implementer reads a few crops of each tracklet on contact sheets and the read
propagates to every crop of that tracklet. A crop is a ``number`` box of the config's detect output scoring at least
--min-score that lies at least 90% inside exactly one row of the link output, a non-referee; a box inside two
overlapping people is dropped, since it may be either's (counted in ``ambiguous_dropped``); boxes inside no person
box at all are kept per shot as ``strays``, the candidates for not-a-number examples. The manifest stores boxes
only; the pictures are cut from the clips at training time and never committed. Gold games are refused.

Sheets show one tracklet per row: --per-row of its largest crops spread over its life, upscaled, then the person box
of the first one. Rows are captioned with the tracklet's index in the sample only (no team, no roster), so the read is
blind. Ops lines: ``<index> <label> [note...]`` where label is the digits, ``illegible`` (no crop can be read),
``mixed`` (the crops show more than one number: a tracker swap, dropped) or ``not_number`` (the boxes are logos).
`check` lists reads on neither roster of the clip's season, the conflicts sent to Tomasz.

From round 2 a reader does the first pass. `prefill` reads --per-tracklet crops of every tracklet with a config's
numbers backend, restricts each read to the roster of the tracklet's team, and stores the votes in
runs/number_crops/<manifest>/prefill.json. `accept` labels the tracklets whose leading number has at least
--min-share of at least --min-reads reads, and marks them ``by: reader`` with the vote in the note; with --sample it
puts that many of them into the sample instead, unlabelled, so the implementer reads them blind on sheets and the
accepted labels get a measured precision (`agree` compares). Tracklets the reader does not settle go to the sample
through `sample --unsettled`.

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.config import load_config
from bga.numbers.base import number_crop, roster_choice, roster_numbers
from bga.pipeline import git_commit, load_output, make_backend, sequences
from bga.review.sheets import contact_sheet
from bga.stage import FrameReader
from bga.train.leakage import assert_no_gold

SPECIAL = {"illegible", "mixed", "not_number"}
OUT = Path("runs/number_crops")


def ios(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def spread(items, k):
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def tracklets_of(seq, link, dets, teams, a):
    """Number crops per linked track of one shot, and the number boxes inside no person box."""
    by_id, team_votes, person, strays, ambiguous = {}, {}, {}, [], [0]
    for f in seq.frames:
        rows = link["frames"].get(str(f), [])
        tf = teams.get(str(f), [])
        for d in dets.get(str(f), []):
            if d[5] != "number" or d[4] < a.min_score:
                continue
            box = [round(v, 1) for v in d[:4]]
            holders = [r for r in rows if ios(box, r[1:5]) >= 0.9]
            if not holders:
                if all(ios(box, r[1:5]) < 0.5 for r in rows):
                    strays.append([f, *box, round(d[4], 3)])
                continue
            if len(holders) > 1:
                ambiguous[0] += 1
                continue
            best = holders[0]
            if best[6] == "referee":
                continue
            tid = best[0]
            by_id.setdefault(tid, []).append([f, *box, round(d[4], 3)])
            person.setdefault(tid, [f, *[round(v, 1) for v in best[1:5]]])
            det = best[7]
            if 0 <= det < len(tf) and tf[det]:
                team_votes.setdefault(tid, Counter())[tf[det]] += 1
    out = []
    for tid, crops in sorted(by_id.items()):
        votes = team_votes.get(tid, Counter())
        team = votes.most_common(1)
        out.append({
            "id": f"{seq.name}_t{tid}",
            "clip": seq.clip,
            "video": seq.video,
            "sequence": seq.name,
            "team": team[0][0] if team else "",
            "team_share": round(team[0][1] / sum(votes.values()), 3) if team else 0.0,
            "person": person[tid],
            "crops": crops,
            "label": None,
            "note": "",
        })
    return out, strays, ambiguous[0]


def manifest(a):
    cfg = load_config(a.config)
    seqs = sequences(cfg)
    assert_no_gold({s.clip for s in seqs})
    tracklets, strays, ambiguous = [], {}, 0
    for seq in seqs:
        link = load_output(cfg, Path("runs"), seq, "link")
        dets = load_output(cfg, Path("runs"), seq, "detect")["frames"]
        teams = load_output(cfg, Path("runs"), seq, "team")["frames"]
        t, s, amb = tracklets_of(seq, link, dets, teams, a)
        tracklets += t
        ambiguous += amb
        if s:
            strays[seq.name] = {"clip": seq.clip, "video": seq.video, "boxes": s}
    out = {
        "version": 1,
        "config": cfg.name,
        "config_hash": cfg.config_hash(),
        "detect_weights": cfg.layers["detect"].weights,
        "commit": git_commit(),
        "made": dt.date.today().isoformat(),
        "rule": {"min_score": a.min_score, "min_ios": 0.9, "inside_one_person_only": True},
        "ambiguous_dropped": ambiguous,
        "clips": sorted({s.clip for s in seqs}),
        "sequences": len(seqs),
        "tracklets": tracklets,
        "strays": strays,
        "sample": [],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    n = sum(len(t["crops"]) for t in tracklets)
    n_strays = sum(len(s["boxes"]) for s in strays.values())
    print(f"{a.out}: {len(tracklets)} tracklets, {n} crops, {n_strays} strays, {ambiguous} inside two people dropped")


def sample(a):
    m = json.loads(a.manifest.read_text())
    pool = [t["id"] for t in m["tracklets"] if len(t["crops"]) >= a.min_crops and t["id"] not in m["sample"]]
    if a.unsettled:  # only tracklets the reader read and did not settle, and that carry no label yet
        pre = json.loads(prefill_path(a.manifest).read_text())["tracklets"]
        labelled = {t["id"] for t in m["tracklets"] if t["label"] is not None}
        pool = [i for i in pool if i in pre and not settled(pre[i], a) and i not in labelled]
    pick = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    m["sample"] += sorted(pick)
    a.manifest.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{len(pick)} tracklets added; sample now {len(m['sample'])}")


def best_crops(t, k):
    """The k largest crops by area, then in time order, so the row shows the most readable ones over the tracklet."""
    big = sorted(t["crops"], key=lambda c: -(c[3] - c[1]) * (c[4] - c[2]))[: max(k * 3, k)]
    return spread(sorted(big), k)


def tile(img, box, margin, size):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1, sy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    sx2, sy2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[sy1:sy2, sx1:sx2]
    s = size / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def sheets(a):
    m = json.loads(a.manifest.read_text())
    by_id = {t["id"]: t for t in m["tracklets"]}
    todo = [(k, by_id[i]) for k, i in enumerate(m["sample"]) if by_id[i]["label"] is None]
    out = OUT / a.manifest.stem
    out.mkdir(parents=True, exist_ok=True)
    readers = {}
    cols = a.per_row + 1
    for n in range(0, len(todo), a.per_sheet):
        items = []
        for k, t in todo[n : n + a.per_sheet]:
            read = readers.setdefault(t["video"], FrameReader(t["video"]))
            picks = best_crops(t, a.per_row)
            for j, (f, x1, y1, x2, y2, _s) in enumerate(picks):
                items.append((tile(read(f), (x1, y1, x2, y2), 0.3, a.size), f"#{k}" if j == 0 else ""))
            items += [(np.zeros((2, 2, 3), np.uint8), "")] * (a.per_row - len(picks))
            f, *pbox = t["person"]
            items.append((tile(read(f), pbox, 0.05, a.size), f"n{len(t['crops'])}"))
        p = out / f"sheet_{n // a.per_sheet:02d}.jpg"
        cv2.imwrite(str(p), contact_sheet(items, cols=cols, tile_wh=(a.size, a.size), caption_h=18))
        print(p, len(todo[n : n + a.per_sheet]), "tracklets", f"#{todo[n][0]}-#{todo[n : n + a.per_sheet][-1][0]}")
    for r in readers.values():
        r.close()


def apply(a):
    m = json.loads(a.manifest.read_text())
    by_id = {t["id"]: t for t in m["tracklets"]}
    n = 0
    for ln in a.ops.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        k, label, *note = ln.split()
        if label not in SPECIAL and not re.fullmatch(r"\d{1,2}", label):
            sys.exit(f"bad label {label!r} for #{k}")
        t = by_id[m["sample"][int(k)]]
        t["label"], t["note"] = label, " ".join(note)
        n += 1
    a.manifest.write_text(json.dumps(m, indent=1) + "\n")
    left = sum(by_id[i]["label"] is None for i in m["sample"])
    print(f"{n} labels applied; {left} sampled tracklets unlabelled")


def check(a):
    """Reads on neither side's roster (the conflicts), and how often the read sits on the other side's roster only.

    The side of a tracklet comes from the team layer, which names the sides by the clip meta's kits; clips without
    kits get an arbitrary naming, so a read is checked against both rosters.

    """
    m = json.loads(a.manifest.read_text())
    by_id = {t["id"]: t for t in m["tracklets"]}
    metas, labels, sides = {}, Counter(), Counter()
    for k, tid in enumerate(m["sample"]):
        t = by_id[tid]
        lab = t["label"]
        labels["unlabelled" if lab is None else lab if lab in SPECIAL else "digits"] += 1
        if lab is None or lab in SPECIAL:
            continue
        if t["clip"] not in metas:
            metas[t["clip"]] = json.loads((Path("data/labels") / f"{t['clip']}.json").read_text())["meta"]
        rosters = {s: roster_numbers(metas[t["clip"]], s) for s in ("home", "away")}
        own = lab in rosters.get(t["team"], set())
        other = lab in rosters["away" if t["team"] == "home" else "home"]
        sides["both" if own and other else "own" if own else "other" if other else "neither"] += 1
        if not own and not other:
            print(f"#{k} {tid} read {lab}: on neither roster")
    print(dict(labels), "roster side of digit reads:", dict(sides))


def prefill_path(manifest_path):
    return OUT / manifest_path.stem / "prefill.json"


def prefill(a):
    """Votes of a numbers backend per tracklet: each read restricted to the roster of the tracklet's team."""
    m = json.loads(a.manifest.read_text())
    backend = make_backend(load_config(a.config), "numbers")
    pad = int(backend.p.get("pad", 0))
    metas, jobs = {}, {}
    for t in m["tracklets"]:
        if len(t["crops"]) < a.min_crops:
            continue
        if t["clip"] not in metas:
            metas[t["clip"]] = json.loads((Path("data/labels") / f"{t['clip']}.json").read_text())["meta"]
        for c in best_crops(t, a.per_tracklet):
            jobs.setdefault(t["video"], []).append((c[0], c[1:5], t["id"]))
    by_id = {t["id"]: t for t in m["tracklets"]}
    votes = {}
    backend.load()
    for video, crops in sorted(jobs.items()):
        read = FrameReader(video)
        crops.sort(key=lambda c: c[0])
        for i in range(0, len(crops), 256):
            chunk = crops[i : i + 256]
            images = [number_crop(read(f), box, pad) for f, box, _ in chunk]
            rosters = [roster_numbers(metas[by_id[tid]["clip"]], by_id[tid]["team"]) for _, _, tid in chunk]
            for (_, _, tid), r, roster in zip(chunk, backend.read_crops(images, rosters), rosters, strict=True):
                answer = roster_choice(r, roster)[0]
                votes.setdefault(tid, Counter())[answer or "unreadable"] += 1
        read.close()
    backend.unload()
    out = {"config": str(a.config), "weights": backend.weights, "per_tracklet": a.per_tracklet, "tracklets": {}}
    for tid, v in sorted(votes.items()):
        (top, n_top), n = v.most_common(1)[0], sum(v.values())
        out["tracklets"][tid] = {"proposal": top, "share": round(n_top / n, 3), "reads": n, "votes": dict(v)}
    path = prefill_path(a.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    kinds = Counter("unreadable" if p["proposal"] == "unreadable" else "number" for p in out["tracklets"].values())
    print(f"{path}: {len(out['tracklets'])} tracklets read, proposals {dict(kinds)}")


def settled(p, a):
    """Whether a tracklet's votes settle on a number under the accept rule."""
    return p["proposal"] != "unreadable" and p["share"] >= a.min_share and p["reads"] >= a.min_reads


def accept(a):
    """Label the tracklets the reader settles, or with --sample hold that many back for a blind check on sheets."""
    m = json.loads(a.manifest.read_text())
    pre = json.loads(prefill_path(a.manifest).read_text())
    by_id = {t["id"]: t for t in m["tracklets"]}
    ok = sorted(tid for tid, p in pre["tracklets"].items() if settled(p, a) and by_id[tid]["label"] is None)
    ok = [tid for tid in ok if tid not in m["sample"]]
    if a.sample:
        pick = sorted(random.Random(a.seed).sample(ok, min(a.sample, len(ok))))
        m["sample"] += pick
        m.setdefault("checks", []).extend(pick)
        print(f"{len(pick)} settled tracklets put into the sample for a blind check; sample now {len(m['sample'])}")
    else:
        for tid in ok:
            p = pre["tracklets"][tid]
            by_id[tid]["label"], by_id[tid]["by"] = p["proposal"], "reader"
            by_id[tid]["note"] = f"{pre['weights']} {p['share']} of {p['reads']} reads"
        m["accept_rule"] = {"weights": pre["weights"], "min_share": a.min_share, "min_reads": a.min_reads}
        print(f"{len(ok)} tracklets labelled by the reader ({sum(len(by_id[t]['crops']) for t in ok)} crops)")
    a.manifest.write_text(json.dumps(m, indent=1) + "\n")


def agree(a):
    """The reader's proposal against the implementer's blind sheet label, on the labelled sample."""
    m = json.loads(a.manifest.read_text())
    pre = json.loads(prefill_path(a.manifest).read_text())["tracklets"]
    by_id = {t["id"]: t for t in m["tracklets"]}
    only = set(m.get("checks", [])) if a.checks else None
    rows = Counter()
    for k, tid in enumerate(m["sample"]):
        t, p = by_id[tid], pre.get(tid)
        if t["label"] is None or t.get("by") == "reader" or p is None or (only is not None and tid not in only):
            continue
        kind = "settled" if settled(p, a) else "unsettled"
        same = p["proposal"] == t["label"]
        rows[kind, "agree" if same else f"sheet says {t['label'] if t['label'] in SPECIAL else 'another number'}"] += 1
        if kind == "settled" and not same:
            print(f"#{k} {tid}: reader {p['proposal']} ({p['share']} of {p['reads']}), sheet {t['label']}")
    for key, n in sorted(rows.items()):
        print(key, n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-score", type=float, default=0.3)
    p = sub.add_parser("sample")
    p.add_argument("manifest", type=Path)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--min-crops", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unsettled", action="store_true", help="only tracklets the prefill does not settle")
    p.add_argument("--min-share", type=float, default=0.8)
    p.add_argument("--min-reads", type=int, default=6)
    p = sub.add_parser("sheets")
    p.add_argument("manifest", type=Path)
    p.add_argument("--per-sheet", type=int, default=10)
    p.add_argument("--per-row", type=int, default=4)
    p.add_argument("--size", type=int, default=150)
    p = sub.add_parser("apply")
    p.add_argument("manifest", type=Path)
    p.add_argument("ops", type=Path)
    p = sub.add_parser("check")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("prefill")
    p.add_argument("manifest", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--per-tracklet", type=int, default=12)
    p.add_argument("--min-crops", type=int, default=3)
    for name in ("accept", "agree"):
        p = sub.add_parser(name)
        p.add_argument("manifest", type=Path)
        p.add_argument("--min-share", type=float, default=0.8)
        p.add_argument("--min-reads", type=int, default=6)
        p.add_argument("--sample", type=int, default=0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--checks", action="store_true", help="agree: only the tracklets accept --sample held back")
    a = ap.parse_args(argv)
    cmds = (manifest, sample, sheets, apply, check, prefill, accept, agree)
    {fn.__name__: fn for fn in cmds}[a.cmd](a)


if __name__ == "__main__":
    main()
