"""Cut gold for Layer 0 (M10, #32): candidate cuts from two detectors, review sheets, and the reviewed gold file.

    uv run scripts/cut_review.py candidates <tn.npy> <hd.json> <clip> --out <dir>    # merge both detectors' cuts
    uv run scripts/cut_review.py sheets <dir>/<clip>.candidates.json --only disagree   # strips to read
    uv run scripts/cut_review.py export <dir>/<clip>.candidates.json <verdicts.tsv>    # into data/labels/cut_gold.json
    uv run scripts/cut_review.py unseen <tn.npy> <clip> --threshold 0.3 --out <dir>    # a detector's cuts on no entry
    uv run scripts/cut_review.py add <dir>/<clip>.transnetv2@0.3.candidates.json <verdicts.tsv>  # read, then added

A candidate is a cut either detector makes; cuts of the two within ``tol_s`` are one candidate. Each candidate gets a
strip of four frames (0.4 s before, the frame before the cut, the cut frame, 0.4 s after), read by eye. Verdicts are
``cut`` (hard, including an edit's jump cut within one camera), ``gradual <first> <last>`` (the blended frames of a
dissolve or wipe), or ``no`` (no camera change or time jump). Every disagreement is read; agreements are read on a
random sample. Cuts neither detector makes are only found by reading central shots.

Only game time is reviewed: the spans from the first to the last ``central``-tagged label shot, split where tagged
central shots are more than ``break_s`` apart (half time and long breaks). Intros, studio and ads outside them are left
out, as Layer 0 drops them whatever their cuts.

"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

GOLD = Path("data/labels/cut_gold.json")
#: Strip frames around a candidate: seconds before and after, and the frame before the cut (``None``).
OFFSETS = (-0.4, None, 0.0, 0.4)
THUMB = (320, 180)


def game_spans(shots, fps, break_s=150.0):
    central = [s for s in shots if s.get("camera") == "central"]
    spans = []
    for s in central:
        if spans and s["start"] - spans[-1][1] <= break_s * fps:
            spans[-1][1] = s["end"]
        else:
            spans.append([s["start"], s["end"]])
    return spans


def cuts_of(shots):
    return [s["start"] for s in shots[1:]]


def transnet_cuts(probs, threshold=0.5):
    flags = probs > threshold
    return [i for i in range(1, len(flags)) if flags[i - 1] and not flags[i]]


def candidates(a):
    label = json.loads(Path(f"data/labels/{a.clip}.json").read_text())
    fps, n = float(label["fps"]), int(label["frame_count"])
    tol = max(1, round(a.tol_s * fps))
    probs = np.load(a.tn)
    tn = transnet_cuts(probs)
    hd = cuts_of(json.loads(Path(a.hd).read_text()))
    events = sorted([(f, "transnetv2") for f in tn] + [(f, "hist_diff") for f in hd])
    merged = []
    for f, who in events:
        if merged and f - merged[-1]["frames"][-1] <= tol:
            merged[-1]["frames"].append(f)
            merged[-1]["by"].add(who)
        else:
            merged.append({"frames": [f], "by": {who}})
    spans = game_spans(label["shots"], fps, a.break_s)
    out = []
    for i, m in enumerate(merged):
        frame = min(m["frames"])
        lo, hi = max(0, frame - tol), min(len(probs), frame + tol + 1)
        out.append({
            "id": i,
            "frame": frame,
            "frames": sorted(set(m["frames"])),
            "by": sorted(m["by"]),
            "tn_max_prob": round(float(probs[lo:hi].max()), 3) if hi > lo else None,
            "in_game": any(s <= frame <= e for s, e in spans),
        })
    doc = {
        "clip": a.clip,
        "video": label["clip"],
        "fps": fps,
        "frame_count": n,
        "tol_s": a.tol_s,
        "spans": spans,
        "candidates": out,
    }
    path = Path(a.out) / f"{a.clip}.candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    game = [c for c in out if c["in_game"]]
    both = sum(len(c["by"]) == 2 for c in game)
    print(f"{a.clip}: {len(game)} of {len(out)} candidates in game time, {both} by both, {len(game) - both} by one")


def strip(cap, frame, fps, n):
    tiles = []
    for off in OFFSETS:
        f = min(max(0, frame - 1 if off is None else frame + round(off * fps)), n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        img = cv2.resize(img, THUMB) if ok else np.zeros((THUMB[1], THUMB[0], 3), np.uint8)
        cv2.putText(img, f"{f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(img)
    return np.hstack(tiles)


def sheets(a):
    doc = json.loads(Path(a.candidates).read_text())
    cands = [c for c in doc["candidates"] if c["in_game"]]
    if a.only == "disagree":
        cands = [c for c in cands if len(c["by"]) == 1]
    elif a.only == "agree-sample":
        agree = [c for c in cands if len(c["by"]) == 2]
        cands = sorted(random.Random(a.seed).sample(agree, min(a.sample, len(agree))), key=lambda c: c["frame"])
    out = Path(a.out or Path(a.candidates).with_suffix("")) / a.only
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(doc["video"])
    rows, k = [], 0
    for c in cands:
        row = strip(cap, c["frame"], doc["fps"], doc["frame_count"])
        label = f"#{c['id']} f{c['frame']} {'+'.join(b[:2] for b in c['by'])} p{c['tn_max_prob']}"
        bar = np.zeros((18, row.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(np.vstack([bar, row]))
        if len(rows) == a.per_sheet:
            cv2.imwrite(str(out / f"sheet_{k:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows, k = [], k + 1
    if rows:
        cv2.imwrite(str(out / f"sheet_{k:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        k += 1
    print(f"{len(cands)} candidates on {k} sheets in {out}")


def dense(a):
    doc = json.loads(Path(a.candidates).read_text())
    cap = cv2.VideoCapture(doc["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    tiles = []
    for f in range(a.start, a.end + 1):
        ok, img = cap.read()
        if not ok:
            break
        if (f - a.start) % a.step:
            continue
        img = cv2.resize(img, (240, 135))
        cv2.putText(img, f"{f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles), 5)]))
    print(out)


def red_share(img):
    hsv = cv2.cvtColor(cv2.resize(img, (96, 54)), cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h <= 4) | (h >= 172)) & (sat > 170) & (val > 140)).mean())


def stinger(a):
    """Print a ``gradual`` verdict line for each candidate on an ORLEN logo stinger, from the red share per frame."""
    doc = json.loads(Path(a.candidates).read_text())
    by_id = {c["id"]: c for c in doc["candidates"]}
    cap = cv2.VideoCapture(doc["video"])
    span = round(2 * doc["fps"])
    for cid in a.ids:
        f = by_id[cid]["frame"]
        lo = max(0, f - span)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        share = []
        for _ in range(2 * span + 1):
            ok, img = cap.read()
            if not ok:
                break
            share.append(red_share(img))
        share = np.array(share)
        peak = int(np.argmax(share))
        if share[peak] < 0.6:
            print(f"{cid}\t?\tno stinger card within 2 s (max red share {share[peak]:.2f})")
            continue
        on = share > np.percentile(share, 20) + 0.05
        first = last = peak
        while first > 0 and on[first - 1]:
            first -= 1
        while last < len(share) - 1 and on[last + 1]:
            last += 1
        print(f"{cid}\tgradual {lo + first} {lo + last}\tORLEN stinger, frames by red share")


def dissolve(a):
    """Print a ``gradual`` verdict line for each candidate on a dissolve: where a frame's progress from the frame
    ``half`` before to the one ``half`` after (by mean absolute difference) climbs steepest."""
    doc = json.loads(Path(a.candidates).read_text())
    by_id = {c["id"]: c for c in doc["candidates"]}
    cap = cv2.VideoCapture(doc["video"])
    half = round(a.half_s * doc["fps"])
    for cid in a.ids:
        f = by_id[cid]["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - half)
        frames = [cv2.resize(cap.read()[1], (96, 54)).astype(np.float32) for _ in range(2 * half + 1)]
        da = np.array([np.abs(x - frames[0]).mean() for x in frames])
        db = np.array([np.abs(x - frames[-1]).mean() for x in frames])
        progress = np.convolve(da / np.maximum(da + db, 1e-6), np.ones(3) / 3, mode="same")
        step = np.gradient(progress)
        near = round(0.3 * doc["fps"])
        step[: half - near] = step[half + near + 1 :] = 0  # the strip shows the blend at the candidate
        peak = int(np.argmax(step))
        first = last = peak
        while first > 1 and step[first - 1] > step[peak] / 4:
            first -= 1
        while last < len(step) - 2 and step[last + 1] > step[peak] / 4:
            last += 1
        print(f"{cid}\tgradual {f - half + first} {f - half + last}\tdissolve, frames by progress between its ends")


def read_verdicts(path):
    verdicts = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cid, verdict, *rest = line.split("\t")
        kind, *frames = verdict.split()
        if kind not in ("cut", "gradual", "no") or (kind == "gradual") != (len(frames) == 2):
            raise SystemExit(f"{path}: bad verdict {verdict!r} for candidate {cid}")
        rec = {"verdict": kind}
        if kind == "gradual":
            rec["first"], rec["last"] = int(frames[0]), int(frames[1])
        if rest and rest[0].strip():
            rec["note"] = rest[0].strip()
        verdicts[int(cid)] = rec
    return verdicts


def absorb(transitions, tol):
    """Give an unread cut on a gradual transition (within ``tol``) that transition's extent, so it counts once."""
    gradual = [t for t in transitions if t["verdict"] == "gradual"]
    n = 0
    for t in transitions:
        if t["reviewed"] or t["verdict"] != "cut":
            continue
        for g in gradual:
            if g["first"] - tol <= t["frame"] <= g["last"] + 1 + tol:
                t.update(
                    verdict="gradual",
                    first=g["first"],
                    last=g["last"],
                    note=f"extent of the read entry at {g['frame']}",
                )
                n += 1
                break
    return n


def export(a):
    doc = json.loads(Path(a.candidates).read_text())
    verdicts = {}
    for path in a.verdicts:
        verdicts.update(read_verdicts(path))
    game = [c for c in doc["candidates"] if c["in_game"]]
    unread = [c["id"] for c in game if len(c["by"]) == 1 and c["id"] not in verdicts]
    if unread:
        raise SystemExit(f"{len(unread)} disagreements have no verdict, e.g. {unread[:5]}")
    transitions, read, real = [], 0, 0
    for c in game:
        rec = {"frame": c["frame"], "by": c["by"], "reviewed": c["id"] in verdicts}
        rec.update(verdicts.get(c["id"], {"verdict": "cut"}))
        if rec["reviewed"] and len(c["by"]) == 2:
            read += 1
            real += rec["verdict"] != "no"
        transitions.append(rec)
    absorbed = absorb(transitions, max(1, round(doc["tol_s"] * doc["fps"])))
    gold = json.loads(GOLD.read_text()) if GOLD.exists() else {"version": 1, "tol_s": doc["tol_s"], "games": {}}
    if gold["tol_s"] != doc["tol_s"]:
        raise SystemExit(f"candidates merged at tol_s {doc['tol_s']}, gold uses {gold['tol_s']}")
    gold["games"][doc["clip"]] = {
        "fps": doc["fps"],
        "spans": doc["spans"],
        "agree_sample": {"read": read, "real": real},
        "transitions": transitions,
    }
    gold["games"] = dict(sorted(gold["games"].items()))
    GOLD.write_text(json.dumps(gold, indent=1) + "\n")
    print(
        f"{doc['clip']}: {len(transitions)} entries, {len(verdicts)} read, agreement sample {real}/{read} real, "
        f"{absorbed} unread cuts on gradual transitions"
    )


def unseen(a):
    """Write a candidates file of a detector's in-game cuts that fall on no gold entry, for reading and ``add``."""
    from bga.eval.ingest import gold_transitions

    gold = json.loads(GOLD.read_text())
    game = gold["games"][a.clip]
    label = json.loads(Path(f"data/labels/{a.clip}.json").read_text())
    tol = max(1, round(gold["tol_s"] * game["fps"]))
    windows = gold_transitions(game, tol)
    probs = np.load(a.tn)
    by = f"transnetv2@{a.threshold}"
    out = []
    for f in transnet_cuts(probs, a.threshold):
        in_game = any(s <= f <= e for s, e in game["spans"])
        if in_game and not any(t.holds(f) for t in windows) and not (out and f - out[-1]["frame"] <= tol):
            lo, hi = max(0, f - tol), min(len(probs), f + tol + 1)
            out.append({
                "id": len(out),
                "frame": f,
                "frames": [f],
                "by": [by],
                "in_game": True,
                "tn_max_prob": round(float(probs[lo:hi].max()), 3),
            })
    doc = {
        "clip": a.clip,
        "video": label["clip"],
        "fps": game["fps"],
        "frame_count": int(label["frame_count"]),
        "tol_s": gold["tol_s"],
        "spans": game["spans"],
        "candidates": out,
    }
    path = Path(a.out) / f"{a.clip}.{by}.candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    print(f"{a.clip}: {len(out)} {by} cuts on no gold entry -> {path}")


def add(a):
    """Add read verdicts of an ``unseen`` candidates file to the game's gold entries."""
    doc = json.loads(Path(a.candidates).read_text())
    verdicts = {}
    for path in a.verdicts:
        verdicts.update(read_verdicts(path))
    unread = [c["id"] for c in doc["candidates"] if c["id"] not in verdicts]
    if unread:
        raise SystemExit(f"{len(unread)} candidates have no verdict, e.g. {unread[:5]}")
    gold = json.loads(GOLD.read_text())
    game = gold["games"][doc["clip"]]
    known = {(t["frame"], tuple(t["by"])) for t in game["transitions"]}
    new = [{"frame": c["frame"], "by": c["by"], "reviewed": True, **verdicts[c["id"]]} for c in doc["candidates"]]
    new = [t for t in new if (t["frame"], tuple(t["by"])) not in known]
    game["transitions"] = sorted(game["transitions"] + new, key=lambda t: t["frame"])
    absorbed = absorb(game["transitions"], max(1, round(gold["tol_s"] * game["fps"])))
    GOLD.write_text(json.dumps(gold, indent=1) + "\n")
    real = sum(t["verdict"] != "no" for t in new)
    print(f"{doc['clip']}: {len(new)} entries added, {real} real, {absorbed} unread cuts on gradual transitions")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("tn")
    c.add_argument("hd")
    c.add_argument("clip")
    c.add_argument("--out", required=True)
    c.add_argument("--tol-s", type=float, default=0.2)
    c.add_argument("--break-s", type=float, default=150.0)
    s = sub.add_parser("sheets")
    s.add_argument("candidates")
    s.add_argument("--only", choices=["disagree", "agree-sample", "all"], default="disagree")
    s.add_argument("--sample", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--per-sheet", type=int, default=6)
    s.add_argument("--out")
    d = sub.add_parser("dense", help="every step-th frame of a range, to settle a candidate by eye")
    d.add_argument("candidates")
    d.add_argument("start", type=int)
    d.add_argument("end", type=int)
    d.add_argument("--step", type=int, default=1)
    d.add_argument("--out", required=True)
    t = sub.add_parser("stinger", help="verdict lines for candidates on an ORLEN logo stinger")
    t.add_argument("candidates")
    t.add_argument("ids", type=int, nargs="+")
    x = sub.add_parser("dissolve", help="verdict lines for candidates on a dissolve")
    x.add_argument("candidates")
    x.add_argument("ids", type=int, nargs="+")
    x.add_argument("--half-s", type=float, default=0.75)
    u = sub.add_parser("unseen", help="a detector's in-game cuts on no gold entry, as a candidates file")
    u.add_argument("tn")
    u.add_argument("clip")
    u.add_argument("--threshold", type=float, required=True)
    u.add_argument("--out", required=True)
    ad = sub.add_parser("add", help="add read verdicts of an unseen candidates file to the gold")
    ad.add_argument("candidates")
    ad.add_argument("verdicts", nargs="+")
    e = sub.add_parser("export")
    e.add_argument("candidates")
    e.add_argument("verdicts", nargs="+", help="TSVs of candidate id, verdict and an optional note")
    a = ap.parse_args(argv)
    {
        "candidates": candidates,
        "sheets": sheets,
        "dense": dense,
        "stinger": stinger,
        "dissolve": dissolve,
        "export": export,
        "unseen": unseen,
        "add": add,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
