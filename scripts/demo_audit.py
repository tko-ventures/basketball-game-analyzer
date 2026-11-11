#!/usr/bin/env python3
"""Check what the web demo shows against gold track labels, before showing it to anyone who knows the players.

    uv run --extra prefill --extra demo scripts/demo_web.py     # writes runs/demo/cache/*_shown.json
    uv run scripts/demo_audit.py [--gold v0] [--step 2]

For every gold box (every 5th frame at 50 fps) that a demo box matches (IoU >= 0.5), it compares:
  team     the demo's side against the gold team (referees count as their own team)
  number   the number the demo shows against the gold number, where gold has one
It prints per-shot counts and, for each wrong or missing number, the frame span, so the errors can be
looked at in the video. A demo aid, not a metrics row (the demo is a showcase, docs/decisions.md).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

GOLD_CLASS = {1: "home", 2: "away", 3: "ref", 4: "other"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gold(seq: Path):
    frames = [int(x) for x in (seq / "frames.txt").read_text().split()]
    ids = {int(r["tid"]): r for r in csv.DictReader((seq / "identities.csv").open())}
    boxes = defaultdict(list)
    for line in (seq / "gt" / "gt.txt").read_text().splitlines():
        fi, tid, x, y, w, h, _, cls, vis = line.split(",")
        if float(vis) < 1.0:  # interpolated gap, not a reviewed box
            continue
        f = frames[int(fi) - 1]
        boxes[f].append(
            (int(tid), GOLD_CLASS[int(cls)], (float(x), float(y), float(x) + float(w), float(y) + float(h)))
        )
    return boxes, ids


def match(gold, shown, thr=0.5):
    """Greedy one-to-one matching by IoU."""
    pairs = sorted(
        ((iou(g[2], s[3:7]), gi, si) for gi, g in enumerate(gold) for si, s in enumerate(shown)), reverse=True
    )
    used_g, used_s, out = set(), set(), []
    for v, gi, si in pairs:
        if v < thr:
            break
        if gi in used_g or si in used_s:
            continue
        used_g.add(gi)
        used_s.add(si)
        out.append((gold[gi], shown[si]))
    return out


def spans(frames, gap=20):
    frames = sorted(frames)
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > gap:
            out.append((start, prev))
            start = f
        prev = f
    out.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="v0")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--cache", type=Path, default=Path("runs/demo/cache"))
    a = ap.parse_args(argv)
    gold_dir = Path("data/gold") / a.gold
    for shown_path in sorted(a.cache.glob(f"*_step{a.step}_shown.json")):
        key = shown_path.name.split(f"_step{a.step}")[0]  # <youtube_id>_s<shot>
        seq = gold_dir / key
        if not seq.exists():
            print(f"{key}: no gold sequence")
            continue
        shown = json.loads(shown_path.read_text())
        gboxes, ids = load_gold(seq)
        team, num = Counter(), Counter()
        wrong = defaultdict(list)  # (gold tid, gold label, shown label) -> frames
        for f, gold in sorted(gboxes.items()):
            if str(f) not in shown:
                continue
            for (gtid, gcls, _), s in match(gold, shown[str(f)]):
                if gcls == "other":
                    continue
                team["agree" if s[1] == gcls else "differ"] += 1
                gnum = ids.get(gtid, {}).get("number") or ""
                if gcls == "ref":
                    continue
                if s[2] and s[1] != gcls:
                    num["number on the wrong team"] += 1
                    wrong[(gtid, f"{gcls} #{gnum or '?'}", f"{s[1]} #{s[2]}")].append(f)
                elif not gnum:
                    num["gold has no number" + (", demo shows one" if s[2] else "")] += 1
                    if s[2]:
                        wrong[(gtid, f"{gcls} #?", f"{s[1]} #{s[2]}")].append(f)
                elif not s[2]:
                    num["right player, no number shown"] += 1
                elif s[2] == gnum:
                    num["correct"] += 1
                else:
                    num["wrong number"] += 1
                    wrong[(gtid, f"{gcls} #{gnum} {ids[gtid]['name']}", f"#{s[2]}")].append(f)
        print(f"\n{key}  (gold {a.gold}, {sum(team.values())} matched boxes)")
        print(f"  team    {dict(team)}")
        print(f"  number  {dict(num)}")
        for (gtid, g, s), fs in sorted(wrong.items(), key=lambda kv: -len(kv[1])):
            print(f"    gold {gtid:>5} {g:<38} shown {s:<12} {len(fs):>3} boxes, frames {spans(fs)}")


if __name__ == "__main__":
    main()
