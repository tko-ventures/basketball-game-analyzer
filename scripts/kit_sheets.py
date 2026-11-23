#!/usr/bin/env python3
"""Kit labels for the clip metadata: which side wears the light kit (``meta.home.kit`` and ``meta.away.kit``).

The team layer gives the brighter kit cluster to the side marked ``light`` and refuses a clip that does not mark
exactly one side. Three steps, each a subcommand:

    uv run scripts/kit_sheets.py sheets     # runs/kits/<clip>.jpg: four frames of each clip without kits
    uv run scripts/kit_sheets.py evidence   # round-1 number reads per kit cluster against both rosters
    uv run scripts/kit_sheets.py apply      # write the kits of data/labels/review/kits.ops.txt into the label files

``sheets`` takes the middle frame of each of a clip's four longest shots (highlights hold their main-camera shots
longest). ``evidence`` reads ``data/labels/numbers/train_r1.json``, whose tracklets were made while the team layer
still named the brighter cluster ``away`` on clips without kits: a cluster whose numbers are only on one side's
roster is that side. ``apply`` reads one line per clip, ``<youtube id> <home kit> <away kit>`` with ``light`` or
``dark``, and changes nothing else in a label file.

"""

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bga.numbers.base import roster_numbers  # noqa: E402
from bga.paths import write_atomic  # noqa: E402
from bga.stage import FrameReader  # noqa: E402

LABELS = ROOT / "data/labels"
OPS = LABELS / "review/kits.ops.txt"
OUT = ROOT / "runs/kits"
KITS = ("light", "dark")


def clip_files():
    return sorted(p for p in LABELS.glob("2*.json") if "meta" in json.loads(p.read_text()))


def has_kits(meta):
    return sorted((meta.get(s) or {}).get("kit") or "" for s in ("home", "away")) == ["dark", "light"]


def sheets(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    for path in clip_files():
        m = json.loads(path.read_text())
        if has_kits(m["meta"]):
            continue
        longest = sorted(m["shots"], key=lambda s: s["end"] - s["start"], reverse=True)[:4]
        read = FrameReader(str(ROOT / m["clip"]))
        try:
            tiles = []
            for s in sorted(longest, key=lambda s: s["start"]):
                frame = cv2.resize(read((s["start"] + s["end"]) // 2), (960, 540), interpolation=cv2.INTER_AREA)
                tiles.append(frame)
        finally:
            read.close()
        tiles += [np.zeros((540, 960, 3), np.uint8)] * (4 - len(tiles))
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        label = f"{path.stem[9:20]}  home: {m['meta']['home']['team']}   away: {m['meta']['away']['team']}"
        cv2.rectangle(sheet, (0, 0), (1920, 44), (0, 0, 0), -1)
        cv2.putText(sheet, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{path.stem[9:20]}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(OUT / f"{path.stem[9:20]}.jpg")


def evidence(_args):
    manifest = json.loads((LABELS / "numbers/train_r1.json").read_text())
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in manifest["tracklets"]:
        if (t.get("label") or "").isdigit():
            by[t["clip"]][t["team"]].append(t["label"])
    for clip, clusters in sorted(by.items()):
        meta = json.loads((LABELS / f"{clip}.json").read_text())["meta"]
        rosters = ROOT / "data/rosters"
        home, away = roster_numbers(meta, "home", rosters), roster_numbers(meta, "away", rosters)
        print(clip)
        for name, numbers in sorted(clusters.items()):
            only_home = sum(n in home and n not in away for n in numbers)
            only_away = sum(n in away and n not in home for n in numbers)
            kind = "brighter" if name == "away" else "darker"
            print(
                f"  {kind} cluster: {len(numbers)} tracklets, {only_home} only on home's roster, {only_away} on away's"
            )


def apply(_args):
    stems = {p.stem[9:20]: p for p in clip_files()}
    for line in OPS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        clip, home, away = line.split()[:3]
        if sorted((home, away)) != sorted(KITS):
            raise SystemExit(f"{clip}: one side must be light and the other dark, got {home} {away}")
        path = stems[clip]
        m = json.loads(path.read_text())
        if (m["meta"]["home"].get("kit"), m["meta"]["away"].get("kit")) == (home, away):
            continue
        m["meta"]["home"]["kit"], m["meta"]["away"]["kit"] = home, away
        write_atomic(path, json.dumps(m, indent=1, ensure_ascii=False))  # as the labeller writes them
        print(f"{clip}: home {home}, away {away}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for fn in (sheets, evidence, apply):
        sub.add_parser(fn.__name__).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)
