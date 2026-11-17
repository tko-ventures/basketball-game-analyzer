#!/usr/bin/env python3
"""Export the gold set (data/labels/gold.json) as MOTChallenge sequences, one per gold shot.

    uv run scripts/export_gold.py --version v0 [--max-gap 3]

Writes data/gold/<version>/<youtube_id>_s<shot>/{gt/gt.txt,frames.txt,seqinfo.ini,identities.csv} and
data/gold/<version>/manifest.json (sequences, held-out flags, label-file hashes, counts).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.eval.gold import build_sequence, load_manifest, write_sequence  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--gold", type=Path, default=Path("data/labels/gold.json"))
    ap.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    ap.add_argument("--out", type=Path, default=Path("data/gold"))
    ap.add_argument("--max-gap", type=int, default=3, help="missing sampled frames filled by interpolation")
    a = ap.parse_args(argv)
    gold = load_manifest(a.gold)
    out = a.out / a.version
    manifest = {
        "version": a.version,
        "exported": dt.date.today().isoformat(),
        "stride": gold["stride"],
        "max_gap": a.max_gap,
        "sequences": [],
    }
    for game in gold["games"]:
        path = a.labels_dir / f"{game['clip']}.json"
        raw = path.read_bytes()
        labels = json.loads(raw)
        cap = cv2.VideoCapture(labels["clip"])
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080)
        cap.release()
        for shot in game["shots"]:
            seq = build_sequence(labels, game, shot, gold["stride"], a.max_gap)
            write_sequence(seq, out, labels.get("meta", {}), game["fps"], gold["stride"], wh)
            filled = sum(1 for r in seq.rows if r[8] == 0.5)
            manifest["sequences"].append(
                {
                    "name": seq.name,
                    "clip": seq.clip,
                    "shot": seq.shot,
                    "held_out": seq.held_out,
                    "frames": len(seq.frames),
                    "boxes": len(seq.rows),
                    "boxes_interpolated_or_occluded": filled,
                    "identities": len(seq.identities),
                    "labels_sha256_8": hashlib.sha256(raw).hexdigest()[:8],
                }
            )
            print(f"{seq.name}: {len(seq.frames)} frames, {len(seq.rows)} boxes, {len(seq.identities)} ids")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
