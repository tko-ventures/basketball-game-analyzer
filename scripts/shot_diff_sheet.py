#!/usr/bin/env python3
"""Contact sheet of the frames around every cut on which two shot files disagree.

    uv run scripts/shot_diff_sheet.py --git-rev c8ef83f data/labels runs/_shots_regen/diff.jpg

One row per disagreeing cut: frames c-2 .. c+1, captioned with clip, frame and which side has it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bga.review.sheets import contact_sheet  # noqa: E402
from scripts.compare_shots import cuts  # noqa: E402


def unmatched(a: list[int], b: list[int], tol: int) -> list[int]:
    return [x for x in a if not any(abs(x - y) <= tol for y in b)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-rev", required=True)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tol", type=int, default=2)
    a = ap.parse_args(argv)
    items = []
    for p in sorted(a.labels.glob("*.json")):
        new = json.loads(p.read_text())
        rev = subprocess.run(["git", "show", f"{a.git_rev}:{p.as_posix()}"], capture_output=True)
        if rev.returncode:
            continue
        co, cn = cuts(json.loads(rev.stdout)), cuts(new)
        diffs = [(c, "old only") for c in unmatched(co, cn, a.tol)] + [
            (c, "new only") for c in unmatched(cn, co, a.tol)
        ]
        if not diffs:
            continue
        cap = cv2.VideoCapture(new["clip"])
        clip_id = p.stem.split("_")[1]
        for c, side in sorted(diffs):
            for f in range(c - 2, c + 2):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = cap.read()
                if ok:
                    items.append((img, f"{clip_id} f{f} {side if f == c else ''}"))
        cap.release()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), contact_sheet(items, cols=4), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(items) // 4} disagreeing cuts -> {a.out}")


if __name__ == "__main__":
    main()
