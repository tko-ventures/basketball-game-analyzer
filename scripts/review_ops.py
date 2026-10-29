#!/usr/bin/env python3
"""Apply a list of tracklet review operations to a clip's labels.

    uv run scripts/review_ops.py data/labels/review/<clip-id>.ops.txt

Ops file, one operation per line, first line ``clip <labels stem>``. Comments are lines starting
with ``#`` or text after two or more spaces and ``#``; a ``#`` inside a note is kept:
    merge 1012 1003            fold identity 1012 into 1003 in every frame
    split 1003 4410 1077       frames >= 4410 of identity 1003 become new identity 1077
    delete 1044                remove identity 1044 everywhere (bench, staff, crowd)
    team 1003 home             home | away | ref | other
    number 1003 23
    note 1003 switches to another player after ~3340   free text for the human pass (identities.csv)
    confirm 1003               mark the track reviewed
Operations run in file order, so the file is the audit trail of a review pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.labeler import server  # noqa: E402


def apply_ops(labels: dict, lines: list[str]) -> dict[str, int]:
    ids, frames = labels["identities"], labels["frames"]
    counts: dict[str, int] = {}
    for raw in lines:
        # Comments: whole lines starting with "#", or "#" after two or more spaces; "#7" in a note stays.
        line = re.split(r"\s{2,}#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("#") or line.startswith("clip "):
            continue
        op, *args = line.split()
        counts[op] = counts.get(op, 0) + 1
        if op == "merge":
            src, dst = args
            for f in frames.values():
                for b in f["boxes"]:
                    if str(b["tid"]) == src:
                        b["tid"] = int(dst)
            s, d = ids.pop(src, {}), ids.setdefault(dst, {"team": "", "number": "", "name": ""})
            d["team"] = d.get("team") or s.get("team", "")
            d["number"] = d.get("number") or s.get("number", "")
        elif op == "split":
            src, start, new = args
            ids.setdefault(new, {**ids[src], "reviewed": False})
            for k, f in frames.items():
                if int(k) >= int(start):
                    for b in f["boxes"]:
                        if str(b["tid"]) == src:
                            b["tid"] = int(new)
        elif op == "delete":
            (tid,) = args
            for f in frames.values():
                f["boxes"] = [b for b in f["boxes"] if str(b["tid"]) != tid]
            ids.pop(tid, None)
        elif op == "team":
            tid, team = args
            assert team in server.TEAMS, team
            ids[tid]["team"] = team
        elif op == "number":
            tid, num = args
            ids[tid]["number"] = num
        elif op == "note":
            tid, *text = args
            ids[tid]["note"] = " ".join(text)
        elif op == "confirm":
            (tid,) = args
            ids[tid]["reviewed"] = True
        else:
            raise ValueError(f"unknown op {op!r} in {raw!r}")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ops", type=Path)
    ap.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    a = ap.parse_args(argv)
    lines = a.ops.read_text().splitlines()
    stem = next(line.split(None, 1)[1].strip() for line in lines if line.startswith("clip "))
    path = a.labels_dir / f"{stem}.json"
    labels = json.loads(path.read_text())
    counts = apply_ops(labels, lines)
    for ident in labels["identities"].values():
        side = labels["meta"].get(ident.get("team"), {})
        roster = server.REPO / "data/rosters" / f"{side.get('roster', '')}.json"
        if ident.get("number") and roster.is_file():
            season = json.loads(roster.read_text())["seasons"].get(labels["meta"]["season"], {})
            ident["name"] = next(
                (p["name"] for p in season.get("players", []) if p["number"] == ident["number"]), ""
            )
    server.save_labels(a.labels_dir, labels)
    print(f"{stem}: {counts}, {len(labels['identities'])} identities")


if __name__ == "__main__":
    main()
