#!/usr/bin/env python3
"""Holder gold v1: the ball holder of each box-gold v1 frame under the 2026-11-20 possession definition.

    uv run scripts/holders_gold_v1.py            # writes data/gold/v1/holders.json

Box gold v1 (data/gold/v1/boxes.json) was labelled on 2026-11-18, before the possession schema of 2026-11-20
(docs/adr/2026-11-20-detector-classes-and-possession-schema.md) made the holder whoever holds or dribbles the ball,
a shooter still holding it included. Its ``player-in-possession`` boxes are the starting point; the ops file
data/labels/review/holders_v1.ops.txt changes the frames where the two definitions differ. Box gold itself is left
alone, so detection rows keep scoring the boxes Tomasz drew.

Ops lines: ``holder <youtube_id> <frame> <box index|none> <implementer|flagged|tomasz>  # reason``. ``implementer``
and ``tomasz`` lines are applied; a ``flagged`` line only records the proposed answer, and box gold stands until
Tomasz turns it into a ``tomasz`` line.

"""

from __future__ import annotations

import json
from pathlib import Path

BOXES = Path("data/gold/v1/boxes.json")
OPS = Path("data/labels/review/holders_v1.ops.txt")
OUT = Path("data/gold/v1/holders.json")
DEFINITION = (
    "Whoever holds or dribbles the ball, mid-bounce and inbounders included; a shooter still holding the ball is "
    "the holder. A ball in flight has no holder (2026-11-20 possession schema)."
)


def read_ops(path: Path) -> dict[tuple[str, int], dict]:
    """Parse the ops file into ``{(youtube_id, frame): {"holder", "status", "note"}}``."""
    ops = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line, _, note = raw.partition("#")
        parts = line.split()
        if not parts:
            continue
        if parts[0] != "holder" or len(parts) != 5 or parts[4] not in ("implementer", "flagged", "tomasz"):
            raise SystemExit(f"bad ops line: {raw}")
        _, yid, frame, holder, status = parts
        ops[yid, int(frame)] = {
            "holder": None if holder == "none" else int(holder),
            "status": status,
            "note": note.strip(),
        }
    return ops


def main() -> None:
    frames = json.loads(BOXES.read_text(encoding="utf-8"))["frames"]
    ops = read_ops(OPS)
    out, seen = [], set()
    for fr in frames:
        key = (fr["youtube_id"], int(fr["frame"]))
        box_gold = next((i for i, b in enumerate(fr["boxes"]) if b["cls"] == "player-in-possession"), None)
        row = {"youtube_id": fr["youtube_id"], "shot": fr["shot"], "frame": fr["frame"], "holder": box_gold}
        row["status"] = "box_gold"
        op = ops.get(key)
        if op is not None:
            seen.add(key)
            row["note"] = op["note"]
            if op["status"] == "flagged":
                row["status"], row["proposed"] = "flagged", op["holder"]
            else:
                row["holder"], row["status"] = op["holder"], op["status"]
        if row["holder"] is not None and fr["boxes"][row["holder"]]["cls"] in ("ball", "rim", "number", "referee"):
            raise SystemExit(f"{key}: box {row['holder']} is a {fr['boxes'][row['holder']]['cls']}, not a player")
        out.append(row)
    missing = set(ops) - seen
    if missing:
        raise SystemExit(f"ops name frames not in box gold: {sorted(missing)}")
    doc = {"version": "v1", "definition": DEFINITION, "boxes": str(BOXES), "frames": out}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    changed = sum(r["status"] in ("implementer", "tomasz") for r in out)
    flagged = sum(r["status"] == "flagged" for r in out)
    print(f"{OUT}: {len(out)} frames, {changed} changed from box gold, {flagged} flagged for Tomasz")


if __name__ == "__main__":
    main()
