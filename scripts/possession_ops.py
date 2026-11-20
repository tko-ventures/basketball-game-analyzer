#!/usr/bin/env python3
"""Record each possession-queue answer as an explicit holder, then apply an ops file of review decisions.

    uv run scripts/possession_ops.py migrate                                   # holder from the class, once
    uv run scripts/possession_ops.py apply data/labels/review/possession.ops.txt

``migrate`` gives every frame of ``data/labels/box_queue_possession.json`` without a ``holder`` field the id of its
``player-in-possession`` box (or null) and ``dead_ball: false``. Frames answered before the page saved the holder
could not tell "nobody" from "the shooter holds it", so those come through ops.

Ops file, one operation per line; comments are lines starting with ``#`` or text after two or more spaces and ``#``.
``<i>`` is the frame's index in the queue (``possqueue`` only appends, so indices are stable):
    holder <i> <box id>    that box holds or dribbles the ball; a shooter keeps the shot class, any other
                           player-in-possession goes back to player
    nobody <i>             nobody holds the ball
    dead <i>               the ball is out of live play (a throw-in from behind a line, the referee has it)
    live <i>               the ball is in live play
    review <i> <note>      back to the reviewer: status todo, the note and the answer so far shown on the page;
                           put it before the frame's other ops so the note keeps the reviewer's answer
Operations run in file order, so the file is the audit trail of the pass. Each file is applied once: its sha256 goes
into the queue's ``meta.ops_applied``, and ``apply`` refuses a file already there (``review`` would undo the answers
given since). A later pass is a new file.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

QUEUE = Path("data/labels/box_queue_possession.json")
SHOOTING = {"player-jump-shot", "player-layup-dunk"}


def answer(f):
    holder = "nobody" if f.get("holder") is None else f"box {f['holder']}"
    return holder + (", dead ball" if f.get("dead_ball") else "")


def set_holder(f, hid):
    ids = {b["id"]: b for b in f["boxes"]}
    if hid is not None and hid not in ids:
        raise SystemExit(f"frame {f['clip']} {f['frame']}: no box {hid}")
    for b in f["boxes"]:
        if b["cls"] == "player-in-possession" and b["id"] != hid:
            b["cls"] = "player"
    if hid is not None and ids[hid]["cls"] not in SHOOTING:
        ids[hid]["cls"] = "player-in-possession"
    f["holder"] = hid


def migrate(q):
    n = 0
    for f in q["frames"]:
        if "holder" in f:
            continue
        pip = [b["id"] for b in f["boxes"] if b["cls"] == "player-in-possession"]
        if len(pip) > 1 and f["status"] == "done":
            raise SystemExit(f"frame {f['clip']} {f['frame']}: {len(pip)} holders")
        f["holder"] = pip[0] if len(pip) == 1 and f["status"] == "done" else None
        f["dead_ball"] = False
        n += 1
    print(f"{n} frames given an explicit holder")


def apply(q, ops_path):
    sha = hashlib.sha256(ops_path.read_bytes()).hexdigest()
    done = q.setdefault("meta", {}).setdefault("ops_applied", {})
    if sha in done.values():
        raise SystemExit(f"{ops_path} was applied already; write a later pass as a new file")
    counts = {}
    for raw in ops_path.read_text().splitlines():
        line = re.split(r"\s{2,}#", raw)[0].strip()
        if not line or line.startswith("#"):
            continue
        op, i, *rest = line.split(maxsplit=2) if line.startswith("review") else line.split()
        f = q["frames"][int(i)]
        if op == "holder":
            set_holder(f, int(rest[0]))
        elif op == "nobody":
            set_holder(f, None)
        elif op in ("dead", "live"):
            f["dead_ball"] = op == "dead"
        elif op == "review":
            f["note"] = f"{rest[0]} (answer before: {answer(f)})"
            f["status"] = "todo"
        else:
            raise SystemExit(f"unknown op {op!r} in {raw!r}")
        counts[op] = counts.get(op, 0) + 1
    done[ops_path.name] = sha
    print(", ".join(f"{k} {v}" for k, v in counts.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["migrate", "apply"])
    ap.add_argument("ops", nargs="?", type=Path)
    a = ap.parse_args()
    q = json.loads(QUEUE.read_text())
    if a.cmd == "migrate":
        migrate(q)
    else:
        apply(q, a.ops)
    QUEUE.write_text(json.dumps(q, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
