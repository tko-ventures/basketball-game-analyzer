#!/usr/bin/env python3
"""The implementer's first pass on an auto-labelling round (plan section 7), applied to its draft.

    uv run scripts/firstpass.py --round 4

Inputs, committed under data/labels/review/:
  r<N>_reads.tsv      one line per draft frame (frame index ``idx``): ball (B, c<k>, none, missing, unsure), drop
                      (draft people indexes with d duplicate, n not on court, x two people merged), missing people,
                      holder (draft index, nobody, unsure, nobox), action (hold, jump-shot, layup-dunk), note. Read on
                      runs/autolabel/r<N>/sheets/ (scripts/r4_sheets.py) by reader agents, spot-checked.
  r<N>_proposals.tsv  for the frames with people missing: the draft's weaker people boxes accepted as ``p<j>:<class>``
                      (scripts/r4_sheets.py --proposals) and the people still missing.

Outputs:
  runs/autolabel/r<N>/draft_fp.json   the draft with the drops, accepted proposals and holder classes applied, and
                                      per frame ``fp``: what the first pass settled.
  data/labels/box_queue_r<N>balls.json   ball queue (/boxes?set=r<N>balls): settled balls done, the rest to do.
                                         A ball not seen in the crop is to do: the crop may not show it.
  data/labels/box_queue_r<N>people.json  people queue (/boxes?set=r<N>people): frames with a merged box or people
                                         still missing, for Tomasz to fix geometry.

The manifest (autolabel.py manifest) then reads draft_fp.json and the people queue, and the possession queue
(autolabel.py possqueue) starts each frame from the first pass's holder.

"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from autolabel import PERSON, Round, proposals

#: Holder action from the first pass to the holder's class (the labeller's rule: a shooter keeps the shot class).
ACTION_CLASS = {"hold": "player-in-possession", "jump-shot": "player-jump-shot", "layup-dunk": "player-layup-dunk"}
PEOPLE_INSTRUCTIONS = (
    "<b>People in a crowded frame.</b> Flagged boxes cover two people: split them (drag the box onto one, "
    "<kbd>⇧</kbd>-drag a box for the other). The note says who has no box yet: <kbd>⇧</kbd>-drag one for each "
    "on-court player or referee. <kbd>Enter</kbd>: frame done · <kbd>S</kbd> skip · <kbd>[</kbd> <kbd>]</kbd> "
    "previous / next frame."
)


#: Positions in the readers' notes are in the review crop's coordinates, not the frame's: dropped from queue notes,
#: with any bracket holding one.
CROP_XY = re.compile(r"\s*\([^)]*~?[xy]~?\d[^)]*\)|\s*~?\b[xy]~?\d+(?:-\d+)?(?:,\s*y~?\d+(?:-\d+)?)?")


def clean(note: str) -> str:
    """A reader's note without crop coordinates, for the labeller.

    >>> clean("referee at baseline (x~470, below rim) has no box; shooter (~x1400-1470,y1075-1310) too; ref at x~600")
    'referee at baseline has no box; shooter too; ref at'

    """
    return CROP_XY.sub("", note).strip()


def _tsv(path: Path) -> dict[int, dict]:
    return {int(r["idx"]): r for r in csv.DictReader(path.open(), delimiter="\t")} if path.exists() else {}


def _drops(spec: str) -> dict[int, str]:
    return {int(d[:-1]): d[-1] for d in spec.replace(" ", "").split(",") if d}


def apply_frame(f: dict, read: dict, prop: dict | None) -> dict:
    """One draft frame with its first-pass read (and proposal read, if any) applied."""
    drops = _drops(read["drop"])
    people = [(i, list(b)) for i, b in enumerate(f["boxes"]) if b[5] in PERSON and drops.get(i) not in ("d", "n")]
    merged = [b[:4] for i, b in people if drops.get(i) == "x"]
    props = proposals(f)
    for spec in (prop or {}).get("accept", "").replace(" ", "").split(","):
        if spec:
            j, cls = spec.split(":")
            people.append((None, [*props[int(j[1:])][:5], cls, "first-pass proposal"]))
    holder, decided = None, read["holder"] == "nobody"
    if read["holder"].isdigit() and int(read["holder"]) not in drops:
        holder, decided = int(read["holder"]), True
    if decided:
        for i, b in people:
            if b[5] == "player-in-possession":
                b[5] = "player"
            if i == holder:
                b[5] = ACTION_CLASS.get(read["action"], "player-in-possession")
    balls = [b for b in f["boxes"] if b[5] == "ball"]
    cands = sorted((c for c in f["candidates"] if c[5] == "ball"), key=lambda c: -c[4])[:3]
    ball = None
    if read["ball"] == "B" and len(balls) == 1:
        ball = balls[0]
    elif read["ball"] in ("c0", "c1", "c2") and int(read["ball"][1]) < len(cands):
        ball = [*cands[int(read["ball"][1])][:5], "ball", "first-pass candidate"]
    still = int(prop["still_missing"]) if prop else int(read["missing"])
    others = [b for b in f["boxes"] if b[5] not in PERSON and b[5] != "ball"]
    return {
        **{k: f[k] for k in ("clip", "video", "shot", "frame", "bucket", "mapped")},
        "boxes": others + [b for _, b in people] + balls,
        "candidates": f["candidates"],
        "fp": {
            "ball": ball[:4] if ball else None,
            "ball_settled": ball is not None,
            "holder": (next(b[:4] for i, b in people if i == holder) if holder is not None else None),
            "holder_decided": decided,
            "merged": merged,
            "people_todo": bool(merged) or still > 0,
            "note": clean(read["note"]),
            "people_note": f"{still} without a box, {len(merged)} merged box(es) flagged. Reader: "
            + clean(prop["note"] if prop else read["note"]),
        },
    }


def _box(i, xyxy, cls, src, flags=()):
    return {"id": i, "xyxy": xyxy, "cls": cls, "team": "", "tid": None, "src": src, "flags": list(flags), "hints": []}


def ball_queue(r: Round, frames: list[dict]) -> dict:
    """The round's ball queue: first-pass balls done, every other frame to do with the draft's guesses."""
    out = []
    for f in frames:
        fp, balls = f["fp"], [b for b in f["boxes"] if b[5] == "ball"]
        cands = sorted((c for c in f["candidates"] if c[5] == "ball"), key=lambda c: -c[4])[:3]
        boxes = [fp["ball"]] if fp["ball_settled"] else [b[:4] for b in balls]
        out.append({
            "youtube_id": f["clip"],
            "clip": f["clip"],
            "video": f["video"],
            "shot": f["shot"],
            "frame": f["frame"],
            "status": "done" if fp["ball_settled"] else "todo",
            "note": "first pass (implementer)" if fp["ball_settled"] else fp["note"],
            "boxes": [
                {**_box(i, b, "ball", "first-pass" if fp["ball_settled"] else "det"), "alt": {}}
                for i, b in enumerate(boxes)
            ],
            "candidates": [] if fp["ball_settled"] else [{"xyxy": c[:4], "cls": "ball", "score": c[4]} for c in cands],
        })
    from autolabel import INSTRUCTIONS

    return {
        "version": f"r{r.n}balls",
        "mode": "ball",
        "title": f"Ball queue, round {r.n}",
        "instructions": INSTRUCTIONS,
        "classes": ["ball"],
        "frames": out,
        "meta": {"review_seconds": {}},
    }


def people_queue(r: Round, frames: list[dict]) -> dict:
    """The round's people queue: frames with a merged box or people still missing, every people box shown."""
    out = []
    for f in (f for f in frames if f["fp"]["people_todo"]):
        people = [b for b in f["boxes"] if b[5] in PERSON]
        out.append({
            "youtube_id": f["clip"],
            "clip": f["clip"],
            "video": f["video"],
            "shot": f["shot"],
            "frame": f["frame"],
            "status": "todo",
            "note": f["fp"]["people_note"],
            "boxes": [
                {**_box(i, b[:4], b[5], "draft", ["merged"] if b[:4] in f["fp"]["merged"] else []), "alt": {}}
                for i, b in enumerate(people)
            ],
            "candidates": [],
        })
    return {
        "version": f"r{r.n}people",
        "title": f"People queue, round {r.n}",
        "instructions": PEOPLE_INSTRUCTIONS,
        "classes": sorted(PERSON),
        "frames": out,
        "meta": {"review_seconds": {}},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite queues that already exist")
    a = ap.parse_args(argv)
    r = Round(a.round)
    reads = _tsv(Path(f"data/labels/review/r{r.n}_reads.tsv"))
    props = _tsv(Path(f"data/labels/review/r{r.n}_proposals.tsv"))
    draft = json.loads((r.out / "draft.json").read_text())["frames"]
    if sorted(reads) != list(range(len(draft))):
        raise SystemExit(f"reads cover {len(reads)} of {len(draft)} draft frames")
    frames = [apply_frame(f, reads[i], props.get(i)) for i, f in enumerate(draft)]
    (r.out / "draft_fp.json").write_text(json.dumps({"frames": frames}, indent=1))
    for path, q in (
        (r.ball_queue, ball_queue(r, frames)),
        (Path(f"data/labels/box_queue_r{r.n}people.json"), people_queue(r, frames)),
    ):
        if path.exists() and not a.force:
            raise SystemExit(f"{path} exists; --force")
        path.write_text(json.dumps(q, indent=1) + "\n")
        todo = sum(f["status"] == "todo" for f in q["frames"])
        print(f"{path}: {len(q['frames'])} frames, {todo} to do")
    held = sum(f["fp"]["holder_decided"] for f in frames)
    print(f"holder settled on {held} of {len(frames)} frames; draft with the first pass -> {r.out / 'draft_fp.json'}")


if __name__ == "__main__":
    main()
