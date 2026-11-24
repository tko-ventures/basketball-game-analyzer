"""Holder gold v1: the ball holder of each box-gold frame under the 2026-11-20 possession definition.

Box gold v1 (``data/gold/v1/boxes.json``) predates the 2026-11-20 possession schema, so its ``player-in-possession``
boxes are only the starting point. An ops file of ``holder`` lines changes the frames where the two definitions
differ, and :func:`build_holders` writes ``data/gold/v1/holders.json``, which ``bga eval-ball`` scores against.

Ops lines: ``holder <youtube_id> <frame> <box index|none> <implementer|flagged|tomasz>  # reason``. ``implementer``
and ``tomasz`` lines are applied; a ``flagged`` line records the proposed answer only, and box gold stands until
Tomasz decides it (the labeller's ``/holders`` page turns it into a ``tomasz`` line).

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from bga.paths import write_atomic

#: Box gold v1, the frames and person boxes the holders index into.
BOXES_PATH = Path("data/gold/v1/boxes.json")
#: The implementer's and Tomasz's holder decisions.
OPS_PATH = Path("data/labels/review/holders_v1.ops.txt")
#: The holder gold written from box gold and the ops file.
HOLDERS_PATH = Path("data/gold/v1/holders.json")
#: The holder definition recorded in the holder gold file.
DEFINITION = (
    "Whoever holds or dribbles the ball, mid-bounce and inbounders included; a shooter still holding the ball is "
    "the holder. A ball in flight has no holder (2026-11-20 possession schema)."
)
#: Box classes that can never be a holder.
NOT_PEOPLE = frozenset({"ball", "rim", "number", "referee", "ball-in-basket"})


class OpStatus(StrEnum):
    """Who decided a holder line, and whether it is applied."""

    IMPLEMENTER = "implementer"
    FLAGGED = "flagged"
    TOMASZ = "tomasz"


@dataclass(frozen=True)
class HolderOp:
    """One ``holder`` line of the ops file.

    Attributes:
        holder: Index of the holder's box in the frame's ``boxes.json`` entry, or ``None`` for no holder.
        status: Who decided it; a flagged line is a proposal only.
        note: The reason after ``#``.

    """

    holder: int | None
    status: OpStatus
    note: str = ""


def parse_op(line: str) -> tuple[tuple[str, int], HolderOp] | None:
    """Parse one ops line into ``((youtube_id, frame), op)``, or ``None`` for a blank or comment line.

    Args:
        line: A line of the ops file.

    Returns:
        The frame key and its op, or ``None``.

    Raises:
        ValueError: When the line is not a well-formed ``holder`` line.

    Examples:
        >>> parse_op("holder abc 120 4 flagged  # held?")
        (('abc', 120), HolderOp(holder=4, status=<OpStatus.FLAGGED: 'flagged'>, note='held?'))
        >>> parse_op("# a comment") is None
        True

    """
    body, _, note = line.partition("#")
    parts = body.split()
    if not parts:
        return None
    if parts[0] != "holder" or len(parts) != 5 or parts[4] not in OpStatus.__members__.values():
        raise ValueError(f"bad ops line: {line}")
    _, yid, frame, holder, status = parts
    op = HolderOp(None if holder == "none" else int(holder), OpStatus(status), note.strip())
    return (yid, int(frame)), op


def read_op_history(path: Path = OPS_PATH) -> dict[tuple[str, int], list[HolderOp]]:
    """Read every ops line per frame, in file order: the first is the original call, the last is in force.

    Args:
        path: The ops file.

    Returns:
        ``{(youtube_id, frame): [op, ...]}``.

    Examples:
        >>> all(len(ops) >= 1 for ops in read_op_history().values())
        True

    """
    history: dict[tuple[str, int], list[HolderOp]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_op(raw)
        if parsed is not None:
            history.setdefault(parsed[0], []).append(parsed[1])
    return history


def read_ops(path: Path = OPS_PATH) -> dict[tuple[str, int], HolderOp]:
    """Read the ops file as ``{(youtube_id, frame): op}``: the last line of each frame is the one in force.

    Args:
        path: The ops file.

    Returns:
        One op per frame, in the order frames first appear.

    Examples:
        >>> len(read_ops()) >= 10
        True

    """
    return {key: ops[-1] for key, ops in read_op_history(path).items()}


def format_op(youtube_id: str, frame: int, op: HolderOp) -> str:
    """Render one ops line.

    Args:
        youtube_id: Clip id.
        frame: Source frame index.
        op: The decision.

    Returns:
        The line, without a newline.

    Examples:
        >>> format_op("abc", 120, HolderOp(None, OpStatus.TOMASZ, "loose ball"))
        'holder abc 120 none tomasz  # loose ball'

    """
    holder = "none" if op.holder is None else str(op.holder)
    line = f"holder {youtube_id} {frame} {holder} {op.status.value}"
    return f"{line}  # {op.note}" if op.note else line


def set_answer(youtube_id: str, frame: int, holder: int | None, path: Path = OPS_PATH) -> None:
    r"""Record Tomasz's answer for one frame, keeping the original line above it.

    The answer is a ``tomasz`` line right under the frame's other lines (replacing an earlier answer), so the file
    keeps both the original call and the decision.

    Args:
        youtube_id: Clip id.
        frame: Source frame index.
        holder: The holder's box index, or ``None`` for no holder.
        path: The ops file.

    Raises:
        ValueError: When the frame has no line in the ops file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = Path(tmp) / "ops.txt"
        ...     _ = p.write_text("# head\nholder abc 120 4 flagged  # held?\n", encoding="utf-8")
        ...     set_answer("abc", 120, 4, p)
        ...     set_answer("abc", 120, None, p)
        ...     p.read_text(encoding="utf-8").splitlines()
        ['# head', 'holder abc 120 4 flagged  # held?', 'holder abc 120 none tomasz  # held?']

    """
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [
        (i, p[1]) for i, raw in enumerate(lines) if (p := parse_op(raw)) is not None and p[0] == (youtube_id, frame)
    ]
    if not hits:
        raise ValueError(f"no holder line for {youtube_id} frame {frame}")
    last, last_op = hits[-1]
    replace = last_op.status is OpStatus.TOMASZ and len(hits) > 1
    note = last_op.note if replace else hits[0][1].note  # a re-answer keeps the note written on the answer
    new = format_op(youtube_id, frame, HolderOp(holder, OpStatus.TOMASZ, note))
    if replace:
        lines[last] = new
    else:
        lines.insert(last + 1, new)
    write_atomic(path, "\n".join(lines) + "\n")


def gold_frames(boxes_path: Path = BOXES_PATH) -> dict[tuple[str, int], dict[str, Any]]:
    """Read box gold v1 as ``{(youtube_id, frame): frame entry}``.

    Args:
        boxes_path: Box gold v1.

    Returns:
        Every box-gold frame by key.

    Examples:
        >>> len(gold_frames())
        100

    """
    frames = json.loads(boxes_path.read_text(encoding="utf-8"))["frames"]
    return {(fr["youtube_id"], int(fr["frame"])): fr for fr in frames}


def box_gold_holder(frame: dict[str, Any]) -> int | None:
    """Return the index of the frame's ``player-in-possession`` box, or ``None``.

    Args:
        frame: One ``boxes.json`` frame entry.

    Returns:
        The box index, or ``None`` when no possession box is drawn.

    Examples:
        >>> box_gold_holder({"boxes": [{"cls": "ball"}, {"cls": "player-in-possession"}]})
        1

    """
    return next((i for i, b in enumerate(frame["boxes"]) if b["cls"] == "player-in-possession"), None)


def check_holder(frame: dict[str, Any], holder: int | None) -> None:
    """Raise unless ``holder`` is ``None`` or the index of a player box in ``frame``.

    Args:
        frame: One ``boxes.json`` frame entry.
        holder: A box index, or ``None``.

    Raises:
        ValueError: When the index is out of range or names a ball, rim, number or referee box.

    Examples:
        >>> check_holder({"boxes": [{"cls": "ball"}]}, 0)
        Traceback (most recent call last):
        ...
        ValueError: box 0 is a ball, not a player

    """
    if holder is None:
        return
    if not 0 <= holder < len(frame["boxes"]):
        raise ValueError(f"box {holder} does not exist")
    cls = frame["boxes"][holder]["cls"]
    if cls in NOT_PEOPLE:
        raise ValueError(f"box {holder} is a {cls}, not a player")


def build_holders(
    boxes_path: Path = BOXES_PATH, ops_path: Path = OPS_PATH, out_path: Path = HOLDERS_PATH
) -> dict[str, int]:
    """Write holder gold from box gold and the ops file.

    Args:
        boxes_path: Box gold v1.
        ops_path: The ops file.
        out_path: Where to write ``holders.json``.

    Returns:
        Counts ``frames``, ``decided`` (applied lines), ``differs`` (holder differs from box gold), ``flagged`` and
        ``tomasz``.

    Raises:
        ValueError: When an op names a frame missing from box gold or a box that is not a player.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     build_holders(out_path=Path(tmp) / "holders.json")["frames"]
        100

    """
    frames = gold_frames(boxes_path)
    ops = read_ops(ops_path)
    missing = sorted(set(ops) - set(frames))
    if missing:
        raise ValueError(f"ops name frames not in box gold: {missing}")
    rows = []
    differs = 0
    for key, fr in frames.items():
        row: dict[str, Any] = {"youtube_id": key[0], "shot": fr["shot"], "frame": key[1]}
        row["holder"], row["status"] = box_gold_holder(fr), "box_gold"
        op = ops.get(key)
        if op is not None:
            check_holder(fr, op.holder)
            row["note"] = op.note
            if op.status is OpStatus.FLAGGED:
                row["status"], row["proposed"] = op.status.value, op.holder
            else:
                row["holder"], row["status"] = op.holder, op.status.value
        differs += row["holder"] != box_gold_holder(fr)
        rows.append(row)
    doc = {"version": "v1", "definition": DEFINITION, "boxes": boxes_path.as_posix(), "frames": rows}
    write_atomic(out_path, json.dumps(doc, indent=1) + "\n")
    return {
        "frames": len(rows),
        "decided": sum(r["status"] in ("implementer", "tomasz") for r in rows),
        "differs": differs,
        "flagged": sum(r["status"] == "flagged" for r in rows),
        "tomasz": sum(r["status"] == "tomasz" for r in rows),
    }
