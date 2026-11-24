"""Backend of the labeller's ``/holders`` page, where Tomasz decides the holder-gold frames flagged for him.

Every ``holder`` line of the ops file is one item: flagged lines first (the questions), then the implementer's
applied changes (for a second look). Answering an item rewrites its line as a ``tomasz`` line and rebuilds
``data/gold/v1/holders.json`` (:mod:`bga.eval.holders`).

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bga.eval.holders import (
    BOXES_PATH,
    HOLDERS_PATH,
    NOT_PEOPLE,
    OPS_PATH,
    OpStatus,
    box_gold_holder,
    build_holders,
    check_holder,
    gold_frames,
    read_op_history,
    set_answer,
)

#: Folder of per-clip label files, which name each clip's video and frame rate.
LABELS_DIR = Path("data/labels")
#: Clip shown to the held-out split.
HELD_OUT_ID = "Tnuscrm9RyU"
#: Neighbouring frames shown around each gold frame, in seconds.
STRIP_OFFSETS_S = (-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6)
#: Item order by original call: questions first.
_ORDER = {OpStatus.FLAGGED: 0, OpStatus.TOMASZ: 1, OpStatus.IMPLEMENTER: 2}


def _clip_video(youtube_id: str, labels_dir: Path) -> tuple[str, float]:
    """Return the video path (relative to the repo) and frame rate of a clip from its label file."""
    for path in sorted(labels_dir.glob(f"*_{youtube_id}_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "clip" in data:
            return data["clip"], float(data.get("fps") or 50.0)
    raise FileNotFoundError(f"no label file names a video for {youtube_id}")


def _item(key: tuple[str, int], ops: list[Any], frame: dict[str, Any], labels_dir: Path) -> dict[str, Any]:
    """Build one page item for a frame and its ops lines (the first is the original call, the last in force)."""
    video, fps = _clip_video(key[0], labels_dir)
    people = [
        {"idx": i, "cls": b["cls"], "team": b.get("team", ""), "tid": b.get("tid"), "xyxy": b["xyxy"]}
        for i, b in enumerate(frame["boxes"])
        if b["cls"] not in NOT_PEOPLE
    ]
    ball = next((b["xyxy"] for b in frame["boxes"] if b["cls"] == "ball"), None)
    first, last = ops[0], ops[-1]
    return {
        "key": f"{key[0]}:{key[1]}",
        "youtube_id": key[0],
        "shot": frame["shot"],
        "frame": key[1],
        "video": video,
        "fps": fps,
        "held_out": key[0] == HELD_OUT_ID,
        "people": people,
        "ball": ball,
        "box_gold": box_gold_holder(frame),
        "origin": first.status.value,
        "status": last.status.value,
        "question": first.note,
        "proposed": first.holder,
        "answer": last.holder if last.status is OpStatus.TOMASZ else None,
        "strip": sorted({max(0, key[1] + round(s * fps)) for s in STRIP_OFFSETS_S}),
    }


def load_holders_state(
    boxes_path: Path = BOXES_PATH, ops_path: Path = OPS_PATH, labels_dir: Path = LABELS_DIR
) -> dict[str, Any]:
    """Build the payload of ``/api/holders/state``: one item per ops line, flagged ones first.

    Args:
        boxes_path: Box gold v1.
        ops_path: The holder ops file.
        labels_dir: Folder of per-clip label files.

    Returns:
        ``{"items": [...], "ops_path": str}``.

    Examples:
        >>> st = load_holders_state()
        >>> st["items"][0]["origin"]
        'flagged'

    """
    frames = gold_frames(boxes_path)
    history = read_op_history(ops_path)
    items = [_item(key, ops, frames[key], labels_dir) for key, ops in history.items()]
    items.sort(key=lambda it: (_ORDER[OpStatus(it["origin"])], it["youtube_id"], it["frame"]))
    return {"items": items, "ops_path": ops_path.as_posix()}


def save_holder_answer(
    body: dict[str, Any],
    boxes_path: Path = BOXES_PATH,
    ops_path: Path = OPS_PATH,
    out_path: Path = HOLDERS_PATH,
) -> dict[str, Any]:
    """Record Tomasz's answer for one frame and rebuild holder gold.

    Args:
        body: ``{"youtube_id", "frame", "holder": box index | null}``.
        boxes_path: Box gold v1.
        ops_path: The holder ops file.
        out_path: Where holder gold is written.

    Returns:
        The rebuild counts from :func:`bga.eval.holders.build_holders`.

    Raises:
        ValueError: When the frame has no ops line or the holder is not a player box of that frame.

    """
    key = (str(body["youtube_id"]), int(body["frame"]))
    holder = None if body.get("holder") is None else int(body["holder"])
    frames = gold_frames(boxes_path)
    if key not in frames:
        raise ValueError(f"{key[0]} frame {key[1]} is not a box-gold frame")
    check_holder(frames[key], holder)
    set_answer(key[0], key[1], holder, ops_path)
    return build_holders(boxes_path, ops_path, out_path)
