"""Backend of the labeller's ``/shotgold`` page, where Tomasz labels the shot-gold shots.

The page has two jobs per shot (:mod:`bga.eval.shot_gold`): the shot attempts, and the ball and holder on one frame a
second. This module builds the page's payload from the working file, serves the detector boxes offered for clicking
(a cache written by ``scripts/shot_gold.py prefill``), and records each answer in the working file at once.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bga.eval.shot_gold import (
    DEFINITION,
    LABELS_DIR,
    QUEUE_PATH,
    STRIDE,
    load_queue,
    progress,
    save_queue,
    set_frame_label,
    set_shot_events,
)
from bga.ingest.clip_shots import shot_frames

#: Detector boxes of every grid frame of each shot, one file per sequence (labeller cache, not committed).
BOXES_DIR = Path("data/labels/.cache/shot_gold")
#: Neighbouring frames shown around each labelled frame, in seconds.
STRIP_OFFSETS_S = (-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6)


def _side(meta: dict[str, Any], side: str) -> dict[str, str]:
    """Return a side's team name and kit from a clip's meta."""
    info = meta.get(side) or {}
    return {"team": str(info.get("team") or side), "kit": str(info.get("kit") or "")}


def _strip(frame: int, shot: dict[str, Any]) -> list[int]:
    """List the frames shown around a labelled frame, kept inside the shot."""
    near = {frame + round(s * shot["fps"]) for s in STRIP_OFFSETS_S}
    return sorted(f for f in near if shot["start"] <= f <= shot["end"])


def load_shotgold_state(queue_path: Path = QUEUE_PATH, labels_dir: Path = LABELS_DIR) -> dict[str, Any]:
    """Build the payload of ``/api/shotgold/state``: every shot with its attempts, frames, video and teams.

    Args:
        queue_path: The shot-gold working file.
        labels_dir: Folder of per-clip label files.

    Returns:
        ``{"definition", "progress", "shots": [...]}``; each shot adds ``video``, ``home``, ``away``, ``grid`` (its
        frames on the stride grid) and, per labelled frame, ``strip`` (the neighbouring frames to show).

    Examples:
        >>> st = load_shotgold_state()
        >>> st["shots"][0]["grid"][0] % 5, st["shots"][0]["home"]["kit"]
        (0, 'light')

    """
    queue = load_queue(queue_path)
    shots = []
    for shot in queue["shots"]:
        labels = json.loads((labels_dir / f"{shot['clip']}.json").read_text(encoding="utf-8"))
        meta = labels.get("meta", {})
        shots.append({
            **shot,
            "video": labels["clip"],
            "home": _side(meta, "home"),
            "away": _side(meta, "away"),
            "grid": shot_frames(shot["start"], shot["end"], STRIDE),
            "frames": [{**f, "strip": _strip(f["frame"], shot)} for f in shot["frames"]],
        })
    return {"definition": DEFINITION, "progress": progress(queue), "shots": shots}


def load_boxes(sequence: str, boxes_dir: Path = BOXES_DIR) -> dict[str, Any]:
    """Return the detector boxes of a sequence's grid frames, for clicking players and snapping the ball.

    Args:
        sequence: ``<youtube id>_s<shot>``.
        boxes_dir: Folder of the per-sequence box caches.

    Returns:
        ``{"frames": {"<frame>": {"people": [[x1, y1, x2, y2, score, cls], ...], "balls": [[x1, y1, x2, y2, score],
        ...]}}}``; empty when the cache is missing, in which case the page still takes drawn boxes.

    Raises:
        ValueError: When the sequence name is not a plain file stem.

    Examples:
        >>> load_boxes("no_such_s1")
        {'frames': {}}

    """
    if not sequence or any(c in sequence for c in "/\\.") or sequence != Path(sequence).name:
        raise ValueError(f"bad sequence name: {sequence!r}")
    path = boxes_dir / f"{sequence}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"frames": {}}


def save_frame_answer(body: dict[str, Any], queue_path: Path = QUEUE_PATH) -> dict[str, Any]:
    """Record Tomasz's ball and holder for one frame.

    Args:
        body: ``{"sequence", "frame", "ball": box | null, "holder": {"xyxy", "team"} | null}``.
        queue_path: The shot-gold working file.

    Returns:
        The progress counts after the save.

    Raises:
        ValueError: When the frame is not in the queue or a box or team is not valid.

    Examples:
        >>> save_frame_answer({"sequence": "nope_s1", "frame": 5})
        Traceback (most recent call last):
        ...
        ValueError: nope_s1 is not a shot-gold sequence

    """
    queue = load_queue(queue_path)
    set_frame_label(queue, body)
    save_queue(queue, queue_path)
    return progress(queue)


def save_shot_events(body: dict[str, Any], queue_path: Path = QUEUE_PATH) -> dict[str, Any]:
    """Record Tomasz's shot attempts for one shot, replacing the earlier list.

    Args:
        body: ``{"sequence", "events": [{"release_frame", "rim_frame", "outcome", "points", "shooter"}, ...]}``.
        queue_path: The shot-gold working file.

    Returns:
        The progress counts after the save.

    Raises:
        ValueError: When the shot is not in the queue or an attempt is not valid.

    Examples:
        >>> save_shot_events({"sequence": "nope_s1", "events": []})
        Traceback (most recent call last):
        ...
        ValueError: nope_s1 is not a shot-gold sequence

    """
    queue = load_queue(queue_path)
    set_shot_events(queue, body)
    save_queue(queue, queue_path)
    return progress(queue)
