"""Shot gold: further shots of the gold games, labelled for Layer 7 only (ball, holder and shot attempts).

Gold v1 scores Layer 7 on 16 shot attempts and 100 frames, too few to tell one more right shooter from noise. The
gold games hold many more central shots that no training set may use (:mod:`bga.train.leakage` refuses the whole
clip), so this module picks some of them and keeps their labels in one working file, ``data/labels/shot_gold.json``,
which the labeller's ``/shotgold`` page edits:

- per shot, its **shot attempts**: release frame, rim frame, made or missed, 2 or 3 points, and the shooter's box and
  team at the release frame;
- per sampled frame (one a second), the **ball** box and the **holder**'s box and team, or no holder.

Person and ball boxes offered for clicking come from the PLK detector (``scripts/shot_gold.py prefill``); a label is
whatever the implementer's first pass proposed and Tomasz confirmed or changed.

"""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Any

from bga.paths import write_atomic

#: The working file: the chosen shots, the boxes offered for clicking, the proposals and the labels.
QUEUE_PATH = Path("data/labels/shot_gold.json")
#: The exported Layer 7 shot-gold file read by ``bga eval-ball`` (confirmed answers only).
EXPORT_PATH = Path("data/gold/v1/shot_gold.json")
#: The gold-set index naming the gold games and the shots already in gold v0 and v1.
GOLD_INDEX = Path("data/labels/gold.json")
#: Folder of per-clip label files (shot cuts, camera tags, teams and kits).
LABELS_DIR = Path("data/labels")
#: Grid step of the pipeline's frames, in source frames.
STRIDE = 5
#: Seconds between two labelled frames of a shot.
FRAME_STEP_S = 1.0
#: Seconds left unlabelled at each end of a shot, where the cut makes holds hard to judge.
EDGE_S = 0.5
#: Shortest shot picked, in seconds.
MIN_SHOT_S = 6.0
#: Shot outcomes.
OUTCOMES = ("made", "missed")
#: Points of an attempt; 1 marks a free throw (decided 2026-11-26, #53), 2 and 3 field goals.
POINTS = (1, 2, 3)
#: ``free_throws`` value of a shot whose free throws were proposed for labelling, and once Tomasz confirmed them.
FT_PROPOSED, FT_LABELLED = "proposed", "labelled"
#: Sides a holder or shooter can belong to.
TEAMS = ("home", "away")
#: The holder definition shown on the page and recorded in the file.
DEFINITION = (
    "Whoever holds or dribbles the ball, mid-bounce and inbounders included; a shooter still holding the ball is "
    "the holder. A ball in flight, loose, or in a referee's hands has no holder (2026-11-20 possession schema)."
)


class Status(StrEnum):
    """Where a frame's labels or a shot's attempts stand."""

    TODO = "todo"
    FLAGGED = "flagged"
    FIRST_PASS = "first_pass"  # noqa: S105  # a status name, not a password
    DONE = "done"


def pick_shots(shots: list[dict[str, Any]], taken: set[int], fps: float, count: int) -> list[int]:
    """Pick ``count`` central shots spread evenly over a clip, skipping shots already in gold and short ones.

    The choice reads only the shot list, never a model's output, so it cannot favour shots the pipeline handles well.

    Args:
        shots: The clip's shots (``start``, ``end``, ``camera``), in order.
        taken: Shot indices already in the gold set.
        fps: The clip's frame rate.
        count: How many shots to pick.

    Returns:
        The picked shot indices, ascending; every candidate when there are no more than ``count``.

    Examples:
        >>> shots = [{"start": 400 * i, "end": 400 * i + 399, "camera": "central"} for i in range(6)]
        >>> pick_shots(shots, {1}, 50.0, 2)
        [2, 4]

    """
    cand = [
        i
        for i, s in enumerate(shots)
        if s.get("camera") == "central" and i not in taken and (s["end"] - s["start"] + 1) / fps >= MIN_SHOT_S
    ]
    if len(cand) <= count:
        return cand
    return [cand[int((k + 0.5) * len(cand) / count)] for k in range(count)]


def sample_frames(start: int, end: int, fps: float, step_s: float = FRAME_STEP_S) -> list[int]:
    """List the frames of a shot to label: one every ``step_s`` on the stride grid, away from the cuts.

    Args:
        start: The shot's first frame.
        end: The shot's last frame, included.
        fps: The clip's frame rate.
        step_s: Seconds between two labelled frames.

    Returns:
        Source frame indices, ascending.

    Examples:
        >>> sample_frames(546, 800, 50.0)
        [575, 625, 675, 725, 775]
        >>> sample_frames(546, 800, 60.0, step_s=2.0)
        [580, 700]

    """
    edge, step = round(EDGE_S * fps), max(STRIDE, round(step_s * fps / STRIDE) * STRIDE)
    first = math.ceil((start + edge) / STRIDE) * STRIDE
    return list(range(first, end - edge + 1, step))


def build_queue(
    per_dev_game: int, per_held_out_game: int, gold_index: Path = GOLD_INDEX, labels_dir: Path = LABELS_DIR
) -> dict[str, Any]:
    """Build an empty working file: the picked shots of every highlight gold game and their frames to label.

    Full-game gold games (``full_game`` in the index) join the working file through :func:`add_game` instead.

    Args:
        per_dev_game: Shots picked from each dev game.
        per_held_out_game: Shots picked from each held-out game.
        gold_index: The gold-set index.
        labels_dir: Folder of per-clip label files.

    Returns:
        The working file's content, with no boxes, proposals or labels yet.

    Examples:
        >>> q = build_queue(1, 2)
        >>> [s["held_out"] for s in q["shots"]].count(True), q["shots"][0]["frames"][0]["status"]
        (2, 'todo')

    """
    out: list[dict[str, Any]] = []
    for game in json.loads(gold_index.read_text(encoding="utf-8"))["games"]:
        if game.get("full_game"):
            continue
        labels = json.loads((labels_dir / f"{game['clip']}.json").read_text(encoding="utf-8"))
        fps = float(labels["fps"])
        count = per_held_out_game if game["held_out"] else per_dev_game
        for i in pick_shots(labels["shots"], {s["shot"] for s in game["shots"]}, fps, count):
            start, end = int(labels["shots"][i]["start"]), int(labels["shots"][i]["end"])
            out.append(_shot_record(game, i, start, end, fps, FRAME_STEP_S))
    return {"version": 1, "definition": DEFINITION, "stride": STRIDE, "shots": out}


def trim_window(start: int, end: int, fps: float, max_s: float) -> tuple[int, int]:
    """Keep the middle ``max_s`` seconds of a shot; a shot no longer than that is kept whole.

    Full-game shots of the main camera can run for minutes; the middle window keeps a sequence's labelling load fixed
    without looking at what happens in it.

    Args:
        start: The shot's first frame.
        end: The shot's last frame, included.
        fps: The clip's frame rate.
        max_s: Longest window kept, in seconds.

    Returns:
        The window's ``(start, end)``, inclusive.

    Examples:
        >>> trim_window(0, 5999, 60.0, 40.0)
        (1800, 4199)
        >>> trim_window(100, 999, 60.0, 40.0)
        (100, 999)

    """
    keep = round(max_s * fps)
    if end - start + 1 <= keep:
        return start, end
    first = start + (end - start + 1 - keep) // 2
    return first, first + keep - 1


def add_game(
    queue: dict[str, Any], game: dict[str, Any], labels: dict[str, Any], count: int, step_s: float, max_s: float
) -> list[str]:
    """Append ``count`` shots of one more game to the working file, keeping every shot and answer already in it.

    Shots are chosen by :func:`pick_shots` from the game's tagged shot list (so never from a model's output), cut to
    their middle ``max_s`` seconds, and labelled every ``step_s`` seconds.

    Args:
        queue: The working file's content; edited in place.
        game: The game's entry in the gold-set index (``clip``, ``youtube_id``, ``held_out``, ``shots``).
        labels: The game's label file (``fps`` and the tagged ``shots``).
        count: Shots to pick.
        step_s: Seconds between two labelled frames.
        max_s: Longest window kept of a shot, in seconds.

    Returns:
        The sequence names added (none when the game is already in the working file).

    Raises:
        ValueError: When the game already has shots in the working file.

    Examples:
        >>> shots = [{"start": 3000 * i, "end": 3000 * i + 2999, "camera": "central"} for i in range(4)]
        >>> game = {"clip": "c", "youtube_id": "yt", "held_out": False, "shots": []}
        >>> q = {"shots": []}
        >>> add_game(q, game, {"fps": 60.0, "shots": shots}, 2, 2.0, 20.0)
        ['yt_s1', 'yt_s3']
        >>> q["shots"][0]["start"], q["shots"][0]["end"], len(q["shots"][0]["frames"])
        (3900, 5099, 10)

    """
    if any(s["youtube_id"] == game["youtube_id"] for s in queue["shots"]):
        raise ValueError(f"{game['youtube_id']} already has shots in the working file")
    fps = float(labels["fps"])
    added = []
    for i in pick_shots(labels["shots"], {s["shot"] for s in game["shots"]}, fps, count):
        start, end = trim_window(int(labels["shots"][i]["start"]), int(labels["shots"][i]["end"]), fps, max_s)
        queue["shots"].append(_shot_record(game, i, start, end, fps, step_s))
        added.append(queue["shots"][-1]["sequence"])
    return added


def _shot_record(game: dict[str, Any], shot: int, start: int, end: int, fps: float, step_s: float) -> dict[str, Any]:
    """Return an unlabelled shot record of the working file."""
    return {
        "sequence": f"{game['youtube_id']}_s{shot}",
        "youtube_id": game["youtube_id"],
        "clip": game["clip"],
        "shot": shot,
        "start": start,
        "end": end,
        "fps": fps,
        "held_out": bool(game["held_out"]),
        "events_status": Status.TODO.value,
        "events": [],
        "frames": [_empty_frame(f) for f in sample_frames(start, end, fps, step_s)],
    }


def _empty_frame(frame: int) -> dict[str, Any]:
    """Return an unlabelled frame record; ``origin`` keeps what the first pass made of it once Tomasz answers."""
    return {
        "frame": frame,
        "status": Status.TODO.value,
        "proposal": None,
        "ball": None,
        "holder": None,
        "origin": Status.TODO.value,
        "note": "",
    }


def load_queue(path: Path = QUEUE_PATH) -> dict[str, Any]:
    """Read the working file.

    Args:
        path: The working file.

    Returns:
        Its content.

    Examples:
        >>> load_queue()["version"]
        1

    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_queue(queue: dict[str, Any], path: Path = QUEUE_PATH) -> None:
    """Write the working file whole, so a stopped write never leaves half a file.

    Args:
        queue: The working file's content.
        path: Where to write it.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     save_queue({"version": 1, "shots": []}, Path(tmp) / "q.json")
        ...     load_queue(Path(tmp) / "q.json")["shots"]
        []

    """
    write_atomic(path, json.dumps(queue, indent=1, ensure_ascii=False) + "\n")


def find_shot(queue: dict[str, Any], sequence: str) -> dict[str, Any]:
    """Return the shot record of a sequence.

    Args:
        queue: The working file's content.
        sequence: ``<youtube id>_s<shot>``.

    Returns:
        The shot record (the object inside ``queue``, so edits to it are edits to the queue).

    Raises:
        ValueError: When the queue has no such sequence.

    Examples:
        >>> find_shot({"shots": [{"sequence": "a_s1"}]}, "a_s1")
        {'sequence': 'a_s1'}

    """
    for shot in queue["shots"]:
        if shot["sequence"] == sequence:
            return shot
    raise ValueError(f"{sequence} is not a shot-gold sequence")


def clean_box(box: object) -> list[float]:
    """Validate an ``[x1, y1, x2, y2]`` box and round it to a tenth of a pixel.

    Args:
        box: Four numbers, top-left then bottom-right.

    Returns:
        The box as a list of four floats.

    Raises:
        ValueError: When it is not four finite numbers with a positive width and height.

    Examples:
        >>> clean_box([10, 20.26, 30, 40])
        [10.0, 20.3, 30.0, 40.0]

    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"a box is four numbers, got {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"not a box with a positive size: {box!r}")
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def clean_person(person: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate a holder or shooter: a box and a side, or nobody.

    Args:
        person: ``{"xyxy": box, "team": "home" | "away"}`` or ``None``.

    Returns:
        The cleaned record, or ``None`` for nobody.

    Raises:
        ValueError: When the box or the team is not valid.

    Examples:
        >>> clean_person({"xyxy": [0, 0, 10, 20], "team": "away"})
        {'xyxy': [0.0, 0.0, 10.0, 20.0], 'team': 'away'}
        >>> clean_person(None) is None
        True

    """
    if person is None:
        return None
    if person.get("team") not in TEAMS:
        raise ValueError(f"team must be one of {TEAMS}, got {person.get('team')!r}")
    return {"xyxy": clean_box(person.get("xyxy")), "team": person["team"]}


def clean_event(event: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
    """Validate one shot attempt of a shot.

    Args:
        event: ``release_frame``, ``rim_frame``, ``outcome``, ``points`` (1 for a free throw), ``shooter`` and an
            optional ``note``.
        shot: The shot record it belongs to.

    Returns:
        The cleaned attempt.

    Raises:
        ValueError: When a frame lies outside the shot, the release is after the rim frame, or a field is not valid.

    Examples:
        >>> shot = {"sequence": "a_s1", "start": 100, "end": 400}
        >>> ev = {"release_frame": 200, "rim_frame": 240, "outcome": "made", "points": 3, "shooter": None}
        >>> clean_event(ev, shot)["points"]
        3
        >>> clean_event({**ev, "points": 1}, shot)["points"]
        1
        >>> clean_event({**ev, "rim_frame": 150}, shot)
        Traceback (most recent call last):
        ...
        ValueError: a_s1: release frame 200 is after rim frame 150

    """
    release, rim = int(event["release_frame"]), int(event["rim_frame"])
    for name, f in (("release", release), ("rim", rim)):
        if not shot["start"] <= f <= shot["end"]:
            raise ValueError(f"{shot['sequence']}: {name} frame {f} is outside the shot")
    if release > rim:
        raise ValueError(f"{shot['sequence']}: release frame {release} is after rim frame {rim}")
    if event.get("outcome") not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {event.get('outcome')!r}")
    if event.get("points") not in POINTS:
        raise ValueError(f"points must be one of {POINTS}, got {event.get('points')!r}")
    return {
        "release_frame": release,
        "rim_frame": rim,
        "outcome": event["outcome"],
        "points": int(event["points"]),
        "shooter": clean_person(event.get("shooter")),
        "note": str(event.get("note") or ""),
    }


def set_frame_label(queue: dict[str, Any], body: dict[str, Any], status: Status = Status.DONE) -> dict[str, Any]:
    """Record the ball and holder of one frame.

    Args:
        queue: The working file's content, changed in place.
        body: ``{"sequence", "frame", "ball": box | None, "holder": person | None}`` and an optional ``note``.
        status: ``DONE`` for Tomasz's answer, ``FIRST_PASS`` for the implementer's settled read.

    Returns:
        The changed frame record.

    Raises:
        ValueError: When the sequence or frame is not in the queue, or a box or team is not valid.

    Examples:
        >>> q = {"shots": [{"sequence": "a_s1", "frames": [{"frame": 5, "status": "todo", "ball": None}]}]}
        >>> body = {"sequence": "a_s1", "frame": 5, "ball": [1, 1, 9, 9], "holder": None}
        >>> set_frame_label(q, body)["status"]
        'done'

    """
    shot = find_shot(queue, str(body["sequence"]))
    rec = next((f for f in shot["frames"] if f["frame"] == int(body["frame"])), None)
    if rec is None:
        raise ValueError(f"{shot['sequence']} has no labelled frame {body['frame']}")
    rec["ball"] = None if body.get("ball") is None else clean_box(body["ball"])
    rec["holder"] = clean_person(body.get("holder"))
    rec["note"] = str(body.get("note") or rec.get("note") or "")
    rec["status"] = status.value
    if status is not Status.DONE:
        rec["origin"] = status.value
    return rec


def set_shot_events(queue: dict[str, Any], body: dict[str, Any], status: Status = Status.DONE) -> dict[str, Any]:
    """Record the shot attempts of one shot, replacing the earlier list.

    Tomasz's answer on a shot whose free throws were proposed (``free_throws == "proposed"``) also marks its free
    throws as labelled, so free-throw scores only count shots where free throws were looked for.

    Args:
        queue: The working file's content, changed in place.
        body: ``{"sequence", "events": [attempt, ...]}``.
        status: ``DONE`` for Tomasz's answer, ``FIRST_PASS`` for the implementer's proposal.

    Returns:
        The changed shot record.

    Raises:
        ValueError: When the sequence is not in the queue or an attempt is not valid.

    Examples:
        >>> q = {"shots": [{"sequence": "a_s1", "start": 0, "end": 99, "events": [], "events_status": "todo"}]}
        >>> set_shot_events(q, {"sequence": "a_s1", "events": []})["events_status"]
        'done'
        >>> q["shots"][0]["free_throws"] = "proposed"
        >>> set_shot_events(q, {"sequence": "a_s1", "events": []})["free_throws"]
        'labelled'

    """
    shot = find_shot(queue, str(body["sequence"]))
    events = [clean_event(ev, shot) for ev in body.get("events", [])]
    shot["events"] = sorted(events, key=lambda ev: ev["rim_frame"])
    shot["events_status"] = status.value
    if status == Status.DONE and shot.get("free_throws") == FT_PROPOSED:
        shot["free_throws"] = FT_LABELLED
    return shot


def progress(queue: dict[str, Any]) -> dict[str, int]:
    """Count the frames and shots Tomasz has answered and those still open.

    Args:
        queue: The working file's content.

    Returns:
        ``frames``, ``frames_done``, ``frames_flagged``, ``shots`` and ``shots_done``.

    Examples:
        >>> q = {"shots": [{"events_status": "done", "frames": [{"status": "done"}, {"status": "flagged"}]}]}
        >>> progress(q)
        {'frames': 2, 'frames_done': 1, 'frames_flagged': 1, 'shots': 1, 'shots_done': 1}

    """
    frames = [f for s in queue["shots"] for f in s["frames"]]
    return {
        "frames": len(frames),
        "frames_done": sum(f["status"] == Status.DONE for f in frames),
        "frames_flagged": sum(f["status"] == Status.FLAGGED for f in frames),
        "shots": len(queue["shots"]),
        "shots_done": sum(s["events_status"] == Status.DONE for s in queue["shots"]),
    }


def export_shot_gold(queue_path: Path = QUEUE_PATH, out_path: Path = EXPORT_PATH) -> dict[str, Any]:
    """Export confirmed shot-gold answers to ``data/gold/v1/shot_gold.json`` for ``bga eval-ball``.

    Only shots with ``events_status == "done"`` enter ``verified_sequences`` and ``events``, and only frames with
    ``status == "done"`` enter ``frames``. ``free_throw_sequences`` lists the verified shots whose free throws were
    labelled too (1-point attempts); elsewhere free throws were left out. Working-file draft fields (``proposal``,
    ``origin``, ``events_note``) are omitted.

    Args:
        queue_path: Working file edited by ``/shotgold``.
        out_path: Destination gold file under ``data/gold/v1/``.

    Returns:
        The exported document.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     qp = Path(tmp) / "q.json"
        ...     op = Path(tmp) / "out.json"
        ...     shot = {"sequence": "a_s1", "youtube_id": "a", "clip": "c", "shot": 1, "start": 0, "end": 100,
        ...             "fps": 50.0, "held_out": False, "events_status": "done",
        ...             "events": [{"release_frame": 20, "rim_frame": 40, "outcome": "made", "points": 2,
        ...                         "shooter": {"xyxy": [1.0, 2.0, 3.0, 4.0], "team": "home"}, "note": ""}],
        ...             "frames": [{"frame": 50, "status": "done", "ball": [5.0, 5.0, 9.0, 9.0], "holder": None}]}
        ...     save_queue({"version": 1, "definition": DEFINITION, "stride": STRIDE, "shots": [shot]}, qp)
        ...     doc = export_shot_gold(qp, op)
        ...     (doc["verified_sequences"], len(doc["frames"]), len(doc["events"]), doc["events"][0]["shooter_team"])
        (['a_s1'], 1, 1, 'home')

    """
    queue = load_queue(queue_path)
    shots: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    verified: list[str] = []

    for shot in queue["shots"]:
        done_frames = [f for f in shot["frames"] if f["status"] == Status.DONE]
        ev_done = shot["events_status"] == Status.DONE
        if not done_frames and not ev_done:
            continue
        shots.append({
            "sequence": shot["sequence"],
            "youtube_id": shot["youtube_id"],
            "clip": shot["clip"],
            "shot": int(shot["shot"]),
            "start": int(shot["start"]),
            "end": int(shot["end"]),
            "fps": float(shot["fps"]),
            "held_out": bool(shot["held_out"]),
        })
        frames.extend(
            {
                "sequence": shot["sequence"],
                "youtube_id": shot["youtube_id"],
                "shot": int(shot["shot"]),
                "frame": int(fr["frame"]),
                "held_out": bool(shot["held_out"]),
                "ball": fr["ball"],
                "holder": fr["holder"],
            }
            for fr in done_frames
        )
        if ev_done:
            verified.append(shot["sequence"])
            for ev in shot["events"]:
                shooter = ev.get("shooter")
                events.append({
                    "sequence": shot["sequence"],
                    "youtube_id": shot["youtube_id"],
                    "shot": int(shot["shot"]),
                    "held_out": bool(shot["held_out"]),
                    "release_frame": int(ev["release_frame"]),
                    "rim_frame": int(ev["rim_frame"]),
                    "outcome": ev["outcome"],
                    "points": int(ev["points"]),
                    "shooter_xyxy": shooter["xyxy"] if shooter else None,
                    "shooter_team": shooter["team"] if shooter else None,
                    "note": str(ev.get("note") or ""),
                })

    doc: dict[str, Any] = {
        "version": "v1",
        "definition": queue.get("definition", DEFINITION),
        "stride": int(queue.get("stride", STRIDE)),
        "verified_sequences": sorted(verified),
        "free_throw_sequences": sorted(
            s["sequence"] for s in queue["shots"] if s.get("free_throws") == FT_LABELLED and s["sequence"] in verified
        ),
        "shots": shots,
        "frames": frames,
        "events": events,
    }
    write_atomic(out_path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc
