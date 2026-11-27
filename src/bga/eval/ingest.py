"""Score Layer 0 on the gold games (M10, #32): camera tags now, shot cuts once cut gold exists.

**Camera:** every shot of a gold game's label file tagged ``central`` or ``other`` is scored by the configured
camera backend on the label file's own shot bounds, so a camera row does not depend on the cut detector. The split
follows ``data/labels/gold.json``: dev games unless ``held_out``.

"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from bga.ingest.camera import TAGS, read_shot_frames
from bga.ingest.video import make_component

#: The gold games file naming dev and held-out games.
GOLD_JSON = Path("data/labels/gold.json")


@dataclass(frozen=True, slots=True)
class CameraScore:
    """Camera accuracy over the tagged shots of one split.

    Examples:
        >>> s = CameraScore(shots=10, games=2, acc=0.9, central_p=0.8, central_r=1.0, per_game={}, ms_per_shot=5.0)
        >>> s.acc
        0.9

    """

    shots: int
    games: int
    acc: float
    central_p: float
    central_r: float
    per_game: dict[str, float]
    ms_per_shot: float


def split_clips(held_out: bool, gold_json: Path = GOLD_JSON) -> list[str]:
    """Label stems of the gold games of one split.

    Args:
        held_out: The held-out games instead of the dev games.
        gold_json: The gold games file.

    Returns:
        Label stems in file order.

    Examples:
        >>> "20260131_vHfwZeG4BPw_ZAPIS_MECZU_Energa_Trefl_Sopot_-_Dziki_Warszawa_31.01.20" in split_clips(False)
        True

    """
    games = json.loads(gold_json.read_text())["games"]
    return [g["clip"] for g in games if bool(g["held_out"]) == held_out]


def score_camera(camera: dict, samples: int, held_out: bool = False, labels: Path = Path("data/labels")) -> CameraScore:
    """Score a camera backend on every tagged shot of the split's games.

    Args:
        camera: The backend entry ``{"backend", "params"}``.
        samples: Frames per shot.
        held_out: Score the held-out games (once per milestone).
        labels: The label folder.

    Returns:
        Accuracy over ``central`` / ``other`` tags, central precision and recall, accuracy per YouTube id, and the
        classifier's time per shot (frame reading excluded).

    Examples:
        >>> score_camera({"backend": "siglip_zeroshot", "params": {}}, samples=3).acc  # doctest: +SKIP
        0.8

    """
    cam = make_component("camera", camera)
    cam.load()
    pairs: list[tuple[str, bool, bool]] = []  # (youtube id, gold central, predicted central)
    seconds = 0.0
    try:
        for clip in split_clips(held_out):
            label = json.loads((labels / f"{clip}.json").read_text())
            shots = [s for s in label["shots"] if s.get("camera") in TAGS]
            frames = read_shot_frames(label["clip"], shots, samples)
            t = time.perf_counter()
            scores = cam.score_shots(frames)
            seconds += time.perf_counter() - t
            pairs += [
                (clip.split("_")[1], s["camera"] == "central", p >= 0.5) for s, p in zip(shots, scores, strict=True)
            ]
    finally:
        cam.unload()
    tp = sum(g and p for _, g, p in pairs)
    per_game = {}
    for yid in dict.fromkeys(y for y, _, _ in pairs):
        hits = [g == p for y, g, p in pairs if y == yid]
        per_game[yid] = sum(hits) / len(hits)
    return CameraScore(
        shots=len(pairs),
        games=len(per_game),
        acc=sum(g == p for _, g, p in pairs) / max(len(pairs), 1),
        central_p=tp / max(sum(p for _, _, p in pairs), 1),
        central_r=tp / max(sum(g for _, g, _ in pairs), 1),
        per_game=per_game,
        ms_per_shot=1000 * seconds / max(len(pairs), 1),
    )
