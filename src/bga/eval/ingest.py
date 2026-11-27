"""Score Layer 0 on the gold games (M10, #32): camera tags and shot cuts.

**Camera:** every shot of a gold game's label file tagged ``central`` or ``other`` is scored by the configured
camera backend on the label file's own shot bounds, so a camera row does not depend on the cut detector. The split
follows ``data/labels/gold.json``: dev games unless ``held_out``.

**Cuts:** the configured cut detector's cuts inside each game's game-time spans are matched to the cut gold
``data/labels/cut_gold.json`` (``scripts/cut_review.py``). The gold holds every cut either of two detectors made in
game time: each one only one detector made is read by eye, and the ones both made are read on a random sample and
otherwise taken as real; the further cuts ``transnetv2`` makes at threshold 0.3 are all read and added. A detected cut
is a hit when it falls within ``tol_s`` of a real hard cut, or on a gradual transition's blended frames (or the frame
after them) within ``tol_s``. A second cut on the same transition is a duplicate, neither hit nor false positive, since
``shots/video`` drops the sliver of a shot between them. Recall is over the cuts in the gold; a cut every gold detector
missed is not in it.

"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from bga.ingest.camera import TAGS, read_shot_frames
from bga.ingest.video import VideoShots, make_component

#: The gold games file naming dev and held-out games.
GOLD_JSON = Path("data/labels/gold.json")
#: The reviewed cut gold of the gold games, written by ``scripts/cut_review.py``.
CUT_GOLD_JSON = Path("data/labels/cut_gold.json")


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


@dataclass(frozen=True, slots=True)
class Transition:
    """One cut gold entry: the frames a detected cut may start a shot on, and whether the entry is a real cut.

    Examples:
        >>> Transition(lo=95, hi=105, real=True, reviewed=True).holds(100)
        True

    """

    lo: int
    hi: int
    real: bool
    reviewed: bool

    def holds(self, frame: int) -> bool:
        """Whether a cut starting a shot on ``frame`` falls on this entry.

        Args:
            frame: First frame of a detected shot.

        Returns:
            ``lo <= frame <= hi``.

        Examples:
            >>> Transition(lo=95, hi=105, real=False, reviewed=True).holds(106)
            False

        """
        return self.lo <= frame <= self.hi


@dataclass(frozen=True, slots=True)
class CutCounts:
    """Detected cuts matched to the cut gold of one or more games.

    Examples:
        >>> c = CutCounts(tp=8, fp=2, fn=2, dup=1, unseen=0) + CutCounts(tp=2, fp=0, fn=0, dup=0, unseen=0)
        >>> (c.tp, c.precision, c.recall)
        (10, 0.8333333333333334, 0.8333333333333334)

    """

    tp: int
    fp: int
    fn: int
    dup: int
    unseen: int

    def __add__(self, other: CutCounts) -> CutCounts:
        """Counts of two sets of games together."""
        return CutCounts(*(a + b for a, b in zip(self._values(), other._values(), strict=True)))

    def _values(self) -> tuple[int, int, int, int, int]:
        return (self.tp, self.fp, self.fn, self.dup, self.unseen)

    @property
    def precision(self) -> float:
        """Hits over hits and false positives."""
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        """Hits over real gold cuts."""
        return self.tp / max(self.tp + self.fn, 1)


def gold_transitions(game: dict, tol: int) -> list[Transition]:
    """The windows of one game's cut gold entries, a gradual transition listed once however many entries name it.

    Args:
        game: One game of the cut gold, with ``transitions`` of ``frame``, ``verdict`` (``cut``, ``gradual`` with
            ``first`` and ``last`` blended frames, or ``no``) and ``reviewed``.
        tol: Tolerance in frames.

    Returns:
        One :class:`Transition` per hard cut, distinct gradual transition, or rejected candidate.

    Examples:
        >>> game = {"transitions": [
        ...     {"frame": 100, "verdict": "cut", "reviewed": False},
        ...     {"frame": 200, "verdict": "gradual", "first": 198, "last": 205, "reviewed": True},
        ...     {"frame": 206, "verdict": "gradual", "first": 198, "last": 205, "reviewed": True},
        ...     {"frame": 300, "verdict": "no", "reviewed": True}]}
        >>> [(t.lo, t.hi, t.real) for t in gold_transitions(game, tol=2)]
        [(98, 102, True), (196, 208, True), (298, 302, False)]

    """
    out: dict[tuple[int, int, bool], Transition] = {}
    for t in game["transitions"]:
        lo, hi = (t["first"], t["last"] + 1) if t["verdict"] == "gradual" else (t["frame"], t["frame"])
        key = (lo - tol, hi + tol, t["verdict"] != "no")
        out.setdefault(key, Transition(*key, reviewed=t["reviewed"]))
    return list(out.values())


def match_cuts(cuts: list[int], gold: list[Transition], spans: list[list[int]]) -> CutCounts:
    """Match a detector's cuts inside the game-time spans to the cut gold.

    Args:
        cuts: First frames of every detected shot but the first.
        gold: The game's :func:`gold_transitions`.
        spans: Game-time ``[first, last]`` frame spans; cuts outside them are not scored.

    Returns:
        Hits, false positives, missed real cuts, duplicates, and the false positives on no gold entry at all
        (``unseen``: cuts nobody read, which a new detector may need reviewed).

    Examples:
        >>> gold = [Transition(98, 102, True, True), Transition(196, 208, True, True)]
        >>> gold.append(Transition(298, 302, False, True))
        >>> match_cuts([50, 101, 199, 207, 300, 400, 999], gold, [[0, 500]])
        CutCounts(tp=2, fp=3, fn=0, dup=1, unseen=2)

    """
    real = [t for t in gold if t.real]
    hit = [False] * len(real)
    tp = fp = dup = unseen = 0
    for c in cuts:
        if not any(s <= c <= e for s, e in spans):
            continue
        free = [i for i, t in enumerate(real) if t.holds(c) and not hit[i]]
        if free:
            hit[free[0]] = True
            tp += 1
        elif any(t.holds(c) for t in real):
            dup += 1
        else:
            fp += 1
            unseen += not any(t.holds(c) for t in gold)
    return CutCounts(tp=tp, fp=fp, fn=hit.count(False), dup=dup, unseen=unseen)


@dataclass(frozen=True, slots=True)
class CutScore:
    """A cut detector's score over the cut gold of one split.

    Examples:
        >>> s = CutScore(counts=CutCounts(9, 1, 1, 0, 0), games=2, per_game={}, sample_read=40, sample_real=40)
        >>> round(s.counts.precision, 2)
        0.9

    """

    counts: CutCounts
    games: int
    per_game: dict[str, CutCounts]
    sample_read: int
    sample_real: int


def score_cuts(
    shots_params: dict, held_out: bool = False, labels: Path = Path("data/labels"), cut_gold: Path = CUT_GOLD_JSON
) -> CutScore:
    """Score a ``shots/video`` config's cut detector on the cut gold of the split's games.

    Cuts come from the config's cache (:meth:`bga.ingest.video.VideoShots.game_cuts`), computed once per video.

    Args:
        shots_params: The ``shots`` layer params of a ``shots/video`` config.
        held_out: Score the held-out games (once per milestone).
        labels: The label folder.
        cut_gold: The cut gold file.

    Returns:
        Counts over all games and per YouTube id, and the agreement sample read (entries and real cuts among them).

    Raises:
        ValueError: When a game of the split has no cut gold.

    Examples:
        >>> score_cuts({"cuts": {"backend": "hist_diff", "params": {}}}).counts.recall  # doctest: +SKIP
        0.9

    """
    gold = json.loads(cut_gold.read_text())
    shots = VideoShots(shots_params)
    total, per_game, read, real = CutCounts(0, 0, 0, 0, 0), {}, 0, 0
    for clip in split_clips(held_out):
        if clip not in gold["games"]:
            raise ValueError(f"{clip} has no cut gold in {cut_gold}; review it with scripts/cut_review.py")
        game = gold["games"][clip]
        label = json.loads((labels / f"{clip}.json").read_text())
        found = shots.game_cuts(clip, label["clip"])
        tol = max(1, round(gold["tol_s"] * game["fps"]))
        counts = match_cuts([s["start"] for s in found[1:]], gold_transitions(game, tol), game["spans"])
        per_game[clip.split("_")[1]] = counts
        total += counts
        read += game["agree_sample"]["read"]
        real += game["agree_sample"]["real"]
    return CutScore(counts=total, games=len(per_game), per_game=per_game, sample_read=read, sample_real=real)
