"""Appearance scores on gold boxes (plan section 8): can an encoder tell teammates apart, within a shot and across cuts.

The config's appearance backend embeds the clean gold boxes (visibility 1: neither interpolated nor occluded) of the
selected gold shots, so the scores depend on neither the detector nor the tracker. Each question ranks candidates by
cosine distance to a query and reports rank-1 (the right player is nearest) and MRR:

- **re-association**: one crop of a player against the crops of his teammates 1 s and 3 s later in the same shot.
  Queries are taken every 0.5 s.
- **cross-shot**: the mean embedding of a numbered player in one shot against the mean embeddings of his numbered
  teammates in another shot of the same game.

This is the metric form of ``scripts/probe_reid.py`` (2026-11-10), which also scores the pedestrian encoders.

"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bga.config import RunConfig
from bga.pipeline import make_backend, sequences
from bga.stage import FrameReader, Sequence

#: Re-association gaps in seconds.
GAPS_S = (1, 3)
#: Seconds between query frames, so neighbouring queries differ.
QUERY_EVERY_S = 0.5
#: Boxes a player needs in a shot for a mean embedding.
MIN_BOXES = 5
#: Gold classes of players: 1 home, 2 away (3 referee, 4 other).
PLAYER_CLASSES = (1, 2)


@dataclass(frozen=True)
class GoldBoxes:
    """The clean gold boxes of one gold shot, one entry per box.

    Attributes:
        name: Sequence name.
        game: Clip stem; shots of one clip are one game.
        step: Index of the box's frame on the gold grid (every 5th source frame).
        tid: Gold track id.
        cls: Gold class (1 home, 2 away, 3 referee, 4 other).
        xyxy: Boxes as an ``(n, 4)`` array in pixels.
        source: Source frame of each box.
        number: Jersey number per gold track id, for tracks with one.
        stride: Source frames between grid steps.

    """

    name: str
    game: str
    step: np.ndarray
    tid: np.ndarray
    cls: np.ndarray
    xyxy: np.ndarray
    source: np.ndarray
    number: dict[int, str]
    stride: int = 5


def read_gold_boxes(root: Path, name: str, game: str) -> GoldBoxes:
    r"""Read the clean boxes and jersey numbers of one exported gold sequence.

    Args:
        root: Gold version directory (``data/gold/<version>``).
        name: Sequence name.
        game: The sequence's clip stem.

    Returns:
        Its boxes with visibility 1.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     seq = Path(d) / "s"
        ...     (seq / "gt").mkdir(parents=True)
        ...     _ = (seq / "frames.txt").write_text("100\n105\n")
        ...     _ = (seq / "gt" / "gt.txt").write_text("1,7,0,0,10,20,1,1,1.0\n2,7,1,0,10,20,1,1,0.5\n")
        ...     _ = (seq / "identities.csv").write_text("tid,team,number\n7,home,23\n")
        ...     g = read_gold_boxes(Path(d), "s", "game")
        >>> g.step.tolist(), g.source.tolist(), g.xyxy.tolist(), g.number, g.stride
        ([0], [100], [[0.0, 0.0, 10.0, 20.0]], {7: '23'}, 5)

    """
    d = root / name
    grid = np.array([int(x) for x in (d / "frames.txt").read_text().split()])
    rows = np.loadtxt(d / "gt" / "gt.txt", delimiter=",", ndmin=2)
    rows = rows[rows[:, 8] >= 1.0]
    with (d / "identities.csv").open() as f:
        number = {int(r["tid"]): r["number"] for r in csv.DictReader(f) if r["number"]}
    step = rows[:, 0].astype(int) - 1
    x, y, w, h = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
    return GoldBoxes(
        name=name,
        game=game,
        step=step,
        tid=rows[:, 1].astype(int),
        cls=rows[:, 7].astype(int),
        xyxy=np.column_stack([x, y, x + w, y + h]),
        source=grid[step],
        number=number,
        stride=int(grid[1] - grid[0]) if len(grid) > 1 else 1,
    )


@dataclass(frozen=True)
class RankSummary:
    """Rank-1 and MRR over a set of queries, with the chance level.

    Attributes:
        n: Number of queries.
        candidates: Mean number of candidates per query.
        chance: Mean rank-1 a random ranking would score.
        rank1: Share of queries whose right candidate is nearest.
        mrr: Mean reciprocal rank of the right candidate.

    """

    n: int
    candidates: float
    chance: float
    rank1: float
    mrr: float


@dataclass
class Ranks:
    """Collects the rank of the right candidate per query.

    Examples:
        >>> r = Ranks()
        >>> r.add(np.array([0.1, 0.3]), 0)
        >>> r.add(np.array([0.1, 0.3, 0.2]), 2)
        >>> r.summary()
        RankSummary(n=2, candidates=2.5, chance=0.417, rank1=0.5, mrr=0.75)

    """

    ranks: list[int] = field(default_factory=list)
    candidates: list[int] = field(default_factory=list)

    def add(self, distances: np.ndarray, correct: int) -> None:
        """Record one query: its distances to the candidates and the index of the right one (ties count in favour).

        Args:
            distances: Distance from the query to each candidate.
            correct: Index of the right candidate.

        """
        self.ranks.append(1 + int((distances < distances[correct]).sum()))
        self.candidates.append(len(distances))

    def summary(self) -> RankSummary:
        """Summarise the recorded queries, rounded to 3 places; all zeros when there are none.

        Returns:
            The summary.

        """
        if not self.ranks:
            return RankSummary(0, 0.0, 0.0, 0.0, 0.0)
        r, c = np.array(self.ranks), np.array(self.candidates)
        return RankSummary(
            n=len(r),
            candidates=round(float(c.mean()), 1),
            chance=round(float((1 / c).mean()), 3),
            rank1=round(float((r == 1).mean()), 3),
            mrr=round(float((1 / r).mean()), 3),
        )


def reassociate(g: GoldBoxes, emb: np.ndarray, gap: int, every: int, ranks: Ranks) -> None:
    """Rank each queried player's teammates ``gap`` grid steps later in the same shot by appearance.

    Args:
        g: The shot's gold boxes.
        emb: Unit-length embeddings, one row per box of ``g``.
        gap: Grid steps between query and candidates.
        every: Grid steps between query frames.
        ranks: Collector the queries are added to.

    Examples:
        >>> g = GoldBoxes("s", "g", np.array([0, 0, 1, 1]), np.array([1, 2, 1, 2]), np.array([1, 1, 1, 1]),
        ...               np.zeros((4, 4)), np.zeros(4), {})
        >>> emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])  # the two swap looks
        >>> r = Ranks()
        >>> reassociate(g, emb, 1, 1, r)
        >>> r.summary().rank1
        0.0

    """
    for s in np.unique(g.step[g.step % every == 0]):
        later = np.flatnonzero(g.step == s + gap)
        for q in np.flatnonzero((g.step == s) & np.isin(g.cls, PLAYER_CLASSES)):
            cand = later[g.cls[later] == g.cls[q]]
            hit = np.flatnonzero(g.tid[cand] == g.tid[q])
            if len(cand) >= 2 and len(hit):
                ranks.add(1 - emb[cand] @ emb[q], int(hit[0]))


def numbered_means(g: GoldBoxes, emb: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    """Mean embedding of each numbered player in a shot, keyed by ``(class, number)``.

    Args:
        g: The shot's gold boxes.
        emb: Unit-length embeddings, one row per box of ``g``.

    Returns:
        Unit-length means of the players with at least ``MIN_BOXES`` clean boxes.

    Examples:
        >>> g = GoldBoxes("s", "g", np.arange(5), np.full(5, 7), np.ones(5, int), np.zeros((5, 4)), np.zeros(5),
        ...               {7: "23"})
        >>> numbered_means(g, np.tile([0.0, 2.0], (5, 1)))[(1, "23")].tolist()
        [0.0, 1.0]

    """
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for tid, number in g.number.items():
        mine = np.flatnonzero((g.tid == tid) & np.isin(g.cls, PLAYER_CLASSES))
        if len(mine):
            groups[(int(g.cls[mine[0]]), number)].extend(mine.tolist())
    out = {}
    for key, idx in groups.items():
        if len(idx) >= MIN_BOXES:
            m = emb[idx].mean(axis=0)
            out[key] = m / np.linalg.norm(m)
    return out


def rank_across(queries: dict, gallery: dict, ranks: Ranks) -> None:
    """Rank each query player's numbered teammates in another shot by mean-embedding distance.

    Args:
        queries: ``(class, number)`` to mean embedding in one shot.
        gallery: The same for another shot of the game.
        ranks: Collector the queries are added to.

    Examples:
        >>> r = Ranks()
        >>> q = {(1, "5"): np.array([1.0, 0.0])}
        >>> rank_across(q, {(1, "5"): np.array([0.9, 0.1]), (1, "7"): np.array([0.0, 1.0])}, r)
        >>> r.summary().rank1
        1.0

    """
    for (team, number), q in queries.items():
        keys = [k for k in gallery if k[0] == team]
        if (team, number) in gallery and len(keys) >= 2:
            ranks.add(1 - np.stack([gallery[k] for k in keys]) @ q, keys.index((team, number)))


def embed_gold(backend: object, seq: Sequence, g: GoldBoxes) -> tuple[np.ndarray, float]:
    """Embed the gold boxes of one shot with an appearance backend, frame by frame.

    Args:
        backend: A loaded appearance backend with an ``embed(image, xyxy)`` method.
        seq: The shot's sequence (for its video).
        g: Its gold boxes.

    Returns:
        The embeddings in box order, and the seconds spent in ``embed``.

    """
    out, secs = np.zeros((len(g.tid), 0), np.float32), 0.0
    read = FrameReader(seq.video)
    try:
        for src in np.unique(g.source):
            idx = np.flatnonzero(g.source == src)
            image = read(int(src))
            t0 = time.perf_counter()
            e = backend.embed(image, g.xyxy[idx])
            secs += time.perf_counter() - t0
            if not out.shape[1]:
                out = np.zeros((len(g.tid), e.shape[1]), np.float32)
            out[idx] = e
    finally:
        read.close()
    return out, secs


def score(cfg: RunConfig, gold: str = "v0", root: Path = Path("data/gold")) -> dict:
    """Score a config's appearance backend on the gold boxes of the shots its shots layer selects.

    Args:
        cfg: The run config; its ``appearance`` backend must have an ``embed`` method.
        gold: Gold set version.
        root: Gold set root.

    Returns:
        ``sequences``, ``boxes``, ``ms_per_crop``, ``reassoc`` (gap in seconds to a :class:`RankSummary`),
        ``cross_shot`` (a :class:`RankSummary`) and ``cross_shot_by_game``.

    Raises:
        ValueError: When the appearance backend cannot embed (``none``).

    Examples:
        >>> score(load_config("configs/appearance-reid-sportsmot.yaml"))["reassoc"][1].rank1  # doctest: +SKIP
        0.86

    """
    backend = make_backend(cfg, "appearance")
    if not hasattr(backend, "embed"):
        raise ValueError(f"appearance/{cfg.layers['appearance'].backend} has no embeddings to score")
    seqs = sequences(cfg)
    reassoc = {g: Ranks() for g in GAPS_S}
    means, boxes, secs = [], 0, 0.0
    backend.load()
    try:
        for seq in seqs:
            g = read_gold_boxes(root / gold, seq.name, seq.clip)
            emb, t = embed_gold(backend, seq, g)
            boxes, secs = boxes + len(g.tid), secs + t
            steps_per_s = seq.fps / g.stride
            for gap, ranks in reassoc.items():
                reassociate(g, emb, round(gap * steps_per_s), round(QUERY_EVERY_S * steps_per_s), ranks)
            means.append((seq.clip, numbered_means(g, emb)))
    finally:
        backend.unload()
    total, by_game = Ranks(), defaultdict(Ranks)
    for a, (game_a, qa) in enumerate(means):
        for b, (game_b, qb) in enumerate(means):
            if a != b and game_a == game_b:
                rank_across(qa, qb, total)
                rank_across(qa, qb, by_game[game_a])
    return {
        "sequences": len(seqs),
        "boxes": boxes,
        "ms_per_crop": 1000 * secs / max(1, boxes),
        "reassoc": {gap: r.summary() for gap, r in reassoc.items()},
        "cross_shot": total.summary(),
        "cross_shot_by_game": {game: r.summary() for game, r in sorted(by_game.items())},
    }
