"""Number-reader scores on the gold number crops (plan section 5, layer 5): crop by crop, without detector or tracker.

The gold crops (``data/gold/v1/numbers.json``, ``docs/gold.md``) are Tomasz's hand-drawn ``number`` boxes plus the
PLK detector's number boxes that match no gold box. Each is labelled with its digits, ``illegible`` or
``not_number``. The config's numbers backend reads every crop of the selected shots through ``read_crops``, cut with
its own ``pad``, and the reads are scored:

- **legible crops:** top-1 of the free read, and top-1 under the roster of the crop's team
  (:func:`bga.numbers.base.roster_choice`); the share answered under the roster;
- **refusal:** on the ``illegible`` and ``not_number`` crops, the share the reader declines under the roster;
- **calibration:** for readers that report probabilities, the expected calibration error of the roster answer's
  probability, where an answer is right when it equals the digits, or declines a crop with none;
- **speed:** milliseconds per crop on this machine.

"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bga.config import RunConfig
from bga.numbers.base import CropRead, number_crop, roster_choice, roster_numbers
from bga.pipeline import make_backend, sequences
from bga.stage import FrameReader

#: Gold number crops, written by ``scripts/number_gold.py export``.
GOLD_CROPS = Path("data/gold/v1/numbers.json")
#: Labels of crops a reader should decline.
REFUSE = frozenset({"illegible", "not_number"})
#: Equal-width confidence bins of the calibration error.
ECE_BINS = 10


@dataclass(frozen=True, slots=True)
class GoldCrop:
    """One gold number crop.

    Examples:
        >>> GoldCrop("n001", "clip_s2", 100, (1.0, 2.0, 30.0, 40.0), "23", "home").legible
        True

    """

    id: str
    sequence: str
    frame: int
    xyxy: tuple[float, float, float, float]
    label: str
    team: str

    @property
    def legible(self) -> bool:
        """Whether the crop carries digits."""
        return self.label not in REFUSE


def load_gold_crops(path: Path = GOLD_CROPS) -> list[GoldCrop]:
    """Read the gold number crops.

    Args:
        path: The export of ``scripts/number_gold.py``.

    Returns:
        Every crop, in file order.

    Examples:
        >>> crops = load_gold_crops()
        >>> len(crops) > 0, crops[0].id
        (True, 'n001')

    """
    return [
        GoldCrop(c["id"], c["sequence"], int(c["frame"]), tuple(c["xyxy"]), c["label"], c.get("team", ""))
        for c in json.loads(path.read_text())["crops"]
    ]


@dataclass(slots=True)
class Tally:
    """Running counts of one reader's answers on gold crops.

    Examples:
        >>> t = Tally()
        >>> t.add(GoldCrop("a", "s", 0, (0, 0, 1, 1), "23", "home"), CropRead("23"), ("23", None))
        >>> t.add(GoldCrop("b", "s", 0, (0, 0, 1, 1), "illegible", "home"), CropRead("8"), (None, None))
        >>> t.summary()["top1_roster"], t.summary()["refused"], t.summary()["ece"] is None
        (1.0, 1.0, True)

    """

    legible: int = 0
    free_right: int = 0
    roster_right: int = 0
    answered: int = 0
    refuse: int = 0
    refused: int = 0
    conf: list[float] = field(default_factory=list)
    right: list[bool] = field(default_factory=list)
    wrong: list[tuple[str, str, str | None]] = field(default_factory=list)

    def add(self, crop: GoldCrop, read: CropRead, choice: tuple[str | None, float | None]) -> None:
        """Count one crop's read and its roster answer.

        Args:
            crop: The gold crop.
            read: The reader's free answer.
            choice: The roster answer and its probability, from :func:`bga.numbers.base.roster_choice`.

        """
        answer, p = choice
        if crop.legible:
            self.legible += 1
            self.free_right += read.value == crop.label
            self.roster_right += answer == crop.label
            self.answered += answer is not None
            ok = answer == crop.label
        else:
            self.refuse += 1
            self.refused += answer is None
            ok = answer is None
        if not ok:
            self.wrong.append((crop.id, crop.label, answer))
        if p is not None:
            self.conf.append(p)
            self.right.append(ok)

    def summary(self) -> dict:
        """Shares and counts of the answers so far.

        Returns:
            ``legible`` and ``refuse`` (crop counts), ``top1_free``, ``top1_roster`` and ``answered`` (shares of the
            legible crops), ``refused`` (share of the others), ``ece`` (``None`` without probabilities) and ``wrong``
            (``(id, label, roster answer)`` of every wrong answer).

        """
        return {
            "legible": self.legible,
            "top1_free": self.free_right / self.legible if self.legible else float("nan"),
            "top1_roster": self.roster_right / self.legible if self.legible else float("nan"),
            "answered": self.answered / self.legible if self.legible else float("nan"),
            "refuse": self.refuse,
            "refused": self.refused / self.refuse if self.refuse else float("nan"),
            "ece": ece(self.conf, self.right) if self.conf else None,
            "wrong": list(self.wrong),
        }


def ece(conf: list[float], right: list[bool], bins: int = ECE_BINS) -> float:
    """Expected calibration error: the mean gap between confidence and accuracy, over equal-width bins, weighted.

    Args:
        conf: The probability the reader gave each answer.
        right: Whether each answer was right.
        bins: Number of bins over [0, 1].

    Returns:
        The error, from 0 (calibrated) to 1.

    Examples:
        >>> ece([0.9, 0.9, 0.9, 0.9], [True, True, True, False])
        0.15
        >>> ece([1.0, 1.0], [True, True])
        0.0

    """
    c, r = np.asarray(conf, float), np.asarray(right, float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    gap = sum(abs(c[idx == b].mean() - r[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any())
    return round(float(gap / len(c)), 6)


def score(cfg: RunConfig, path: Path = GOLD_CROPS) -> dict:
    """Score a config's numbers backend on the gold crops of the shots its shots layer selects.

    Args:
        cfg: The run config; its ``numbers`` backend must have a ``read_crops`` method.
        path: Gold number crops.

    Returns:
        :meth:`Tally.summary` plus ``crops``, ``sequences`` and ``ms_per_crop``.

    Raises:
        ValueError: When the numbers backend cannot read single crops.

    Examples:
        >>> score(load_config("configs/numbers-smolvlm-nba.yaml"))["top1_roster"]  # doctest: +SKIP
        0.5

    """
    backend = make_backend(cfg, "numbers")
    if not hasattr(backend, "read_crops"):
        raise ValueError(f"numbers/{cfg.layers['numbers'].backend} cannot read single crops")
    seqs = {s.name: s for s in sequences(cfg)}
    by_seq: dict[str, list[GoldCrop]] = defaultdict(list)
    for c in load_gold_crops(path):
        if c.sequence in seqs:
            by_seq[c.sequence].append(c)
    tally, secs, n = Tally(), 0.0, 0
    pad = int(backend.p.get("pad", 0))
    backend.load()
    try:
        for name, crops in by_seq.items():
            seq = seqs[name]
            rosters = {side: roster_numbers(seq.meta, side) for side in ("home", "away")}
            read = FrameReader(seq.video)
            try:
                images = [number_crop(read(c.frame), c.xyxy, pad) for c in crops]
            finally:
                read.close()
            teams = [rosters.get(c.team, set()) for c in crops]
            t0 = time.perf_counter()
            reads = backend.read_crops(images, teams)
            secs, n = secs + time.perf_counter() - t0, n + len(crops)
            for c, r, roster in zip(crops, reads, teams, strict=True):
                tally.add(c, r, roster_choice(r, roster))
    finally:
        backend.unload()
    return {**tally.summary(), "crops": n, "sequences": len(by_seq), "ms_per_crop": 1000 * secs / max(1, n)}
