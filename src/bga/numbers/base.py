"""Layer 5 (numbers).

Output per sequence:
    {"tracks": {"<track_id>": "23" | null}, "log": [[frame, track_id, "23" | null], ...], "reads": int,
     "ms_per_read": float}

A track's number is confirmed by the backend's own rule (for Skalski's stack, N consecutive equal
reads); null means no confirmed number. ``log`` lists every read in order; outputs written before 2026-11-21
lack it.

Readers are also scored crop by crop (``bga eval-numbers``): a backend with ``read_crops`` takes padded ``number``
crops (:func:`number_crop`) and returns a :class:`CropRead` per crop. A reader that reports a distribution gives
probabilities over candidate numbers plus :data:`UNREADABLE`; :func:`roster_choice` restricts a read to a team's
roster.

"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bga.stage import Backend as NumbersBackend  # noqa: F401  # re-exported: the numbers layer's backend protocol

#: Folder of the PLK roster files (``scripts/fetch_rosters.py``).
ROSTERS = Path("data/rosters")


def roster_numbers(meta: dict, side: str, rosters: Path = ROSTERS) -> set[str]:
    """Jersey numbers on one side's roster for the clip's season.

    Args:
        meta: A sequence's clip metadata: ``season`` and ``home``/``away`` with a ``roster`` file stem.
        side: ``home`` or ``away``.
        rosters: Folder of roster files.

    Returns:
        The numbers, or an empty set when the clip names no roster or season for that side.

    Examples:
        >>> meta = {"season": "2025/26", "home": {"roster": "dziki_warszawa"}}
        >>> "2" in roster_numbers(meta, "home"), roster_numbers(meta, "away")
        (True, set())

    """
    stem, season = (meta.get(side) or {}).get("roster"), meta.get("season")
    if not stem or not season or not (rosters / f"{stem}.json").exists():
        return set()
    players = json.loads((rosters / f"{stem}.json").read_text())["seasons"].get(season, {}).get("players", [])
    return {p["number"] for p in players}


def confirm_consecutive(values: Iterable[str | None], n: int) -> str | None:
    """The first value read ``n`` times in a row, skipping ``None`` reads (``ConsecutiveValueTracker``'s rule).

    Args:
        values: Reads in time order.
        n: Equal reads in a row needed.

    Returns:
        The confirmed value, or ``None``.

    Examples:
        >>> confirm_consecutive(["2", None, "2", "8", "8", "8", "2"], 3)
        '8'
        >>> confirm_consecutive(["2", "8", "2"], 2) is None
        True

    """
    last, streak = None, 0
    for v in values:
        if v is None:
            continue
        streak = streak + 1 if v == last else 1
        last = v
        if streak >= n:
            return v
    return None


#: The answer of a reader that declines to read a crop, as a key of :attr:`CropRead.probs`.
UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class CropRead:
    """One reader's answer on one crop.

    Attributes:
        value: The number read (one or two digits), or ``None`` when the reader declines.
        probs: Probabilities over candidate numbers and :data:`UNREADABLE`, summing to 1, for readers that report a
            distribution; ``None`` for readers that only decode text.

    Examples:
        >>> CropRead("23", {"23": 0.9, UNREADABLE: 0.1}).value
        '23'

    """

    value: str | None
    probs: dict[str, float] | None = None


def number_crop(image: np.ndarray, xyxy: Iterable[float], pad: int) -> np.ndarray:
    """Cut a ``number`` box out of a frame with ``pad`` pixels on each side, clipped to the frame.

    Args:
        image: The frame, H x W x 3.
        xyxy: The box as x1, y1, x2, y2.
        pad: Margin in pixels.

    Returns:
        The crop (a view into ``image``); empty when the box lies outside the frame.

    Examples:
        >>> number_crop(np.zeros((100, 200, 3), np.uint8), (10, 20, 30, 50), pad=10).shape
        (50, 40, 3)

    """
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    h, w = image.shape[:2]
    return image[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]


def roster_choice(read: CropRead, roster: set[str]) -> tuple[str | None, float | None]:
    """A read restricted to a roster: the number answered and the reader's probability for it.

    With probabilities, the candidates are the roster numbers plus :data:`UNREADABLE`, renormalised, and the most
    probable one wins (``None`` when it is :data:`UNREADABLE`, which also wins ties). Without, a decoded value off the
    roster becomes ``None``. An empty roster (a side with no roster file) leaves the read unrestricted.

    Args:
        read: The reader's answer.
        roster: Numbers on the team's roster.

    Returns:
        The answer, or ``None``, and its probability (``None`` for readers without probabilities).

    Examples:
        >>> roster_choice(CropRead("8", {"8": 0.5, "3": 0.3, UNREADABLE: 0.2}), {"3", "23"})
        ('3', 0.6)
        >>> roster_choice(CropRead("8"), {"3", "23"}), roster_choice(CropRead("8"), set())
        ((None, None), ('8', None))

    """
    if read.probs is None:
        return (read.value if not roster or read.value in roster else None), None
    keep = {k: v for k, v in read.probs.items() if not roster or k in roster or k == UNREADABLE}
    total = sum(keep.values())
    if total <= 0:
        return None, None
    best = max(keep, key=lambda k: (keep[k], k == UNREADABLE))
    return (None if best == UNREADABLE else best), round(keep[best] / total, 6)
