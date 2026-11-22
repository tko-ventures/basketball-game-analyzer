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
import time
from collections.abc import Iterable
from collections.abc import Sequence as Seq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bga.stage import Backend as NumbersBackend  # noqa: F401  # re-exported: the numbers layer's backend protocol
from bga.stage import BaseBackend, FrameReader, Sequence

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


#: Classes of the crop classifiers: :data:`UNREADABLE`, then the numbers 0 to 99.
NUMBER_CLASSES = (UNREADABLE, *(str(i) for i in range(100)))


def ios(inner: Seq[float], outer: Seq[float]) -> float:
    """Intersection over the smaller (``inner``) box's area, for xyxy boxes.

    Args:
        inner: Box whose coverage is measured, as x1, y1, x2, y2 (extra items are ignored).
        outer: Covering box, as x1, y1, x2, y2 (extra items are ignored).

    Returns:
        Covered share of ``inner``; 0 when ``inner`` has no area.

    Examples:
        >>> ios((10, 10, 20, 20), (0, 0, 15, 100))
        0.5
        >>> ios((0, 0, 0, 10), (0, 0, 10, 10))
        0.0

    """
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


class TrackNumbers(BaseBackend):
    """Base of the numbers backends: reads ``number`` crops matched to tracks and confirms a number per track.

    A subclass implements ``read_crop`` (one crop to a number or ``None``) and ``read_crops`` (crops to
    :class:`CropRead`, for ``bga eval-numbers``); :meth:`run` does the matching, sampling and confirmation.

    Matching (plan section 3): a number box belongs to the track box that contains it with IoS >= ``min_ios``; crops
    padded ``pad`` px; reads every ``read_every`` frames per track; ``ConsecutiveValueTracker`` with
    ``n_consecutive`` equal reads confirms. Referee tracks are skipped.

    """

    defaults: ClassVar[dict[str, Any]] = {
        "min_score": 0.3,
        "min_ios": 0.9,
        "pad": 10,
        "n_consecutive": 3,
        "read_every": 1,
        "max_reads_per_track": 12,
        "stop_on_confirm": True,
    }

    def read_crop(self, crop: np.ndarray) -> str | None:
        """Read the number on one crop; implemented by each backend.

        Args:
            crop: BGR crop of a ``number`` box.

        Raises:
            NotImplementedError: Always, on the base class.

        """
        raise NotImplementedError

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Read and confirm a number per track over the sequence.

        Each ``read_every``-th frame, every ``number`` box scoring at least ``min_score`` is given to the non-referee
        track box that contains it best, if its IoS is at least ``min_ios``; tracks read ``max_reads_per_track``
        times are skipped, and so are confirmed tracks unless ``stop_on_confirm`` is off (the log then holds reads
        after a confirmation, for rules downstream).

        Args:
            seq: Sequence to read.
            upstream: ``detect`` and ``track`` outputs of the sequence.
            read: Frame reader of the sequence's video.

        Returns:
            ``{"tracks": {"<track_id>": "23" | None}, "log", "reads", "ms_per_read", "ms_per_frame"}``, where ``log``
            lists every read as ``[frame, track_id, value]``.

        """
        from sports.common.temporal import ConsecutiveValueTracker

        dets, tracks = upstream["detect"]["frames"], upstream["track"]["frames"]
        confirm = ConsecutiveValueTracker(n_consecutive=self.p["n_consecutive"])
        done, reads, log, t, n = {}, {}, [], 0.0, 0
        for j, f in enumerate(seq.frames):
            if j % self.p["read_every"]:
                continue
            nums = [d for d in dets[str(f)] if d[5] == "number" and d[4] >= self.p["min_score"]]
            rows = [r for r in tracks[str(f)] if r[6] != "referee"]
            ids, values, img = [], [], None
            for d in nums:
                best = max(rows, key=lambda r: ios(d[:4], r[1:5]), default=None)
                if best is None or ios(d[:4], best[1:5]) < self.p["min_ios"]:
                    continue
                tid = best[0]
                if (self.p["stop_on_confirm"] and tid in done) or reads.get(tid, 0) >= self.p["max_reads_per_track"]:
                    continue
                img = read(f) if img is None else img
                crop = number_crop(img, d[:4], self.p["pad"])
                if crop.size == 0:
                    continue
                t0 = time.perf_counter()
                value = self.read_crop(crop)
                t += time.perf_counter() - t0
                n += 1
                reads[tid] = reads.get(tid, 0) + 1
                ids.append(tid)
                values.append(value)
                log.append([f, tid, value])
            done.update({tid: v for tid, v in zip(ids, confirm.update(ids, values), strict=True) if v is not None})
        all_ids = {r[0] for f in seq.frames for r in tracks[str(f)]}
        return {
            "tracks": {str(tid): done.get(tid) for tid in sorted(all_ids)},
            "log": log,
            "reads": n,
            "ms_per_read": round(1000 * t / max(1, n), 1),
            "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1),
        }
