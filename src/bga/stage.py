"""What every layer backend shares: the sequence it runs on, frame access, and the backend protocol.

Each layer's ``base.py`` narrows ``Backend`` with that layer's output type. Outputs are JSON-friendly
dicts written to ``runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json``.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import cv2
import numpy as np


@dataclass
class Sequence:
    """A run unit: one shot of one clip, processed on ``frames`` (source frame indices).

    Examples:
        >>> seq = Sequence(name="clip_s2", clip="clip", video="data/clips/clip.mp4", shot=2, frames=[0, 5], fps=50.0)
        >>> seq.meta, seq.held_out
        ({}, False)

    """

    name: str
    clip: str  # labels stem, also the runs/<clip> directory
    video: str
    shot: int
    frames: list[int]
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)  # season, home/away team, kit, roster
    held_out: bool = False


class FrameReader:
    """Random access to decoded frames with a small cache (sequential and strided reads stay cheap).

    A read a short way ahead of the last one decodes and drops the frames between instead of seeking. A seek restarts
    decoding at the previous keyframe: 122 ms a frame on the 12 gold shots at stride 5 against 3.7 ms this way, with
    all 1,216 frames identical (M1 Max, 2026-11-22; the same in the Modal runner image).

    Examples:
        >>> read = FrameReader("missing.mp4")
        >>> read(0)
        Traceback (most recent call last):
        ...
        IndexError: frame 0 unreadable
        >>> read.close()

    """

    #: Longest forward step, in frames, read by dropping the frames between; longer steps and steps back seek.
    MAX_SKIP: ClassVar[int] = 64

    def __init__(self, video: str, cache: int = 64) -> None:
        """Open the video for reading.

        Args:
            video: Path to the video file.
            cache: Number of decoded frames kept; the oldest is evicted first.

        """
        self.cap = cv2.VideoCapture(video)
        self.pos: int | None = -1  # the last frame decoded; None when the position is unknown (after a failed read)
        self.cache: dict[int, np.ndarray] = {}
        self.size = cache

    def _advance(self, i: int) -> bool:
        """Move the decoder so the next read returns frame ``i``: drop frames for a short step ahead, else seek."""
        step = i - self.pos if self.pos is not None else 0
        if 0 < step <= self.MAX_SKIP:
            return all(self.cap.grab() for _ in range(step - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # a failed seek shows in the read that follows
        return True

    def __call__(self, i: int) -> np.ndarray:
        """Return frame ``i`` as a BGR image, seeking only for a step back or one longer than ``MAX_SKIP``.

        Args:
            i: Source frame index.

        Returns:
            The decoded frame, shared with the cache (do not modify it in place).

        Raises:
            IndexError: When the frame cannot be decoded.

        """
        if i in self.cache:
            return self.cache[i]
        ok = self._advance(i)
        ok, img = self.cap.read() if ok else (False, None)
        if not ok:
            self.pos = None
            raise IndexError(f"frame {i} unreadable")
        self.pos = i
        if len(self.cache) >= self.size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = img
        return img

    def close(self) -> None:
        """Release the video capture."""
        self.cap.release()


class Backend(Protocol):
    """Interface every layer backend implements; the registry sets ``layer`` and ``name``.

    ``requires`` is ``"cuda"`` for backends that run only remotely, else ``None``. ``scope`` is ``"sequence"``
    (default; :meth:`run` is called per sequence) or ``"clip"`` (:meth:`run_clip` is called once per clip with
    every sequence of that clip).

    """

    layer: ClassVar[str]
    name: ClassVar[str]
    requires: ClassVar[str | None]  # "cuda" for remote-only backends
    scope: ClassVar[str]  # "sequence" or "clip"

    def __init__(self, params: dict[str, Any], weights: str | None = None) -> None:
        """Take the config entry's params and weights id without loading anything heavy."""
        ...

    def load(self) -> None:
        """Bring heavy models into memory; the pipeline keeps one heavy model resident at a time."""

    def unload(self) -> None:
        """Free what ``load`` brought into memory."""
        ...

    def run(self, seq: Sequence, upstream: dict[str, dict], read: FrameReader) -> dict:
        """Process one sequence given upstream layers' outputs (keyed by layer name)."""
        ...


class BaseBackend:
    """Convenience base: stores params, no-op load/unload, local execution.

    Subclasses declare every param they accept, with its default, in ``defaults``. Clip-level backends set
    ``scope = "clip"`` and implement ``run_clip(seqs, upstreams) -> dict[str, dict]``.

    Examples:
        >>> class Demo(BaseBackend):
        ...     layer, name, defaults = "demo", "demo", {"k": 1}
        >>> Demo({"k": 2}).p
        {'k': 2}
        >>> Demo({"j": 2})
        Traceback (most recent call last):
        ...
        ValueError: demo/demo: unknown params ['j']

    """

    layer: ClassVar[str]
    name: ClassVar[str]
    requires: ClassVar[str | None] = None
    scope: ClassVar[str] = "sequence"
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any], weights: str | None = None) -> None:
        """Merge ``params`` over ``defaults`` into ``self.p`` and keep the weights id.

        Args:
            params: The config entry's params; every key must be in ``defaults``.
            weights: The config entry's weights id, if any.

        Raises:
            ValueError: On a param the backend does not declare.

        """
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.layer}/{self.name}: unknown params {sorted(unknown)}")
        self.p = {**self.defaults, **params}
        self.weights = weights

    def load(self) -> None:
        """Do nothing; backends with heavy models override this."""

    def unload(self) -> None:
        """Do nothing; backends with heavy models override this."""
