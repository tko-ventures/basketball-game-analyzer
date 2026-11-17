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
    """A run unit: one shot of one clip, processed on ``frames`` (source frame indices)."""

    name: str
    clip: str  # labels stem, also the runs/<clip> directory
    video: str
    shot: int
    frames: list[int]
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)  # season, home/away team, kit, roster
    held_out: bool = False


class FrameReader:
    """Random access to decoded frames with a small cache (sequential reads stay cheap)."""

    def __init__(self, video: str, cache: int = 64):
        self.cap = cv2.VideoCapture(video)
        self.pos = -1
        self.cache: dict[int, np.ndarray] = {}
        self.size = cache

    def __call__(self, i: int) -> np.ndarray:
        if i in self.cache:
            return self.cache[i]
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        if not ok:
            raise IndexError(f"frame {i} unreadable")
        self.pos = i
        if len(self.cache) >= self.size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = img
        return img

    def close(self) -> None:
        self.cap.release()


class Backend(Protocol):
    layer: ClassVar[str]
    name: ClassVar[str]
    requires: ClassVar[str | None]  # "cuda" for remote-only backends

    def __init__(self, params: dict[str, Any], weights: str | None = None) -> None: ...

    def load(self) -> None:
        """Bring heavy models into memory; the pipeline keeps one heavy model resident at a time."""

    def unload(self) -> None: ...

    def run(self, seq: Sequence, upstream: dict[str, dict], read: FrameReader) -> dict:
        """Process one sequence given upstream layers' outputs (keyed by layer name)."""
        ...


class BaseBackend:
    """Convenience base: stores params, no-op load/unload, local execution."""

    layer: ClassVar[str]
    name: ClassVar[str]
    requires: ClassVar[str | None] = None
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any], weights: str | None = None) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.layer}/{self.name}: unknown params {sorted(unknown)}")
        self.p = {**self.defaults, **params}
        self.weights = weights

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass
