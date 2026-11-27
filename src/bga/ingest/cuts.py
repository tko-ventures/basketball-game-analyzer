"""Shot-cut detectors (Layer 0): where a video's camera shots start (M10, #32).

A cuts backend reads a whole video and returns its shots as ``{"start", "end"}`` with inclusive frame bounds, the
format of a label file's ``shots``. Backends register under the pseudo-layer ``cuts`` and are named in the ``cuts``
param of the ``shots/video`` backend (:mod:`bga.ingest.video`).

"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from bga.ingest.hist_diff import DEFAULT_MIN_LEN_S, DEFAULT_THRESHOLD, detect_shots
from bga.registry import register
from bga.stage import BaseBackend


class CutsBackend(BaseBackend):
    """Base of the cut detectors: split one video into shots."""

    layer: ClassVar[str] = "cuts"

    def shots(self, video: str | Path, progress: Callable[[float], None] | None = None) -> list[dict]:
        """Split ``video`` into shots.

        Args:
            video: Path of the video.
            progress: Called with the share of the video done, if given.

        Returns:
            Shots ``{"start", "end"}`` covering every frame, in order, with inclusive bounds.

        """
        raise NotImplementedError


@register("cuts", "hist_diff")
class HistDiffCuts(CutsBackend):
    """The prior labeller's detector (:mod:`bga.ingest.hist_diff`), which cut every label file so far.

    Params (``defaults``): ``threshold`` (HSV histogram distance scale) and ``min_len_s`` (shortest shot).

    Examples:
        >>> HistDiffCuts({}).p
        {'threshold': 0.25, 'min_len_s': 0.4}

    """

    defaults: ClassVar[dict[str, Any]] = {"threshold": DEFAULT_THRESHOLD, "min_len_s": DEFAULT_MIN_LEN_S}

    def shots(self, video: str | Path, progress: Callable[[float], None] | None = None) -> list[dict]:
        """Decode every frame and cut where :class:`bga.ingest.hist_diff.ShotCutDetector` does.

        Args:
            video: Path of the video.
            progress: Called with the share of the video decoded.

        Returns:
            Shots ``{"start", "end"}``.

        """
        found = detect_shots(video, self.p["threshold"], self.p["min_len_s"], progress)
        return [{"start": s["start"], "end": s["end"]} for s in found]
