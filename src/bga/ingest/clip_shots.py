"""``shots/clips``: the shots of non-gold clips from their label files, for training data such as re-ID crops.

A sequence is one shot of ``data/labels/<clip>.json`` lasting at least ``min_s`` seconds, framed on the multiples of
``stride`` inside it, named ``<youtube id>_s<shot index>`` like the gold sequences. Every gold game is refused
(:func:`bga.train.leakage.assert_no_gold`), so nothing made from these sequences can contain a gold frame.

"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence
from bga.train.leakage import assert_no_gold


def shot_frames(start: int, end: int, stride: int) -> list[int]:
    """Frames of a shot on the multiples of ``stride``, as the gold grid places them.

    Args:
        start: The shot's first frame.
        end: The shot's last frame, included.
        stride: Grid step in source frames.

    Returns:
        The grid frames inside the shot.

    Examples:
        >>> shot_frames(1902, 1921, 5)
        [1905, 1910, 1915, 1920]

    """
    return list(range(-(-start // stride) * stride, end + 1, stride))


@register("shots", "clips")
class ClipShots(BaseBackend):
    """Shots backend over the label files of named non-gold clips.

    Params (``defaults``): ``clips`` (label stems), ``min_s`` (shortest shot kept, in seconds), ``stride`` and
    ``labels`` (the label folder). ``clips`` and ``min_s`` only choose sequences, so they stay out of the hashes.

    Examples:
        >>> ClipShots({"clips": ["20250213_4XMD9FWmQws_PGE_Start_Dziki_Skrot_meczu"]}).sequences()
        Traceback (most recent call last):
        ...
        ValueError: gold clips in a training manifest: [...]

    """

    defaults: ClassVar[dict[str, Any]] = {"clips": [], "min_s": 3.0, "stride": 5, "labels": "data/labels"}

    def sequences(self) -> list[Sequence]:
        """List the shots of the configured clips lasting at least ``min_s``.

        Returns:
            One sequence per kept shot, with video, fps and meta from the clip's label file.

        Raises:
            ValueError: When a clip belongs to a gold game.

        """
        assert_no_gold(self.p["clips"])
        out = []
        for clip in self.p["clips"]:
            labels = json.loads((Path(self.p["labels"]) / f"{clip}.json").read_text())
            fps, ytid = float(labels["fps"]), clip.split("_")[1]
            for i, s in enumerate(labels["shots"]):
                if (s["end"] - s["start"] + 1) / fps < self.p["min_s"]:
                    continue
                out.append(
                    Sequence(
                        name=f"{ytid}_s{i}",
                        clip=clip,
                        video=labels["clip"],
                        shot=i,
                        frames=shot_frames(s["start"], s["end"], self.p["stride"]),
                        fps=fps,
                        meta=labels.get("meta", {}),
                    )
                )
        return out

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Return the sequence's fields as the layer's output.

        Args:
            seq: The sequence.
            upstream: Unused.
            read: Unused.

        Returns:
            ``dataclasses.asdict(seq)``.

        """
        return asdict(seq)
