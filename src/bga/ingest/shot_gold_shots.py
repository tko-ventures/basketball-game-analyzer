"""``shots/shot_gold``: the shot-gold sequences (``data/labels/shot_gold.json``), one per picked shot.

These are further shots of the gold games, labelled for Layer 7 only (:mod:`bga.eval.shot_gold`). Frames sit on the
multiples of ``stride`` inside the shot, as for the other shots backends, so every labelled frame is processed.

"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

from bga.ingest.clip_shots import shot_frames
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("shots", "shot_gold")
class ShotGoldShots(BaseBackend):
    """Shots backend over the shot-gold working file: one sequence per picked shot.

    ``held_out`` is ``exclude`` (default), ``only``, or anything else to keep every sequence. ``queue``, ``labels``
    and ``held_out`` only choose sequences, so they stay out of the hashes.

    Examples:
        >>> seqs = ShotGoldShots({"held_out": "only"}).sequences()
        >>> len(seqs) > 0 and all(s.held_out for s in seqs)
        True

    """

    defaults: ClassVar[dict[str, Any]] = {
        "queue": "data/labels/shot_gold.json",
        "stride": 5,
        "held_out": "exclude",
        "labels": "data/labels",
    }

    def sequences(self) -> list[Sequence]:
        """List the shot-gold sequences selected by ``held_out``, framed at ``stride``.

        Returns:
            One sequence per picked shot, with video, fps and meta from the clip's label file.

        """
        out = []
        for s in json.loads(Path(self.p["queue"]).read_text(encoding="utf-8"))["shots"]:
            if (self.p["held_out"] == "exclude" and s["held_out"]) or (
                self.p["held_out"] == "only" and not s["held_out"]
            ):
                continue
            labels = json.loads((Path(self.p["labels"]) / f"{s['clip']}.json").read_text(encoding="utf-8"))
            out.append(
                Sequence(
                    name=s["sequence"],
                    clip=s["clip"],
                    video=labels["clip"],
                    shot=int(s["shot"]),
                    frames=shot_frames(int(s["start"]), int(s["end"]), self.p["stride"]),
                    fps=float(labels["fps"]),
                    meta=labels.get("meta", {}),
                    held_out=bool(s["held_out"]),
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
