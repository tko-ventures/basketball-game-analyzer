"""``shots/gold``: the gold-set sequences (data/gold/<version>), one per gold shot.

Frames run from the shot's first gold frame to its last at ``stride``; with a stride that divides the
gold stride (5), every gold frame is processed, so tracks are scored on exactly the gold frames.

"""

from __future__ import annotations

import configparser
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("shots", "gold")
class GoldShots(BaseBackend):
    """Shots backend over the exported gold set: one sequence per gold shot.

    ``held_out`` is ``exclude`` (default), ``only``, or anything else to keep every sequence.

    Examples:
        >>> seqs = GoldShots({"held_out": "only"}).sequences()
        >>> len(seqs) > 0 and all(s.held_out for s in seqs)
        True

    """

    defaults: ClassVar[dict[str, Any]] = {
        "version": "v0",
        "stride": 5,
        "held_out": "exclude",
        "root": "data/gold",
        "labels": "data/labels",
    }

    def sequences(self) -> list[Sequence]:
        """List the gold sequences selected by ``held_out``, framed at ``stride``.

        Returns:
            One sequence per gold shot, with video, fps and meta from the clip's label file.

        Raises:
            ValueError: ``stride`` does not divide the gold set's stride.

        """
        root = Path(self.p["root"]) / self.p["version"]
        manifest = json.loads((root / "manifest.json").read_text())
        out = []
        for s in manifest["sequences"]:
            if (self.p["held_out"] == "exclude" and s["held_out"]) or (
                self.p["held_out"] == "only" and not s["held_out"]
            ):
                continue
            ini = configparser.ConfigParser()
            ini.read(root / s["name"] / "seqinfo.ini")
            grid = [int(x) for x in (root / s["name"] / "frames.txt").read_text().split()]
            if self.p["stride"] > manifest["stride"] or manifest["stride"] % self.p["stride"]:
                raise ValueError(f"stride {self.p['stride']} must divide the gold stride {manifest['stride']}")
            labels = json.loads((Path(self.p["labels"]) / f"{s['clip']}.json").read_text())
            out.append(
                Sequence(
                    name=s["name"],
                    clip=s["clip"],
                    video=labels["clip"],
                    shot=s["shot"],
                    frames=list(range(grid[0], grid[-1] + 1, self.p["stride"])),
                    fps=float(ini["Sequence"]["sourceFrameRate"]),
                    meta=labels.get("meta", {}),
                    held_out=s["held_out"],
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
