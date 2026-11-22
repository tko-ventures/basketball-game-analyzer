"""``numbers/smolvlm_ocr``: Skalski's jersey-number reader on ``number`` boxes matched to tracks.

The reader is basketball-jersey-numbers-ocr/3, a SmolVLM2 fine-tune on NBA crops; a track's number is confirmed by
consecutive agreement.

Matching (plan section 3): a number box belongs to the track box that contains it with IoS >= 0.9. Crops padded 10 px;
reads every ``read_every`` frames per track; ``ConsecutiveValueTracker`` with ``n_consecutive`` equal reads confirms.
Runs on MPS with ``max_new_tokens=4`` (2 s per crop instead of 8 s, docs/handoff.md). Referee tracks are skipped.

"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence as Seq
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.detect.roboflow_inference import roboflow_key
from bga.device import pick_device
from bga.numbers.base import CropRead, number_crop
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


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


@register("numbers", "smolvlm_ocr")
class SmolVLMOcr(BaseBackend):
    """Numbers backend: SmolVLM2 reads of matched ``number`` crops, confirmed by ``ConsecutiveValueTracker``.

    Examples:
        >>> ocr = SmolVLMOcr({"read_every": 5})  # doctest: +SKIP
        >>> ocr.load()  # doctest: +SKIP
        >>> ocr.run(seq, {"detect": detect_out, "track": track_out}, read)["tracks"]  # doctest: +SKIP
        {'1': '23', '2': None, ...}

    """

    defaults: ClassVar[dict[str, Any]] = {
        "model_id": "basketball-jersey-numbers-ocr/3",
        "min_score": 0.3,
        "min_ios": 0.9,
        "pad": 10,
        "n_consecutive": 3,
        "read_every": 1,
        "max_reads_per_track": 12,
        "stop_on_confirm": True,
    }

    def load(self) -> None:
        """Load the reader through ``inference_models`` on MPS when available, else CPU."""
        import os

        import torch
        from inference_models import AutoModel

        roboflow_key()
        device = torch.device(pick_device("auto"))
        self.model = AutoModel.from_pretrained(
            self.p["model_id"], api_key=os.environ.get("ROBOFLOW_API_KEY"), device=device
        )

    def unload(self) -> None:
        """Drop the reader."""
        self.model = None

    def read_crop(self, crop: np.ndarray) -> str | None:
        """Read the number on one crop.

        Args:
            crop: BGR crop of a ``number`` box, resized to 224 x 224 for the reader.

        Returns:
            The digits the reader answered when there are one or two of them, else None.

        """
        text = self.model.prompt(images=cv2.resize(crop, (224, 224)), prompt="Read the number.", max_new_tokens=4)[0]
        digits = re.sub(r"\D", "", str(text))
        return digits if 0 < len(digits) <= 2 else None

    def read_crops(self, crops: list[np.ndarray], rosters: list[set[str]]) -> list[CropRead]:
        """Read a batch of crops for ``bga eval-numbers``, one generation each; the reader reports no probabilities.

        Args:
            crops: BGR crops cut by :func:`bga.numbers.base.number_crop` with this backend's ``pad``.
            rosters: The team's roster per crop; unused, since the reader decodes freely.

        Returns:
            One :class:`bga.numbers.base.CropRead` per crop, without ``probs``.

        """
        return [CropRead(self.read_crop(c)) for c in crops]

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
