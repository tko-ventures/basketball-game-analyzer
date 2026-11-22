"""``numbers/smolvlm_ocr``: Skalski's jersey-number reader on ``number`` boxes matched to tracks.

The reader is basketball-jersey-numbers-ocr/3, a SmolVLM2 fine-tune on NBA crops; a track's number is confirmed by
consecutive agreement.

Matching (plan section 3): a number box belongs to the track box that contains it with IoS >= 0.9. Crops padded 10 px;
reads every ``read_every`` frames per track; ``ConsecutiveValueTracker`` with ``n_consecutive`` equal reads confirms.
Runs on MPS with ``max_new_tokens=4`` (2 s per crop instead of 8 s, docs/handoff.md). Referee tracks are skipped.

"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.detect.roboflow_inference import roboflow_key
from bga.device import pick_device
from bga.numbers.base import CropRead, TrackNumbers
from bga.registry import register


@register("numbers", "smolvlm_ocr")
class SmolVLMOcr(TrackNumbers):
    """Numbers backend: SmolVLM2 reads of matched ``number`` crops, confirmed by ``ConsecutiveValueTracker``.

    Examples:
        >>> ocr = SmolVLMOcr({"read_every": 5})  # doctest: +SKIP
        >>> ocr.load()  # doctest: +SKIP
        >>> ocr.run(seq, {"detect": detect_out, "track": track_out}, read)["tracks"]  # doctest: +SKIP
        {'1': '23', '2': None, ...}

    """

    defaults: ClassVar[dict[str, Any]] = {"model_id": "basketball-jersey-numbers-ocr/3", **TrackNumbers.defaults}

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
