"""Detector backend over RF-DETR weights trained by ``bga train`` (plan section 5, layer 2), run with PyTorch.

The weight id names ``weights/detect/rfdetr_torch/<name>@<hash8>/`` with ``checkpoint_best_total.pth`` and the
``train.json`` written by ``bga train``. The class names come from the checkpoint, the tiling from ``train.json``:
weights trained on square tiles (``recipe.tiles``) are run on the same tiles and their detections merged
(``bga.detect.tiles``), so training and inference see objects at the same scale.

"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.detect.tiles import merge, square_tiles
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence
from bga.train.config import ROOT


@register("detect", "rfdetr_torch")
class RFDETRTorch(BaseBackend):
    """Detector backend over trained RF-DETR weights; ``device`` defaults to MPS on the Mac.

    Params (``defaults``): ``confidence`` (score threshold) and ``device``.

    Examples:
        >>> det = RFDETRTorch({}, "detect/rfdetr_torch/nba-704@00000000")
        >>> det.p["confidence"], det.p["device"]
        (0.05, 'mps')

    """

    defaults: ClassVar[dict[str, Any]] = {"confidence": 0.05, "device": "mps"}

    def load(self) -> None:
        """Load the checkpoint of the trained weight id and read its class names and tiling from ``train.json``.

        Raises:
            ValueError: When ``weights`` is missing or a ``pretrained:`` id.

        Examples:
            >>> RFDETRTorch({}, "detect/rfdetr_torch/nba-704@00000000").load()  # doctest: +SKIP

        """
        from rfdetr import RFDETR

        if not self.weights or self.weights.startswith("pretrained:"):
            raise ValueError("needs a trained weight id: detect/rfdetr_torch/<name>@<hash8>")
        wdir = ROOT / "weights" / self.weights
        tj = json.loads((wdir / "train.json").read_text())
        self.tiles = tj["config"]["recipe"].get("tiles") or 1
        self.model = RFDETR.from_checkpoint(
            str(wdir / "checkpoint_best_total.pth"), device=self.p["device"], trust_checkpoint=True
        )
        # the checkpoint's own names, indexed by class id; the COCO categories in train.json start with the
        # "basketball" placeholder, which the model drops, so indexing those would shift every class by one
        self.class_names = self.model.class_names

    def unload(self) -> None:
        """Drop the model so the next layer has the memory.

        Examples:
            >>> det = RFDETRTorch({}, "detect/rfdetr_torch/nba-704@00000000")
            >>> det.unload()
            >>> det.model is None
            True

        """
        self.model = None

    def _predict(self, rgb: np.ndarray) -> list[list]:
        """Detections ``[x1, y1, x2, y2, score, class name]`` of one RGB image."""
        d = self.model.predict(rgb, threshold=self.p["confidence"])
        return [
            [*(round(float(v), 1) for v in box), round(float(s), 4), self.class_names[int(c)]]
            for box, s, c in zip(d.xyxy, d.confidence, d.class_id, strict=True)
            if 0 <= int(c) < len(self.class_names)
        ]

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Detect on every frame of the sequence, on the whole frame or on the tiles the weights were trained on.

        Args:
            seq: Sequence whose frames are detected.
            upstream: Unused; detection needs no upstream layer.
            read: Frame reader of the sequence's video.

        Returns:
            ``{"frames": {"<frame>": [[x1, y1, x2, y2, score, class_name], ...]}, "ms_per_frame": float}``.

        Examples:
            >>> det.run(seq, {}, FrameReader(seq.video))["frames"]  # doctest: +SKIP
            {'<frame>': [[x1, y1, x2, y2, score, class_name], ...], ...}

        """
        frames, t = {}, 0.0
        for f in seq.frames:
            rgb = cv2.cvtColor(read(f), cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            if self.tiles > 1:
                tiles = square_tiles(rgb.shape[1], rgb.shape[0], self.tiles)
                per = [self._predict(rgb[y1:y2, x1:x2]) for x1, y1, x2, y2 in tiles]
                dets = merge(per, tiles)
            else:
                dets = self._predict(rgb)
            t += time.perf_counter() - t0
            frames[str(f)] = [[*(round(v, 1) for v in d[:4]), d[4], d[5]] for d in dets]
        return {"frames": frames, "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1)}
