"""``detect/roboflow_inference``: a Roboflow-hosted model run locally through ``inference``.

The baseline uses Skalski's public RF-DETR checkpoints of basketball-player-detection-3 (ONNX on
the CoreML provider, downloaded and cached by ``inference``; ROBOFLOW_API_KEY from
~/.config/bga/roboflow.env). Weights are declared pretrained by model id in the run config.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def roboflow_key() -> None:
    if os.environ.get("ROBOFLOW_API_KEY"):
        return
    env = Path.home() / ".config/bga/roboflow.env"
    if env.is_file():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k == "ROBOFLOW_API_KEY" and v:
                os.environ[k] = v.strip().strip('"')


@register("detect", "roboflow_inference")
class RoboflowInference(BaseBackend):
    defaults = {"model_id": "basketball-player-detection-3-ycjdo/6", "confidence": 0.4, "iou": 0.9}

    def load(self) -> None:
        roboflow_key()
        from inference import get_model

        self.model = get_model(model_id=self.p["model_id"])

    def unload(self) -> None:
        self.model = None

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        frames, t = {}, 0.0
        for f in seq.frames:
            img = read(f)
            t0 = time.perf_counter()
            r = self.model.infer(img, confidence=self.p["confidence"], iou_threshold=self.p["iou"])[0]
            t += time.perf_counter() - t0
            frames[str(f)] = [
                [
                    round(p.x - p.width / 2, 1),
                    round(p.y - p.height / 2, 1),
                    round(p.x + p.width / 2, 1),
                    round(p.y + p.height / 2, 1),
                    round(float(p.confidence), 4),
                    p.class_name,
                ]
                for p in r.predictions
            ]
        return {"frames": frames, "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1)}
