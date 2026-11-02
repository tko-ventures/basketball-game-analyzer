"""``court/roboflow_keypoints``: Skalski's NBA court keypoint model, homography to the FIBA court.

Landmarks with confidence above ``kp_conf`` (at least ``min_points``) map to
``CourtConfiguration(League.FIBA)`` vertices in centimetres with RANSAC (plan section 5, layer 1).
No smoothing in the baseline. Known: the NBA model misplaces landmarks in Lublin and mostly fails in
Sopot (docs/decisions.md 2026-10-31); M3 fine-tunes it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from bga.detect.roboflow_inference import roboflow_key
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("court", "roboflow_keypoints")
class RoboflowKeypoints(BaseBackend):
    defaults = {
        "model_id": "basketball-court-detection-2/14",
        "confidence": 0.3,
        "kp_conf": 0.5,
        "min_points": 4,
        "ransac_cm": 40.0,
    }

    def load(self) -> None:
        roboflow_key()
        from inference import get_model
        from sports.basketball import CourtConfiguration, League

        self.model = get_model(model_id=self.p["model_id"])
        self.vertices = np.array(CourtConfiguration(league=League.FIBA).vertices, dtype=np.float32)

    def unload(self) -> None:
        self.model = None

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        out, t = {}, 0.0
        for f in seq.frames:
            img = read(f)
            t0 = time.perf_counter()
            kp = self.model.infer(img, confidence=self.p["confidence"])[0]
            H = None
            if kp.predictions and len(kp.predictions[0].keypoints) == len(self.vertices):
                pts = kp.predictions[0].keypoints
                mask = np.array([p.confidence for p in pts]) > self.p["kp_conf"]
                if mask.sum() >= self.p["min_points"]:
                    src = np.array([[p.x, p.y] for p in pts], dtype=np.float32)[mask]
                    m, _ = cv2.findHomography(src, self.vertices[mask], cv2.RANSAC, self.p["ransac_cm"])
                    H = None if m is None else np.round(m, 8).tolist()
            t += time.perf_counter() - t0
            out[str(f)] = H
        return {"frames": out, "units": "cm", "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1)}
