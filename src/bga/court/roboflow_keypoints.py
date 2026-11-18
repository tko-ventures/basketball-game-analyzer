"""``court/roboflow_keypoints``: Skalski's NBA court keypoint model, homography to the FIBA court.

Landmarks with confidence above ``kp_conf`` (at least ``min_points``) map to
``CourtConfiguration(League.FIBA)`` vertices in centimetres with RANSAC (plan section 5, layer 1).
``smooth_window`` > 0 median-smooths the mappings within the shot (``bga.court.smooth``); off in the
baseline. Known: the NBA model misplaces landmarks in Lublin and mostly fails in Sopot
(docs/decisions.md 2026-10-31); M3 fine-tunes it.

"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.court.smooth import smooth_homographies
from bga.detect.roboflow_inference import roboflow_key
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("court", "roboflow_keypoints")
class RoboflowKeypoints(BaseBackend):
    """Court layer backend: Roboflow's hosted NBA court keypoint model, its landmarks fitted to the FIBA court.

    Params (``defaults``): ``model_id`` (Roboflow model), ``confidence`` (detection score), ``kp_conf`` (landmark
    confidence), ``min_points`` (fewest confident landmarks to fit), ``ransac_cm`` (RANSAC threshold on the court) and
    ``smooth_window`` (0 is off).

    Examples:
        >>> backend = RoboflowKeypoints({"kp_conf": 0.6})
        >>> backend.p["model_id"], backend.p["kp_conf"]
        ('basketball-court-detection-2/14', 0.6)

    """

    defaults: ClassVar[dict[str, Any]] = {
        "model_id": "basketball-court-detection-2/14",
        "confidence": 0.3,
        "kp_conf": 0.5,
        "min_points": 4,
        "ransac_cm": 40.0,
        "smooth_window": 0,
    }

    def load(self) -> None:
        """Load the Roboflow model (needs ``ROBOFLOW_API_KEY``) and the ``sports`` FIBA court vertices.

        Examples:
            >>> backend = RoboflowKeypoints({})
            >>> backend.load()  # doctest: +SKIP

        """
        roboflow_key()
        from inference import get_model
        from sports.basketball import CourtConfiguration, League

        self.model = get_model(model_id=self.p["model_id"])
        self.vertices = np.array(CourtConfiguration(league=League.FIBA).vertices, dtype=np.float32)

    def unload(self) -> None:
        """Drop the model so the next layer has the memory.

        Examples:
            >>> backend = RoboflowKeypoints({})
            >>> backend.unload()
            >>> backend.model is None
            True

        """
        self.model = None

    def homography(self, img: np.ndarray) -> list[list[float]] | None:
        """Fit the image->court homography of one frame from the first court prediction.

        Args:
            img: BGR frame.

        Returns:
            The 3x3 homography (pixels to centimetres, rounded to 8 decimals), or None when no court is predicted, fewer
            than ``min_points`` landmarks are confident or RANSAC fails.

        Examples:
            >>> backend.homography(read(0))  # doctest: +SKIP
            [[...], [...], [...]]

        """
        kp = self.model.infer(img, confidence=self.p["confidence"])[0]
        if not kp.predictions or len(kp.predictions[0].keypoints) != len(self.vertices):
            return None
        pts = kp.predictions[0].keypoints
        mask = np.array([p.confidence for p in pts]) > self.p["kp_conf"]
        if mask.sum() < self.p["min_points"]:
            return None
        src = np.array([[p.x, p.y] for p in pts], dtype=np.float32)[mask]
        m, _ = cv2.findHomography(src, self.vertices[mask], cv2.RANSAC, self.p["ransac_cm"])
        return None if m is None else np.round(m, 8).tolist()

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict[str, Any]:
        """Map every frame of a sequence, then smooth within the shot when ``smooth_window`` > 0.

        Args:
            seq: the sequence (one shot) to process.
            upstream: upstream layer outputs (unused; the court layer reads frames only).
            read: returns the BGR image of a frame index.

        Returns:
            ``{"frames": {"<frame>": H | None}, "units": "cm", "ms_per_frame": float}``; inference time only.

        Examples:
            >>> backend.run(seq, {}, read)["units"]  # doctest: +SKIP
            'cm'

        """
        out, t = {}, 0.0
        for f in seq.frames:
            img = read(f)
            t0 = time.perf_counter()
            out[str(f)] = self.homography(img)
            t += time.perf_counter() - t0
        if self.p.get("smooth_window", 0) and seq.frames:
            h, w = read(seq.frames[0]).shape[:2]
            out = smooth_homographies(seq.frames, out, (w, h), int(self.p["smooth_window"]))
        return {"frames": out, "units": "cm", "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1)}
