"""``court/rfdetr_keypoint``: an RF-DETR Keypoint court model trained by ``bga train``, homography to FIBA.

Weights are a trained weight id (``court/rfdetr_keypoint/<name>@<hash8>``, whose ``train.json`` gives the
class and keypoint counts the model was built with). Per frame, the most confident court detection
gives 33 landmarks; those with confidence above ``kp_conf`` (at least ``min_points``) map to the FIBA
vertices in centimetres (``tools/labeler/static/fiba_court.json``, the same order as the model's
keypoints) with RANSAC, as in ``court/roboflow_keypoints``. ``smooth_window`` > 0 median-smooths the
mappings within the shot (``bga.court.smooth``). Runs on MPS on the Mac.

"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.court.smooth import smooth_homographies
from bga.device import pick_device
from bga.paths import weights_dir
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

#: Repository root; weights live under ``weights/`` and the court geometry under ``tools/``.
ROOT = Path(__file__).resolve().parents[3]


@register("court", "rfdetr_keypoint")
class RFDETRKeypoint(BaseBackend):
    """Court layer backend: a fine-tuned RF-DETR Keypoint model, its landmarks fitted to the FIBA court.

    Params (``defaults``): ``threshold`` (detection score), ``kp_conf`` (landmark confidence), ``min_points``
    (fewest confident landmarks to fit), ``ransac_cm`` (RANSAC threshold on the court), ``device`` and
    ``smooth_window`` (0 is off).

    Examples:
        >>> backend = RFDETRKeypoint({}, "court/rfdetr_keypoint/plk-v1@c26b7841")
        >>> backend.p["kp_conf"], backend.p["device"]
        (0.5, 'mps')

    """

    defaults: ClassVar[dict[str, Any]] = {
        "threshold": 0.3,
        "kp_conf": 0.5,
        "min_points": 4,
        "ransac_cm": 40.0,
        "device": "mps",
        "smooth_window": 0,
    }

    def load(self) -> None:
        """Load the checkpoint of the trained weight id, built with its ``train.json`` schema, and the FIBA vertices.

        Raises:
            ValueError: when ``weights`` is missing or a ``pretrained:`` id rather than a trained weight id.

        Examples:
            >>> backend = RFDETRKeypoint({}, "court/rfdetr_keypoint/plk-v1@c26b7841")
            >>> backend.load()  # doctest: +SKIP

        """
        from rfdetr import RFDETRKeypointPreview

        if not self.weights or self.weights.startswith("pretrained:"):
            raise ValueError("needs a trained weight id: court/rfdetr_keypoint/<name>@<hash8>")
        wdir = weights_dir(self.weights)
        schema = json.loads((wdir / "train.json").read_text())["trainer"]["schema"]
        self.model = RFDETRKeypointPreview.from_checkpoint(
            str(wdir / "checkpoint_best_total.pth"),
            num_classes=len(schema["class_names"]),
            num_keypoints_per_class=schema["num_keypoints_per_class"],
            device=pick_device(self.p["device"]),
            trust_checkpoint=True,
        )
        geo = json.loads((ROOT / "tools/labeler/static/fiba_court.json").read_text())
        self.vertices = np.array(geo["vertices"], dtype=np.float32)

    def unload(self) -> None:
        """Drop the model so the next layer has the memory.

        Examples:
            >>> backend = RFDETRKeypoint({}, "court/rfdetr_keypoint/plk-v1@c26b7841")
            >>> backend.unload()
            >>> backend.model is None
            True

        """
        self.model = None

    def homography(self, img_bgr: np.ndarray) -> list[list[float]] | None:
        """Fit the image->court homography of one frame from the most confident court detection.

        Args:
            img_bgr: BGR frame.

        Returns:
            The 3x3 homography (pixels to centimetres, rounded to 8 decimals), or None when no court is detected, fewer
            than ``min_points`` landmarks are confident or RANSAC fails.

        Examples:
            >>> backend.homography(read(0))  # doctest: +SKIP
            [[...], [...], [...]]

        """
        kp = self.model.predict(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), threshold=self.p["threshold"])
        if len(kp) == 0 or kp.xy.shape[1] != len(self.vertices):
            return None
        best = int(np.argmax(kp.detection_confidence)) if kp.detection_confidence is not None else 0
        conf, xy = kp.keypoint_confidence[best], kp.xy[best]
        mask = conf > self.p["kp_conf"]
        if mask.sum() < self.p["min_points"]:
            return None
        src = xy[mask].astype(np.float32)
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
