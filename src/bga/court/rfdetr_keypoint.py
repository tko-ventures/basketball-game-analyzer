"""``court/rfdetr_keypoint``: an RF-DETR Keypoint court model trained by ``bga train``, homography to FIBA.

Weights are a trained weight id (``court/rfdetr_keypoint/<name>@<hash8>``, whose ``train.json`` gives the
class and keypoint counts the model was built with). Per frame, the most confident court detection
gives 33 landmarks; those with confidence above ``kp_conf`` (at least ``min_points``) map to the FIBA
vertices in centimetres (``tools/labeler/static/fiba_court.json``, the same order as the model's
keypoints) with RANSAC, as in ``court/roboflow_keypoints``. No smoothing. Runs on MPS on the Mac.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

ROOT = Path(__file__).resolve().parents[3]


@register("court", "rfdetr_keypoint")
class RFDETRKeypoint(BaseBackend):
    defaults = {"threshold": 0.3, "kp_conf": 0.5, "min_points": 4, "ransac_cm": 40.0, "device": "mps"}

    def load(self) -> None:
        from rfdetr import RFDETRKeypointPreview

        if not self.weights or self.weights.startswith("pretrained:"):
            raise ValueError("needs a trained weight id: court/rfdetr_keypoint/<name>@<hash8>")
        wdir = ROOT / "weights" / self.weights
        schema = json.loads((wdir / "train.json").read_text())["trainer"]["schema"]
        self.model = RFDETRKeypointPreview.from_checkpoint(
            str(wdir / "checkpoint_best_total.pth"),
            num_classes=len(schema["class_names"]),
            num_keypoints_per_class=schema["num_keypoints_per_class"],
            device=self.p["device"],
            trust_checkpoint=True,
        )
        geo = json.loads((ROOT / "tools/labeler/static/fiba_court.json").read_text())
        self.vertices = np.array(geo["vertices"], dtype=np.float32)

    def unload(self) -> None:
        self.model = None

    def homography(self, img_bgr: np.ndarray) -> list | None:
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

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        out, t = {}, 0.0
        for f in seq.frames:
            img = read(f)
            t0 = time.perf_counter()
            out[str(f)] = self.homography(img)
            t += time.perf_counter() - t0
        return {"frames": out, "units": "cm", "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1)}
