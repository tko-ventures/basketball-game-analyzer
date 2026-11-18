"""``team/siglip_kmeans``: Skalski's ``sports.TeamClassifier`` (SigLIP + UMAP + KMeans), fitted per sequence.

Fitted on central crops (40% of the box) of player boxes sampled at ``fit_fps``; referees come from the
detector class, never from a cluster. The brighter cluster takes the side whose kit is "light" in the
clip meta (plan section 5, layer 3).

"""

from __future__ import annotations

import time
from collections.abc import Sequence as Seq
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.detect.base import PERSON_CLASSES
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def central_crop(img: np.ndarray, box: Seq[float], factor: float) -> np.ndarray:
    """Cut the centre of a box, scaled by ``factor`` in both sides, clipped to the image.

    Args:
        img: Image to crop.
        box: Box as x1, y1, x2, y2 (extra items are ignored).
        factor: Share of the box's width and height kept around its centre.

    Returns:
        A view into ``img``; empty when the scaled box falls outside the image.

    Examples:
        >>> img = np.zeros((100, 200, 3), np.uint8)
        >>> central_crop(img, (0, 0, 100, 50, 0.9, "player"), 0.4).shape
        (20, 40, 3)
        >>> central_crop(img, (190, 90, 210, 110), 1.0).shape
        (10, 10, 3)

    """
    x1, y1, x2, y2 = box[:4]
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    img_h, img_w = img.shape[:2]
    return img[
        max(0, int(cy - h / 2)) : min(img_h, int(cy + h / 2)), max(0, int(cx - w / 2)) : min(img_w, int(cx + w / 2))
    ]


@register("team", "siglip_kmeans")
class SiglipKMeans(BaseBackend):
    """Team backend: ``sports.TeamClassifier`` fitted on the sequence's own player crops.

    Examples:
        >>> team = SiglipKMeans({"device": "cpu"})  # doctest: +SKIP
        >>> team.load()  # doctest: +SKIP
        >>> team.run(seq, {"detect": detect_out}, read)["frames"]  # doctest: +SKIP
        {'<frame>': ['home', 'away', 'ref', '', ...], ...}

    """

    defaults: ClassVar[dict[str, Any]] = {"fit_fps": 1.0, "crop": 0.4, "min_score": 0.4, "device": "mps"}

    def load(self) -> None:
        """Import ``sports.TeamClassifier``; the classifier itself is built per sequence in :meth:`run`."""
        from sports import TeamClassifier

        self.TeamClassifier = TeamClassifier

    def unload(self) -> None:
        """Drop the classifier class."""
        self.TeamClassifier = None

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Label each detected box ``home``, ``away``, ``ref`` or ``""``.

        Player boxes scoring at least ``min_score`` are cropped; crops from frames every ``1 / fit_fps`` seconds fit
        the classifier (all crops when fewer than 10), and every crop is predicted. The cluster with the brighter mean
        HSV value goes to home when the clip meta marks the home kit ``light``, else to away. Referees come from the
        detector class. With fewer than 4 crops no clustering is done.

        Args:
            seq: Sequence to label.
            upstream: ``detect`` output of the sequence.
            read: Frame reader of the sequence's video.

        Returns:
            ``{"frames": {"<frame>": [label per detect box]}, "clusters": {"<cluster>": side}, "ms_per_frame": float}``;
            ``ms_per_frame`` is absent when no clustering was done.

        """
        dets = upstream["detect"]["frames"]
        keys, crops, fit_mask = [], [], []
        step = max(1, round(seq.fps / self.p["fit_fps"]))
        t0 = time.perf_counter()
        for f in seq.frames:
            img = None
            for i, d in enumerate(dets[str(f)]):
                if d[5] not in PERSON_CLASSES or d[5] == "referee" or d[4] < self.p["min_score"]:
                    continue
                img = read(f) if img is None else img
                c = central_crop(img, d, self.p["crop"])
                if c.size:
                    keys.append((f, i))
                    crops.append(c)
                    fit_mask.append((f - seq.frames[0]) % step == 0)
        out = {str(f): ["ref" if d[5] == "referee" else "" for d in dets[str(f)]] for f in seq.frames}
        if len(crops) < 4:
            return {"frames": out, "clusters": {}}
        tc = self.TeamClassifier(device=self.p["device"])
        fit = [c for c, m in zip(crops, fit_mask, strict=True) if m]
        tc.fit(fit if len(fit) >= 10 else crops)
        pred = [int(p) for p in tc.predict(crops)]
        bright = [
            np.mean(
                [cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2].mean() for c, p in zip(crops, pred, strict=True) if p == k]
                or [0]
            )
            for k in (0, 1)
        ]
        light = "home" if seq.meta.get("home", {}).get("kit") == "light" else "away"
        dark = "away" if light == "home" else "home"
        side = {int(np.argmax(bright)): light, int(np.argmin(bright)): dark}
        for (f, i), p in zip(keys, pred, strict=True):
            out[str(f)][i] = side[p]
        ms = 1000 * (time.perf_counter() - t0) / max(1, len(seq.frames))
        return {"frames": out, "clusters": {str(k): v for k, v in side.items()}, "ms_per_frame": round(ms, 1)}
