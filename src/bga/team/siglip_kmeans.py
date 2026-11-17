"""``team/siglip_kmeans``: Skalski's ``sports.TeamClassifier`` (SigLIP + UMAP + KMeans), fitted per sequence.

Fitted on central crops (40% of the box) of player boxes sampled at ``fit_fps``; referees come from the
detector class, never from a cluster. The brighter cluster takes the side whose kit is "light" in the
clip meta (plan section 5, layer 3).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from bga.detect.base import PERSON_CLASSES
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def central_crop(img: np.ndarray, box, factor: float) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    H, W = img.shape[:2]
    return img[
        max(0, int(cy - h / 2)) : min(H, int(cy + h / 2)), max(0, int(cx - w / 2)) : min(W, int(cx + w / 2))
    ]


@register("team", "siglip_kmeans")
class SiglipKMeans(BaseBackend):
    defaults = {"fit_fps": 1.0, "crop": 0.4, "min_score": 0.4, "device": "mps"}

    def load(self) -> None:
        from sports import TeamClassifier

        self.TeamClassifier = TeamClassifier

    def unload(self) -> None:
        self.TeamClassifier = None

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
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
                [
                    cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2].mean()
                    for c, p in zip(crops, pred, strict=True)
                    if p == k
                ]
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
