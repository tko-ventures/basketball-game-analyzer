"""``team/siglip_kmeans``: Skalski's ``sports.TeamClassifier`` (SigLIP + UMAP + KMeans), fitted per sequence.

Fitted on central crops (40% of the box) of player boxes sampled at ``fit_fps``; referees come from the
detector class, never from a cluster. The brighter cluster takes the side whose kit is "light" in the
clip meta (plan section 5, layer 3); a clip whose meta does not mark exactly one side ``light`` is refused.
SigLIP is loaded once in ``load``; each sequence gets its own UMAP and KMeans.

"""

from __future__ import annotations

import time
from collections.abc import Sequence as Seq
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.detect.base import PERSON_CLASSES
from bga.device import pick_device
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence
from bga.team.base import REF_TEAM


def central_crop(img: np.ndarray, box: Seq[float], factor: float, cy: float = 0.5) -> np.ndarray:
    """Cut a part of a box, scaled by ``factor`` around a point on its vertical axis, clipped to the image.

    Args:
        img: Image to crop.
        box: Box as x1, y1, x2, y2 (extra items are ignored).
        factor: Share of the box's width and height kept around the crop centre.
        cy: Height of the crop centre as a share of the box height from its top; 0.5 is the box centre, about 0.35
            the chest of a standing player.

    Returns:
        A view into ``img``; empty when the scaled box falls outside the image.

    Examples:
        >>> img = np.zeros((100, 200, 3), np.uint8)
        >>> central_crop(img, (0, 0, 100, 50, 0.9, "player"), 0.4).shape
        (20, 40, 3)
        >>> central_crop(img, (190, 90, 210, 110), 1.0).shape
        (10, 10, 3)
        >>> img[10:30] = 255  # the top of the box is white
        >>> int(central_crop(img, (0, 0, 100, 100), 0.2, cy=0.2).mean())
        255

    """
    x1, y1, x2, y2 = box[:4]
    cx, cy, w, h = (x1 + x2) / 2, y1 + (y2 - y1) * cy, (x2 - x1) * factor, (y2 - y1) * factor
    img_h, img_w = img.shape[:2]
    return img[
        max(0, int(cy - h / 2)) : min(img_h, int(cy + h / 2)), max(0, int(cx - w / 2)) : min(img_w, int(cx + w / 2))
    ]


@dataclass(frozen=True)
class Separation:
    """How well the two kit clusters of one sequence separate, measured on the points they were fitted on.

    Attributes:
        silhouette: Mean silhouette over the fitted points, from -1 to 1; near 1 when the clusters are compact and
            far apart.
        minor_share: Share of the fitted points in the smaller cluster; 0.5 is an even split.

    """

    silhouette: float
    minor_share: float


def cluster_separation(points: np.ndarray, labels: np.ndarray) -> Separation:
    """Score the separation of a two-cluster labelling of points.

    Args:
        points: Points as an ``(n, d)`` array.
        labels: Cluster label per point, 0 or 1.

    Returns:
        The silhouette and the smaller cluster's share, both rounded to 3 places. A single cluster scores silhouette
        0 and share 0.

    Examples:
        >>> pts = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
        >>> cluster_separation(pts, np.array([0, 0, 1, 1]))
        Separation(silhouette=0.9, minor_share=0.5)
        >>> cluster_separation(pts, np.array([0, 0, 0, 0]))
        Separation(silhouette=0.0, minor_share=0.0)

    """
    from sklearn.metrics import silhouette_score

    sizes = np.bincount(labels, minlength=2)
    if sizes.min() == 0:
        return Separation(0.0, 0.0)
    return Separation(round(float(silhouette_score(points, labels)), 3), round(float(sizes.min() / sizes.sum()), 3))


def light_side(meta: dict) -> str:
    """The side whose kit the clip meta marks ``light``; the brighter cluster is given to it.

    Args:
        meta: A sequence's clip metadata, with ``home`` and ``away`` records.

    Returns:
        ``home`` or ``away``.

    Raises:
        ValueError: Unless exactly one side's ``kit`` is ``light``. Guessing would name the teams the wrong way
            round without a sign, as it did on four non-gold games on 2026-11-22.

    Examples:
        >>> light_side({"home": {"kit": "dark"}, "away": {"kit": "light"}})
        'away'
        >>> light_side({"home": {"kit": ""}, "away": {}})
        Traceback (most recent call last):
        ...
        ValueError: clip meta must mark one side's kit light and the other dark; got home '', away None

    """
    kits = {side: (meta.get(side) or {}).get("kit") for side in ("home", "away")}
    light = [side for side, kit in kits.items() if kit == "light"]
    if len(light) != 1:
        raise ValueError(
            "clip meta must mark one side's kit light and the other dark;"
            f" got home {kits['home']!r}, away {kits['away']!r}"
        )
    return light[0]


@register("team", "siglip_kmeans")
class SiglipKMeans(BaseBackend):
    """Team backend: ``sports.TeamClassifier`` fitted on the sequence's own player crops.

    Examples:
        >>> team = SiglipKMeans({"device": "cpu"})  # doctest: +SKIP
        >>> team.load()  # doctest: +SKIP
        >>> team.run(seq, {"detect": detect_out}, read)["frames"]  # doctest: +SKIP
        {'<frame>': ['home', 'away', 'ref', '', ...], ...}

    """

    defaults: ClassVar[dict[str, Any]] = {
        "fit_fps": 1.0,
        "crop": 0.4,
        "crop_cy": 0.5,
        "min_score": 0.4,
        "device": "mps",
        "seed": None,
    }

    def load(self) -> None:
        """Build ``sports.TeamClassifier`` once: its constructor loads SigLIP, which every sequence then shares."""
        from sports import TeamClassifier

        self.classifier = TeamClassifier(device=pick_device(self.p["device"]))

    def unload(self) -> None:
        """Drop the classifier and its SigLIP model."""
        self.classifier = None

    def _fit_predict(self, crops: list[np.ndarray], fit: list[np.ndarray]) -> tuple[list[int], Separation]:
        """Fit the classifier on ``fit``, predict a cluster per crop, and score the fitted clusters' separation."""
        import umap
        from sklearn.cluster import KMeans

        tc = self.classifier
        # a new UMAP and KMeans per sequence, as ``TeamClassifier`` builds them, so no fit carries over; they draw
        # random starts, and a seed makes reruns identical
        tc.reducer = umap.UMAP(n_components=3, random_state=self.p["seed"])
        tc.cluster_model = KMeans(n_clusters=2, random_state=self.p["seed"])
        tc.fit(fit)
        sep = cluster_separation(tc.reducer.embedding_, tc.cluster_model.labels_)
        return [int(p) for p in tc.predict(crops)], sep

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Label each detected box ``home``, ``away``, ``ref`` or ``""``.

        Player boxes scoring at least ``min_score`` are cropped (``crop`` of the box around the point ``crop_cy`` of its
        height from the top); crops from frames every ``1 / fit_fps`` seconds fit
        the classifier (all crops when fewer than 10), and every crop is predicted. The cluster with the brighter mean
        HSV value goes to the side whose kit the clip meta marks ``light`` (:func:`light_side`). Referees come from the
        detector class. With fewer than 4 crops no clustering is done. The ``seed`` param, when set, seeds UMAP and
        KMeans. The separation of the fitted clusters is reported and changes no label.

        Args:
            seq: Sequence to label.
            upstream: ``detect`` output of the sequence.
            read: Frame reader of the sequence's video.

        Returns:
            ``{"frames": {"<frame>": [label per detect box]}, "clusters": {"<cluster>": side}, "separation":
            {"silhouette": float, "minor_share": float}, "ms_per_frame": float}``; ``separation`` and ``ms_per_frame``
            are absent when no clustering was done.

        Raises:
            ValueError: When the clip meta does not mark exactly one side's kit ``light``.

        """
        light = light_side(seq.meta)  # before any work: a clip without kits cannot be labelled
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
                c = central_crop(img, d, self.p["crop"], self.p["crop_cy"])
                if c.size:
                    keys.append((f, i))
                    crops.append(c)
                    fit_mask.append((f - seq.frames[0]) % step == 0)
        out = {str(f): [REF_TEAM if d[5] == "referee" else "" for d in dets[str(f)]] for f in seq.frames}
        if len(crops) < 4:
            return {"frames": out, "clusters": {}}
        fit = [c for c, m in zip(crops, fit_mask, strict=True) if m]
        pred, sep = self._fit_predict(crops, fit if len(fit) >= 10 else crops)
        bright = [
            np.mean(
                [cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2].mean() for c, p in zip(crops, pred, strict=True) if p == k]
                or [0]
            )
            for k in (0, 1)
        ]
        dark = "away" if light == "home" else "home"
        side = {int(np.argmax(bright)): light, int(np.argmin(bright)): dark}
        for (f, i), p in zip(keys, pred, strict=True):
            out[str(f)][i] = side[p]
        ms = 1000 * (time.perf_counter() - t0) / max(1, len(seq.frames))
        return {
            "frames": out,
            "clusters": {str(k): v for k, v in side.items()},
            "separation": asdict(sep),
            "ms_per_frame": round(ms, 1),
        }
