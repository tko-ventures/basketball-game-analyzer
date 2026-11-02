"""Carry a keyframe's court mapping through its shot (labelling aid for M3, decision 2026-11-02).

Between consecutive sampled frames the camera's motion over the floor is a homography G (the floor is
a plane). It is estimated from SIFT matches inside the court area, which the previous mapping
predicts; players move and RANSAC drops their matches. Chaining G from the keyframe outwards gives
an image-to-court mapping for every sampled frame: ``H_j = H_k @ G_{j->k}``. Propagation stops when
matches get too few. Drift is measured, not assumed (scripts/court_propagate.py --check).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

COURT_CM = (2800.0, 1500.0)


def court_mask(H: np.ndarray, shape: tuple[int, int], margin_cm: float = -50.0) -> np.ndarray:
    """Pixels inside the court rectangle (grown by margin_cm; negative keeps clear of ad boards) under H."""
    G = np.linalg.inv(H)
    L, W = COURT_CM
    corners = np.array(
        [
            [-margin_cm, -margin_cm],
            [L + margin_cm, -margin_cm],
            [L + margin_cm, W + margin_cm],
            [-margin_cm, W + margin_cm],
        ],
        dtype=np.float64,
    )
    # Subdivide the outline so a court edge behind the camera does not fold the polygon.
    pts = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        for t in np.linspace(0, 1, 20, endpoint=False):
            pts.append(a + (b - a) * t)
    p = np.c_[np.array(pts), np.ones(len(pts))] @ G.T
    keep = p[:, 2] > 1e-9
    poly = (p[keep, :2] / p[keep, 2:3]).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [cv2.convexHull(poly)], 255)
    return mask


@dataclass
class Step:
    frame: int
    H: np.ndarray | None
    inliers: int


class Propagator:
    def __init__(self, min_inliers: int = 40, ratio: float = 0.75, ransac_px: float = 3.0):
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers, self.ratio, self.ransac_px = min_inliers, ratio, ransac_px

    def features(self, img: np.ndarray, mask: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, mask)

    def step(self, a, b) -> tuple[np.ndarray | None, int]:
        """Homography mapping image b to image a, from features (kp, desc) of each."""
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None, 0
        good = [
            m
            for m, n in (p for p in self.matcher.knnMatch(db, da, k=2) if len(p) == 2)
            if m.distance < self.ratio * n.distance
        ]
        if len(good) < 8:
            return None, 0
        src = np.float32([kb[m.queryIdx].pt for m in good])
        dst = np.float32([ka[m.trainIdx].pt for m in good])
        G, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        return (G, int(inl.sum())) if G is not None else (None, 0)

    def run(self, read, frames: list[int], key: int, H_key: np.ndarray) -> dict[int, Step]:
        """Propagate H_key (image->court at frame ``key``) forwards and backwards over ``frames``."""
        out = {key: Step(key, H_key, -1)}
        k = frames.index(key)
        for order in (frames[k + 1 :], frames[:k][::-1]):
            H, prev_f = H_key, key
            img = read(prev_f)
            prev = self.features(img, court_mask(H, img.shape[:2]))
            for f in order:
                img = read(f)
                cur = self.features(img, court_mask(H, img.shape[:2]))
                G, n = self.step(prev, cur)
                if G is None or n < self.min_inliers:
                    out[f] = Step(f, None, n)
                    break  # lost: the rest of this direction stays unlabelled
                H = H @ G
                H = H / H[2, 2]
                out[f] = Step(f, H, n)
                prev, prev_f = cur, f
        return out


def landmarks(H: np.ndarray, vertices: np.ndarray, wh: tuple[int, int], margin: int = 0) -> list[list[float]]:
    """Image position and visibility (2 inside the frame, 0 outside) of each court vertex."""
    G = np.linalg.inv(H)
    p = np.c_[vertices, np.ones(len(vertices))] @ G.T
    out = []
    for x, y, w in p:
        if w <= 1e-9:
            out.append([0.0, 0.0, 0])
            continue
        x, y = x / w, y / w
        vis = 2 if -margin <= x < wh[0] + margin and -margin <= y < wh[1] + margin else 0
        out.append([round(float(x), 1), round(float(y), 1), vis])
    return out
