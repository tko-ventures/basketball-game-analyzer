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


def court_to_image(H: np.ndarray, floor_px: tuple[float, float] = (960.0, 648.0)) -> np.ndarray:
    """Inverse of an image->court H, signed so that court points in front of the camera have w > 0.

    A homography and its negative are the same map, but "w > 0" is how a point in front of the camera
    is told from one behind it. Normalising H to h33 = 1 flips the sign whenever the image origin lies
    above the floor's horizon (stands, roof). The sign is fixed at ``floor_px``, an image point on the
    floor (default: below the centre of a 1920x1080 frame), not at a court point: from a low or
    baseline camera part of the court is behind the camera. The labeller page does the same.
    """
    H = np.asarray(H, dtype=np.float64)
    G = np.linalg.inv(H)
    return -G if (H @ [floor_px[0], floor_px[1], 1.0])[2] < 0 else G


def court_mask(H: np.ndarray, shape: tuple[int, int], margin_cm: float = -50.0) -> np.ndarray:
    """Pixels inside the court rectangle (grown by margin_cm; negative keeps clear of ad boards) under H."""
    G = court_to_image(H, (shape[1] / 2, shape[0] * 0.6))
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
    # a court partly behind the camera reaches far out of the frame: clip before converting
    poly = np.clip(p[keep, :2] / p[keep, 2:3], -1e5, 1e5).astype(np.int32)
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
    G = court_to_image(H, (wh[0] / 2, wh[1] * 0.6))
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


def disagreement_m(Ha: np.ndarray, Hb: np.ndarray, vertices: np.ndarray, wh: tuple[int, int]) -> float | None:
    """Median distance in metres between where Ha and Hb put the court vertices visible under Ha."""
    pts = [(x, y) for x, y, v in landmarks(Ha, vertices, wh) if v]
    if len(pts) < 4:
        return None
    p = np.c_[np.array(pts), np.ones(len(pts))]
    a, b = p @ np.asarray(Ha).T, p @ np.asarray(Hb).T
    a, b = a[:, :2] / a[:, 2:3], b[:, :2] / b[:, 2:3]
    return float(np.median(np.linalg.norm(a - b, axis=1)) / 100.0)


def round_trip_m(
    prop: Propagator, read, key: int, H_key: np.ndarray, far: int, vertices: np.ndarray, wh: tuple[int, int]
) -> float | None:
    """Drift of carrying H_key from ``key`` to ``far`` in steps of 5 frames and back in steps of 10.

    The way back uses other frame pairs, so errors do not cancel. Returns the court distance in metres
    between H_key and the mapping that comes back, or None when the chain is lost on the way.
    """
    sgn = 1 if far >= key else -1
    fwd = prop.run(read, sorted(range(key, far + sgn, 5 * sgn)), key, H_key)
    if any(st.H is None for st in fwd.values()):
        return None
    last = max(fwd) if sgn > 0 else min(fwd)
    if last == key:
        return 0.0
    back = prop.run(read, sorted({key, last} | set(range(last, key, -10 * sgn))), last, fwd[last].H)
    if back.get(key) is None or back[key].H is None:
        return None
    return disagreement_m(H_key, back[key].H, vertices, wh)


def court_line_points(geo: dict, step_cm: float = 10.0) -> np.ndarray:
    """Points every ``step_cm`` along the painted lines and arcs of the court geometry (fiba_court.json)."""
    verts = np.asarray(geo["vertices"], dtype=np.float64)
    segs = [(verts[a], verts[b]) for a, b in geo["edges"]]
    for pl in geo["polylines"]:
        pl = np.asarray(pl, dtype=np.float64)
        segs += list(zip(pl[:-1], pl[1:], strict=True))
    pts = []
    for a, b in segs:
        n = max(2, int(np.linalg.norm(b - a) / step_cm))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n)]
    return np.array(pts)


def line_support(
    img: np.ndarray, H: np.ndarray, line_pts: np.ndarray, tol_px: float = 2.0, bottom: float = 0.83
) -> float | None:
    """Share of the projected court line points that fall within ``tol_px`` of a painted line in the frame.

    Painted lines are bright thin structures, found with a white top-hat (15 px ellipse, threshold 25).
    Players and the score bug (the bottom ``1 - bottom`` of the frame, skipped) hide some lines, and floors
    with faint or coloured lines score low even when H is right, so compare a frame with its keyframe:
    a court that has drifted by tens of pixels loses most of its support. None when fewer than 50 line
    points are in view.
    """
    h, w = img.shape[:2]
    G = court_to_image(H, (w / 2, h * 0.6))
    p = np.c_[line_pts, np.ones(len(line_pts))] @ G.T
    p = p[p[:, 2] > 1e-9]
    p = p[:, :2] / p[:, 2:3]
    p = p[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h * bottom)]
    if len(p) < 50:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dist = cv2.distanceTransform((tophat <= 25).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist[p[:, 1].astype(int), p[:, 0].astype(int)] <= tol_px).mean())
