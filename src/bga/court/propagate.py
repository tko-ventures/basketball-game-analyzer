"""Carry a keyframe's court mapping through its shot (labelling aid for M3, decision 2026-11-02).

Between consecutive sampled frames the camera's motion over the floor is a homography G (the floor is
a plane). It is estimated from SIFT matches inside the court area, which the previous mapping
predicts; players move and RANSAC drops their matches. Chaining G from the keyframe outwards gives
an image-to-court mapping for every sampled frame: ``H_j = H_k @ G_{j->k}``. Propagation stops when
matches get too few. Drift is measured, not assumed (scripts/court_propagate.py --check).

Homographies are named ``hom`` (image pixels to court centimetres) in code; ``H`` in prose.

"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

#: FIBA court length and width in centimetres.
COURT_CM = (2800.0, 1500.0)

#: SIFT keypoints and their descriptors (None when the frame has none), as ``cv2.SIFT.detectAndCompute`` returns them.
type Features = tuple[Sequence[cv2.KeyPoint], np.ndarray | None]


def court_to_image(hom: np.ndarray, floor_px: tuple[float, float] = (960.0, 648.0)) -> np.ndarray:
    """Invert an image->court homography, signed so that court points in front of the camera have w > 0.

    A homography and its negative are the same map, but "w > 0" is how a point in front of the camera
    is told from one behind it. Normalising H to h33 = 1 flips the sign whenever the image origin lies
    above the floor's horizon (stands, roof). The sign is fixed at ``floor_px``, an image point on the
    floor (default: below the centre of a 1920x1080 frame), not at a court point: from a low or
    baseline camera part of the court is behind the camera. The labeller page does the same.

    Args:
        hom: image->court homography (3x3, pixels to centimetres).
        floor_px: an image point known to lie on the floor.

    Returns:
        The court->image homography (3x3).

    Examples:
        >>> hom = np.diag([2.0, 2.0, 1.0])
        >>> court_to_image(hom) @ [1400.0, 750.0, 1.0]
        array([700., 375.,   1.])
        >>> court_to_image(-hom) @ [1400.0, 750.0, 1.0]
        array([700., 375.,   1.])

    """
    hom = np.asarray(hom, dtype=np.float64)
    inv = np.linalg.inv(hom)
    return -inv if (hom @ [floor_px[0], floor_px[1], 1.0])[2] < 0 else inv


def court_mask(hom: np.ndarray, shape: tuple[int, int], margin_cm: float = -50.0) -> np.ndarray:
    """Mask the image pixels that lie inside the court rectangle under an image->court homography.

    Args:
        hom: image->court homography (3x3).
        shape: image ``(height, width)``.
        margin_cm: grows the court rectangle by this much; the negative default keeps clear of ad boards.

    Returns:
        A ``uint8`` mask of ``shape``, 255 inside the court and 0 elsewhere.

    Examples:
        >>> mask = court_mask(np.diag([10.0, 10.0, 1.0]), (150, 280))
        >>> mask.shape, int(mask[75, 140]), int(mask[2, 2])
        ((150, 280), 255, 0)

    """
    to_img = court_to_image(hom, (shape[1] / 2, shape[0] * 0.6))
    length, width = COURT_CM
    corners = np.array(
        [
            [-margin_cm, -margin_cm],
            [length + margin_cm, -margin_cm],
            [length + margin_cm, width + margin_cm],
            [-margin_cm, width + margin_cm],
        ],
        dtype=np.float64,
    )
    # Subdivide the outline so a court edge behind the camera does not fold the polygon.
    pts = [
        a + (b - a) * t
        for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True)
        for t in np.linspace(0, 1, 20, endpoint=False)
    ]
    p = np.c_[np.array(pts), np.ones(len(pts))] @ to_img.T
    keep = p[:, 2] > 1e-9
    # a court partly behind the camera reaches far out of the frame: clip before converting
    poly = np.clip(p[keep, :2] / p[keep, 2:3], -1e5, 1e5).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [cv2.convexHull(poly)], 255)
    return mask


@dataclass
class Step:
    """One sampled frame's propagated mapping.

    Attributes:
        frame: frame index.
        H: image->court homography, or None where propagation was lost.
        inliers: RANSAC inliers of the step that reached this frame; -1 at the keyframe.

    """

    frame: int
    H: np.ndarray | None
    inliers: int


def _texture(seed: int = 0, shape: tuple[int, int] = (150, 280)) -> np.ndarray:
    """Blurred random BGR noise, a feature-rich stand-in for a frame in the doctests."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((*shape, 3)) * 255).astype(np.uint8), (5, 5), 0)


class Propagator:
    """Chain frame-to-frame floor homographies from SIFT matches inside the court.

    Args:
        min_inliers: fewest RANSAC inliers a step needs before propagation counts as lost.
        ratio: Lowe's ratio-test threshold for the two nearest descriptor matches.
        ransac_px: RANSAC reprojection threshold in pixels.

    Examples:
        >>> prop = Propagator()
        >>> prop.min_inliers, prop.ratio, prop.ransac_px
        (40, 0.75, 3.0)

    """

    def __init__(self, min_inliers: int = 40, ratio: float = 0.75, ransac_px: float = 3.0) -> None:
        """Create the SIFT detector (3000 features) and the brute-force L2 matcher."""
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers, self.ratio, self.ransac_px = min_inliers, ratio, ransac_px

    def features(self, img: np.ndarray, mask: np.ndarray | None) -> Features:
        """Detect SIFT keypoints and descriptors of a BGR frame within ``mask``.

        Args:
            img: BGR frame.
            mask: ``uint8`` mask of the pixels to search (non-zero) or None for the whole frame.

        Returns:
            Keypoints and descriptors (None when no keypoint is found).

        Examples:
            >>> kp, desc = Propagator().features(_texture(), None)
            >>> len(kp) > 100, desc.shape[1]
            (True, 128)

        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, mask)

    def step(self, a: Features, b: Features) -> tuple[np.ndarray | None, int]:
        """Estimate the homography mapping image b to image a from the features of each.

        Args:
            a: features of the reference frame.
            b: features of the frame to map onto it.

        Returns:
            The 3x3 homography and its RANSAC inlier count, or ``(None, 0)`` when there are too few matches.

        Examples:
            >>> prop, img = Propagator(), _texture()
            >>> motion, n = prop.step(prop.features(img, None), prop.features(np.roll(img, 5, axis=1), None))
            >>> round(float(motion[0, 2])), n > 40
            (-5, True)

        """
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
        motion, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        return (motion, int(inl.sum())) if motion is not None else (None, 0)

    def run(
        self, read: Callable[[int], np.ndarray], frames: list[int], key: int, hom_key: np.ndarray
    ) -> dict[int, Step]:
        """Propagate ``hom_key`` (image->court at frame ``key``) forwards and backwards over ``frames``.

        Args:
            read: returns the BGR image of a frame index.
            frames: sampled frame indices in order; must contain ``key``.
            key: the keyframe.
            hom_key: image->court homography at the keyframe.

        Returns:
            A step per frame reached, keyed by frame. A direction stops at the first lost frame, which gets a step with
            ``H=None``; frames beyond it are absent.

        Examples:
            >>> img, hom = _texture(), np.diag([10.0, 10.0, 1.0])
            >>> steps = Propagator().run(lambda f: np.roll(img, 2 * f, axis=1), [0, 1, 2], 1, hom)
            >>> sorted(steps), steps[1].inliers, round(float(steps[2].H[0, 2]))
            ([0, 1, 2], -1, -20)

        """
        out = {key: Step(key, hom_key, -1)}
        k = frames.index(key)
        for order in (frames[k + 1 :], frames[:k][::-1]):
            hom, prev_f = hom_key, key
            img = read(prev_f)
            prev = self.features(img, court_mask(hom, img.shape[:2]))
            for f in order:
                img = read(f)
                cur = self.features(img, court_mask(hom, img.shape[:2]))
                motion, n = self.step(prev, cur)
                if motion is None or n < self.min_inliers:
                    out[f] = Step(f, None, n)
                    break  # lost: the rest of this direction stays unlabelled
                hom = hom @ motion
                hom = hom / hom[2, 2]
                out[f] = Step(f, hom, n)
                prev, prev_f = cur, f
        return out


def landmarks(hom: np.ndarray, vertices: np.ndarray, wh: tuple[int, int], margin: int = 0) -> list[list[float]]:
    """Project court vertices into the image and flag which are in the frame.

    Args:
        hom: image->court homography (3x3).
        vertices: court points in centimetres, shape ``(n, 2)``.
        wh: image ``(width, height)``.
        margin: pixels beyond the frame edge that still count as inside.

    Returns:
        ``[x, y, visibility]`` per vertex: visibility 2 inside the frame, 0 outside; ``[0.0, 0.0, 0]`` behind the
        camera.

    Examples:
        >>> landmarks(np.diag([10.0, 10.0, 1.0]), np.array([[0.0, 0.0], [1400.0, 750.0], [5000.0, 0.0]]), (280, 150))
        [[0.0, 0.0, 2], [140.0, 75.0, 2], [500.0, 0.0, 0]]

    """
    to_img = court_to_image(hom, (wh[0] / 2, wh[1] * 0.6))
    p = np.c_[vertices, np.ones(len(vertices))] @ to_img.T
    out = []
    for x, y, w in p:
        if w <= 1e-9:
            out.append([0.0, 0.0, 0])
            continue
        x, y = x / w, y / w
        vis = 2 if -margin <= x < wh[0] + margin and -margin <= y < wh[1] + margin else 0
        out.append([round(float(x), 1), round(float(y), 1), vis])
    return out


def disagreement_m(hom_a: np.ndarray, hom_b: np.ndarray, vertices: np.ndarray, wh: tuple[int, int]) -> float | None:
    """Measure how far apart two mappings put the court vertices visible under the first.

    Args:
        hom_a: reference image->court homography; picks the visible vertices.
        hom_b: image->court homography to compare.
        vertices: court points in centimetres, shape ``(n, 2)``.
        wh: image ``(width, height)``.

    Returns:
        Median court distance in metres, or None when fewer than 4 vertices are visible under ``hom_a``.

    Examples:
        >>> hom_a = np.diag([10.0, 10.0, 1.0])
        >>> hom_b = np.array([[10.0, 0.0, 100.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
        >>> square = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
        >>> disagreement_m(hom_a, hom_b, square, (280, 150))
        1.0
        >>> disagreement_m(hom_a, hom_b, square[:3], (280, 150)) is None
        True

    """
    pts = [(x, y) for x, y, v in landmarks(hom_a, vertices, wh) if v]
    if len(pts) < 4:
        return None
    p = np.c_[np.array(pts), np.ones(len(pts))]
    a, b = p @ np.asarray(hom_a).T, p @ np.asarray(hom_b).T
    a, b = a[:, :2] / a[:, 2:3], b[:, :2] / b[:, 2:3]
    return float(np.median(np.linalg.norm(a - b, axis=1)) / 100.0)


def round_trip_m(
    prop: Propagator,
    read: Callable[[int], np.ndarray],
    key: int,
    hom_key: np.ndarray,
    far: int,
    vertices: np.ndarray,
    wh: tuple[int, int],
) -> float | None:
    """Measure the drift of carrying ``hom_key`` from ``key`` to ``far`` in steps of 5 frames and back in steps of 10.

    The way back uses other frame pairs, so errors do not cancel.

    Args:
        prop: the propagator to test.
        read: returns the BGR image of a frame index.
        key: the keyframe.
        hom_key: image->court homography at the keyframe.
        far: the frame to walk to (before or after ``key``).
        vertices: court points in centimetres used to compare the mappings.
        wh: image ``(width, height)``.

    Returns:
        The court distance in metres between ``hom_key`` and the mapping that comes back, or None when the chain is lost
        on the way (or fewer than 4 vertices are in view).

    Examples:
        >>> img, square = _texture(), np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
        >>> drift = round_trip_m(Propagator(), lambda f: img, 0, np.diag([10.0, 10.0, 1.0]), 10, square, (280, 150))
        >>> round(drift, 3)
        0.0

    """
    sgn = 1 if far >= key else -1
    fwd = prop.run(read, sorted(range(key, far + sgn, 5 * sgn)), key, hom_key)
    if any(st.H is None for st in fwd.values()):
        return None
    last = max(fwd) if sgn > 0 else min(fwd)
    if last == key:
        return 0.0
    back = prop.run(read, sorted({key, last} | set(range(last, key, -10 * sgn))), last, fwd[last].H)
    if back.get(key) is None or back[key].H is None:
        return None
    return disagreement_m(hom_key, back[key].H, vertices, wh)


def court_line_points(geo: dict, step_cm: float = 10.0) -> np.ndarray:
    """Sample points along the painted lines and arcs of the court geometry.

    Args:
        geo: court geometry as in ``tools/labeler/static/fiba_court.json``: ``vertices`` (cm), ``edges`` (vertex index
            pairs) and ``polylines`` (point lists, for the arcs).
        step_cm: target spacing; every segment gets at least its two end points.

    Returns:
        Points in centimetres, shape ``(n, 2)``.

    Examples:
        >>> geo = {"vertices": [[0, 0], [100, 0]], "edges": [[0, 1]], "polylines": [[[0, 0], [0, 50], [0, 100]]]}
        >>> court_line_points(geo).shape
        (20, 2)

    """
    verts = np.asarray(geo["vertices"], dtype=np.float64)
    segs = [(verts[a], verts[b]) for a, b in geo["edges"]]
    for pl in geo["polylines"]:
        line = np.asarray(pl, dtype=np.float64)
        segs += list(itertools.pairwise(line))
    pts = []
    for a, b in segs:
        n = max(2, int(np.linalg.norm(b - a) / step_cm))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n)]
    return np.array(pts)


def line_support(
    img: np.ndarray, hom: np.ndarray, line_pts: np.ndarray, tol_px: float = 2.0, bottom: float = 0.83
) -> float | None:
    """Score a mapping by the share of projected court line points that land on a painted line in the frame.

    Painted lines are bright thin structures, found with a white top-hat (15 px ellipse, threshold 25).
    Players and the score bug (the bottom ``1 - bottom`` of the frame, skipped) hide some lines, and floors
    with faint or coloured lines score low even when H is right, so compare a frame with its keyframe:
    a court that has drifted by tens of pixels loses most of its support.

    Args:
        img: BGR frame.
        hom: image->court homography (3x3).
        line_pts: court line points in centimetres (``court_line_points``).
        tol_px: largest distance to a painted line that counts as on it.
        bottom: share of the frame height kept from the top.

    Returns:
        The share in [0, 1], or None when fewer than 50 line points are in view.

    Examples:
        >>> line = np.array([[x, 500.0] for x in range(0, 2800, 10)])  # a court line 5 m from the side line
        >>> img = np.zeros((150, 280, 3), np.uint8)
        >>> line_support(img, np.diag([10.0, 10.0, 1.0]), line)
        0.0
        >>> _ = cv2.line(img, (0, 50), (279, 50), (255, 255, 255), 1)
        >>> line_support(img, np.diag([10.0, 10.0, 1.0]), line)
        1.0

    """
    h, w = img.shape[:2]
    to_img = court_to_image(hom, (w / 2, h * 0.6))
    p = np.c_[line_pts, np.ones(len(line_pts))] @ to_img.T
    p = p[p[:, 2] > 1e-9]
    p = p[:, :2] / p[:, 2:3]
    p = p[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h * bottom)]
    if len(p) < 50:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dist = cv2.distanceTransform((tophat <= 25).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist[p[:, 1].astype(int), p[:, 0].astype(int)] <= tol_px).mean())
