"""Temporal smoothing of per-frame court homographies within one shot (plan section 5, layer 1).

A sequence is one shot, so nothing here crosses a cut. Each frame's image->court mapping is evaluated at
a fixed grid of image points; for every grid point the court positions it gets from the frames in a
centred window are reduced to their median, and the frame's homography is refitted from the grid to
those medians. A frame whose mapping jumps is outvoted by its neighbours; a steady camera pan passes
through a centred median without lag; a frame with no mapping gets one from its neighbours.

Grid points count only where a mapping puts them on the floor (inside the court grown by ``margin_m``),
so points in the stands, which a homography sends far away, never vote.

"""

from __future__ import annotations

import cv2
import numpy as np

#: FIBA court length and width in centimetres.
COURT_CM = (2800.0, 1500.0)


def _grid(wh: tuple[int, int], nx: int = 12, ny: int = 8, top: float = 0.2, bottom: float = 0.83) -> np.ndarray:
    """Image points of an ``nx`` by ``ny`` grid between ``top`` and ``bottom`` of the frame, row by row."""
    xs = np.linspace(0.04, 0.96, nx) * wh[0]
    ys = np.linspace(top, bottom, ny) * wh[1]
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def _map(hom: np.ndarray | list[list[float]], pts: np.ndarray, margin_cm: float) -> np.ndarray:
    """Court positions (cm) of image points under H; NaN where off the floor or behind the horizon."""
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(hom, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        c = p[:, :2] / p[:, 2:3]
    sign = np.sign(np.median(p[:, 2]))
    ok = (p[:, 2] * sign > 1e-9) & (c[:, 0] > -margin_cm) & (c[:, 0] < COURT_CM[0] + margin_cm)
    ok &= (c[:, 1] > -margin_cm) & (c[:, 1] < COURT_CM[1] + margin_cm)
    c[~ok] = np.nan
    return c


def smooth_homographies(
    frames: list[int], mappings: dict[str, list | None], wh: tuple[int, int], window: int, margin_m: float = 2.0
) -> dict[str, list | None]:
    """Median-smooth the image->court mappings of one shot over ``window`` frames each way.

    Args:
        frames: the shot's frame indices, in order.
        mappings: image->court homography (3x3 nested list) or None per frame, keyed by the frame as a string.
        wh: image ``(width, height)``.
        window: frames each way in the median; 0 or less returns ``mappings`` unchanged.
        margin_m: grid points count only where a mapping puts them inside the court grown by this many metres.

    Returns:
        Smoothed homographies keyed like ``mappings`` (rounded to 8 decimals), for every frame in ``frames``. A frame
        gets None when no neighbour has a mapping, and keeps its own mapping when fewer than 8 grid points have enough
        votes.

    Examples:
        >>> hom = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]
        >>> out = smooth_homographies([0, 1, 2], {"0": hom, "1": None, "2": hom}, (280, 150), window=1)
        >>> np.allclose(out["1"], hom, atol=1e-3)  # the gap gets its neighbours' mapping
        True

    """
    if window <= 0:
        return mappings
    grid = _grid(wh)
    mapped = [None if mappings.get(str(f)) is None else _map(mappings[str(f)], grid, margin_m * 100) for f in frames]
    out: dict[str, list | None] = {}
    for i, f in enumerate(frames):
        near = [m for m in mapped[max(0, i - window) : i + window + 1] if m is not None]
        if not near:
            out[str(f)] = None
            continue
        stack = np.stack(near)  # (frames, points, 2)
        with np.errstate(all="ignore"):
            counts = np.sum(~np.isnan(stack[:, :, 0]), axis=0)
            med = np.nanmedian(stack, axis=0)
        keep = counts >= max(2, (len(near) + 1) // 2)
        if keep.sum() < 8:
            out[str(f)] = mappings.get(str(f))
            continue
        fit, _ = cv2.findHomography(grid[keep].astype(np.float32), med[keep].astype(np.float32), 0)
        out[str(f)] = None if fit is None else np.round(fit, 8).tolist()
    return out
