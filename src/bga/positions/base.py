"""Layer 6 (positions): shared projection and path-smoothing helpers.

Output per sequence:
    {
        "tracks": {"<track_id>": [[frame, x_m, y_m], ...]},
        "foot_px": {"<track_id>": [[frame, fx, fy], ...]},
        "court_m": [28.0, 15.0],
        "paths_left_uncleaned": 0,
    }

Coordinates are in metres on the FIBA court ``[0..28] x [0..15]``. ``foot_px`` records the raw image ground
point (pixels) that was projected on each observed frame; backends with airborne detection also include
``"airborne": {"<track_id>": [frame, ...]}``.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from bga.stage import Backend as PositionsBackend  # noqa: F401  # re-exported: the positions layer's backend protocol
from bga.stage import Sequence

#: FIBA court dimensions in metres (CourtConfiguration(League.FIBA) / 100).
COURT_M: list[float] = [28.0, 15.0]
#: Minimum projective homogeneous weight ``w`` before a point is treated as singular (on/behind the horizon).
MIN_W: float = 1e-6


def active_tracks(upstream: dict[str, dict[str, Any]]) -> dict[str, list]:
    """Return the per-frame track rows from ``link`` when configured, falling back to ``track``.

    Args:
        upstream: Upstream layer outputs for one sequence.

    Returns:
        Mapping from frame string to list of track rows ``[tid, x1, y1, x2, y2, conf, cls, det_idx]``.

    Examples:
        >>> active_tracks({"track": {"frames": {"0": [[1]]}}})
        {'0': [[1]]}
        >>> active_tracks({"track": {"frames": {"0": [[1]]}}, "link": {"frames": {"0": [[2]]}}})
        {'0': [[2]]}

    """
    return (upstream.get("link") or upstream["track"])["frames"]


def court_scale(court_out: dict[str, Any]) -> float:
    """Scale factor from court homography units to metres (all built-in court backends output centimetres)."""
    return 1.0 if court_out.get("units") == "m" else 0.01


def project(homography: list[list[float]] | np.ndarray, pts: np.ndarray, *, min_w: float = MIN_W) -> np.ndarray:
    """Map ``(N, 2)`` image points through a 3x3 homography, returning ``NaN`` on ``w <= min_w`` singularities.

    Args:
        homography: 3x3 matrix mapping image pixels to court coordinates.
        pts: ``(N, 2)`` array of image points.
        min_w: Minimum positive homogeneous denominator ``w`` accepted; points on or behind the camera horizon
            (``w <= min_w`` or non-finite) yield ``[nan, nan]`` instead of ``Inf`` or sign-flipped coordinates.

    Returns:
        ``(N, 2)`` array of projected points in the homography's target units, with ``NaN`` for singular points.

    Examples:
        >>> project([[2, 0, 10], [0, 2, 20], [0, 0, 1]], np.array([[1.0, 2.0]])).tolist()
        [[12.0, 24.0]]
        >>> bool(np.all(np.isnan(project([[1, 0, 0], [0, 1, 0], [0, 0, 0]], np.array([[1.0, 2.0]])))))
        True

    """
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(arr) == 0:
        return np.empty((0, 2), dtype=np.float64)
    p = np.c_[arr, np.ones(len(arr), dtype=np.float64)] @ np.asarray(homography, dtype=np.float64).T
    w = p[:, 2:3]
    out = np.full((len(arr), 2), np.nan, dtype=np.float64)
    ok = np.isfinite(w[:, 0]) & (w[:, 0] > min_w) & np.all(np.isfinite(p[:, :2]), axis=1)
    if np.any(ok):
        out[ok] = p[ok, :2] / w[ok]
    return out


def clean_track_paths(
    seq: Sequence,
    xy: np.ndarray,
    ids: list[int],
    *,
    clean: bool,
    smooth_window: int,
    min_jump_m: float,
) -> tuple[dict[str, list[list[float]]], int]:
    """Interpolate gaps inside each track's valid span and optionally smooth with ``sports.common.path.clean_paths``.

    Args:
        seq: Sequence whose ``frames`` align with axis 0 of ``xy``.
        xy: ``(T, K, 2)`` array of raw court coordinates in metres, ``NaN`` where absent or unmapped.
        ids: Length-``K`` sorted list of track ids matching axis 1 of ``xy``.
        clean: Whether to apply ``clean_paths`` smoothing.
        smooth_window: Savitzky-Golay / rolling window length for ``clean_paths``.
        min_jump_m: Minimum jump distance in metres flagged by ``clean_paths``.

    Returns:
        Tuple ``(tracks, paths_left_uncleaned)`` where ``tracks`` maps ``str(tid)`` to ``[[frame, x_m, y_m], ...]``.

    Examples:
        >>> seq = Sequence("s", "c", "v.mp4", 0, [0, 5, 10], 50.0)
        >>> xy = np.array([[[1.0, 2.0]], [[np.nan, np.nan]], [[3.0, 4.0]]])
        >>> tracks, bad = clean_track_paths(seq, xy, [7], clean=False, smooth_window=9, min_jump_m=0.7)
        >>> tracks["7"], bad
        ([[0, 1.0, 2.0], [5, 2.0, 3.0], [10, 3.0, 4.0]], 0)

    """
    if not ids:
        return {}, 0
    clean_paths = None
    if clean:
        from sports.common.path import clean_paths as _clean_paths

        clean_paths = _clean_paths
    tracks_out: dict[str, list[list[float]]] = {}
    uncleaned = 0
    for j, tid in enumerate(ids):
        seen = np.where(np.isfinite(xy[:, j, 0]) & np.isfinite(xy[:, j, 1]))[0]
        if len(seen) == 0:
            continue
        span = np.arange(seen[0], seen[-1] + 1)
        path = np.column_stack([np.interp(span, seen, xy[seen, j, c]) for c in (0, 1)])
        if clean_paths is not None and len(path) >= smooth_window:
            try:
                cleaned, _ = clean_paths(
                    path[:, None, :],
                    min_jump_dist=min_jump_m,
                    smooth_window=smooth_window,
                )
                path = cleaned[:, 0, :]
            except (ValueError, IndexError):
                uncleaned += 1
        tracks_out[str(tid)] = [
            [seq.frames[k], round(float(path[n, 0]), 3), round(float(path[n, 1]), 3)]
            for n, k in enumerate(span)
            if np.isfinite(path[n, 0]) and np.isfinite(path[n, 1])
        ]
    return tracks_out, uncleaned
