"""``positions/bbox_bottom``: box bottom-centre through the frame's homography, then ``clean_paths``.

The plan's baseline ground point (``ankles_pose`` is the candidate). Frames without a homography leave
gaps, which ``clean_paths`` fills by interpolation inside a track's span.

"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def project(homography: list, pts: np.ndarray) -> np.ndarray:
    """Map image points through a 3x3 homography.

    Args:
        homography: 3x3 matrix, image pixels to court cm.
        pts: Points of shape (N, 2).

    Returns:
        Mapped points of shape (N, 2).

    Examples:
        >>> project([[2, 0, 10], [0, 2, 0], [0, 0, 1]], np.array([[1.0, 1.0], [0.0, 5.0]]))
        array([[12.,  2.],
               [10., 10.]])

    """
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(homography).T
    return p[:, :2] / p[:, 2:3]


@register("positions", "bbox_bottom")
class BBoxBottom(BaseBackend):
    """Positions backend: each track box's bottom centre through the frame's homography, in metres.

    Examples:
        >>> pos = BBoxBottom({"smooth_window": 9})  # doctest: +SKIP
        >>> pos.run(seq, {"track": track_out, "court": court_out}, read)["tracks"]  # doctest: +SKIP
        {'<track_id>': [[frame, x_m, y_m], ...], ...}

    """

    defaults: ClassVar[dict[str, Any]] = {"clean": True, "smooth_window": 9, "min_jump_m": 0.7}

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Project every track's ground point to the court and clean each track's path.

        Frames without a homography are interpolated inside a track's span; with ``clean``, paths at least
        ``smooth_window`` long go through ``sports`` ``clean_paths``.

        Args:
            seq: Sequence to project.
            upstream: ``track`` and ``court`` outputs of the sequence.
            read: Unused; positions need no pixels.

        Returns:
            ``{"tracks": {"<track_id>": [[frame, x_m, y_m], ...]}, "court_m": [28.0, 15.0],
            "paths_left_uncleaned": int}``; the last counts paths ``clean_paths`` could not clean.

        """
        from sports.common.path import clean_paths

        tracks, homographies = upstream["track"]["frames"], upstream["court"]["frames"]
        ids = sorted({r[0] for f in seq.frames for r in tracks[str(f)]})
        col = {t: j for j, t in enumerate(ids)}
        xy = np.full((len(seq.frames), len(ids), 2), np.nan)
        for i, f in enumerate(seq.frames):
            homography, rows = homographies.get(str(f)), tracks[str(f)]
            if homography is None or not rows:
                continue
            feet = np.array([[(r[1] + r[3]) / 2, r[4]] for r in rows])
            for r, p in zip(rows, project(homography, feet) / 100.0, strict=True):
                xy[i, col[r[0]]] = p
        out, unclean = {}, 0
        for t, j in col.items():
            seen = np.where(np.isfinite(xy[:, j, 0]))[0]
            if len(seen) == 0:
                continue
            lo, hi = seen[0], seen[-1] + 1
            path = xy[lo:hi, j : j + 1].copy()
            idx = np.arange(hi - lo)
            ok = np.isfinite(path[:, 0, 0])
            for d in range(2):  # clean_paths needs a gap-free path: interpolate missing frames first
                path[:, 0, d] = np.interp(idx, idx[ok], path[ok, 0, d])
            if self.p["clean"] and hi - lo >= self.p["smooth_window"]:  # shorter: sports bug, n+1 out
                try:
                    path, _ = clean_paths(
                        path, min_jump_dist=self.p["min_jump_m"], smooth_window=self.p["smooth_window"]
                    )
                except ValueError:  # every point flagged as a jump leaves nothing to interpolate from
                    unclean += 1
            out[str(t)] = [
                [seq.frames[lo + k], round(float(x), 3), round(float(y), 3)]
                for k, (x, y) in enumerate(path[:, 0])
                if np.isfinite(x)
            ]
        return {"tracks": out, "court_m": [28.0, 15.0], "paths_left_uncleaned": unclean}
