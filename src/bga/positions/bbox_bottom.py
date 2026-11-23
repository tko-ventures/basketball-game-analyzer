"""``positions/bbox_bottom``: bottom-centre of each track box, projected through the homography and cleaned.

Consumes ``link`` when configured (falling back to ``track``). The baseline layer-6 backend (plan section 5):
projects ``((x1 + x2) / 2, y2)`` through the per-frame homography (guarded against projective ``w <= 0``
singularities) and runs ``sports.common.path.clean_paths`` per track within the shot.

"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from bga.positions.base import COURT_M, active_tracks, clean_track_paths, court_scale, project
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

__all__ = ["BBoxBottom", "project"]


@register("positions", "bbox_bottom")
class BBoxBottom(BaseBackend):
    """Positions layer backend: box bottom-centre through the court homography, with ``clean_paths`` smoothing.

    Params (``defaults``): ``clean`` (whether to run ``clean_paths``), ``smooth_window`` and ``min_jump_m``.

    Examples:
        >>> BBoxBottom({}).p
        {'clean': True, 'smooth_window': 9, 'min_jump_m': 0.7}

    """

    defaults: ClassVar[dict[str, Any]] = {"clean": True, "smooth_window": 9, "min_jump_m": 0.7}

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Project every tracked person's box bottom-centre onto the FIBA court for ``seq``.

        Args:
            seq: Sequence to run on.
            upstream: ``link`` (when configured) or ``track``, and ``court`` outputs for ``seq``.
            read: Unused; positions are derived from track boxes and homographies.

        Returns:
            ``{"tracks", "foot_px", "court_m", "paths_left_uncleaned"}``.

        Examples:
            >>> seq = Sequence("s", "c", "v.mp4", 0, [0], 50.0)
            >>> up = {
            ...     "track": {"frames": {"0": [[1, 10, 20, 30, 60, 0.9, "player", 0]]}},
            ...     "court": {"frames": {"0": [[10, 0, 0], [0, 10, 0], [0, 0, 1]]}, "units": "cm"},
            ... }
            >>> out = BBoxBottom({"clean": False}).run(seq, up, None)
            >>> out["tracks"], out["foot_px"]
            ({'1': [[0, 2.0, 6.0]]}, {'1': [[0, 20.0, 60.0]]})

        """
        del read
        tracks = active_tracks(upstream)
        homographies = upstream["court"]["frames"]
        scale = court_scale(upstream["court"])
        ids = sorted({int(r[0]) for rows in tracks.values() for r in rows})
        if not ids:
            return {"tracks": {}, "foot_px": {}, "court_m": list(COURT_M), "paths_left_uncleaned": 0}
        col = {tid: j for j, tid in enumerate(ids)}
        xy = np.full((len(seq.frames), len(ids), 2), np.nan, dtype=np.float64)
        foot_px: dict[str, list[list[float]]] = {str(tid): [] for tid in ids}
        for i, f in enumerate(seq.frames):
            rows = tracks.get(str(f), [])
            if not rows:
                continue
            pts = np.array([[(r[1] + r[3]) / 2.0, r[4]] for r in rows], dtype=np.float64)
            for r, pt in zip(rows, pts, strict=True):
                foot_px[str(int(r[0]))].append([f, round(float(pt[0]), 1), round(float(pt[1]), 1)])
            homography = homographies.get(str(f))
            if homography is None:
                continue
            projected = project(homography, pts) * scale
            for r, p in zip(rows, projected, strict=True):
                xy[i, col[int(r[0])]] = p
        tracks_out, uncleaned = clean_track_paths(
            seq,
            xy,
            ids,
            clean=bool(self.p["clean"]),
            smooth_window=int(self.p["smooth_window"]),
            min_jump_m=float(self.p["min_jump_m"]),
        )
        return {
            "tracks": tracks_out,
            "foot_px": {k: v for k, v in foot_px.items() if v},
            "court_m": list(COURT_M),
            "paths_left_uncleaned": uncleaned,
        }
