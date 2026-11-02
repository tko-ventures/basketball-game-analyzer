"""``positions/bbox_bottom``: box bottom-centre through the frame's homography, then ``clean_paths``.

The plan's baseline ground point (``ankles_pose`` is the candidate). Frames without a homography leave
gaps, which ``clean_paths`` fills by interpolation inside a track's span.
"""

from __future__ import annotations

import numpy as np

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def project(H: list, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


@register("positions", "bbox_bottom")
class BBoxBottom(BaseBackend):
    defaults = {"clean": True, "smooth_window": 9, "min_jump_m": 0.7}

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        from sports.common.path import clean_paths

        tracks, Hs = upstream["track"]["frames"], upstream["court"]["frames"]
        ids = sorted({r[0] for f in seq.frames for r in tracks[str(f)]})
        col = {t: j for j, t in enumerate(ids)}
        xy = np.full((len(seq.frames), len(ids), 2), np.nan)
        for i, f in enumerate(seq.frames):
            H, rows = Hs.get(str(f)), tracks[str(f)]
            if H is None or not rows:
                continue
            feet = np.array([[(r[1] + r[3]) / 2, r[4]] for r in rows])
            for r, p in zip(rows, project(H, feet) / 100.0, strict=True):
                xy[i, col[r[0]]] = p
        out = {}
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
            if self.p["clean"] and hi - lo >= 3:
                path, _ = clean_paths(
                    path, min_jump_dist=self.p["min_jump_m"], smooth_window=self.p["smooth_window"]
                )
            out[str(t)] = [
                [seq.frames[lo + k], round(float(x), 3), round(float(y), 3)]
                for k, (x, y) in enumerate(path[:, 0])
                if np.isfinite(x)
            ]
        return {"tracks": out, "court_m": [28.0, 15.0]}
