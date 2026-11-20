"""``track/trackers``: Roboflow ``trackers`` (2.7) on the detector's person boxes, in pixels.

The maintained successor of supervision's ``ByteTrack`` (decision 2026-11-20): ``ByteTrackTracker``,
``BoTSORTTracker`` (camera-motion compensation, optional re-ID), ``OCSORTTracker`` and ``SORTTracker`` behind one
backend, chosen by the ``tracker`` param. Tracker arguments go through ``args`` unchanged, so ``trackers tune`` results
and the library's docs map onto configs one to one.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from bga.detect.base import PERSON_CLASSES
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

if TYPE_CHECKING:
    from trackers.core.base import BaseTracker

#: ``tracker`` param value to the ``trackers`` class name.
TRACKERS = {
    "bytetrack": "ByteTrackTracker",
    "botsort": "BoTSORTTracker",
    "ocsort": "OCSORTTracker",
    "sort": "SORTTracker",
}


def lost_buffer_frames(seconds: float) -> int:
    """Convert a lost-track buffer in seconds to ``trackers``' unit, frames at 30 fps (it rescales by frame rate).

    Args:
        seconds: How long a lost track is kept.

    Returns:
        The ``lost_track_buffer`` argument.

    Examples:
        >>> lost_buffer_frames(1.0), lost_buffer_frames(2.5), lost_buffer_frames(0)
        (30, 75, 0)

    """
    return round(seconds * 30)


@register("track", "trackers")
class RoboflowTrackers(BaseBackend):
    """Track backend over the ``trackers`` package; outputs the ``track`` layer format, confirmed tracks only.

    Params (``defaults``): ``tracker`` (``bytetrack``, ``botsort``, ``ocsort``, ``sort``), ``min_score`` (boxes below
    are not passed in), ``lost_track_buffer_s``, ``args`` (other constructor arguments, passed as given),
    ``reid_model`` (a ``reid`` model source for BoT-SORT, or ``None``) and ``reid_config`` (``trackers.ReIDConfig``
    fields). BoT-SORT gets each frame, for camera-motion compensation and re-ID; the others get none.

    Examples:
        >>> tracker = RoboflowTrackers({"tracker": "botsort", "args": {"enable_cmc": True}})  # doctest: +SKIP
        >>> tracker.run(seq, {"detect": detect_out}, read)["frames"]  # doctest: +SKIP
        {'<frame>': [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...], ...}

    """

    defaults: ClassVar[dict[str, Any]] = {
        "tracker": "bytetrack",
        "min_score": 0.1,
        "lost_track_buffer_s": 1.0,
        "args": {},
        "reid_model": None,
        "reid_config": {},
    }

    def _make(self, rate: float) -> BaseTracker:
        """Build the configured tracker for a sequence processed at ``rate`` frames per second."""
        import trackers

        name = self.p["tracker"]
        if name not in TRACKERS:
            raise ValueError(f"unknown tracker {name!r}; one of {sorted(TRACKERS)}")
        kwargs = {
            "frame_rate": rate,
            "lost_track_buffer": lost_buffer_frames(self.p["lost_track_buffer_s"]),
            **self.p["args"],
        }
        if self.p["reid_model"]:
            if name != "botsort":
                raise ValueError("reid_model needs tracker: botsort")
            kwargs["reid_model"] = self.p["reid_model"]
            kwargs["reid_config"] = trackers.ReIDConfig(**self.p["reid_config"])
        return getattr(trackers, TRACKERS[name])(**kwargs)

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Track the detector's person boxes scoring at least ``min_score`` through the sequence.

        Track ids are the library's plus one, so they start at 1 as the ``bytetrack`` backend's do.

        Args:
            seq: Sequence to track.
            upstream: ``detect`` output of the sequence.
            read: Frame reader; used by BoT-SORT only.

        Returns:
            ``{"frames": {"<frame>": [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...]},
            "ms_per_frame": float}``.

        """
        import supervision as sv

        stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
        tracker = self._make(seq.fps / stride)
        wants_frame = self.p["tracker"] == "botsort"
        dets, out, t = upstream["detect"]["frames"], {}, 0.0
        for f in seq.frames:
            rows = [
                (i, d) for i, d in enumerate(dets[str(f)]) if d[5] in PERSON_CLASSES and d[4] >= self.p["min_score"]
            ]
            frame = read(f) if wants_frame else None
            det = (
                sv.Detections(
                    xyxy=np.array([d[:4] for _, d in rows], dtype=np.float32),
                    confidence=np.array([d[4] for _, d in rows], dtype=np.float32),
                    class_id=np.zeros(len(rows), dtype=int),
                    data={"idx": np.array([i for i, _ in rows]), "cls": np.array([d[5] for _, d in rows])},
                )
                if rows
                else sv.Detections.empty()
            )
            t0 = time.perf_counter()
            res = tracker.update(det, frame) if wants_frame else tracker.update(det)
            t += time.perf_counter() - t0
            keep = [k for k in range(len(res)) if res.tracker_id is not None and res.tracker_id[k] >= 0]
            out[str(f)] = [
                [
                    int(res.tracker_id[k]) + 1,
                    *[round(float(v), 1) for v in res.xyxy[k]],
                    round(float(res.confidence[k]), 4),
                    str(res.data["cls"][k]),
                    int(res.data["idx"][k]),
                ]
                for k in keep
            ]
        return {"frames": out, "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 2)}
