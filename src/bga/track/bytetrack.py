"""``track/bytetrack``: supervision 0.30.9's ByteTrack (vendored in ``_sv_bytetrack``) on the person boxes, in pixels.

The baseline of plan section 5 (``bytetrack_court`` later runs the motion model in court metres).

"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from bga.detect.base import PERSON_CLASSES
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence
from bga.track._sv_bytetrack import ByteTrack as ByteTrackSV


@register("track", "bytetrack")
class ByteTrack(BaseBackend):
    """Track backend: ``supervision.ByteTrack`` on person boxes, in pixel space.

    ``lost_track_buffer_s`` is converted to frames at the sequence's processed rate (``fps / stride``).

    Examples:
        >>> tracker = ByteTrack({"min_score": 0.2})  # doctest: +SKIP
        >>> tracker.run(seq, {"detect": detect_out}, read)["frames"]  # doctest: +SKIP
        {'<frame>': [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...], ...}

    """

    defaults: ClassVar[dict[str, Any]] = {
        "min_score": 0.1,
        "track_activation_threshold": 0.25,
        "lost_track_buffer_s": 1.0,
        "minimum_matching_threshold": 0.8,
        "minimum_consecutive_frames": 1,
    }

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Track the detector's person boxes scoring at least ``min_score`` through the sequence.

        Args:
            seq: Sequence to track.
            upstream: ``detect`` output of the sequence.
            read: Unused; tracking needs no pixels.

        Returns:
            ``{"frames": {"<frame>": [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...]},
            "ms_per_frame": float}``.

        """
        import supervision as sv

        stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
        rate = seq.fps / stride
        tracker = ByteTrackSV(
            track_activation_threshold=self.p["track_activation_threshold"],
            lost_track_buffer=max(1, round(self.p["lost_track_buffer_s"] * rate)),
            minimum_matching_threshold=self.p["minimum_matching_threshold"],
            frame_rate=rate,
            minimum_consecutive_frames=self.p["minimum_consecutive_frames"],
        )
        dets, out, t = upstream["detect"]["frames"], {}, 0.0
        for f in seq.frames:
            rows = [
                (i, d) for i, d in enumerate(dets[str(f)]) if d[5] in PERSON_CLASSES and d[4] >= self.p["min_score"]
            ]
            t0 = time.perf_counter()
            if rows:
                det = sv.Detections(
                    xyxy=np.array([d[:4] for _, d in rows], dtype=np.float32),
                    confidence=np.array([d[4] for _, d in rows], dtype=np.float32),
                    class_id=np.zeros(len(rows), dtype=int),
                    data={"idx": np.array([i for i, _ in rows]), "cls": np.array([d[5] for _, d in rows])},
                )
            else:
                det = sv.Detections.empty()
            res = tracker.update_with_detections(det)
            t += time.perf_counter() - t0
            out[str(f)] = (
                [
                    [
                        int(res.tracker_id[k]),
                        *[round(float(v), 1) for v in res.xyxy[k]],
                        round(float(res.confidence[k]), 4),
                        str(res.data["cls"][k]),
                        int(res.data["idx"][k]),
                    ]
                    for k in range(len(res))
                ]
                if len(res)
                else []
            )
        return {"frames": out, "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 2)}
