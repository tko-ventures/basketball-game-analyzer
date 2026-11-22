"""``track/trackers``: Roboflow ``trackers`` (2.7) on the detector's person boxes, in pixels.

The maintained successor of supervision's ``ByteTrack`` (decision 2026-11-20): ``ByteTrackTracker``,
``BoTSORTTracker`` (camera-motion compensation, optional re-ID), ``OCSORTTracker`` and ``SORTTracker`` behind one
backend, chosen by the ``tracker`` param. Tracker arguments go through ``args`` unchanged, so ``trackers tune`` results
and the library's docs map onto configs one to one.

"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
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


#: Where precomputed camera-motion transforms are cached, one ``.npy`` per sequence and setting.
CMC_CACHE = Path("runs/_cache/cmc")


def cmc_transforms(
    seq: Sequence, dets: dict, read: FrameReader, method: str, downscale: int, mask_score: float
) -> np.ndarray:
    """Estimate BoT-SORT's camera-motion transforms for every processed frame of a sequence, once, and cache them.

    Person boxes scoring at least ``mask_score`` are masked out of the feature search, as BoT-SORT masks its
    high-confidence boxes; with ``mask_score`` equal to ``high_conf_det_threshold`` the transforms are the ones the
    tracker would estimate itself. The cache key covers the video, frames, masked boxes and CMC settings.

    Args:
        seq: The sequence.
        dets: Its ``detect`` output frames.
        read: Frame reader of the sequence.
        method: ``trackers`` CMC method, e.g. ``sparseOptFlow``.
        downscale: CMC downscale factor.
        mask_score: Minimum score of the person boxes masked out.

    Returns:
        Array ``(frames, 2, 3)`` of previous-to-current affine transforms; the first is the identity.

    Examples:
        >>> h = cmc_transforms(seq, dets, read, "sparseOptFlow", 2, 0.6)  # doctest: +SKIP
        >>> h.shape  # doctest: +SKIP
        (115, 2, 3)

    """
    from trackers import CMC, CMCConfig

    masks = [[d[:4] for d in dets[str(f)] if d[5] in PERSON_CLASSES and d[4] >= mask_score] for f in seq.frames]
    blob = json.dumps([seq.video, seq.frames, masks, method, downscale], separators=(",", ":"))
    path = CMC_CACHE / f"{seq.name}_{hashlib.sha256(blob.encode()).hexdigest()[:12]}.npy"
    if path.exists():
        return np.load(path)
    cmc = CMC(CMCConfig(method=method, downscale=downscale))
    out = np.stack([
        cmc.estimate(read(f), np.array(m, dtype=np.float32) if m else None)
        for f, m in zip(seq.frames, masks, strict=True)
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


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


def reid_source(reid_model: str, weights: str | None) -> str:
    """The ``reid`` model source BoT-SORT loads: the param as given, or the entry's trained encoder folder.

    Args:
        reid_model: The ``reid_model`` param: a ``reid`` source, or ``weights``.
        weights: The config entry's weights id.

    Returns:
        A source ``reid.ReIDModel.from_pretrained`` accepts.

    Raises:
        ValueError: For ``weights`` without an ``appearance/reid`` weight id.

    Examples:
        >>> reid_source("osnet_x1_0_sportsmot", None)
        'osnet_x1_0_sportsmot'
        >>> reid_source("weights", "appearance/reid/plk-r1@553936e9").endswith("appearance/reid/plk-r1@553936e9")
        True

    """
    if reid_model != "weights":
        return reid_model
    if not weights or not weights.startswith("appearance/reid/"):
        raise ValueError(f"reid_model: weights needs an appearance/reid weight id, not {weights!r}")
    from bga.paths import weights_dir

    return str(weights_dir(weights))


@register("track", "trackers")
class RoboflowTrackers(BaseBackend):
    """Track backend over the ``trackers`` package; outputs the ``track`` layer format, confirmed tracks only.

    Params (``defaults``): ``tracker`` (``bytetrack``, ``botsort``, ``ocsort``, ``sort``), ``min_score`` (boxes below
    are not passed in), ``lost_track_buffer_s``, ``args`` (other constructor arguments, passed as given),
    ``reid_model`` (a ``reid`` model source for BoT-SORT, ``weights`` for the entry's trained ``appearance/reid``
    weight id, or ``None``), ``reid_config`` (``trackers.ReIDConfig``
    fields), ``cmc`` and ``cmc_mask_score``. With ``cmc: internal`` BoT-SORT gets each frame and estimates camera
    motion itself; with ``cmc: precomputed`` the transforms come from :func:`cmc_transforms`, cached per sequence, so
    a parameter sweep decodes the frames once. Re-ID still gets the frames.

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
        "cmc": "internal",
        "cmc_mask_score": 0.6,
    }

    def _make(self, rate: float, override: dict | None = None) -> BaseTracker:
        """Build the configured tracker for a sequence processed at ``rate`` frames per second, ``override`` on top."""
        import trackers

        name = self.p["tracker"]
        if name not in TRACKERS:
            raise ValueError(f"unknown tracker {name!r}; one of {sorted(TRACKERS)}")
        kwargs = {
            "frame_rate": rate,
            "lost_track_buffer": lost_buffer_frames(self.p["lost_track_buffer_s"]),
            **self.p["args"],
            **(override or {}),
        }
        if self.p["reid_model"]:
            if name != "botsort":
                raise ValueError("reid_model needs tracker: botsort")
            kwargs["reid_model"] = reid_source(self.p["reid_model"], self.weights)
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
        dets, out, t = upstream["detect"]["frames"], {}, 0.0
        botsort = self.p["tracker"] == "botsort"
        pre = botsort and self.p["cmc"] == "precomputed" and self.p["args"].get("enable_cmc", True)
        if pre and self.p["reid_model"]:
            raise ValueError("cmc: precomputed cannot run with re-ID: trackers takes a frame or a transform, not both")
        if pre:
            a = self.p["args"]
            hs = cmc_transforms(
                seq,
                dets,
                read,
                a.get("cmc_method", "sparseOptFlow"),
                a.get("cmc_downscale", 2),
                self.p["cmc_mask_score"],
            )
        tracker = self._make(seq.fps / stride, {"enable_cmc": False} if pre else None)  # transforms, not frames
        # frames only feed BoT-SORT's own camera-motion estimate and re-ID
        wants_frame = botsort and not pre and (self.p["args"].get("enable_cmc", True) or bool(self.p["reid_model"]))
        for j, f in enumerate(seq.frames):
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
            if pre:
                res = tracker.update(det, cmc_transform=hs[j])
            else:
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
