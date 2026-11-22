"""``track/sam2``: SAM 2.1 video propagation prompted with the detector's boxes (plan section 5, layer 4 ``sam2``).

Skalski's SAM2 tracking path (plan section 3), run with the official SAM 2.1 video predictor on the whole shot instead
of the real-time fork, as the gold prefill did (``bga.remote.sam2_prefill``): the detector's person boxes on the
shot's first grid frame are box prompts; masks propagate through every source frame of the shot; a detection that no
mask covers on two consecutive grid frames becomes a new object there and propagation continues from it, at most
``max_rounds`` times. Frames go to SAM 2 as JPEGs with a ``long_side`` of 1024, its input size.

On each grid frame, each object's mask bounding box is a track row. It takes the class, score and index of the
detection it overlaps best at IoU ``match_iou`` or more, so the team layer's labels reach it; otherwise it is a
``player`` row with score 0.5 and no detection (index -1). Runs on CUDA only (``requires``), on Modal.

"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from bga.detect.base import PERSON_CLASSES
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

#: Score given to a mask row that no detection matches.
UNMATCHED_SCORE = 0.5


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two ``[x1, y1, x2, y2, ...]`` boxes; 0 when their union is empty.

    Args:
        a: First box.
        b: Second box.

    Returns:
        Intersection over union.

    Examples:
        >>> iou([0, 0, 10, 10], [5, 0, 15, 10])
        0.3333333333333333

    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def prompt_boxes(dets: list, min_score: float) -> list[list[float]]:
    """Person boxes a frame's detections give SAM 2 as prompts: score at least ``min_score``, not wider than tall.

    Args:
        dets: The frame's detections, ``[x1, y1, x2, y2, score, class_name, ...]``.
        min_score: Lowest score kept.

    Returns:
        The boxes as ``[x1, y1, x2, y2]``.

    Examples:
        >>> prompt_boxes([[0, 0, 10, 30, 0.9, "player"], [0, 0, 30, 10, 0.9, "player"], [0, 0, 5, 5, 0.9, "ball"]], 0.4)
        [[0, 0, 10, 30]]

    """
    return [
        list(d[:4]) for d in dets if d[5] in PERSON_CLASSES and d[4] >= min_score and (d[2] - d[0]) <= (d[3] - d[1])
    ]


def mask_rows(objects: dict[int, list | None], dets: list, match_iou: float) -> list[list]:
    """Track rows of one grid frame: each object's mask box with the class, score and index of its best detection.

    Args:
        objects: Object id to its mask box ``[x1, y1, x2, y2, ...]`` in frame pixels, or ``None`` without a mask.
        dets: The frame's detections.
        match_iou: Least IoU for a detection to lend the row its class, score and index.

    Returns:
        ``[object_id, x1, y1, x2, y2, score, class_name, det_index]`` rows, by object id.

    Examples:
        >>> mask_rows({1: [0, 0, 10, 20], 2: None, 3: [50, 50, 60, 70]}, [[0, 0, 10, 21, 0.8, "referee"]], 0.5)
        [[1, 0, 0, 10, 20, 0.8, 'referee', 0], [3, 50, 50, 60, 70, 0.5, 'player', -1]]

    """
    rows = []
    for oid in sorted(objects):
        box = objects[oid]
        if not box:
            continue
        scored = [(iou(box, d), i) for i, d in enumerate(dets) if d[5] in PERSON_CLASSES]
        best, i = max(scored, default=(0.0, -1))
        if best >= match_iou:
            rows.append([oid, *box[:4], dets[i][4], dets[i][5], i])
        else:
            rows.append([oid, *box[:4], UNMATCHED_SCORE, "player", -1])
    return rows


@register("track", "sam2")
class Sam2Video(BaseBackend):
    """Track backend: SAM 2.1 video propagation from detector box prompts, re-prompted on uncovered detections.

    Params (``defaults``): ``model_id`` (Hugging Face id of the SAM 2.1 weights), ``min_score`` (prompt boxes),
    ``long_side`` (JPEG size given to SAM 2), ``max_rounds`` (propagation passes) and ``match_iou``.

    Examples:
        >>> Sam2Video({}).run(seq, {"detect": detect_out}, read)["objects"]  # doctest: +SKIP
        14

    """

    requires: ClassVar[str | None] = "cuda"
    #: Modal GPU the backend runs on.
    gpu: ClassVar[str] = "L40S"
    defaults: ClassVar[dict[str, Any]] = {
        "model_id": "facebook/sam2.1-hiera-large",
        "min_score": 0.4,
        "long_side": 1024,
        "max_rounds": 12,
        "match_iou": 0.5,
    }

    def load(self) -> None:
        """Load the SAM 2.1 video predictor on CUDA."""
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        torch.backends.cuda.matmul.allow_tf32 = True
        self.pred = SAM2VideoPredictor.from_pretrained(self.p["model_id"], device="cuda")

    def unload(self) -> None:
        """Drop the predictor."""
        self.pred = None

    def _write_frames(self, seq: Sequence, read: FrameReader, folder: Path) -> float:
        """Write every source frame of the shot as a JPEG at ``long_side``; return the scale from frame pixels."""
        first = read(seq.frames[0])
        scale = self.p["long_side"] / max(first.shape[:2])
        size = (round(first.shape[1] * scale), round(first.shape[0] * scale))
        for j, f in enumerate(range(seq.frames[0], seq.frames[-1] + 1)):
            img = cv2.resize(read(f), size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(folder / f"{j:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return scale

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Track the shot with SAM 2.1 and return its grid frames' mask boxes as track rows.

        Args:
            seq: The sequence.
            upstream: ``detect`` output.
            read: Frame reader of the sequence's video.

        Returns:
            The ``track`` layer output, plus ``objects`` and ``rounds``.

        """
        import torch

        from bga.remote.sam2_prefill import first_persistent_uncovered

        t0 = time.perf_counter()
        dets = upstream["detect"]["frames"]
        f0 = seq.frames[0]
        with tempfile.TemporaryDirectory() as tmp:
            scale = self._write_frames(seq, read, Path(tmp))
            prompts = {
                f - f0: [[v * scale for v in b] for b in prompt_boxes(dets[str(f)], self.p["min_score"])]
                for f in seq.frames
            }
            sampled = sorted(prompts)
            results: dict[int, dict[int, list | None]] = {}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.pred.init_state(video_path=tmp, offload_video_to_cpu=True)
                next_id, rounds, new = 1, 0, (sampled[0], prompts[sampled[0]])
                while new:
                    start, boxes = new
                    for b in boxes:
                        self.pred.add_new_points_or_box(
                            state, frame_idx=start, obj_id=next_id, box=np.array(b, dtype=np.float32)
                        )
                        next_id += 1
                    for fidx, obj_ids, masks in self.pred.propagate_in_video(state, start_frame_idx=start):
                        if fidx in prompts:
                            results[fidx] = _mask_boxes(obj_ids, masks)
                    rounds += 1
                    more = rounds < self.p["max_rounds"]
                    new = first_persistent_uncovered(prompts, results, sampled, start + 1) if more else None
        frames = {
            str(f): mask_rows(
                {o: [v / scale for v in b[:4]] if b else None for o, b in results.get(f - f0, {}).items()},
                dets[str(f)],
                self.p["match_iou"],
            )
            for f in seq.frames
        }
        secs = time.perf_counter() - t0
        return {
            "frames": frames,
            "objects": next_id - 1,
            "rounds": rounds,
            "ms_per_frame": round(1000 * secs / max(1, len(seq.frames)), 1),
        }


def _mask_boxes(obj_ids: list, masks: Any) -> dict[int, list | None]:  # noqa: ANN401  # a torch tensor, imported lazily
    """Bounding box ``[x1, y1, x2, y2, pixels]`` of each object's mask on one frame, or ``None`` when it is empty."""
    import torch

    m = masks[:, 0] > 0
    rows, cols = m.any(dim=2), m.any(dim=1)
    out: dict[int, list | None] = {}
    for i, oid in enumerate(obj_ids):
        if not rows[i].any():
            out[int(oid)] = None
            continue
        ys, xs = torch.nonzero(rows[i])[:, 0], torch.nonzero(cols[i])[:, 0]
        out[int(oid)] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(m[i].sum())]
    return out
