"""``numbers/smolvlm_ocr``: Skalski's number reader (basketball-jersey-numbers-ocr/3, a SmolVLM2
fine-tune on NBA crops) on ``number`` boxes matched to tracks, confirmed by consecutive agreement.

Matching (plan section 3): a number box belongs to the track box that contains it with IoS >= 0.9.
Crops padded 10 px; reads every ``read_every`` frames per track; ``ConsecutiveValueTracker`` with
``n_consecutive`` equal reads confirms. Runs on MPS with ``max_new_tokens=4`` (2 s per crop instead of
8 s, docs/handoff.md). Referee tracks are skipped.
"""

from __future__ import annotations

import re
import time

import cv2

from bga.detect.roboflow_inference import roboflow_key
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def ios(inner, outer) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


@register("numbers", "smolvlm_ocr")
class SmolVLMOcr(BaseBackend):
    defaults = {
        "model_id": "basketball-jersey-numbers-ocr/3",
        "min_score": 0.3,
        "min_ios": 0.9,
        "pad": 10,
        "n_consecutive": 3,
        "read_every": 1,
        "max_reads_per_track": 12,
    }

    def load(self) -> None:
        import os

        import torch
        from inference_models import AutoModel

        roboflow_key()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            self.p["model_id"], api_key=os.environ.get("ROBOFLOW_API_KEY"), device=device
        )

    def unload(self) -> None:
        self.model = None

    def read_crop(self, crop) -> str | None:
        text = self.model.prompt(
            images=cv2.resize(crop, (224, 224)), prompt="Read the number.", max_new_tokens=4
        )[0]
        digits = re.sub(r"\D", "", str(text))
        return digits if 0 < len(digits) <= 2 else None

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        from sports.common.temporal import ConsecutiveValueTracker

        dets, tracks = upstream["detect"]["frames"], upstream["track"]["frames"]
        confirm = ConsecutiveValueTracker(n_consecutive=self.p["n_consecutive"])
        done, reads, t, n = {}, {}, 0.0, 0
        for j, f in enumerate(seq.frames):
            if j % self.p["read_every"]:
                continue
            nums = [d for d in dets[str(f)] if d[5] == "number" and d[4] >= self.p["min_score"]]
            rows = [r for r in tracks[str(f)] if r[6] != "referee"]
            ids, values, img = [], [], None
            for d in nums:
                best = max(rows, key=lambda r: ios(d[:4], r[1:5]), default=None)
                if best is None or ios(d[:4], best[1:5]) < self.p["min_ios"]:
                    continue
                tid = best[0]
                if tid in done or reads.get(tid, 0) >= self.p["max_reads_per_track"]:
                    continue
                img = read(f) if img is None else img
                x1, y1, x2, y2 = (int(v) for v in d[:4])
                pad, (H, W) = self.p["pad"], img.shape[:2]
                crop = img[max(0, y1 - pad) : min(H, y2 + pad), max(0, x1 - pad) : min(W, x2 + pad)]
                if crop.size == 0:
                    continue
                t0 = time.perf_counter()
                value = self.read_crop(crop)
                t += time.perf_counter() - t0
                n += 1
                reads[tid] = reads.get(tid, 0) + 1
                ids.append(tid)
                values.append(value)
            for tid, v in zip(ids, confirm.update(ids, values), strict=True):
                if v is not None:
                    done[tid] = v
        all_ids = {r[0] for f in seq.frames for r in tracks[str(f)]}
        return {
            "tracks": {str(tid): done.get(tid) for tid in sorted(all_ids)},
            "reads": n,
            "ms_per_read": round(1000 * t / max(1, n), 1),
            "ms_per_frame": round(1000 * t / max(1, len(seq.frames)), 1),
        }
