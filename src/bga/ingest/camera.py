"""Camera classifiers (Layer 0): which shots of a game show the main camera on live play of that game (M10, #32).

A camera backend scores every shot of one game at once, from a few frames sampled inside each shot, so a backend may
compare a shot with the rest of its game. It returns one score per shot, the probability-like share that the shot is
``central``. Backends register under the pseudo-layer ``camera`` and are named in the ``camera`` param of the
``shots/video`` backend (:mod:`bga.ingest.video`), never selected by a branch in the pipeline.

``central`` is the label-file tag (``docs/gold.md``, "Full-game shots"): the main camera with the scorebug during this
game. Breaks without the scorebug, replays, close-ups and the half-time highlights of other games are ``other``.

"""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np

from bga.device import pick_device
from bga.registry import register
from bga.stage import BaseBackend, FrameReader

#: Label-file camera tags that are scored; shorter shots stay ``unset``.
TAGS = ("central", "other")

#: Size sampled frames are kept at: SigLIP sees 224 x 224, and a full game samples thousands of frames.
SAMPLE_SIZE = (398, 224)


def sample_positions(start: int, end: int, samples: int) -> list[int]:
    """Frames evenly inside a shot, away from its cuts: the middle for one sample, then quarters, and so on.

    Args:
        start: The shot's first frame.
        end: The shot's last frame, included.
        samples: How many frames to take.

    Returns:
        ``samples`` frame indices, ascending, at ``(k + 1) / (samples + 1)`` of the shot.

    Examples:
        >>> sample_positions(0, 100, 1)
        [50]
        >>> sample_positions(0, 100, 3)
        [25, 50, 75]

    """
    return [start + round((end - start) * (k + 1) / (samples + 1)) for k in range(samples)]


def read_shot_frames(video: str, shots: list[dict], samples: int) -> list[list[np.ndarray]]:
    """Read :func:`sample_positions` frames of every shot, shrunk to ``SAMPLE_SIZE``, in one forward pass.

    Args:
        video: Path of the video.
        shots: Shots ``{"start", "end"}``.
        samples: Frames per shot.

    Returns:
        Per shot, its frames that could be decoded, as BGR images.

    Examples:
        >>> read_shot_frames("data/raw/missing.mp4", [{"start": 0, "end": 10}], 1)
        [[]]

    """
    wanted = sorted({f for s in shots for f in sample_positions(s["start"], s["end"], samples)})
    read, got = FrameReader(video), {}
    try:
        for f in wanted:
            try:
                got[f] = cv2.resize(read(f), SAMPLE_SIZE, interpolation=cv2.INTER_AREA)
            except (IndexError, cv2.error):
                continue
    finally:
        read.close()
    return [[got[f] for f in sample_positions(s["start"], s["end"], samples) if f in got] for s in shots]


class CameraBackend(BaseBackend):
    """Base of the camera backends: score each shot of a game from its sampled frames."""

    layer: ClassVar[str] = "camera"

    def score_shots(self, shots: list[list[np.ndarray]]) -> list[float]:
        """Score every shot of one game.

        Args:
            shots: Per shot, its sampled BGR frames (:func:`sample_positions`).

        Returns:
            One score in ``[0, 1]`` per shot; a shot is ``central`` at 0.5 or more.

        """
        raise NotImplementedError


@register("camera", "siglip_zeroshot")
class SiglipZeroShot(CameraBackend):
    """Zero-shot SigLIP: a frame votes central when its nearest prompt is a ``central`` one.

    Needs no training. The shot's score is the share of its frames voting central. It sees one frame at a time, so it
    cannot tell this game from another game's highlights shot by the same kind of camera.

    Params (``defaults``): ``model`` (a Hugging Face SigLIP id), ``central`` and ``other`` (prompt lists), ``batch``
    and ``device``.

    """

    defaults: ClassVar[dict[str, Any]] = {
        "model": "google/siglip-base-patch16-224",
        "central": [
            (
                "a wide broadcast view of a basketball game from the main camera high above the court, showing the"
                " whole court and many players"
            )
        ],
        "other": [
            "a close-up of a basketball player",
            "a slow motion replay",
            "a crowd of fans in the stands",
            "a television graphic or logo",
            "a coach on the sidelines",
            "a view of the basketball court from behind the basket",
            "an empty arena during a break",
            "a studio presenter",
        ],
        "batch": 32,
        "device": "auto",
    }

    def load(self) -> None:
        """Load SigLIP and embed the prompts once."""
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = pick_device(self.p["device"])
        self.model = AutoModel.from_pretrained(self.p["model"]).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.p["model"])
        prompts = list(self.p["central"]) + list(self.p["other"])
        with torch.no_grad():
            tok = self.processor(text=prompts, padding="max_length", return_tensors="pt").to(self.device)
            out = self.model.get_text_features(**tok)
        out = getattr(out, "pooler_output", out)
        self.text = torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def unload(self) -> None:
        """Free the model."""
        self.model = self.processor = self.text = None

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        """L2-normalised SigLIP image embeddings of BGR frames, one row per frame.

        Args:
            frames: BGR images.

        Returns:
            A float32 array ``(len(frames), dim)``.

        """
        import torch

        rows = []
        for i in range(0, len(frames), self.p["batch"]):
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[i : i + self.p["batch"]]]
            with torch.no_grad():
                x = self.processor(images=rgb, return_tensors="pt").to(self.device)
                out = self.model.get_image_features(**x)
            out = getattr(out, "pooler_output", out)
            rows.append(torch.nn.functional.normalize(out, dim=-1).float().cpu().numpy())
        return np.concatenate(rows) if rows else np.zeros((0, self.text.shape[1]), np.float32)

    def score_shots(self, shots: list[list[np.ndarray]]) -> list[float]:
        """Share of each shot's frames whose nearest prompt is a ``central`` prompt.

        Args:
            shots: Per shot, its sampled BGR frames.

        Returns:
            One share per shot.

        """
        flat = [f for fs in shots for f in fs]
        sims = self.embed(flat) @ self.text.T
        n_central = len(self.p["central"])
        votes = sims[:, :n_central].max(1) > sims[:, n_central:].max(1)
        out, i = [], 0
        for fs in shots:
            out.append(float(votes[i : i + len(fs)].mean()) if fs else 0.0)
            i += len(fs)
        return out
