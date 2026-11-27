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


def hsv_feature(frame: np.ndarray) -> np.ndarray:
    """Arena colour signature of a frame: the square root of its hue–saturation histogram, unit length.

    Floor paint, boards and seats differ between halls, so shots of another game's arena land far from this game's.

    Args:
        frame: A BGR image.

    Returns:
        A float32 vector of 18 x 8 bins.

    Examples:
        >>> v = hsv_feature(np.full((54, 96, 3), (40, 120, 200), np.uint8))
        >>> v.shape, round(float(np.linalg.norm(v)), 3)
        ((144,), 1.0)

    """
    hsv = cv2.cvtColor(cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = np.sqrt(cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).ravel())
    return (h / max(float(np.linalg.norm(h)), 1e-9)).astype(np.float32)


def robust_centroid(x: np.ndarray, keep: float = 0.6, iters: int = 5) -> np.ndarray:
    """Unit centroid of the rows most like each other: refit on the ``keep`` share nearest the last centroid.

    Args:
        x: Unit row vectors, at least one.
        keep: Share of rows kept on each refit.
        iters: Number of refits.

    Returns:
        The unit centroid.

    Examples:
        >>> x = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> robust_centroid(x).round(3).tolist()
        [1.0, 0.0]

    """
    cen = x.mean(0)
    for _ in range(iters):
        sim = x @ (cen / max(float(np.linalg.norm(cen)), 1e-9))
        cen = x[sim >= np.quantile(sim, 1.0 - keep)].mean(0)
    return cen / max(float(np.linalg.norm(cen)), 1e-9)


@register("camera", "siglip_game")
class SiglipGame(SiglipZeroShot):
    """Zero-shot SigLIP, then keep only the central shots that look like the rest of this game's central shots.

    A full recording also shows other games' highlights at half-time, shot by the same kind of main camera with a
    scorebug, which a frame-by-frame classifier tags central. Among the shots zero-shot tags central, this backend
    fits a :func:`robust_centroid` of a per-shot feature (the mean over its frames). A shot whose similarity to it is
    below ``min_sim`` scores 0. The feature is ``hsv`` (:func:`hsv_feature`, the arena's colours) or ``siglip``
    (the image embedding).

    Params (``defaults``): those of :class:`SiglipZeroShot`, plus ``feature``, ``min_sim`` and ``keep`` (the share of
    shots each centroid refit keeps).

    """

    defaults: ClassVar[dict[str, Any]] = {
        **SiglipZeroShot.defaults,
        "feature": "hsv",
        "min_sim": 0.95,
        "keep": 0.6,
    }

    def score_shots(self, shots: list[list[np.ndarray]]) -> list[float]:
        """Zero-shot shares, zeroed for central shots unlike the game's own central shots.

        Args:
            shots: Per shot, its sampled BGR frames.

        Returns:
            One score per shot.

        """
        flat = [f for fs in shots for f in fs]
        emb = self.embed(flat)
        n_central = len(self.p["central"])
        sims = emb @ self.text.T
        votes = sims[:, :n_central].max(1) > sims[:, n_central:].max(1)
        if self.p["feature"] == "siglip":
            frame_feat = emb
        else:
            frame_feat = np.stack([hsv_feature(f) for f in flat] or [np.zeros(144, np.float32)])
        scores, feats, i = [], [], 0
        for fs in shots:
            scores.append(float(votes[i : i + len(fs)].mean()) if fs else 0.0)
            v = frame_feat[i : i + len(fs)].mean(0) if fs else np.zeros(frame_feat.shape[1])
            feats.append(v / max(float(np.linalg.norm(v)), 1e-9))
            i += len(fs)
        cand = [k for k, s in enumerate(scores) if s >= 0.5]
        if not cand:
            return scores
        x = np.stack(feats)
        sim = x @ robust_centroid(x[cand], self.p["keep"])
        return [s if sim[k] >= self.p["min_sim"] else 0.0 for k, s in enumerate(scores)]
