"""Shot-cut detectors (Layer 0): where a video's camera shots start (M10, #32).

A cuts backend reads a whole video and returns its shots as ``{"start", "end"}`` with inclusive frame bounds, the
format of a label file's ``shots``. Backends register under the pseudo-layer ``cuts`` and are named in the ``cuts``
param of the ``shots/video`` backend (:mod:`bga.ingest.video`).

"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from bga.ingest.hist_diff import DEFAULT_MIN_LEN_S, DEFAULT_THRESHOLD, detect_shots
from bga.paths import weights_dir
from bga.registry import register
from bga.stage import BaseBackend

if TYPE_CHECKING:
    import torch

    from bga.ingest._transnetv2.model import Conv3DConfigurable, TransNetV2

#: The converted official TransNet V2 weights (``scripts/convert_transnetv2.py``): upstream commit and file hash.
TRANSNETV2 = "pretrained:github/soCzech/TransNetV2@85cef72af9a9/transnetv2-c639fb6d"
#: TransNet V2's input frame size, height and width.
TRANSNET_HW = (27, 48)


class CutsBackend(BaseBackend):
    """Base of the cut detectors: split one video into shots."""

    layer: ClassVar[str] = "cuts"

    def shots(self, video: str | Path, progress: Callable[[float], None] | None = None) -> list[dict]:
        """Split ``video`` into shots.

        Args:
            video: Path of the video.
            progress: Called with the share of the video done, if given.

        Returns:
            Shots ``{"start", "end"}`` covering every frame, in order, with inclusive bounds.

        """
        raise NotImplementedError


@register("cuts", "hist_diff")
class HistDiffCuts(CutsBackend):
    """The prior labeller's detector (:mod:`bga.ingest.hist_diff`), which cut every label file so far.

    Params (``defaults``): ``threshold`` (HSV histogram distance scale) and ``min_len_s`` (shortest shot).

    Examples:
        >>> HistDiffCuts({}).p
        {'threshold': 0.25, 'min_len_s': 0.4}

    """

    defaults: ClassVar[dict[str, Any]] = {"threshold": DEFAULT_THRESHOLD, "min_len_s": DEFAULT_MIN_LEN_S}

    def shots(self, video: str | Path, progress: Callable[[float], None] | None = None) -> list[dict]:
        """Decode every frame and cut where :class:`bga.ingest.hist_diff.ShotCutDetector` does.

        Args:
            video: Path of the video.
            progress: Called with the share of the video decoded.

        Returns:
            Shots ``{"start", "end"}``.

        """
        found = detect_shots(video, self.p["threshold"], self.p["min_len_s"], progress)
        return [{"start": s["start"], "end": s["end"]} for s in found]


def transitions_to_shots(transition: np.ndarray, min_len: int = 1) -> list[dict]:
    """Shots from per-frame transition flags: a shot starts on the first frame after each run of transition frames.

    A run's frames stay with the shot before it, as upstream ``predictions_to_scenes`` ends a scene on the first
    transition frame. A cut fewer than ``min_len`` frames after the previous one is ignored.

    Args:
        transition: Per frame, whether it is part of a transition.
        min_len: Shortest shot in frames.

    Returns:
        Shots ``{"start", "end"}`` covering every frame.

    Examples:
        >>> flags = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0], bool)
        >>> [(s["start"], s["end"]) for s in transitions_to_shots(flags)]
        [(0, 3), (4, 7), (8, 9)]
        >>> [(s["start"], s["end"]) for s in transitions_to_shots(flags, min_len=5)]
        [(0, 7), (8, 9)]

    """
    n = len(transition)
    cuts = [0]
    for i in range(1, n):
        if transition[i - 1] and not transition[i] and i - cuts[-1] >= min_len:
            cuts.append(i)
    ends = [c - 1 for c in cuts[1:]] + [n - 1]
    return [{"start": s, "end": e} for s, e in zip(cuts, ends, strict=True)] if n else []


def windows(frames: Iterator[np.ndarray]) -> Iterator[tuple[np.ndarray, int]]:
    """Upstream TransNet V2 windows over a frame stream: 100 frames, step 50, the first and last padded.

    The stream is padded with 25 copies of its first frame, and at the end with copies of its last frame up to a
    multiple of 50, as upstream ``predict_frames`` does. Each window's middle 50 predictions belong to the frames
    it is centred on.

    Args:
        frames: Frames ``(27, 48, 3)`` RGB uint8, in order.

    Yields:
        ``(window, n_real)``: a ``(100, 27, 48, 3)`` array, and how many of its middle 50 frames are real.

    Examples:
        >>> stream = (np.full((27, 48, 3), i, np.uint8) for i in range(120))
        >>> [(int(w[25, 0, 0, 0]), n) for w, n in windows(stream)]
        [(0, 50), (50, 50), (100, 20)]

    """
    buf: list[np.ndarray] = []
    n = emitted = 0
    for f in frames:
        if not buf:
            buf = [f] * 25
        buf.append(f)
        n += 1
        if len(buf) == 100:
            yield np.stack(buf), 50
            emitted += 50
            buf = buf[50:]
    while emitted < n:  # the last windows, padded with copies of the last frame
        buf += [buf[-1]] * (100 - len(buf))
        yield np.stack(buf), min(50, n - emitted)
        emitted += 50
        buf = buf[50:]


def ffmpeg_frames(video: str | Path, hw: tuple[int, int] = TRANSNET_HW) -> Iterator[np.ndarray]:
    """Decode every frame of a video to RGB at ``hw`` through ffmpeg, without dropping or repeating frames.

    Args:
        video: Path of the video.
        hw: Output height and width.

    Yields:
        ``(h, w, 3)`` uint8 RGB frames, in order.

    Raises:
        RuntimeError: When ffmpeg fails.

    Examples:
        >>> list(ffmpeg_frames("data/raw/missing.mp4"))
        Traceback (most recent call last):
        ...
        RuntimeError: ffmpeg could not decode data/raw/missing.mp4: ...

    """
    h, w = hw
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough", "-vf", f"scale={w}:{h}"]
    cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:"]
    size = h * w * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:  # noqa: S603  # fixed argv, no shell
        while chunk := proc.stdout.read(size):
            if len(chunk) == size:
                yield np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
        proc.stdout.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg could not decode {video}: {err.decode(errors='replace').strip()[-300:]}")


def _separable_conv_2d(self: Conv3DConfigurable, x: torch.Tensor) -> torch.Tensor:
    """``Conv3DConfigurable.forward`` as two 2D convolutions: (1, 3, 3) over each frame, then (3, 1) dilated in time."""
    import torch.nn.functional as fn

    spatial, temporal = self.layers
    b, c, t, h, w = x.shape
    y = fn.conv2d(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), spatial.weight[:, :, 0], None, padding=(1, 1))
    y = y.reshape(b, t, -1, h, w).permute(0, 2, 1, 3, 4).reshape(b, -1, t, h * w)
    d = temporal.dilation[0]
    y = fn.conv2d(y, temporal.weight[..., 0], temporal.bias, padding=(d, 0), dilation=(d, 1))
    return y.reshape(b, -1, t, h, w)


def use_2d_convs(model: TransNetV2) -> TransNetV2:
    """Run every separable 3D convolution of a TransNet V2 model as 2D convolutions with the same weights.

    The result is the same to float precision, and 13x faster on Apple MPS, whose 3D convolutions are slow (M1 Max,
    2026-11-27: 97 against 1,314 frames a second; the CPU runs both at about 210).

    Args:
        model: A :class:`bga.ingest._transnetv2.TransNetV2`.

    Returns:
        The same model, changed in place.

    Examples:
        >>> from bga.ingest._transnetv2 import TransNetV2
        >>> use_2d_convs(TransNetV2()).SDDCNN[0].DDCNN[0].Conv3D_1.forward.__name__
        '_separable_conv_2d'

    """
    from bga.ingest._transnetv2.model import Conv3DConfigurable

    for m in model.modules():
        if isinstance(m, Conv3DConfigurable) and len(m.layers) == 2:
            m.forward = _separable_conv_2d.__get__(m)
    return model


@register("cuts", "transnetv2")
class TransNetV2Cuts(CutsBackend):
    """TransNet V2 (Souček and Lokoč, 2020): a 3D CNN that flags transition frames, official weights, no fine-tune.

    Frames are decoded at 48 x 27 by ffmpeg and scored in upstream's windows (:func:`windows`); a frame whose
    single-frame transition probability exceeds ``threshold`` is a transition (:func:`transitions_to_shots`). The
    3D convolutions run as 2D ones (:func:`use_2d_convs`).

    Params (``defaults``): ``threshold``, ``min_len_s`` (shortest shot), ``batch`` (windows per forward pass) and
    ``device``.

    Examples:
        >>> TransNetV2Cuts({}).p["threshold"]
        0.5

    """

    defaults: ClassVar[dict[str, Any]] = {"threshold": 0.5, "min_len_s": 0.0, "batch": 8, "device": "auto"}

    def load(self) -> None:
        """Load the converted official weights, checking their SHA-256 against :data:`TRANSNETV2`.

        Raises:
            FileNotFoundError: When the weights have not been converted (``scripts/convert_transnetv2.py``).
            ValueError: When the file's hash differs from the one in :data:`TRANSNETV2`.

        """
        import torch

        from bga.device import pick_device
        from bga.ingest._transnetv2 import TransNetV2

        name = TRANSNETV2.rsplit("/", 1)[1]
        path = weights_dir("pretrained/transnetv2") / f"{name}.pth"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: convert the official weights with scripts/convert_transnetv2.py")
        if not hashlib.sha256(path.read_bytes()).hexdigest().startswith(name.rsplit("-", 1)[1]):
            raise ValueError(f"{path} does not match its hash")
        self.device = pick_device(self.p["device"])
        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        use_2d_convs(self.model).eval().to(self.device)

    def unload(self) -> None:
        """Free the model."""
        self.model = None

    def _score(self, batch: list[np.ndarray]) -> np.ndarray:
        """Single-frame transition probabilities of the middle 50 frames of each window, ``(len(batch), 50)``."""
        import torch

        with torch.no_grad():
            single, _ = self.model(torch.from_numpy(np.stack(batch)).to(self.device))
        return torch.sigmoid(single).float().cpu().numpy()[:, 25:75, 0]

    def predict(self, video: str | Path, progress: Callable[[float], None] | None = None) -> np.ndarray:
        """Per-frame transition probability of a whole video.

        Args:
            video: Path of the video.
            progress: Called with the number of frames scored so far (not a share: the length is not read first).

        Returns:
            A float32 array, one probability per decoded frame.

        """
        out: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        keep: list[int] = []

        def flush() -> None:
            for probs, n in zip(self._score(batch), keep, strict=True):
                out.append(probs[:n])
            batch.clear()
            keep.clear()
            if progress:
                progress(float(sum(len(o) for o in out)))

        for window, n in windows(ffmpeg_frames(video)):
            batch.append(window)
            keep.append(n)
            if len(batch) == self.p["batch"]:
                flush()
        if batch:
            flush()
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def shots(self, video: str | Path, progress: Callable[[float], None] | None = None) -> list[dict]:
        """Score every frame and cut after each run of transition frames.

        Args:
            video: Path of the video.
            progress: Passed to :meth:`predict`.

        Returns:
            Shots ``{"start", "end"}``.

        """
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        self.load()
        try:
            probs = self.predict(video, progress)
        finally:
            self.unload()
        return transitions_to_shots(probs > self.p["threshold"], max(1, round(self.p["min_len_s"] * fps)))
