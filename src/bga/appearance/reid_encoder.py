"""``appearance/reid``: a Roboflow ``reid`` 0.1.0 encoder on the detector's person boxes (plan section 5, layer 3b).

``ReIDModel.extract_embeddings`` crops each box from the BGR frame and returns one raw embedding per crop; this backend
L2-normalises them and stores them as float16. The encoder is a ``reid`` alias, ``osnet_x1_0_sportsmot`` by default
(OSNet x1.0 fine-tuned on SportsMOT; weights treated as Apache 2.0, plan question 15), or a PLK fine-tune: a weight id
``appearance/reid/<name>@<hash8>`` trained by ``bga train`` from the alias ``model`` names, loaded from its folder.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from bga.appearance.base import l2_normalise
from bga.detect.base import PERSON_CLASSES
from bga.device import pick_device
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

if TYPE_CHECKING:
    from reid import ReIDModel


def check_weights(model: str, weights: str | None) -> None:
    """Check that a config's weights id declares the encoder its ``model`` param loads.

    Args:
        model: The ``reid`` alias the backend loads.
        weights: The config entry's weights id.

    Raises:
        ValueError: When ``weights`` is missing, is a ``pretrained:`` id naming another encoder, or is a trained id
            of another layer and backend.

    Examples:
        >>> check_weights("osnet_x1_0_sportsmot", "pretrained:roboflow/reid/osnet_x1_0_sportsmot")
        >>> check_weights("osnet_x1_0_sportsmot", "appearance/reid/plk-r1@00000000")
        >>> check_weights("osnet_x1_0_sportsmot", "pretrained:roboflow/reid/osnet_x1_0_msmt17")
        Traceback (most recent call last):
        ...
        ValueError: appearance/reid loads osnet_x1_0_sportsmot but its weights declare ...

    """
    if not weights:
        raise ValueError("appearance/reid needs weights: pretrained:roboflow/reid/<model>")
    if not weights.startswith("pretrained:"):
        if not weights.startswith("appearance/reid/"):
            raise ValueError(f"appearance/reid cannot load {weights}: not an appearance/reid weight id")
        return
    if weights.rsplit("/", 1)[-1] != model:
        raise ValueError(f"appearance/reid loads {model} but its weights declare {weights}")


@register("appearance", "reid")
class ReIDEncoder(BaseBackend):
    """Appearance backend over a ``reid`` encoder: one unit-length embedding per person box.

    Params (``defaults``): ``model`` (a ``reid`` alias), ``min_score`` (person boxes below are not embedded; the
    trackers' input threshold), ``batch_size`` (crops per forward pass) and ``device`` (``auto``, ``cuda``, ``mps``
    or ``cpu``, through :func:`bga.device.pick_device`).

    Examples:
        >>> enc = ReIDEncoder({}, "pretrained:roboflow/reid/osnet_x1_0_sportsmot")
        >>> enc.load()  # doctest: +SKIP
        >>> enc.embed(frame, np.array([[10, 20, 60, 180]])).shape  # doctest: +SKIP
        (1, 512)

    """

    defaults: ClassVar[dict[str, Any]] = {
        "model": "osnet_x1_0_sportsmot",
        "min_score": 0.1,
        "batch_size": 32,
        "device": "auto",
    }

    def __init__(self, params: dict[str, Any], weights: str | None = None) -> None:
        """Keep the params and check the weights id against the encoder.

        Args:
            params: The config entry's params.
            weights: ``pretrained:roboflow/reid/<model>``.

        """
        super().__init__(params, weights)
        check_weights(self.p["model"], weights)
        self.model: ReIDModel | None = None

    def load(self) -> None:
        """Load the encoder onto the picked device.

        An alias downloads once into ``reid``'s cache; a trained weight id loads from its folder.

        """
        from reid import ReIDModel

        from bga.paths import weights_dir

        trained = not self.weights.startswith("pretrained:")
        source = str(weights_dir(self.weights)) if trained else self.p["model"]
        self.model = ReIDModel.from_pretrained(source, device=pick_device(self.p["device"]))

    def unload(self) -> None:
        """Drop the encoder."""
        self.model = None

    def embed(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """Embed the crops of ``xyxy`` from one frame.

        Args:
            image: BGR frame.
            xyxy: Boxes as an ``(n, 4)`` array in pixels.

        Returns:
            An ``(n, d)`` float32 array of unit-length rows; ``(0, 0)`` when there are no boxes.

        """
        import supervision as sv

        if not len(xyxy):
            return np.zeros((0, 0), np.float32)
        det = sv.Detections(xyxy=np.asarray(xyxy, dtype=np.float32))
        return l2_normalise(self.model.extract_embeddings(det, image, batch_size=self.p["batch_size"]))

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Embed every person box of the sequence scoring at least ``min_score``.

        Args:
            seq: Sequence to embed.
            upstream: ``detect`` output of the sequence.
            read: Frame reader.

        Returns:
            The appearance layer output (``bga.appearance.base``) with the embeddings under ``arrays``.

        """
        dets = upstream["detect"]["frames"]
        boxes, parts, secs = [], [], 0.0
        for f in seq.frames:
            rows = [
                (i, d) for i, d in enumerate(dets[str(f)]) if d[5] in PERSON_CLASSES and d[4] >= self.p["min_score"]
            ]
            if not rows:
                continue
            image = read(f)
            t0 = time.perf_counter()
            parts.append(self.embed(image, np.array([d[:4] for _, d in rows])))
            secs += time.perf_counter() - t0
            boxes += [[f, i] for i, _ in rows]
        emb = np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
        return {
            "boxes": boxes,
            "dim": int(emb.shape[1]),
            "model": self.p["model"] if self.weights.startswith("pretrained:") else self.weights,
            "ms_per_crop": round(1000 * secs / max(1, len(boxes)), 2),
            "arrays": {"embeddings": emb.astype(np.float16)},
        }
