"""``numbers/parseq``: PARSeq, a scene-text recogniser, reading jersey numbers (plan section 5, layer 5).

PARSeq (Bautista and Atienza, ECCV 2022; ``strhub`` from https://github.com/baudm/parseq, Apache 2.0 code) decodes a
string from a 32 x 128 picture, one distribution over its 94 characters and an end token per position. A crop is
stretched to that size; the number read is scored from those distributions (:func:`number_probs`), so every read
reports a probability per class of ``bga.numbers.base.NUMBER_CLASSES`` and ``roster_choice`` can restrict it to a
roster plus ``unreadable``.

Weights are either the released scene-text checkpoint, ``pretrained:github/baudm/parseq/v1.0.0/parseq-bb5792a6``
(downloaded once by ``torch.hub`` and checked against the hash in its name), or a PLK fine-tune addressed by its
weight id, ``weights.safetensors`` next to a ``parseq_config.json`` with the model's hyperparameters.

Needs the ``parseq`` extra (``strhub`` at a pinned commit, ``pytorch-lightning`` and ``nltk``, which ``strhub``
imports without declaring them).

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import cv2
import numpy as np

from bga.device import pick_device
from bga.numbers.base import NUMBER_CLASSES, UNREADABLE, CropRead, TrackNumbers
from bga.paths import weights_dir
from bga.registry import register

if TYPE_CHECKING:
    import torch

#: Weight id of the released scene-text checkpoint, and where ``torch.hub`` fetches it.
PRETRAINED = "pretrained:github/baudm/parseq/v1.0.0/parseq-bb5792a6"
PRETRAINED_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"

#: The 94 characters of the released models, in token order after the end token.
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

#: Hyperparameters of the released ``parseq`` model (``configs/main.yaml`` and ``configs/model/parseq.yaml`` of the
#: repo at commit 1902db04). The training values (``lr`` and after) are unused at inference.
HPARAMS: dict[str, Any] = {
    "charset_train": CHARSET,
    "charset_test": CHARSET,
    "max_label_length": 25,
    "img_size": [32, 128],
    "patch_size": [4, 8],
    "embed_dim": 384,
    "enc_num_heads": 6,
    "enc_mlp_ratio": 4,
    "enc_depth": 12,
    "dec_num_heads": 12,
    "dec_mlp_ratio": 4,
    "dec_depth": 1,
    "perm_num": 6,
    "perm_forward": True,
    "perm_mirrored": True,
    "decode_ar": True,
    "refine_iters": 1,
    "dropout": 0.1,
    "lr": 7e-4,
    "batch_size": 384,
    "warmup_pct": 0.075,
    "weight_decay": 0.0,
}

#: Token id of the end token; digit ``d`` is token ``d + 1``.
EOS = 0


def number_probs(tokens: np.ndarray) -> np.ndarray:
    """A probability per class of ``NUMBER_CLASSES`` from PARSeq's per-position token probabilities.

    A one-digit number is its digit at the first position times the end token at the second; a two-digit number is
    its digits at the first two positions times the end token at the third. ``unreadable`` takes what is left:
    letters, longer strings, a leading zero, the empty string. The positions are scored as PARSeq gives them, each
    conditioned on its own most probable string, so a number that differs from that string is scored approximately.

    Args:
        tokens: Token probabilities, shape (crops, positions, tokens), at least 3 positions; token 0 is the end
            token and tokens 1 to 10 the digits 0 to 9.

    Returns:
        Array of shape (crops, 101), rows summing to 1.

    Examples:
        >>> tokens = np.zeros((1, 3, 95))
        >>> tokens[0, 0, 1 + 2], tokens[0, 1, 1 + 3], tokens[0, 2, EOS] = 0.9, 0.8, 1.0  # "23", then the end
        >>> probs = number_probs(tokens)
        >>> probs.shape, NUMBER_CLASSES[int(probs.argmax())], round(float(probs.max()), 2)
        ((1, 101), '23', 0.72)
        >>> round(float(probs[0, 0]), 2)  # unreadable
        0.28

    """
    first, second = tokens[:, 0, 1:11], tokens[:, 1, 1:11]
    one = first * tokens[:, 1, EOS, None]
    two = first[:, 1:, None] * second[:, None, :] * tokens[:, 2, EOS, None, None]
    numbers = np.concatenate([one, two.reshape(len(tokens), 90)], axis=1)
    unreadable = np.clip(1.0 - numbers.sum(1, keepdims=True), 0.0, 1.0)
    return np.concatenate([unreadable, numbers], axis=1)


def to_input(crop: np.ndarray, size: tuple[int, int] = (32, 128)) -> np.ndarray:
    """Turn a BGR crop into PARSeq's RGB input, stretched to ``size``.

    Args:
        crop: BGR crop, H x W x 3, uint8.
        size: Input height and width.

    Returns:
        RGB uint8 array, height x width x 3.

    Examples:
        >>> to_input(np.zeros((30, 20, 3), np.uint8)).shape
        (32, 128, 3)

    """
    return cv2.cvtColor(cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


def build_system(hparams: dict[str, Any]) -> torch.nn.Module:
    """Build ``strhub``'s PARSeq module from its hyperparameters, with untrained weights.

    Args:
        hparams: As :data:`HPARAMS`.

    Returns:
        The ``strhub.models.parseq.system.PARSeq`` module; its ``model`` attribute holds the network.

    Examples:
        >>> build_system(HPARAMS).model.encoder.embed_dim  # doctest: +SKIP
        384

    """
    from strhub.models.parseq.system import PARSeq

    return PARSeq(**hparams)


@register("numbers", "parseq")
class ParseqReader(TrackNumbers):
    """Numbers backend: PARSeq reads of matched ``number`` crops, confirmed per track as in ``smolvlm_ocr``.

    Examples:
        >>> reader = ParseqReader({}, PRETRAINED)  # doctest: +SKIP
        >>> reader.load()  # doctest: +SKIP
        >>> reader.read_crops([crop], [{"7", "23"}])[0].value  # doctest: +SKIP
        '23'

    """

    defaults: ClassVar[dict[str, Any]] = {**TrackNumbers.defaults, "batch_size": 64}

    def load(self) -> None:
        """Load the released checkpoint or a fine-tune onto MPS when available, else CPU.

        Raises:
            ValueError: When the weights are neither :data:`PRETRAINED` nor a trained weight id.

        """
        import torch

        if self.weights == PRETRAINED:
            hparams = dict(HPARAMS)
            state = torch.hub.load_state_dict_from_url(
                PRETRAINED_URL, map_location="cpu", check_hash=True, weights_only=True
            )
        elif self.weights and not self.weights.startswith("pretrained:"):
            from safetensors.torch import load_file

            d = weights_dir(self.weights)
            hparams = json.loads((d / "parseq_config.json").read_text())["hparams"]
            state = load_file(str(d / "weights.safetensors"))
        else:
            raise ValueError(f"numbers/parseq reads {PRETRAINED} or a trained weight id, not {self.weights!r}")
        system = build_system(hparams)
        system.model.load_state_dict(state)
        self.size = tuple(hparams["img_size"])
        self.device = torch.device(pick_device("auto"))
        self.model = system.eval().to(self.device)

    def unload(self) -> None:
        """Drop the model."""
        self.model = None

    def probs(self, crops: list[np.ndarray]) -> np.ndarray:
        """Class probabilities of each crop, in batches of ``batch_size``.

        Args:
            crops: BGR crops cut with this backend's ``pad``.

        Returns:
            Array of shape (crops, 101), rows summing to 1.

        """
        import torch

        out = []
        for i in range(0, len(crops), self.p["batch_size"]):
            x = np.stack([to_input(c, self.size) for c in crops[i : i + self.p["batch_size"]]])
            t = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                tokens = self.model((t - 0.5) / 0.5).softmax(-1).float().cpu().numpy()
            out.append(number_probs(tokens))
        return np.concatenate(out) if out else np.zeros((0, len(NUMBER_CLASSES)))

    def read_crops(self, crops: list[np.ndarray], rosters: list[set[str]]) -> list[CropRead]:
        """Read crops with a probability per class; the free answer is the most probable class.

        Args:
            crops: BGR crops cut with this backend's ``pad``.
            rosters: The team's roster per crop; unused here (``roster_choice`` applies it).

        Returns:
            One :class:`bga.numbers.base.CropRead` per crop, ``value`` ``None`` when ``unreadable`` wins.

        """
        reads = []
        for row in self.probs(crops):
            best = NUMBER_CLASSES[int(row.argmax())]
            probs = {c: float(p) for c, p in zip(NUMBER_CLASSES, row, strict=True)}
            reads.append(CropRead(None if best == UNREADABLE else best, probs))
        return reads

    def read_crop(self, crop: np.ndarray) -> str | None:
        """Read one crop for the track loop: the most probable class, ``None`` for ``unreadable``.

        Args:
            crop: BGR crop of a ``number`` box.

        Returns:
            The number, or ``None``.

        """
        return self.read_crops([crop], [set()])[0].value
