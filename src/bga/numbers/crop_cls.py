"""``numbers/crop_cls``: a crop classifier over ``unreadable`` and the numbers 0 to 99, fine-tuned on PLK crops.

The plan's ``resnet_cls`` (section 5, layer 5) with any ``timm`` backbone: the architecture, input size and
normalisation come from the trained weights' ``cls_config.json`` (``bga train`` for ``numbers/crop_cls``), so the
ResNet-18 and DINOv2 fine-tunes run through this one backend (decision of 2026-11-22). Every read reports a
probability per class, so ``bga.numbers.base.roster_choice`` can restrict it to a roster plus ``unreadable``.

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import cv2
import numpy as np

from bga.device import pick_device
from bga.numbers.base import UNREADABLE, CropRead, TrackNumbers
from bga.paths import weights_dir
from bga.registry import register

if TYPE_CHECKING:
    import torch


def to_input(crop: np.ndarray, size: int) -> np.ndarray:
    """Turn a BGR crop into the classifier's RGB input, stretched to a square.

    Training cuts and stores crops the same way and calls this before its augmentation, so inference sees what
    training saw.

    Args:
        crop: BGR crop, H x W x 3, uint8.
        size: Side of the square input.

    Returns:
        RGB uint8 array, size x size x 3.

    Examples:
        >>> to_input(np.zeros((30, 20, 3), np.uint8), 64).shape
        (64, 64, 3)

    """
    return cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def create_model(arch: str, num_classes: int, size: int, pretrained: bool) -> torch.nn.Module:
    """Build a ``timm`` classifier for square inputs of ``size`` pixels.

    Args:
        arch: The ``timm`` model name, e.g. ``resnet18.tv_in1k`` or ``vit_small_patch14_dinov2.lvd142m``.
        num_classes: Output classes.
        size: Input side; vision transformers get it as ``img_size`` (a multiple of the patch size).
        pretrained: Load the ``timm`` pretrained weights.

    Returns:
        The model.

    Examples:
        >>> m = create_model("resnet18.tv_in1k", 101, 64, pretrained=False)
        >>> m.num_classes
        101

    """
    import timm

    extra = {"img_size": size} if arch.startswith("vit") else {}
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, **extra)


@register("numbers", "crop_cls")
class CropClassifier(TrackNumbers):
    """Numbers backend: a fine-tuned crop classifier, reads confirmed per track as in ``smolvlm_ocr``.

    Examples:
        >>> cls = CropClassifier({}, "numbers/crop_cls/resnet18-r1@00000000")  # doctest: +SKIP
        >>> cls.load()  # doctest: +SKIP
        >>> cls.read_crops([crop], [{"7", "23"}])[0].value  # doctest: +SKIP
        '23'

    """

    defaults: ClassVar[dict[str, Any]] = {**TrackNumbers.defaults, "batch_size": 64}

    def load(self) -> None:
        """Load the weights and their ``cls_config.json`` onto MPS when available, else CPU."""
        import torch
        from safetensors.torch import load_file

        d = weights_dir(self.weights)
        self.cfg = json.loads((d / "cls_config.json").read_text())
        self.classes = list(self.cfg["classes"])
        self.device = torch.device(pick_device("auto"))
        model = create_model(self.cfg["arch"], len(self.classes), self.cfg["size"], pretrained=False)
        model.load_state_dict(load_file(str(d / "weights.safetensors")))
        self.model = model.eval().to(self.device)
        self.mean = torch.tensor(self.cfg["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.cfg["std"], device=self.device).view(1, 3, 1, 1)

    def unload(self) -> None:
        """Drop the model."""
        self.model = None

    def probs(self, crops: list[np.ndarray]) -> np.ndarray:
        """Class probabilities of each crop, in batches of ``batch_size``.

        Args:
            crops: BGR crops cut with this backend's ``pad``.

        Returns:
            Array of shape (crops, classes), rows summing to 1.

        """
        import torch

        size, out = self.cfg["size"], []
        for i in range(0, len(crops), self.p["batch_size"]):
            x = np.stack([to_input(c, size) for c in crops[i : i + self.p["batch_size"]]])
            t = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                out.append(torch.softmax(self.model((t - self.mean) / self.std), dim=1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

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
            best = self.classes[int(row.argmax())]
            probs = {c: float(p) for c, p in zip(self.classes, row, strict=True)}
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
