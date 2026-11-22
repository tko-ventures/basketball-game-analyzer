"""Number-classifier trainer: a ``timm`` backbone fine-tuned on class folders of PLK number crops.

Runs on the training GPU (``bga.remote.modal_train.train_numbers``). Classes are ``bga.numbers.base.NUMBER_CLASSES``;
a class with no crops simply has no folder. Each crop goes through ``bga.numbers.crop_cls.to_input`` (stretched to
the square input, as at inference), then training augmentation: slant, rotation, shift and scale, blur, and colour
and contrast changes (plan section 5, layer 5: augmentation for slant, blur and contrast instead of test-time shear
and CLAHE). No flips: a mirrored digit is another glyph. Batches are drawn with weights 1 / sqrt(class size), so rare
numbers are seen more often without repeating them as often as full balancing would.

Cross-entropy with label smoothing 0.1, AdamW (weight decay 0.05), one warm-up epoch then a cosine schedule, bf16
autocast. After every epoch the model is scored on ``valid``: ``val/top1`` (all crops), ``val/top1_numbers`` (crops
of numbers) and ``val/refused`` (``unreadable`` crops answered ``unreadable``), written to ``metrics.csv``; the epoch
best on ``select_on`` is saved as ``weights.safetensors`` with its ``cls_config.json``.

"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bga.numbers.base import NUMBER_CLASSES, UNREADABLE
from bga.numbers.crop_cls import create_model, to_input

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

#: ImageNet normalisation, which both the ResNet-18 and the DINOv2 ``timm`` weights expect.
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def class_files(split_dir: Path) -> list[tuple[Path, int]]:
    """Every crop of a split with its class index in ``NUMBER_CLASSES``.

    Args:
        split_dir: ``train`` or ``valid`` folder of class folders.

    Returns:
        ``(file, class index)`` pairs, sorted by path; folders not named after a class are skipped.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     (Path(d) / "23").mkdir()
        ...     _ = (Path(d) / "23" / "000001.jpg").write_bytes(b"")
        ...     [(p.name, i) for p, i in class_files(Path(d))]
        [('000001.jpg', 24)]

    """
    index = {c: i for i, c in enumerate(NUMBER_CLASSES)}
    if not split_dir.exists():
        return []
    return sorted((p, index[p.parent.name]) for p in split_dir.glob("*/*.jpg") if p.parent.name in index)


def sample_weights(labels: list[int]) -> list[float]:
    """Per-crop sampling weights, 1 / sqrt(size of the crop's class).

    Examples:
        >>> sample_weights([0, 0, 0, 0, 1])
        [0.5, 0.5, 0.5, 0.5, 1.0]

    """
    sizes: dict[int, int] = {}
    for y in labels:
        sizes[y] = sizes.get(y, 0) + 1
    return [1.0 / math.sqrt(sizes[y]) for y in labels]


def _augment() -> v2.Compose:
    """Training augmentation on uint8 CHW tensors (torchvision v2)."""
    from torchvision.transforms import v2

    return v2.Compose([
        v2.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.85, 1.1), shear=(-15, 15, -5, 5)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.3),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.04),
        v2.RandomGrayscale(p=0.1),
    ])


class CropSet:
    """A torch dataset of crops: the classifier input as a float CHW tensor, normalised, and the class index."""

    def __init__(self, items: list[tuple[Path, int]], size: int, train: bool) -> None:
        """Keep the items and set up augmentation for training.

        Args:
            items: ``(file, class index)`` pairs.
            size: Input side.
            train: Augment the crops.

        """
        import torch

        self.items, self.size = items, size
        self.aug = _augment() if train else None
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        """Number of crops."""
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """The i-th crop as a normalised float tensor, and its class index."""
        import torch

        path, y = self.items[i]
        x = torch.from_numpy(to_input(cv2.imread(str(path)), self.size)).permute(2, 0, 1)
        if self.aug is not None:
            x = self.aug(x)
        return (x.float() / 255.0 - self.mean) / self.std, y


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Top-1 on all crops, on number crops, and the share of ``unreadable`` crops answered ``unreadable``.

    Args:
        model: The classifier, in eval mode.
        loader: Validation loader.
        device: Torch device.

    Returns:
        ``val/top1``, ``val/top1_numbers`` and ``val/refused`` (NaN when a split has no such crops).

    """
    import torch

    right = numbers = numbers_right = refuse = refused = 0
    n = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            right += int((pred == y).sum())
            n += len(y)
            num = y != NUMBER_CLASSES.index(UNREADABLE)
            numbers += int(num.sum())
            numbers_right += int(((pred == y) & num).sum())
            refuse += int((~num).sum())
            refused += int(((pred == y) & ~num).sum())
    nan = float("nan")
    return {
        "val/top1": right / n if n else nan,
        "val/top1_numbers": numbers_right / numbers if numbers else nan,
        "val/refused": refused / refuse if refuse else nan,
    }


def train(ds: Path, out: Path, arch: str, recipe: dict, num_workers: int = 8) -> dict:
    """Fine-tune ``arch`` on the class folders under ``ds``; write the best epoch's weights and the metrics to ``out``.

    Args:
        ds: Dataset with ``train/`` and ``valid/`` class folders (``bga.train.number_data``).
        out: Output directory: ``weights.safetensors``, ``cls_config.json``, ``metrics.csv``.
        arch: The ``timm`` model name (the config's ``base_weights.name``).
        recipe: ``bga.train.config.Recipe`` as a dict: ``resolution``, ``epochs``, ``batch_size``, ``lr``, ``seed``,
            ``select_on``.
        num_workers: Data-loading processes.

    Returns:
        ``{"crops": {split: n}, "best": the best metrics row}``.

    Examples:
        >>> train(Path("/data/ds"), Path("/data/out"), "resnet18.tv_in1k", recipe)["best"]["epoch"]  # doctest: +SKIP
        11

    """
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, WeightedRandomSampler

    seed, size = recipe.get("seed", 0), recipe["resolution"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = class_files(ds / "train"), class_files(ds / "valid")
    sampler = WeightedRandomSampler(sample_weights([y for _, y in tr]), num_samples=len(tr), replacement=True)
    train_dl = DataLoader(
        CropSet(tr, size, True), batch_size=recipe["batch_size"], sampler=sampler, num_workers=num_workers
    )
    val_dl = DataLoader(CropSet(va, size, False), batch_size=256, num_workers=num_workers)
    model = create_model(arch, len(NUMBER_CLASSES), size, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=0.05)
    steps = recipe["epochs"] * max(1, len(train_dl))
    warm = max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    key = recipe.get("select_on") or "val/top1"
    out.mkdir(parents=True, exist_ok=True)
    rows, best = [], None
    for epoch in range(recipe["epochs"]):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + float(loss.detach()) * len(y), n + len(y)
        model.eval()
        row = {"epoch": epoch, "train/loss": total / max(1, n), **evaluate(model, val_dl, device)}
        rows.append(row)
        with (out / "metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)
        if best is None or row[key] >= best[key]:
            best = row
            save_file(
                {k: v.contiguous().cpu() for k, v in model.state_dict().items()}, str(out / "weights.safetensors")
            )
            cfg = {
                "arch": arch,
                "size": size,
                "classes": list(NUMBER_CLASSES),
                "mean": MEAN,
                "std": STD,
                "epoch": epoch,
            }
            (out / "cls_config.json").write_text(json.dumps(cfg, indent=1))
    return {"crops": {"train": len(tr), "valid": len(va)}, "best": best}
