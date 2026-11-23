"""PARSeq number-reader trainer: the released scene-text model fine-tuned on class folders of PLK number crops.

Runs on the training GPU (``bga.remote.modal_train.train_parseq``). The data are the class folders of
``bga.train.number_data``; a crop's text is its number, and the empty string for ``unreadable``, so the model learns
to end at once on a logo or a blurred number. Each crop goes through ``bga.numbers.parseq.to_input`` (stretched to
32 x 128, as at inference), then the crop classifiers' augmentation (``bga.train.cls_trainer.augmentation``).
Batches are drawn with weights 1 / sqrt(class size).

The loss is PARSeq's own permutation language modelling (:func:`plm_loss`, after ``strhub``'s ``training_step``):
the characters are predicted in several orders, each with the attention masks of that order, so the decoder learns
every context it is asked for when it decodes and refines. AdamW without weight decay, one warm-up epoch then a
cosine schedule. After every epoch the model reads ``valid`` as the backend does (``number_probs``, the most probable
class) and is scored by ``bga.train.cls_trainer.score``; the rows go to ``metrics.csv`` and the epoch best on
``select_on`` is saved as ``weights.safetensors`` with its ``parseq_config.json``.

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
from bga.numbers.parseq import HPARAMS, PRETRAINED_URL, build_system, number_probs, to_input
from bga.train.cls_trainer import augmentation, class_files, sample_weights, score

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

#: Base weights a config may name (``base_weights.name``), with where ``torch.hub`` fetches them.
BASES = {"parseq-bb5792a6": PRETRAINED_URL}


def text_of(class_index: int) -> str:
    """The string PARSeq is trained to write for a class of ``NUMBER_CLASSES``: the number, or nothing.

    Examples:
        >>> text_of(NUMBER_CLASSES.index("23")), text_of(NUMBER_CLASSES.index(UNREADABLE))
        ('23', '')

    """
    name = NUMBER_CLASSES[class_index]
    return "" if name == UNREADABLE else name


class TextCropSet:
    """A torch dataset of crops: PARSeq's input as a float CHW tensor in -1 to 1, and the class index."""

    def __init__(self, items: list[tuple[Path, int]], size: tuple[int, int], train: bool) -> None:
        """Keep the items and set up augmentation for training.

        Args:
            items: ``(file, class index)`` pairs.
            size: Input height and width.
            train: Augment the crops.

        """
        self.items, self.size = items, size
        self.aug = augmentation() if train else None

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
        return (x.float() / 255.0 - 0.5) / 0.5, y


def plm_loss(system: torch.nn.Module, images: torch.Tensor, labels: list[str]) -> torch.Tensor:
    """PARSeq's permutation language modelling loss of a batch, as ``strhub``'s ``PARSeq.training_step`` computes it.

    A batch whose labels are all empty has no characters to permute; it is scored in reading order alone.

    Args:
        system: The ``strhub`` PARSeq module, in train mode, on the images' device.
        images: Normalised inputs, shape (crops, 3, height, width).
        labels: The text of each crop; empty for ``unreadable``.

    Returns:
        The mean cross-entropy over the orders and tokens.

    Examples:
        >>> system = build_system(HPARAMS).train()  # doctest: +SKIP
        >>> float(plm_loss(system, torch.zeros(2, 3, 32, 128), ["23", ""])) > 0  # doctest: +SKIP
        True

    """
    import torch
    import torch.nn.functional as F  # noqa: N812  # torch's own name for this module

    tgt = system.tokenizer.encode(labels, images.device)
    memory = system.model.encode(images)
    reading_order = torch.tensor([[0, 1]], device=images.device)  # all labels empty: only [BOS, EOS]
    perms = system.gen_tgt_perms(tgt) if tgt.shape[1] > 2 else reading_order
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    padding = (tgt_in == system.pad_id) | (tgt_in == system.eos_id)
    loss, numel = images.new_zeros(()), 0
    n = int((tgt_out != system.pad_id).sum())
    for i, perm in enumerate(perms):
        tgt_mask, query_mask = system.generate_attn_masks(perm)
        out = system.model.decode(tgt_in, memory, tgt_mask, padding, tgt_query_mask=query_mask)
        logits = system.model.head(out).flatten(end_dim=1)
        loss = loss + n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=system.pad_id)
        numel += n
        if i == 1:  # after the reading order and its reverse, the end token is no longer a target
            tgt_out = torch.where(tgt_out == system.eos_id, system.pad_id, tgt_out)
            n = int((tgt_out != system.pad_id).sum())
    return loss / numel


def predict(system: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Read a loader's crops as the backend does: the most probable class of ``number_probs``.

    Args:
        system: The PARSeq module, in eval mode.
        loader: Validation loader.
        device: Torch device.

    Returns:
        Predicted and true class indices.

    """
    import torch

    preds, ys = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    with torch.no_grad():
        for x, y in loader:
            tokens = system(x.to(device)).softmax(-1).float().cpu().numpy()
            preds.append(number_probs(tokens).argmax(1))
            ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def _train_epoch(system: torch.nn.Module, loader: DataLoader, opt: object, sched: object, device: object) -> float:
    """Run one training epoch; return the mean loss per crop."""
    system.train()
    total, n = 0.0, 0
    for x, y in loader:
        loss = plm_loss(system, x.to(device), [text_of(int(v)) for v in y])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total, n = total + float(loss.detach()) * len(y), n + len(y)
    return total / max(1, n)


def train(ds: Path, out: Path, base: str, recipe: dict, num_workers: int = 8) -> dict:
    """Fine-tune the released PARSeq on the class folders under ``ds``; write the best epoch and the metrics to ``out``.

    Args:
        ds: Dataset with ``train/`` and ``valid/`` class folders (``bga.train.number_data``).
        out: Output directory: ``weights.safetensors``, ``parseq_config.json``, ``metrics.csv``.
        base: The config's ``base_weights.name``, a key of :data:`BASES`.
        recipe: ``bga.train.config.Recipe`` as a dict: ``epochs``, ``batch_size``, ``lr``, ``seed``, ``select_on``.
            ``resolution`` must be the model's input height, 32.
        num_workers: Data-loading processes.

    Returns:
        ``{"crops": {split: n}, "best": the best metrics row, "unseen": validation numbers without training crops}``.

    Raises:
        ValueError: On base weights not in :data:`BASES`, or a resolution other than the model's.

    Examples:
        >>> train(Path("/data/ds"), Path("/data/out"), "parseq-bb5792a6", recipe)["best"]["epoch"]  # doctest: +SKIP
        7

    """
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, WeightedRandomSampler

    hparams = dict(HPARAMS)
    size = tuple(hparams["img_size"])
    if base not in BASES or recipe["resolution"] != size[0]:
        raise ValueError(f"numbers/parseq trains from {sorted(BASES)} at resolution {size[0]}")
    seed = recipe.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = class_files(ds / "train"), class_files(ds / "valid")
    seen = {y for _, y in tr}
    sampler = WeightedRandomSampler(sample_weights([y for _, y in tr]), num_samples=len(tr), replacement=True)
    train_dl = DataLoader(
        TextCropSet(tr, size, True), batch_size=recipe["batch_size"], sampler=sampler, num_workers=num_workers
    )
    val_dl = DataLoader(TextCropSet(va, size, False), batch_size=256, num_workers=num_workers)
    system = build_system(hparams)
    system.model.load_state_dict(
        torch.hub.load_state_dict_from_url(BASES[base], map_location="cpu", check_hash=True, weights_only=True)
    )
    system = system.to(device)
    opt = torch.optim.AdamW(system.model.parameters(), lr=recipe["lr"], weight_decay=0.0)
    steps, warm = recipe["epochs"] * max(1, len(train_dl)), max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    key = recipe.get("select_on") or "val/top1"
    out.mkdir(parents=True, exist_ok=True)
    rows, best = [], None
    for epoch in range(recipe["epochs"]):
        loss = _train_epoch(system, train_dl, opt, sched, device)
        system.eval()
        row = {"epoch": epoch, "train/loss": loss, **score(*predict(system, val_dl, device), seen)}
        rows.append(row)
        with (out / "metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)
        if best is None or row[key] >= best[key]:
            best = row
            state = {k: v.contiguous().cpu() for k, v in system.model.state_dict().items()}
            save_file(state, str(out / "weights.safetensors"))
            cfg = {"hparams": hparams, "base": base, "epoch": epoch}
            (out / "parseq_config.json").write_text(json.dumps(cfg, indent=1))
    unseen = sorted({NUMBER_CLASSES[y] for _, y in va} - {NUMBER_CLASSES[y] for y in seen} - {UNREADABLE}, key=int)
    return {"crops": {"train": len(tr), "valid": len(va)}, "best": best, "unseen": unseen}
