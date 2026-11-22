"""Fine-tune a ``reid`` encoder on identity folders: P x K batches, identity loss plus batch-hard triplet loss.

The recipe follows the strong re-ID baseline of Luo et al. (CVPR Workshops 2019): the encoder's embedding feeds a
batch-hard triplet loss on cosine distance and, through a batch-norm neck, a bias-free identity classifier trained
with label smoothing 0.1 and weighted by ``id_loss_weight`` (1 by default). Batches hold ``batch_size / crops_per_id``
identities with ``crops_per_id`` crops each, drawn from the whole split or, with ``shot_batches``, from one shot at a
time; an epoch is as many batches as it takes to see as many crops as the training split holds. Crops are stretched to
the encoder's input (its ``reid`` preprocessing), flipped, padded and cropped, and randomly erased.

Weights start from a ``reid`` alias (``base_weights.name``). Every ``eval_interval`` epochs, and before the first,
the encoder is saved in ``reid``'s folder format and scored on the ``valid`` identities with ``RetrievalProtocol``
(two queries per identity); ``metrics.csv`` gets ``val/map``, ``val/rank1`` and ``val/minp`` (mean inverse negative
penalty) in percent, and the best on ``select_on`` is kept as ``weights.safetensors`` with ``reid_config.json``,
which ``ReIDModel.from_pretrained`` loads. Runs on CUDA when available.

"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from reid import ReIDPreprocessing

#: Smoothing of the identity loss (Luo et al.).
LABEL_SMOOTHING = 0.1
#: Weight decay of Adam.
WEIGHT_DECAY = 5e-4
#: Queries per validation identity.
VAL_QUERIES = 2


def identity_folders(root: Path) -> tuple[list[str], list[int]]:
    """List the crops under ``root/<identity>/`` with a contiguous label per identity folder.

    Args:
        root: A split directory.

    Returns:
        Crop paths and their labels, folders in sorted order.

    Examples:
        >>> identity_folders(Path("/no/such/dir"))
        ([], [])

    """
    paths, labels = [], []
    folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for label, folder in enumerate(folders):
        for f in sorted(folder.glob("*.jpg")):
            paths.append(str(f))
            labels.append(label)
    return paths, labels


def pk_batches(labels: list[int], k: int, batch_size: int, rng: np.random.Generator) -> Iterator[list[int]]:
    """Yield one epoch of P x K batches: ``batch_size // k`` identities, ``k`` crops each (with repeats when short).

    Args:
        labels: Label of every crop.
        k: Crops per identity in a batch.
        batch_size: Crops per batch; a multiple of ``k``.
        rng: Random source.

    Yields:
        Crop indices of one batch.

    Examples:
        >>> batches = list(pk_batches([0, 0, 1, 1, 2, 2, 3, 3], 2, 4, np.random.default_rng(0)))
        >>> len(batches), [len(b) for b in batches]
        (2, [4, 4])

    """
    by_label: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    ids, p = sorted(by_label), batch_size // k
    for _ in range(max(1, len(labels) // batch_size)):
        batch = []
        for y in rng.choice(ids, size=min(p, len(ids)), replace=False):
            pool = by_label[int(y)]
            batch += [int(i) for i in rng.choice(pool, size=k, replace=len(pool) < k)]
        yield batch


def shot_of(folder: str) -> str:
    """Shot of an identity folder named ``<sequence>_t<track id>`` (``scripts/reid_crops.py``).

    Args:
        folder: The identity folder's name.

    Returns:
        The sequence name, or the whole name when it has no ``_t`` suffix.

    Examples:
        >>> shot_of("kW4HG-5MLNM_s1_t3"), shot_of("market_0001")
        ('kW4HG-5MLNM_s1', 'market_0001')

    """
    return folder.rsplit("_t", 1)[0] if "_t" in folder else folder


def shot_batches(
    labels: list[int], shots: list[str], k: int, batch_size: int, rng: np.random.Generator
) -> Iterator[list[int]]:
    """Yield one epoch of batches, each from one shot: up to ``batch_size // k`` of its identities, ``k`` crops each.

    Within a shot every identity is another person, while one player's tracklets in different shots are separate
    identities; so a batch never pits a player against himself from another shot. Shots with one identity are skipped.
    The epoch has as many batches as :func:`pk_batches` would.

    Args:
        labels: Label of every crop.
        shots: Shot of every crop.
        k: Crops per identity in a batch.
        batch_size: Most crops per batch; a multiple of ``k``.
        rng: Random source.

    Yields:
        Crop indices of one batch.

    Examples:
        >>> labels, shots = [0, 0, 1, 1, 2, 2], ["a", "a", "a", "a", "b", "b"]
        >>> [sorted({labels[i] for i in b}) for b in shot_batches(labels, shots, 2, 4, np.random.default_rng(0))]
        [[0, 1]]

    """
    by_label: dict[int, list[int]] = {}
    by_shot: dict[str, set[int]] = {}
    for i, (y, s) in enumerate(zip(labels, shots, strict=True)):
        by_label.setdefault(y, []).append(i)
        by_shot.setdefault(s, set()).add(y)
    usable = sorted(s for s, ids in by_shot.items() if len(ids) > 1)
    p = batch_size // k
    for _ in range(max(1, len(labels) // batch_size) if usable else 0):
        ids = sorted(by_shot[usable[int(rng.integers(len(usable)))]])
        batch = []
        for y in rng.choice(ids, size=min(p, len(ids)), replace=False):
            pool = by_label[int(y)]
            batch += [int(i) for i in rng.choice(pool, size=k, replace=len(pool) < k)]
        yield batch


def _epoch_batches(labels: list[int], shots: list[str], recipe: dict, rng: np.random.Generator) -> list[list[int]]:
    """One epoch of batches from the sampler the recipe names: :func:`shot_batches` or :func:`pk_batches`."""
    k, size = recipe["crops_per_id"], recipe["batch_size"]
    if recipe.get("shot_batches"):
        return list(shot_batches(labels, shots, k, size, rng))
    return list(pk_batches(labels, k, size, rng))


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Batch-hard triplet loss on cosine distance: each anchor's farthest positive against its nearest negative.

    Args:
        emb: Embeddings, ``(n, d)``.
        labels: Identity labels, ``(n,)``.
        margin: Margin between the two distances.

    Returns:
        The mean hinge loss over anchors.

    Examples:
        >>> import torch
        >>> e = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        >>> round(float(batch_hard_triplet(e, torch.tensor([0, 0, 1, 1]), 0.3)), 4)
        0.0
        >>> round(float(batch_hard_triplet(e, torch.tensor([0, 1, 0, 1]), 0.3)), 4)
        1.3

    """
    import torch.nn.functional as F  # noqa: N812  # torch's own alias

    e = F.normalize(emb, dim=1)
    d = 1.0 - e @ e.T
    same = labels[:, None] == labels[None, :]
    pos = (d * same).max(dim=1).values
    neg = d.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(pos - neg + margin).mean()


class _Crops:
    """Training crops as a torch dataset: read, BGR to RGB, augmented and normalised to the encoder's input."""

    def __init__(self, paths: list[str], labels: list[int], size: tuple[int, int], mean: tuple, std: tuple) -> None:
        import torchvision.transforms.v2 as T  # noqa: N812  # torchvision's own alias

        self.paths, self.labels = paths, labels
        self.tf = T.Compose([
            T.Resize(size, antialias=True),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop(size),
            T.ToDtype(__import__("torch").float32, scale=True),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.5, value=0),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        import cv2
        import torch

        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        return self.tf(torch.from_numpy(img).permute(2, 0, 1)), self.labels[i]


def _encoder(alias: str, device: torch.device) -> tuple[torch.nn.Module, str, ReIDPreprocessing]:
    """The alias's backbone with its pretrained weights, its architecture name and its preprocessing."""
    from reid.architectures.registry import build_architecture
    from reid.loaders import load_state_dict, resolve_load_plan, resolve_weights

    plan = resolve_load_plan(alias, None, None)
    net = build_architecture(plan.architecture).to(device)
    load_state_dict(
        net,
        resolve_weights(plan.weights),
        device,
        architecture=plan.architecture,
        required_match_fraction=plan.required_match_fraction,
    )
    return net, plan.architecture, plan.preprocessing


def save_encoder(net: torch.nn.Module, architecture: str, preprocessing: ReIDPreprocessing, folder: Path) -> None:
    """Write an encoder in ``reid``'s folder format: ``weights.safetensors`` and ``reid_config.json``.

    Args:
        net: The backbone.
        architecture: Its ``reid`` architecture name.
        preprocessing: Its ``ReIDPreprocessing``.
        folder: Output folder (created).

    Examples:
        >>> save_encoder(net, "osnet_x1_0", prep, Path("out"))  # doctest: +SKIP

    """
    from reid.catalog import ModelCard, save_model_config
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(folder / "weights.safetensors")
    )
    save_model_config(ModelCard(architecture=architecture, weights=None), str(folder), preprocessing=preprocessing)


def validate(folder: Path, valid: Path) -> dict[str, float]:
    """Score a saved encoder on the validation identities with ``RetrievalProtocol``.

    Args:
        folder: The encoder in ``reid``'s folder format.
        valid: The ``valid`` split directory.

    Returns:
        ``val/map``, ``val/rank1`` and ``val/minp`` (mean inverse negative penalty) in percent.

    Examples:
        >>> validate(Path("out/epoch"), Path("dataset/valid"))  # doctest: +SKIP
        {'val/map': 61.2, 'val/rank1': 70.4, 'val/minp': 40.0}

    """
    from reid import ReIDModel, RetrievalProtocol, load_identity_folders

    model = ReIDModel.from_pretrained(str(folder))
    protocol = RetrievalProtocol(queries_per_id=VAL_QUERIES, seed=0)
    m = model.evaluate(load_identity_folders(valid), protocol=protocol, verbose=False).metrics
    return {f"val/{k}": round(float(m[k]), 3) for k in ("map", "rank1", "minp")}


def train(dataset: Path, out: Path, base: str, recipe: dict, num_workers: int = 8) -> dict:
    """Fine-tune the ``base`` encoder on ``dataset/train``, validating on ``dataset/valid``; see the module docstring.

    Args:
        dataset: Directory with ``train`` and ``valid`` identity folders.
        out: Output directory: ``metrics.csv``, the best ``weights.safetensors`` and ``reid_config.json``.
        base: The ``reid`` alias to start from.
        recipe: ``epochs``, ``batch_size``, ``crops_per_id``, ``lr``, ``seed``, ``triplet_margin``,
            ``eval_interval`` and ``select_on``.
        num_workers: Data-loading processes.

    Returns:
        The best metrics row and the number of training identities and crops.

    Examples:
        >>> train(Path("dataset"), Path("output"), "osnet_x1_0_sportsmot", recipe)["best"]  # doctest: +SKIP
        {'epoch': 12, 'val/map': 64.0, ...}

    """
    import torch
    from torch.utils.data import DataLoader

    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, arch, prep = _encoder(base, device)
    paths, labels = identity_folders(dataset / "train")
    shots = [shot_of(Path(q).parent.name) for q in paths]
    id_weight = 1.0 if recipe.get("id_loss_weight") is None else recipe["id_loss_weight"]
    data = _Crops(paths, labels, tuple(prep.input_size), tuple(prep.mean), tuple(prep.std))
    dim = net(torch.zeros(2, 3, *prep.input_size, device=device)).shape[1]
    neck = torch.nn.BatchNorm1d(dim).to(device)
    neck.bias.requires_grad_(False)
    head = torch.nn.Linear(dim, max(labels) + 1, bias=False).to(device)
    params = [*net.parameters(), *neck.parameters(), *head.parameters()]
    opt = torch.optim.Adam(params, lr=recipe["lr"], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe["epochs"])
    ce = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    rng, key, every = np.random.default_rng(recipe["seed"]), recipe["select_on"], recipe.get("eval_interval") or 1
    out.mkdir(parents=True, exist_ok=True)
    rows, best = [], None
    for epoch in range(recipe["epochs"] + 1):
        row: dict[str, Any] = {"epoch": epoch, "lr": opt.param_groups[0]["lr"]}
        if epoch:  # epoch 0 scores the starting encoder
            net.train()
            batches = _epoch_batches(labels, shots, recipe, rng)
            loader = DataLoader(data, batch_sampler=batches, num_workers=num_workers, pin_memory=True)
            sums = np.zeros(2)
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                f = net(x)
                losses = (id_weight * ce(head(neck(f)), y), batch_hard_triplet(f, y, recipe["triplet_margin"]))
                opt.zero_grad()
                sum(losses).backward()
                opt.step()
                sums += [v.item() for v in losses]
            sched.step()
            mean = sums / len(batches)
            row.update({"train/ce": round(float(mean[0]), 4), "train/triplet": round(float(mean[1]), 4)})
        if epoch % every == 0 or epoch == recipe["epochs"]:
            net.eval()
            save_encoder(net, arch, prep, out / "epoch")
            row.update(validate(out / "epoch", dataset / "valid"))
            if best is None or row[key] > best[key]:
                best = row
                for name in ("weights.safetensors", "reid_config.json"):
                    shutil.copy(out / "epoch" / name, out / name)
        rows.append(row)
        _write_csv(out / "metrics.csv", rows)
    return {"best": best, "identities": max(labels) + 1, "crops": len(paths)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write the metrics rows with the union of their keys as columns."""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
