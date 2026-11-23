"""Platform-neutral training entrypoints shared by Modal and Colab runners.

Every trainer takes a dataset directory, an output directory, a job dict (``recipe``, ``base``, ``gpu``, ``commit``,
and optional ``init`` / ``resume`` paths relative to a base root or absolute), and a worker count, and returns a
result dict with timing, hardware, schema, package pin, torch version, and output file names. Platform runners add
their own billing fields (``usd`` on Modal, ``ccu`` on Colab), write ``result.json``, and sync outputs back.

"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

#: The RF-DETR trainer package pinned in every RF-DETR run.
RFDETR_PACKAGE = "rfdetr[train]==1.12.1"
#: RF-DETR XL and 2XL (Platform Model Licence 1.0, treated as Apache 2.0 by Tomasz's ruling of 2026-11-18, question 5).
RFDETR_PLUS_PACKAGE = "rfdetr_plus==1.3.0"
#: The re-ID package pinned in every appearance run.
REID_PACKAGE = "reid==0.1.0"
#: ``strhub`` (PARSeq) pinned to a commit; must come from git so ``strhub.models`` is included.
PARSEQ_PACKAGE = "strhub @ git+https://github.com/baudm/parseq@1902db043c029a7e03a3818c616c06600af574be"


def _gpu_name() -> str | None:
    """CUDA device 0's name when CUDA is available, else ``None``."""
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _base_result(
    job: dict[str, Any],
    out: Path,
    seconds: float,
    package: str,
    schema: dict[str, Any],
    *,
    gpu_label: str | None = None,
    best: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the platform-neutral trainer result dictionary."""
    import torch

    result: dict[str, Any] = {
        "gpu": gpu_label or job["gpu"],
        "commit": job.get("commit"),
        "gpu_name": _gpu_name(),
        "seconds": round(seconds, 1),
        "schema": schema,
        "package": package,
        "torch": torch.__version__,
        "files": sorted(p.name for p in out.iterdir()),
    }
    if best is not None:
        result["best"] = best
    return result


def run_keypoint(ds: Path, out: Path, job: dict[str, Any], num_workers: int = 4) -> dict[str, Any]:
    """Fine-tune RF-DETR Keypoint on a COCO keypoint dataset.

    Args:
        ds: Dataset directory with ``train/_annotations.coco.json``.
        out: Output directory where checkpoints and ``metrics.csv`` are written.
        job: Job parameters with ``recipe``, ``gpu`` and optional ``commit``.
        num_workers: Unused by ``KeypointTrainConfig``; accepted for a uniform trainer signature.

    Returns:
        Platform-neutral result dictionary (without ``usd`` / ``ccu``).

    Examples:
        >>> run_keypoint(Path("/data/ds"), Path("/data/out"), job)["package"]  # doctest: +SKIP
        'rfdetr[train]==1.12.1'

    """
    from rfdetr import RFDETRKeypointPreview
    from rfdetr.config import KeypointTrainConfig
    from rfdetr.datasets import infer_coco_keypoint_schema
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    _ = num_workers
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    r = job["recipe"]
    schema = infer_coco_keypoint_schema(ds / "train" / "_annotations.coco.json")
    variant = RFDETRKeypointPreview(
        num_classes=len(schema.class_names),
        num_keypoints_per_class=schema.num_keypoints_per_class,
        resolution=r["resolution"],
    )
    train_config = KeypointTrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(ds),
        output_dir=str(out),
        epochs=r["epochs"],
        batch_size=r["batch_size"],
        grad_accum_steps=r["grad_accum_steps"],
        lr=r["lr"],
        class_names=schema.class_names,
        keypoint_oks_sigmas=schema.keypoint_oks_sigmas,
        seed=r["seed"],
    )
    datamodule = RFDETRDataModule(variant.model_config, train_config)
    model_module = RFDETRModelModule(variant.model_config, train_config)
    trainer = build_trainer(train_config, variant.model_config)
    trainer.fit(model_module, datamodule=datamodule)
    return _base_result(
        job,
        out,
        time.time() - t0,
        RFDETR_PACKAGE,
        {"class_names": schema.class_names, "num_keypoints_per_class": schema.num_keypoints_per_class},
    )


def run_detect(
    ds: Path,
    out: Path,
    job: dict[str, Any],
    num_workers: int = 8,
    *,
    init_path: Path | None = None,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune an RF-DETR detector on a COCO dataset.

    Args:
        ds: Dataset directory with ``train/_annotations.coco.json``.
        out: Output directory for checkpoints and ``metrics.csv``.
        job: Job parameters with ``recipe`` (including ``variant``), ``gpu`` and optional ``commit``.
        num_workers: Data-loader workers.
        init_path: Optional local path to warm-start checkpoint weights.
        resume_path: Optional local path to ``last.ckpt`` to resume from.

    Returns:
        Platform-neutral result dictionary (without ``usd`` / ``ccu``).

    Examples:
        >>> run_detect(Path("/data/ds"), Path("/data/out"), job)["schema"]["class_names"][1]  # doctest: +SKIP
        'ball'

    """
    import rfdetr

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    r = job["recipe"]
    devices = r.get("devices") or 1
    init = {"pretrain_weights": str(init_path)} if init_path is not None else {}
    model = getattr(rfdetr, r["variant"])(**init)
    extra: dict[str, Any] = {}
    if devices > 1:
        extra.update(devices=devices, strategy=r.get("strategy") or "ddp_spawn")
    if r.get("eval_interval"):
        extra["eval_interval"] = r["eval_interval"]
    if r.get("early_stopping_patience"):
        extra.update(
            early_stopping=True, early_stopping_patience=r["early_stopping_patience"], early_stopping_use_ema=True
        )
    model.train(
        dataset_dir=str(ds),
        output_dir=str(out),
        resolution=r["resolution"],
        epochs=r["epochs"],
        batch_size=r["batch_size"],
        grad_accum_steps=r["grad_accum_steps"],
        lr=r["lr"],
        seed=r["seed"],
        num_workers=job.get("num_workers", num_workers),
        resume=str(resume_path) if resume_path is not None else None,
        **extra,
    )
    cats = json.loads((ds / "train" / "_annotations.coco.json").read_text())["categories"]
    gpu = job["gpu"]
    return _base_result(
        job,
        out,
        time.time() - t0,
        RFDETR_PACKAGE,
        {"class_names": [c["name"] for c in sorted(cats, key=lambda c: c["id"])]},
        gpu_label=f"{gpu}:{devices}" if devices > 1 else gpu,
    )


def run_appearance(ds: Path, out: Path, job: dict[str, Any], num_workers: int = 8) -> dict[str, Any]:
    """Fine-tune a ``reid`` encoder on identity folders.

    Args:
        ds: Dataset directory with ``train/`` and ``valid/`` identity folders.
        out: Output directory for ``weights.safetensors``, ``reid_config.json`` and ``metrics.csv``.
        job: Job parameters with ``base``, ``recipe``, ``gpu`` and optional ``commit``.
        num_workers: Data-loader workers.

    Returns:
        Platform-neutral result dictionary (without ``usd`` / ``ccu``).

    Examples:
        >>> run_appearance(Path("/data/ds"), Path("/data/out"), job)["best"]["val/map"]  # doctest: +SKIP
        64.0

    """
    from bga.train.reid_trainer import train

    t0 = time.time()
    got = train(ds, out, job["base"], job["recipe"], num_workers=num_workers)
    return _base_result(
        job,
        out,
        time.time() - t0,
        REID_PACKAGE,
        {"identities": got["identities"], "crops": got["crops"]},
        best=got["best"],
    )


def run_numbers(ds: Path, out: Path, job: dict[str, Any], num_workers: int = 8) -> dict[str, Any]:
    """Fine-tune a ``timm`` number classifier on class folders.

    Args:
        ds: Dataset directory with ``train/`` and ``valid/`` class folders.
        out: Output directory for ``weights.safetensors``, ``cls_config.json`` and ``metrics.csv``.
        job: Job parameters with ``base``, ``recipe``, ``gpu`` and optional ``commit``.
        num_workers: Data-loader workers.

    Returns:
        Platform-neutral result dictionary (without ``usd`` / ``ccu``).

    Examples:
        >>> run_numbers(Path("/data/ds"), Path("/data/out"), job)["best"]["val/top1"]  # doctest: +SKIP
        0.9

    """
    import timm

    from bga.train.cls_trainer import train

    t0 = time.time()
    got = train(ds, out, job["base"], job["recipe"], num_workers=num_workers)
    return _base_result(
        job,
        out,
        time.time() - t0,
        f"timm=={timm.__version__}",
        {"crops": got["crops"], "arch": job["base"]},
        best=got["best"],
    )


def run_parseq(ds: Path, out: Path, job: dict[str, Any], num_workers: int = 8) -> dict[str, Any]:
    """Fine-tune the released PARSeq on class folders.

    Args:
        ds: Dataset directory with ``train/`` and ``valid/`` class folders.
        out: Output directory for ``weights.safetensors``, ``parseq_config.json`` and ``metrics.csv``.
        job: Job parameters with ``base``, ``recipe``, ``gpu`` and optional ``commit``.
        num_workers: Data-loader workers.

    Returns:
        Platform-neutral result dictionary (without ``usd`` / ``ccu``).

    Examples:
        >>> run_parseq(Path("/data/ds"), Path("/data/out"), job)["best"]["val/top1"]  # doctest: +SKIP
        0.9

    """
    from bga.train.parseq_trainer import train

    t0 = time.time()
    got = train(ds, out, job["base"], job["recipe"], num_workers=num_workers)
    return _base_result(
        job,
        out,
        time.time() - t0,
        PARSEQ_PACKAGE,
        {"crops": got["crops"], "base": job["base"], "unseen": got["unseen"]},
        best=got["best"],
    )


#: Map from trainer function name (as in ``bga.train.run.SETUPS``) to its platform-neutral runner.
TRAINERS = {
    "train_keypoint": run_keypoint,
    "train_detect": run_detect,
    "train_appearance": run_appearance,
    "train_numbers": run_numbers,
    "train_parseq": run_parseq,
}


def run_trainer(
    trainer_name: str,
    ds: Path,
    out: Path,
    job: dict[str, Any],
    *,
    num_workers: int = 8,
    init_path: Path | None = None,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Dispatch a training job by trainer name.

    Args:
        trainer_name: Key in :data:`TRAINERS` (e.g. ``train_parseq``).
        ds: Local dataset directory.
        out: Local output directory.
        job: Job parameters dictionary.
        num_workers: Data-loader workers.
        init_path: Optional warm-start checkpoint path (used by ``train_detect``).
        resume_path: Optional resume checkpoint path (used by ``train_detect``).

    Returns:
        Platform-neutral trainer result dictionary.

    Raises:
        ValueError: When ``trainer_name`` is unknown.

    Examples:
        >>> sorted(TRAINERS)
        ['train_appearance', 'train_detect', 'train_keypoint', 'train_numbers', 'train_parseq']

    """
    if trainer_name == "train_detect":
        return run_detect(ds, out, job, num_workers=num_workers, init_path=init_path, resume_path=resume_path)
    fn = TRAINERS.get(trainer_name)
    if fn is None:
        raise ValueError(f"unknown trainer {trainer_name!r}; expected one of {sorted(TRAINERS)}")
    return fn(ds, out, job, num_workers=num_workers)
