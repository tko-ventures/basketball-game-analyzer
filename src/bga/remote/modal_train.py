"""``bga train`` on Modal: RF-DETR fine-tunes with ``rfdetr``, ``reid`` encoders, and ``timm``/PARSeq number readers.

The dataset goes up to the ``bga-data`` volume under ``train/<weight id>/dataset``; the trainer writes to
``train/<weight id>/output`` on the same volume and commits it every few minutes, so a run that is cut
short keeps its checkpoints. Core training logic lives in :mod:`bga.train.dispatch` so Modal and Colab run
the exact same trainers. Each function returns GPU, wall time and list-price cost; ``bga train`` logs them.

"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import modal

from bga.remote.modal_runner import USD_PER_HOUR
from bga.train.dispatch import (
    PARSEQ_PACKAGE,
    REID_PACKAGE,
    RFDETR_PACKAGE,
    RFDETR_PLUS_PACKAGE,
    run_trainer,
)

#: CPU cores and memory per training container, for data loading and augmentation: about 20% faster epochs than
#: Modal's default reservation on RF-DETR Large at 704 px, for about 30% more cost (Modal billing, 2026-11-19).
CPUS, MEMORY_MB = 8.0, 32768
#: The Modal app of every training run.
app = modal.App("bga-train")
#: Shared volume: datasets and trainer outputs under ``train/<weight id>/``.
vol = modal.Volume.from_name("bga-data", create_if_missing=True)
#: Container image with torch, the pinned trainer and the local ``bga`` package.
image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", RFDETR_PACKAGE, RFDETR_PLUS_PACKAGE)
    # supervision pulls the GUI OpenCV, which needs libGL; swap in the headless wheel (no apt on bullseye).
    .run_commands(
        "pip uninstall -y opencv-python opencv-contrib-python",
        "pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93",
    )
    .add_local_python_source("bga")
)
#: Image of the appearance trainer: torch, ``reid`` at the Mac's versions, headless OpenCV and the local ``bga``.
reid_image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", "safetensors==0.8.0", "timm==1.0.30", REID_PACKAGE)
    .run_commands(
        "pip uninstall -y opencv-python opencv-contrib-python",
        "pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93",
    )
    .add_local_python_source("bga")
)
#: Image of the PARSeq trainer: torch and ``timm`` at the Mac's versions, ``strhub`` with the Lightning and ``nltk``
#: it imports, headless OpenCV and the local ``bga``. On bookworm, as the runner image, because Modal's
#: ``debian_slim`` (bullseye) can no longer install git.
parseq_image = (
    modal.Image
    .from_registry("python:3.12-slim-bookworm")
    .apt_install("git")
    .pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "safetensors==0.8.0",
        "timm==1.0.30",
        "pytorch-lightning==2.6.6",
        "nltk==3.10.4",
        "opencv-python-headless==5.0.0.93",
        "pyyaml",
        PARSEQ_PACKAGE,
    )
    .run_commands("python -c 'import strhub.models.parseq.system'")
    .add_local_python_source("bga")
)


def _run_on_volume(trainer_name: str, job: dict[str, Any]) -> dict[str, Any]:
    """Run ``trainer_name`` against ``/data`` with periodic volume commits and Modal list-price USD."""
    ds, out = Path("/data") / job["dataset"], Path("/data") / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def commit_loop() -> None:  # keep checkpoints even if the run is cut short
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    init_path = Path("/data") / job["init"] if job.get("init") else None
    resume_path = Path("/data") / job["resume"] if job.get("resume") else None
    try:
        result = run_trainer(
            trainer_name,
            ds,
            out,
            job,
            num_workers=int(CPUS),
            init_path=init_path,
            resume_path=resume_path,
        )
    finally:
        stop.set()
    devices = (job.get("recipe", {}).get("devices") or 1) if trainer_name == "train_detect" else 1
    gpu = job["gpu"]
    result["usd"] = round(result["seconds"] / 3600 * USD_PER_HOUR.get(gpu, 0.0) * devices, 4)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=CPUS, memory=MEMORY_MB, timeout=6 * 3600)
def train_keypoint(job: dict) -> dict:
    """Fine-tune RF-DETR Keypoint on a dataset on the volume and write ``result.json`` next to the checkpoints.

    Args:
        job: ``dataset`` and ``output`` (paths on the volume), ``recipe`` (``bga.train.config.Recipe`` as a dict),
            ``gpu`` (prices the run) and ``commit`` (recorded).

    Returns:
        GPU, commit, device name, wall time, list-price USD, keypoint schema, trainer package and torch version, and
        the files in the output directory (without ``result.json``, which is written last).

    Examples:
        >>> "checkpoint_best_total.pth" in train_keypoint.remote(job)["files"]  # doctest: +SKIP
        True

    """
    return _run_on_volume("train_keypoint", job)


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=CPUS, memory=MEMORY_MB, timeout=6 * 3600)
def train_detect(job: dict) -> dict:
    """Fine-tune an RF-DETR detector on a COCO dataset on the volume and write ``result.json`` next to the checkpoints.

    Args:
        job: As for :func:`train_keypoint`; ``recipe["variant"]`` names the ``rfdetr`` class (e.g. ``RFDETRLarge``).

    Returns:
        GPU, commit, device name, wall time, list-price USD, the class names in category-id order, trainer package and
        torch version, and the files in the output directory.

    Examples:
        >>> train_detect.remote(job)["schema"]["class_names"][1]  # doctest: +SKIP
        'ball'

    """
    return _run_on_volume("train_detect", job)


@app.function(image=reid_image, volumes={"/data": vol}, gpu="L4", cpu=CPUS, memory=MEMORY_MB, timeout=6 * 3600)
def train_appearance(job: dict) -> dict:
    """Fine-tune a ``reid`` encoder on identity folders on the volume (``bga.train.reid_trainer``).

    Args:
        job: As for :func:`train_keypoint`, plus ``base``: the ``reid`` alias the fine-tune starts from.

    Returns:
        GPU, commit, device name, wall time, list-price USD, the best validation row, identity and crop counts,
        trainer package and torch version, and the files in the output directory.

    Examples:
        >>> train_appearance.remote(job)["best"]["val/map"]  # doctest: +SKIP
        64.0

    """
    return _run_on_volume("train_appearance", job)


@app.function(image=reid_image, volumes={"/data": vol}, gpu="L4", cpu=CPUS, memory=MEMORY_MB, timeout=6 * 3600)
def train_numbers(job: dict) -> dict:
    """Fine-tune a ``timm`` number classifier on class folders on the volume (``bga.train.cls_trainer``).

    Args:
        job: As for :func:`train_keypoint`, plus ``base``: the ``timm`` model name the fine-tune starts from.

    Returns:
        GPU, commit, device name, wall time, list-price USD, the best validation row, crop counts, ``timm`` and
        torch versions, and the files in the output directory.

    Examples:
        >>> train_numbers.remote(job)["best"]["val/top1"]  # doctest: +SKIP
        0.9

    """
    return _run_on_volume("train_numbers", job)


@app.function(image=parseq_image, volumes={"/data": vol}, gpu="L4", cpu=CPUS, memory=MEMORY_MB, timeout=6 * 3600)
def train_parseq(job: dict) -> dict:
    """Fine-tune the released PARSeq on class folders on the volume (``bga.train.parseq_trainer``).

    Args:
        job: As for :func:`train_numbers`; ``base`` names the released checkpoint the fine-tune starts from.

    Returns:
        GPU, commit, device name, wall time, list-price USD, the best validation row, crop counts, the ``strhub``
        pin and torch version, and the files in the output directory.

    Examples:
        >>> train_parseq.remote(job)["best"]["val/top1"]  # doctest: +SKIP
        0.9

    """
    return _run_on_volume("train_parseq", job)
