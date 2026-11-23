"""``bga train`` on Modal: RF-DETR fine-tunes with ``rfdetr``, ``reid`` encoders, and ``timm`` number classifiers.

The dataset goes up to the ``bga-data`` volume under ``train/<weight id>/dataset``; the trainer writes to
``train/<weight id>/output`` on the same volume and commits it every few minutes, so a run that is cut
short keeps its checkpoints. The function follows the June 2026 keypoint post: schema from the COCO
annotations, ``RFDETRKeypointPreview`` with that schema, ``KeypointTrainConfig``, the Lightning modules,
``trainer.fit``. It returns GPU, wall time and list-price cost; ``bga train`` logs them.

"""

from __future__ import annotations

import modal

from bga.remote.modal_runner import USD_PER_HOUR

#: The trainer package pinned in the image and recorded in every result.
RFDETR_PACKAGE = "rfdetr[train]==1.12.1"
#: RF-DETR XL and 2XL (Platform Model Licence 1.0, treated as Apache 2.0 by Tomasz's ruling of 2026-11-18, question 5).
RFDETR_PLUS_PACKAGE = "rfdetr_plus==1.3.0"

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
    import json
    import threading
    import time
    from pathlib import Path

    import torch
    from rfdetr import RFDETRKeypointPreview
    from rfdetr.config import KeypointTrainConfig
    from rfdetr.datasets import infer_coco_keypoint_schema
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    t0 = time.time()
    ds, out = Path("/data") / job["dataset"], Path("/data") / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def commit_loop() -> None:  # keep checkpoints even if the run is cut short
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
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
    stop.set()

    secs = time.time() - t0
    gpu = job["gpu"]
    result = {
        "gpu": gpu,
        "commit": job.get("commit"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR.get(gpu, 0.0), 4),
        "schema": {
            "class_names": schema.class_names,
            "num_keypoints_per_class": schema.num_keypoints_per_class,
        },
        "package": RFDETR_PACKAGE,
        "torch": torch.__version__,
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


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
    import json
    import threading
    import time
    from pathlib import Path

    import rfdetr
    import torch

    t0 = time.time()
    ds, out = Path("/data") / job["dataset"], Path("/data") / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def commit_loop() -> None:  # keep checkpoints even if the run is cut short
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    r = job["recipe"]
    devices = r.get("devices") or 1
    init = {"pretrain_weights": str(Path("/data") / job["init"])} if job.get("init") else {}
    model = getattr(rfdetr, r["variant"])(**init)
    extra = {}
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
        num_workers=job.get("num_workers", int(CPUS)),
        resume=str(Path("/data") / job["resume"]) if job.get("resume") else None,
        **extra,
    )
    stop.set()
    cats = json.loads((ds / "train" / "_annotations.coco.json").read_text())["categories"]
    secs = time.time() - t0
    gpu = job["gpu"]
    result = {
        "gpu": f"{gpu}:{devices}" if devices > 1 else gpu,
        "commit": job.get("commit"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR.get(gpu, 0.0) * devices, 4),  # GPU list price only
        "schema": {"class_names": [c["name"] for c in sorted(cats, key=lambda c: c["id"])]},
        "package": RFDETR_PACKAGE,
        "torch": torch.__version__,
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


#: The re-ID package pinned in the appearance trainer image and recorded in every result.
REID_PACKAGE = "reid==0.1.0"
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
    import json
    import threading
    import time
    from pathlib import Path

    import torch

    from bga.train.reid_trainer import train

    t0 = time.time()
    ds, out = Path("/data") / job["dataset"], Path("/data") / job["output"]
    stop = threading.Event()

    def commit_loop() -> None:  # keep the metrics and best weights even if the run is cut short
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    got = train(ds, out, job["base"], job["recipe"], num_workers=int(CPUS))
    stop.set()
    secs, gpu = time.time() - t0, job["gpu"]
    result = {
        "gpu": gpu,
        "commit": job.get("commit"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR.get(gpu, 0.0), 4),  # GPU list price only
        "schema": {"identities": got["identities"], "crops": got["crops"]},
        "best": got["best"],
        "package": REID_PACKAGE,
        "torch": torch.__version__,
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


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
    import json
    import threading
    import time
    from pathlib import Path

    import timm
    import torch

    from bga.train.cls_trainer import train

    t0 = time.time()
    ds, out = Path("/data") / job["dataset"], Path("/data") / job["output"]
    stop = threading.Event()

    def commit_loop() -> None:  # keep the metrics and best weights even if the run is cut short
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    got = train(ds, out, job["base"], job["recipe"], num_workers=int(CPUS))
    stop.set()
    secs, gpu = time.time() - t0, job["gpu"]
    result = {
        "gpu": gpu,
        "commit": job.get("commit"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR.get(gpu, 0.0), 4),  # GPU list price only
        "schema": {"crops": got["crops"], "arch": job["base"]},
        "best": got["best"],
        "package": f"timm=={timm.__version__}",
        "torch": torch.__version__,
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


#: ``strhub`` (PARSeq) pinned to a commit in the PARSeq trainer image and recorded in every result.
PARSEQ_PACKAGE = "strhub @ git+https://github.com/baudm/parseq@1902db043c029a7e03a3818c616c06600af574be"
#: Image of the PARSeq trainer: torch and ``timm`` at the Mac's versions, ``strhub`` with the Lightning and ``nltk``
#: it imports, headless OpenCV and the local ``bga``.
parseq_image = (
    modal.Image
    .debian_slim(python_version="3.12")
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
    .add_local_python_source("bga")
)


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
    import json
    import threading
    import time
    from pathlib import Path

    import torch

    from bga.train.parseq_trainer import train

    t0 = time.time()
    ds, out = Path("/data") / job["dataset"], Path("/data") / job["output"]
    stop = threading.Event()

    def commit_loop() -> None:  # keep the metrics and best weights even if the run is cut short
        while not stop.wait(300):
            vol.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    got = train(ds, out, job["base"], job["recipe"], num_workers=int(CPUS))
    stop.set()
    secs, gpu = time.time() - t0, job["gpu"]
    result = {
        "gpu": gpu,
        "commit": job.get("commit"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR.get(gpu, 0.0), 4),  # GPU list price only
        "schema": {"crops": got["crops"], "base": job["base"], "unseen": got["unseen"]},
        "best": got["best"],
        "package": PARSEQ_PACKAGE,
        "torch": torch.__version__,
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result
