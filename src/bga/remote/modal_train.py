"""``bga train`` on Modal: RF-DETR Keypoint fine-tunes with the open-source ``rfdetr`` trainer.

The dataset goes up to the ``bga-data`` volume under ``train/<weight id>/dataset``; the trainer writes to
``train/<weight id>/output`` on the same volume and commits it every few minutes, so a run that is cut
short keeps its checkpoints. The function follows the June 2026 keypoint post: schema from the COCO
annotations, ``RFDETRKeypointPreview`` with that schema, ``KeypointTrainConfig``, the Lightning modules,
``trainer.fit``. It returns GPU, wall time and list-price cost; ``bga train`` logs them.
"""

from __future__ import annotations

import modal

from bga.remote.modal_runner import USD_PER_HOUR

RFDETR_PACKAGE = "rfdetr[train]==1.12.1"

app = modal.App("bga-train")
vol = modal.Volume.from_name("bga-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", RFDETR_PACKAGE)
    # supervision pulls the GUI OpenCV, which needs libGL; swap in the headless wheel (no apt on bullseye).
    .run_commands(
        "pip uninstall -y opencv-python opencv-contrib-python",
        "pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93",
    )
    .add_local_python_source("bga")
)


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", timeout=6 * 3600)
def train_keypoint(job: dict) -> dict:
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
