#!/usr/bin/env python3
"""Measure seconds per epoch of a detection fine-tune on Modal with the current container resources.

    uv run --extra remote scripts/probe_train_speed.py configs/train/detect-rfdetr_torch-nba-704.yaml --epochs 2

Runs the config's dataset (already on the bga-data volume from its real run) for a few epochs into
train/_probe/<name>/, not the real output, and prints wall time per epoch, GPU, and cost. A probe, not a training run:
no weights are collected and no docs/train.md row is written; the cost row goes to docs/costs.md.

"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modal

from bga.remote.modal_runner import append_cost_rows
from bga.remote.modal_train import CPUS, app, train_detect
from bga.train.config import load_train_config
from bga.train.run import _remote

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--workers", type=int, default=int(CPUS))
a = ap.parse_args()
cfg = load_train_config(a.config)
recipe = {**cfg.recipe.model_dump(), "epochs": a.epochs}
job = {
    "commit": "probe",
    "dataset": f"{_remote(cfg)}/dataset",
    "output": f"train/_probe/{cfg.name}-w{a.workers}-{int(time.time())}",
    "recipe": recipe,
    "gpu": cfg.platform.gpu,
    "num_workers": a.workers,
}
fn = train_detect.with_options(gpu=cfg.platform.gpu, timeout=3600)
with modal.enable_output(), app.run(detach=True):
    call = fn.spawn(job)
while True:
    try:
        res = modal.FunctionCall.from_id(call.object_id).get(timeout=120)
        break
    except (TimeoutError, modal.exception.TimeoutError):
        continue
    except modal.exception.AuthError:
        time.sleep(5)
per = res["seconds"] / a.epochs
print(
    json.dumps(
        {
            "seconds": res["seconds"],
            "per_epoch_s": round(per, 1),
            "gpu": res["gpu"],
            "usd": res["usd"],
            "workers": a.workers,
            "cpus": CPUS,
        },
        indent=1,
    )
)
row = (
    f"| {time.strftime('%Y-%m-%d')} | modal | {res['gpu']} | probe: {cfg.name} {a.epochs} epochs, "
    f"{a.workers} workers, {CPUS:g} CPUs ({per:.0f} s per epoch incl. start-up) | {res['seconds']} | {res['usd']} |"
)
append_cost_rows([row])
