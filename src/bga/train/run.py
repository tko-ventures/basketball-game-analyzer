"""``bga train``: build the dataset, guard against gold leakage, train remotely, pull the weights back.

Steps for ``configs/train/<name>.yaml``:

1. Build the merged dataset under ``runs/train/<weight id>/dataset`` (``bga.train.court_data``); every
   manifest is checked against its sha256 and against the gold clips.
2. Upload it to the ``bga-data`` Modal volume and run the trainer there (``bga.remote.modal_train``) in a
   detached app, so the run continues if this process stops.
3. Collect: the best checkpoint, ``metrics.csv`` and the trainer's ``result.json`` come back into
   ``weights/<weight id>/``, next to a ``train.json`` (config, dataset sha256 and counts, git commit,
   platform, GPU, wall time, cost, the trainer's validation numbers). A row goes to docs/train.md and the
   cost to docs/costs.md.

``--collect`` repeats step 3 alone, for a run whose local process ended early.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import subprocess
from pathlib import Path

from bga.train.config import ROOT, TrainConfig

log = logging.getLogger("bga")
TRAIN_MD = ROOT / "docs/train.md"
COSTS_MD = ROOT / "docs/costs.md"
PULL = ("checkpoint_best_total.pth", "metrics.csv", "result.json")


def _work(cfg: TrainConfig) -> Path:
    return ROOT / "runs/train" / cfg.weight_id().replace("/", "__")


def _remote(cfg: TrainConfig) -> str:
    return f"train/{cfg.weight_id().replace('/', '__')}"


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def build(cfg: TrainConfig) -> dict:
    if cfg.layer != "court" or cfg.backend != "rfdetr_keypoint":
        raise NotImplementedError(f"no dataset builder for {cfg.layer}/{cfg.backend}")
    from bga.train import court_data

    info = court_data.build(cfg, _work(cfg) / "dataset")
    (_work(cfg) / "dataset_info.json").write_text(json.dumps(info, indent=1))
    return info


def launch(cfg: TrainConfig) -> dict:
    import modal

    from bga.remote.modal_train import app, train_keypoint, vol

    ds = _work(cfg) / "dataset"
    remote = _remote(cfg)
    log.info("uploading %s to bga-data:/%s/dataset", ds, remote)
    with vol.batch_upload(force=True) as up:
        up.put_directory(str(ds), f"{remote}/dataset")
    job = {
        "dataset": f"{remote}/dataset",
        "output": f"{remote}/output",
        "recipe": cfg.recipe.model_dump(),
        "gpu": cfg.platform.gpu,
    }
    fn = train_keypoint.with_options(gpu=cfg.platform.gpu, timeout=int(cfg.platform.timeout_h * 3600))
    with modal.enable_output(), app.run(detach=True):
        return fn.remote(job)


def _read_remote(path: str) -> bytes:
    from bga.remote.modal_train import vol

    buf = io.BytesIO()
    for chunk in vol.read_file(path):
        buf.write(chunk)
    return buf.getvalue()


def _best_row(metrics_csv: str, key: str) -> tuple[dict, dict]:
    rows = [r for r in csv.DictReader(io.StringIO(metrics_csv)) if r.get(key) not in (None, "")]
    if not rows:
        return {}, {}
    best = max(rows, key=lambda r: float(r[key]))
    return best, rows[-1]


def collect(cfg: TrainConfig) -> Path:
    remote, wdir = _remote(cfg), cfg.weights_dir()
    wdir.mkdir(parents=True, exist_ok=True)
    for name in PULL:
        (wdir / name).write_bytes(_read_remote(f"{remote}/output/{name}"))
    result = json.loads((wdir / "result.json").read_text())
    best, last = _best_row((wdir / "metrics.csv").read_text(), cfg.recipe.select_on)
    info_path = _work(cfg) / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    train_json = {
        "weight_id": cfg.weight_id(),
        "config": cfg.model_dump(),
        "config_hash": cfg.config_hash(),
        "dataset": info,
        "commit": _commit(),
        "platform": cfg.platform.name,
        "gpu": result.get("gpu"),
        "seconds": result.get("seconds"),
        "usd": result.get("usd"),
        "trainer": {
            "package": result.get("package"),
            "torch": result.get("torch"),
            "schema": result.get("schema"),
        },
        "select_on": cfg.recipe.select_on,
        "best": best,
        "last": last,
        "collected": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (wdir / "train.json").write_text(json.dumps(train_json, indent=1))
    _log_rows(cfg, train_json)
    return wdir


def _fmt(row: dict, key: str) -> str:
    try:
        return f"{float(row[key]):.3f}"
    except (KeyError, TypeError, ValueError):
        return "n/a"


def _append_row(path: Path, row: str) -> None:
    lines = path.read_text().rstrip("\n").split("\n")
    last = max(i for i, line in enumerate(lines) if line.startswith("|"))
    path.write_text("\n".join(lines[: last + 1] + [row] + lines[last + 1 :]) + "\n")


def _log_rows(cfg: TrainConfig, t: dict) -> None:
    counts = t["dataset"].get("counts", {})
    n = {s: sum(counts.get(s, {}).values()) for s in ("train", "valid")}
    epoch = t["best"].get("epoch", "?")
    cfg_file = f"configs/train/{cfg.layer}-{cfg.backend}-{cfg.name}.yaml"
    score = f"{cfg.recipe.select_on} {_fmt(t['best'], cfg.recipe.select_on)} (epoch {epoch})"
    cost = f"{t['gpu']}, {t['seconds']} s, {t['usd']} USD"
    _append_row(
        TRAIN_MD,
        f"| {dt.date.today().isoformat()} | {t['commit']} | `{cfg_file}` | `{cfg.weight_id()}` "
        f"| {n['train']} / {n['valid']} | {score} | {cost} | |",
    )
    if COSTS_MD.exists():
        _append_row(
            COSTS_MD,
            f"| {dt.date.today().isoformat()} | modal | {t['gpu']} | bga train {cfg.weight_id()} "
            f"| {t['seconds']} | {t['usd']} |",
        )
