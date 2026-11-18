"""``bga train``: build the dataset, guard against gold leakage, train remotely, pull the weights back.

Steps for ``configs/train/<name>.yaml``:

1. Build the merged dataset under ``runs/train/<weight id>/dataset`` (``bga.train.court_data`` or
   ``bga.train.detect_data``); every
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
#: Trainer validation numbers, one row per collected run.
TRAIN_MD = ROOT / "docs/train.md"
#: Cost per remote job.
COSTS_MD = ROOT / "docs/costs.md"
#: Files collected from the trainer's output on the volume into ``weights/<weight id>/``.
PULL = ("checkpoint_best_total.pth", "metrics.csv", "result.json")


def _work(cfg: TrainConfig) -> Path:
    """Local working directory of a training run: ``runs/train/<weight id with / as __>``."""
    return ROOT / "runs/train" / cfg.weight_id().replace("/", "__")


def _remote(cfg: TrainConfig) -> str:
    """Directory of a training run on the ``bga-data`` volume."""
    return f"train/{cfg.weight_id().replace('/', '__')}"


def _commit() -> str:
    """Short hash of the checked-out commit; empty outside a git checkout."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607  # git from PATH on purpose, as on the command line
        capture_output=True,
        text=True,
    ).stdout.strip()


def build(cfg: TrainConfig) -> dict:
    """Build a config's dataset under its working directory and record ``dataset_info.json`` beside it.

    Args:
        cfg: the training config.

    Returns:
        Image counts per split and source, and the dataset sha256 (``bga.train.court_data.build``).

    Raises:
        NotImplementedError: for any layer and backend other than ``court/rfdetr_keypoint`` and ``detect/rfdetr_torch``.

    Examples:
        >>> from bga.train.config import load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/court-rfdetr_keypoint-plk-v1.yaml")
        >>> build(cfg)["sha256"]  # doctest: +SKIP
        '...'

    """
    from bga.train import court_data, detect_data

    builders = {("court", "rfdetr_keypoint"): court_data.build, ("detect", "rfdetr_torch"): detect_data.build}
    if (cfg.layer, cfg.backend) not in builders:
        raise NotImplementedError(f"no dataset builder for {cfg.layer}/{cfg.backend}")
    info = builders[cfg.layer, cfg.backend](cfg, _work(cfg) / "dataset")
    (_work(cfg) / "dataset_info.json").write_text(json.dumps(info, indent=1))
    return info


def _remote_exists(path: str) -> bool:
    """Whether a file exists on the ``bga-data`` volume."""
    from bga.remote.modal_train import vol

    try:
        return any(e.path == path for e in vol.listdir(str(Path(path).parent)))
    except Exception:  # a missing directory raises; it means the file is not there either
        return False


def launch(cfg: TrainConfig, resume: bool = True) -> dict:
    """Upload the built dataset to the ``bga-data`` volume, start training on Modal and wait for the result.

    The call is spawned in a detached app and its id saved to ``call.json`` in the working directory, so the run goes
    on if this process stops; :func:`wait` picks it up again. With ``resume``, a run cut short continues from the
    ``last.ckpt`` its trainer left on the volume.

    Args:
        cfg: the training config; its dataset must be built (``build``).
        resume: Continue from ``output/last.ckpt`` when the volume has one.

    Returns:
        The trainer's result: GPU, commit, wall time, cost, schema and output files.

    Examples:
        >>> launch(cfg)["gpu"]  # doctest: +SKIP
        'L40S'

    """
    import modal

    from bga.remote.modal_train import app, train_detect, train_keypoint, vol

    ds = _work(cfg) / "dataset"
    remote = _remote(cfg)
    ckpt = f"{remote}/output/last.ckpt"
    resume_from = ckpt if resume and _remote_exists(ckpt) else None
    if resume_from is None:
        log.info("uploading %s to bga-data:/%s/dataset", ds, remote)
        with vol.batch_upload(force=True) as up:
            up.put_directory(str(ds), f"{remote}/dataset")
    else:
        log.info("resuming from bga-data:/%s", resume_from)
    job = {
        "commit": _commit(),  # the code that trains; HEAD may move before collect
        "dataset": f"{remote}/dataset",
        "output": f"{remote}/output",
        "recipe": cfg.recipe.model_dump(),
        "gpu": cfg.platform.gpu,
        "resume": resume_from,
    }
    trainer = train_detect if cfg.layer == "detect" else train_keypoint
    fn = trainer.with_options(gpu=cfg.platform.gpu, timeout=int(cfg.platform.timeout_h * 3600))
    with modal.enable_output(), app.run(detach=True):
        call = fn.spawn(job)
    (_work(cfg) / "call.json").write_text(json.dumps({"call_id": call.object_id, "resume": resume_from}))
    log.info("spawned %s; waiting (bga train --wait picks it up again)", call.object_id)
    return wait(cfg)


def wait(cfg: TrainConfig, poll_s: float = 120.0) -> dict:
    """Wait for the spawned training call of a config, polling in short calls that survive expired auth tokens.

    Args:
        cfg: the training config; ``call.json`` in its working directory names the call.
        poll_s: Seconds per poll.

    Returns:
        The trainer's result.

    Examples:
        >>> wait(cfg)["gpu"]  # doctest: +SKIP
        'L40S'

    """
    import time

    import modal

    cid = json.loads((_work(cfg) / "call.json").read_text())["call_id"]
    while True:
        try:
            return modal.FunctionCall.from_id(cid).get(timeout=poll_s)
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except modal.exception.AuthError as e:  # long waits outlive the client's token; a fresh poll gets a new one
            log.warning("auth expired while waiting (%s); polling again", e)
            time.sleep(5)


def _read_remote(path: str) -> bytes:
    """Read a whole file from the ``bga-data`` volume."""
    from bga.remote.modal_train import vol

    buf = io.BytesIO()
    for chunk in vol.read_file(path):
        buf.write(chunk)
    return buf.getvalue()


def _best_row(metrics_csv: str, key: str) -> tuple[dict, dict]:
    """Rows of the trainer's metrics CSV with the highest ``key`` and the last with any ``key``; empty when none."""
    rows = [r for r in csv.DictReader(io.StringIO(metrics_csv)) if r.get(key) not in (None, "")]
    if not rows:
        return {}, {}
    best = max(rows, key=lambda r: float(r[key]))
    return best, rows[-1]


def collect(cfg: TrainConfig) -> Path:
    """Pull a finished run's files into ``weights/<weight id>/``, write ``train.json`` and log the docs rows.

    Args:
        cfg: the training config.

    Returns:
        The weights directory.

    Examples:
        >>> collect(cfg).name  # doctest: +SKIP
        'plk-v1@c26b7841'

    """
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
        "commit": result.get("commit") or _commit(),
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
    """``row[key]`` to three decimals, or ``n/a``."""
    try:
        return f"{float(row[key]):.3f}"
    except (KeyError, TypeError, ValueError):
        return "n/a"


def _append_row(path: Path, row: str) -> None:
    """Insert a table row after the last table row of a Markdown file, above any text that follows the table."""
    lines = path.read_text().rstrip("\n").split("\n")
    last = max(i for i, line in enumerate(lines) if line.startswith("|"))
    path.write_text("\n".join([*lines[: last + 1], row, *lines[last + 1 :]]) + "\n")


def _log_rows(cfg: TrainConfig, t: dict) -> None:
    """Append the run's row to docs/train.md and, when it exists, its cost to docs/costs.md."""
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
