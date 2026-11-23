"""``bga train``: build the dataset, guard against gold leakage, train remotely, pull the weights back.

Steps for ``configs/train/<name>.yaml``:

1. Build the merged dataset under ``runs/train/<weight id>/dataset`` (``bga.train.court_data``,
   ``bga.train.detect_data``, ``bga.train.appearance_data`` or ``bga.train.number_data``); every manifest is checked
   against its sha256 and against the gold clips.
2. Run the trainer on the chosen platform (``colab`` via ``bga.remote.colab_train``, or ``modal`` via
   ``bga.remote.modal_train``). Platform and GPU default to the config's ``platform`` block and can be overridden at
   runtime via ``--platform`` / ``--gpu`` (or ``BGA_TRAIN_PLATFORM`` / ``BGA_TRAIN_GPU``) without changing the
   weight id.
3. Collect: the best checkpoint, ``metrics.csv`` and the trainer's ``result.json`` come back into
   ``weights/<weight id>/``, next to a ``train.json`` (config, dataset sha256 and counts, git commit,
   platform, GPU, wall time, cost, the trainer's validation numbers). A row goes to docs/train.md and the
   cost to docs/costs.md.

``--collect`` repeats step 3 alone, for a run whose local process ended early. Collecting replaces the weights
under their id, so it also removes what was made with the earlier files: cached layer outputs in ``runs/``
(``invalidate_runs``) and the copy of the weights on the Modal volume that remote runs use (``drop_remote_weights``).

"""

from __future__ import annotations

import csv
import datetime as dt
import importlib
import io
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from bga.train.config import ROOT, TrainConfig

log = logging.getLogger("bga")
#: Trainer validation numbers, one row per collected run.
TRAIN_MD = ROOT / "docs/train.md"
#: Cost per remote job.
COSTS_MD = ROOT / "docs/costs.md"
#: Supported remote training platforms.
PLATFORMS: tuple[str, ...] = ("colab", "modal")
#: Files collected from the trainer's output into ``weights/<weight id>/``.
PULL = ("checkpoint_best_total.pth", "metrics.csv", "result.json")
#: Training setups by (layer, backend): the ``bga.train`` dataset module and the trainer name in ``bga.train.dispatch``.
SETUPS = {
    ("court", "rfdetr_keypoint"): ("court_data", "train_keypoint"),
    ("detect", "rfdetr_torch"): ("detect_data", "train_detect"),
    ("appearance", "reid"): ("appearance_data", "train_appearance"),
    ("numbers", "crop_cls"): ("number_data", "train_numbers"),
    ("numbers", "parseq"): ("number_data", "train_parseq"),
}
#: Files collected per setup when they differ from ``PULL``.
PULLS = {
    ("appearance", "reid"): ("weights.safetensors", "reid_config.json", "metrics.csv", "result.json"),
    ("numbers", "crop_cls"): ("weights.safetensors", "cls_config.json", "metrics.csv", "result.json"),
    ("numbers", "parseq"): ("weights.safetensors", "parseq_config.json", "metrics.csv", "result.json"),
}
#: Trainer logs among the collected files; every other collected file is one a backend loads to run the weights.
LOGS = ("metrics.csv", "result.json")


def weight_files() -> tuple[str, ...]:
    """Name the files a backend may load from a weight id's directory, whichever setup trained it.

    These are ``train.json`` and every file some setup collects (``PULL``, ``PULLS``) that is not a trainer log, so
    a setup added there is covered by the remote upload (``bga.remote.modal_runner.upload_weights``) without a
    second list to keep in step.

    Returns:
        File names, ``train.json`` first.

    Examples:
        >>> weight_files()[:2]
        ('train.json', 'checkpoint_best_total.pth')
        >>> "weights.safetensors" in weight_files(), "metrics.csv" in weight_files()
        (True, False)

    """
    names = dict.fromkeys(("train.json", *PULL, *(f for files in PULLS.values() for f in files)))
    return tuple(n for n in names if n not in LOGS)


def resolve_platform(
    cfg: TrainConfig,
    platform: str | None = None,
    gpu: str | None = None,
) -> tuple[str, str]:
    """Resolve the execution platform and GPU for ``bga train`` without mutating ``cfg`` or its weight id.

    Precedence: explicit CLI override, then environment variable (``BGA_TRAIN_PLATFORM`` / ``BGA_TRAIN_GPU``),
    then ``cfg.platform``. Colab GPUs are validated against :data:`bga.remote.colab_train.COLAB_GPUS`.

    Args:
        cfg: The training config.
        platform: Optional ``"colab"`` or ``"modal"`` override.
        gpu: Optional GPU override (e.g. ``"L4"``, ``"A100"``, ``"L40S"``).

    Returns:
        ``(platform_name, gpu_name)``.

    Raises:
        ValueError: When the platform is unknown or the GPU is unsupported on Colab.

    Examples:
        >>> from bga.train.config import load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/court-rfdetr_keypoint-plk-v1.yaml")
        >>> resolve_platform(cfg, platform="colab", gpu="a100")
        ('colab', 'A100')
        >>> resolve_platform(cfg, platform="modal")
        ('modal', 'L40S')

    """
    resolved_platform = (platform or os.environ.get("BGA_TRAIN_PLATFORM") or cfg.platform.name).strip().lower()
    if resolved_platform not in PLATFORMS:
        raise ValueError(f"unsupported training platform {resolved_platform!r} (expected one of {PLATFORMS})")
    resolved_gpu = (gpu or os.environ.get("BGA_TRAIN_GPU") or cfg.platform.gpu).strip()
    if resolved_platform == "colab":
        from bga.remote.colab_train import validate_colab_gpu

        resolved_gpu = validate_colab_gpu(resolved_gpu)
    return resolved_platform, resolved_gpu


def _setup(cfg: TrainConfig) -> tuple[str, str]:
    """Dataset module and trainer function names of a config's layer and backend.

    Examples:
        >>> from types import SimpleNamespace
        >>> _setup(SimpleNamespace(layer="detect", backend="rfdetr_torch"))
        ('detect_data', 'train_detect')

    """
    if (cfg.layer, cfg.backend) not in SETUPS:
        raise NotImplementedError(f"no training setup for {cfg.layer}/{cfg.backend}")
    return SETUPS[cfg.layer, cfg.backend]


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
        NotImplementedError: for a layer and backend without a setup in ``SETUPS``.

    Examples:
        >>> from bga.train.config import load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/court-rfdetr_keypoint-plk-v1.yaml")
        >>> build(cfg)["sha256"]  # doctest: +SKIP
        '...'

    """
    builder = importlib.import_module(f"bga.train.{_setup(cfg)[0]}").build
    info = builder(cfg, _work(cfg) / "dataset")
    (_work(cfg) / "dataset_info.json").write_text(json.dumps(info, indent=1))
    return info


def _remote_exists(path: str) -> bool:
    """Whether a file exists on the ``bga-data`` volume.

    Args:
        path: file path on the volume.

    Returns:
        True when the file is listed in its directory; False when it or its directory is missing. Any other error
        (auth, network) propagates, so a run is never restarted from scratch because the volume could not be read.

    Examples:
        >>> _remote_exists("train/detect/rfdetr_torch/plk-r2@b13bfb19/output/last.ckpt")  # doctest: +SKIP
        True

    """
    from modal.exception import NotFoundError

    from bga.remote.modal_train import vol

    try:
        return any(e.path == path for e in vol.listdir(str(Path(path).parent)))
    except NotFoundError:  # the directory is missing, so the file is too
        return False


def drop_remote_weights(weight_id: str) -> bool:
    """Remove a weight id's copy from the ``bga-data`` volume, so the next remote run uploads the local files again.

    Remote runs upload a weight id once and then reuse the volume's copy. Collecting again (a partial collect, then
    the finished run) changes the local files under the same id, and the copy on the volume would stay the old
    checkpoint.

    Args:
        weight_id: The weight id being (re)written locally.

    Returns:
        True when the volume held a copy, now removed.

    Examples:
        >>> drop_remote_weights("detect/rfdetr_torch/plk-r3-poss@c62744fe")  # doctest: +SKIP
        True

    """
    from bga.remote.modal_train import vol

    path = f"weights/{weight_id}"
    if not _remote_exists(path):
        return False
    vol.remove_file(path, recursive=True)
    return True


def _launch_modal(cfg: TrainConfig, resume: bool, gpu: str) -> dict[str, Any]:
    """Upload the dataset to Modal's ``bga-data`` volume, spawn the detached trainer and wait for it."""
    import modal

    from bga.remote import modal_train
    from bga.remote.modal_train import CPUS, MEMORY_MB, app, vol

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
    init = None
    if cfg.recipe.init_weights:  # warm start: the earlier model's checkpoint goes up next to the dataset
        init = f"{remote}/init.pth"
        with vol.batch_upload(force=True) as up:
            up.put_file(str(ROOT / "weights" / cfg.recipe.init_weights / "checkpoint_best_total.pth"), init)
    job = {
        "commit": _commit(),  # the code that trains; HEAD may move before collect
        "dataset": f"{remote}/dataset",
        "output": f"{remote}/output",
        "recipe": cfg.recipe.model_dump(),
        "gpu": gpu,
        "resume": resume_from,
        "init": init,
        "base": cfg.base_weights.name,
    }
    trainer = getattr(modal_train, _setup(cfg)[1])
    n = cfg.recipe.devices or 1
    gpu_spec = f"{gpu}:{n}" if n > 1 else gpu
    fn = trainer.with_options(  # CPU and memory scale with the GPUs: each DDP process runs its own data loaders
        gpu=gpu_spec, cpu=CPUS * n, memory=MEMORY_MB * n, timeout=int(cfg.platform.timeout_h * 3600)
    )
    with modal.enable_output(), app.run(detach=True):
        call = fn.spawn(job)
    (_work(cfg) / "call.json").write_text(
        json.dumps({"platform": "modal", "gpu": gpu, "call_id": call.object_id, "resume": resume_from})
    )
    log.info("spawned %s; waiting (bga train --wait picks it up again)", call.object_id)
    return wait(cfg)


def _launch_colab(cfg: TrainConfig, gpu: str) -> dict[str, Any]:
    """Run training synchronously on a Google Colab GPU and download outputs into ``runs/train/<id>/output/``."""
    from bga.remote.colab_train import ColabJobSpec, launch_colab

    output_dir = _work(cfg) / "output"
    init_file = (
        ROOT / "weights" / cfg.recipe.init_weights / "checkpoint_best_total.pth" if cfg.recipe.init_weights else None
    )
    spec = ColabJobSpec(
        cfg=cfg,
        trainer_name=_setup(cfg)[1],
        dataset_dir=_work(cfg) / "dataset",
        output_dir=output_dir,
        pull_files=PULLS.get((cfg.layer, cfg.backend), PULL),
        gpu=gpu,
        commit=_commit(),
        init_file=init_file,
    )
    result = launch_colab(spec)
    (_work(cfg) / "call.json").write_text(json.dumps({"platform": "colab", "gpu": gpu, "output_dir": str(output_dir)}))
    return result


def launch(
    cfg: TrainConfig,
    resume: bool = True,
    *,
    platform: str | None = None,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Upload the built dataset, run training on Colab or Modal, and return the trainer's result.

    On Modal, the call is spawned in a detached app and its id saved to ``call.json`` so :func:`wait` can pick it
    up again and ``resume`` continues from ``last.ckpt`` on the volume. On Colab, the session streams output live,
    pulls checkpoints into ``runs/train/<id>/output/``, and terminates the VM before returning.

    Args:
        cfg: The training config; its dataset must be built (:func:`build`).
        resume: Modal only: continue from ``output/last.ckpt`` when the volume has one.
        platform: Optional ``"colab"`` or ``"modal"`` override (falls back to ``BGA_TRAIN_PLATFORM`` then config).
        gpu: Optional GPU override (falls back to ``BGA_TRAIN_GPU`` then config).

    Returns:
        The trainer's result: GPU, commit, wall time, cost/CCU, schema and output files.

    Examples:
        >>> launch(cfg, platform="colab", gpu="L4")["gpu"]  # doctest: +SKIP
        'L4'

    """
    target, target_gpu = resolve_platform(cfg, platform, gpu)
    if target == "colab":
        return _launch_colab(cfg, target_gpu)
    return _launch_modal(cfg, resume, target_gpu)


def wait(cfg: TrainConfig, poll_s: float = 120.0) -> dict[str, Any]:
    """Wait for the spawned training call of a config, polling in short calls that survive expired auth tokens.

    Args:
        cfg: the training config; ``call.json`` in its working directory names the call.
        poll_s: Seconds per poll.

    Returns:
        The trainer's result.

    Raises:
        RuntimeError: When ``call.json`` belongs to a Colab run without a completed ``result.json``.

    Examples:
        >>> wait(cfg)["gpu"]  # doctest: +SKIP
        'L40S'

    """
    import time

    call = json.loads((_work(cfg) / "call.json").read_text())
    if call.get("platform") == "colab":
        result_path = _work(cfg) / "output" / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        raise RuntimeError("Colab runs stream synchronously and cannot be detached with --wait")

    import modal

    cid = call["call_id"]
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


def _call_platform(cfg: TrainConfig) -> str:
    """Platform that produced the outputs currently waiting to be collected for ``cfg``."""
    call_path = _work(cfg) / "call.json"
    if call_path.exists():
        call = json.loads(call_path.read_text())
        if call.get("platform"):
            return str(call["platform"])
    return cfg.platform.name


def _read_output(cfg: TrainConfig, platform: str, name: str) -> bytes:
    """Read a trainer output file from local Colab downloads or from the Modal volume."""
    if platform == "colab":
        return (_work(cfg) / "output" / name).read_bytes()
    return _read_remote(f"{_remote(cfg)}/output/{name}")


def _best_row(metrics_csv: str, key: str) -> tuple[dict, dict]:
    """Rows of the trainer's metrics CSV with the highest ``key`` and the last with any ``key``; empty when none."""
    rows = [r for r in csv.DictReader(io.StringIO(metrics_csv)) if r.get(key) not in (None, "")]
    if not rows:
        return {}, {}
    best = max(rows, key=lambda r: float(r[key]))
    return best, rows[-1]


def invalidate_runs(weight_id: str, runs: Path) -> list[Path]:
    """Delete cached layer outputs made with a weight id's earlier checkpoint, and every output downstream of them.

    A weight id names its training config, not the checkpoint file, so re-collecting a run (a partial collect, then
    the finished one) changes the weights under the same id while the layer hashes stay the same. Outputs cached
    under those hashes would then mix old and new weights; this removes them so the next run recomputes them.

    Args:
        weight_id: The weight id being (re)written.
        runs: Root of the run outputs (``runs/<clip>/<layer>/<backend>@<hash8>/`` with ``meta.json``).

    Returns:
        The removed output directories.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     det = Path(tmp, "clip/detect/x@aaaa")
        ...     trk = Path(tmp, "clip/track/y@bbbb")
        ...     for d, meta in [(det, {"layer_hash": "aaaa", "entry": {"weights": "detect/x/n@1"}}),
        ...                     (trk, {"layer_hash": "bbbb", "upstream": {"detect": "aaaa"}})]:
        ...         d.mkdir(parents=True)
        ...         _ = (d / "meta.json").write_text(json.dumps(meta))
        ...     removed = invalidate_runs("detect/x/n@1", Path(tmp))
        ...     sorted(p.parent.name for p in removed), det.exists(), trk.exists()
        (['detect', 'track'], False, False)

    """
    import shutil

    metas = {}
    for m in runs.glob("*/*/*@*/meta.json"):
        try:
            metas[m.parent] = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
    stale = {d for d, m in metas.items() if (m.get("entry") or {}).get("weights") == weight_id}
    hashes = {metas[d].get("layer_hash") for d in stale}
    grew = True
    while grew:  # follow upstream links: an output built on a stale one is stale too
        more = {d for d, m in metas.items() if d not in stale and hashes & set((m.get("upstream") or {}).values())}
        grew = bool(more)
        stale |= more
        hashes |= {metas[d].get("layer_hash") for d in more}
    for d in stale:
        shutil.rmtree(d)
    return sorted(stale)


def _pull_outputs(cfg: TrainConfig, wdir: Path, platform: str, partial: bool) -> dict[str, Any]:
    """Copy a run's checkpoints and logs into ``wdir`` and return its ``result.json`` payload."""
    if partial:
        (wdir / PULL[0]).write_bytes(_read_output(cfg, platform, "checkpoint_best_ema.pth"))
        (wdir / "metrics.csv").write_bytes(_read_output(cfg, platform, "metrics.csv"))
        cats = json.loads((_work(cfg) / "dataset/train/_annotations.coco.json").read_text())["categories"]
        epochs = [
            int(float(r["epoch"]))
            for r in csv.DictReader(io.StringIO((wdir / "metrics.csv").read_text()))
            if r.get("epoch")
        ]
        result: dict[str, Any] = {
            "platform": platform,
            "schema": {"class_names": [c["name"] for c in sorted(cats, key=lambda c: c["id"])]},
            "partial": {
                "last_epoch": max(epochs, default=None),
                "of": cfg.recipe.epochs,
                "checkpoint": "checkpoint_best_ema.pth",
            },
        }
        (wdir / "result.json").write_text(json.dumps(result, indent=1))
        return result
    for name in PULLS.get((cfg.layer, cfg.backend), PULL):
        (wdir / name).write_bytes(_read_output(cfg, platform, name))
    return json.loads((wdir / "result.json").read_text())


def _maybe_drop_remote_weights(weight_id: str, platform: str) -> None:
    """Drop stale weights from Modal's volume; warn rather than fail when Modal is unavailable on Colab runs."""
    try:
        dropped = drop_remote_weights(weight_id)
    except Exception as exc:  # Colab-only environments may lack Modal credentials
        if platform != "colab":
            raise
        log.warning("could not check Modal volume for stale %s (%s)", weight_id, exc)
        return
    if dropped:
        log.info("removed the volume's copy of %s; the next remote run uploads the collected files", weight_id)


def collect(cfg: TrainConfig, partial: bool = False) -> Path:
    """Pull a finished run's files into ``weights/<weight id>/``, write ``train.json`` and log the docs rows.

    With ``partial``, a run that was cut short is collected as it stands: its best EMA checkpoint so far, the class
    names from the local dataset, and ``partial`` in ``train.json`` with the last epoch done. Its cost is not logged
    again (it was logged by hand when the run stopped).

    Args:
        cfg: the training config.
        partial: Collect a run that never returned a result.

    Returns:
        The weights directory.

    Examples:
        >>> collect(cfg).name  # doctest: +SKIP
        'plk-v1@c26b7841'

    """
    platform = _call_platform(cfg)
    wdir = cfg.weights_dir(ROOT)
    if wdir.exists():  # re-collecting changes the weights under the same id: drop outputs cached with the old ones
        for d in invalidate_runs(cfg.weight_id(), ROOT / "runs"):
            log.info("removed stale cached output %s", d)
    wdir.mkdir(parents=True, exist_ok=True)
    result = _pull_outputs(cfg, wdir, platform, partial)
    actual_platform = result.get("platform") or platform
    best, last = _best_row((wdir / "metrics.csv").read_text(), cfg.recipe.select_on)
    info_path = _work(cfg) / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    train_json: dict[str, Any] = {
        "weight_id": cfg.weight_id(),
        "config": cfg.model_dump(),
        "config_hash": cfg.config_hash(),
        "dataset": info,
        "commit": result.get("commit") or _commit(),
        "platform": actual_platform,
        "gpu": result.get("gpu"),
        "seconds": result.get("seconds"),
        "usd": result.get("usd"),
        "trainer": {
            "package": result.get("package"),
            "torch": result.get("torch"),
            "schema": result.get("schema"),
        },
        "select_on": cfg.recipe.select_on,
        "partial": result.get("partial"),
        "best": best,
        "last": last,
        "collected": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if "ccu" in result:
        train_json["ccu"] = result["ccu"]
    (wdir / "train.json").write_text(json.dumps(train_json, indent=1))
    _maybe_drop_remote_weights(cfg.weight_id(), actual_platform)
    _log_rows(cfg, train_json, cost=not partial)
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


def _format_spend(t: dict[str, Any]) -> tuple[str, str]:
    """Format the ``docs/train.md`` spent column and the ``docs/costs.md`` cost cell for Modal or Colab."""
    platform = t.get("platform") or "modal"
    if platform == "colab":
        ccu = f"{t['ccu']} CCU" if t.get("ccu") is not None else "n/a CCU"
        return f"colab {t['gpu']}, {t['seconds']} s, {ccu}", ccu
    return f"{t['gpu']}, {t['seconds']} s, {t['usd']} USD", f"{t['usd']}"


def _log_rows(cfg: TrainConfig, t: dict, cost: bool = True) -> None:
    """Append the run's row to docs/train.md and, when it exists, its cost to docs/costs.md."""
    counts = t["dataset"].get("counts", {})
    n = {s: sum(counts.get(s, {}).values()) for s in ("train", "valid")}
    epoch = t["best"].get("epoch", "?")
    cfg_file = f"configs/train/{cfg.layer}-{cfg.backend}-{cfg.name}.yaml"
    score = f"{cfg.recipe.select_on} {_fmt(t['best'], cfg.recipe.select_on)} (epoch {epoch})"
    platform = t.get("platform") or cfg.platform.name
    spent_train, cost_cell = _format_spend(t)
    spent = spent_train if cost else "cut short; cost in docs/costs.md"
    part = t.get("partial")
    note = f"partial: best of epochs 0 to {part['last_epoch']} of {part['of']}" if part else ""
    _append_row(
        TRAIN_MD,
        f"| {dt.date.today().isoformat()} | {t['commit']} | `{cfg_file}` | `{cfg.weight_id()}` "
        f"| {n['train']} / {n['valid']} | {score} | {spent} | {note} |",
    )
    if cost and COSTS_MD.exists():
        _append_row(
            COSTS_MD,
            f"| {dt.date.today().isoformat()} | {platform} | {t['gpu']} | bga train {cfg.weight_id()} "
            f"| {t['seconds']} | {cost_cell} |",
        )
