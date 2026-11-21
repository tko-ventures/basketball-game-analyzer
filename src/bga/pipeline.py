"""Run a config layer by layer over its sequences (plan section 5).

Each layer runs over every sequence before the next layer starts, so one heavy model is resident at a
time. Outputs go to ``runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json`` with a ``meta.json``;
arrays a backend returns under ``arrays`` go next to it as ``<sequence>.<name>.npy``. An existing output
is reused unless ``force`` names that layer, so changing one layer's config reruns only that layer and
the layers downstream of it.

"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from bga import registry
from bga.config import LAYERS, RunConfig
from bga.stage import Backend, FrameReader, Sequence

log = logging.getLogger("bga")


def git_commit() -> str:
    """Short hash of the checked-out commit, recorded in every ``meta.json`` and metrics row.

    Returns:
        The output of ``git rev-parse --short HEAD`` (empty outside a checkout), or ``""`` when git is not installed.

    Examples:
        >>> isinstance(git_commit(), str)
        True

    """
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()  # noqa: S607  # git from PATH on purpose
    except OSError:
        return ""


def make_backend(cfg: RunConfig, layer: str) -> Backend:
    """Instantiate the backend a config names for one layer, with its params and weights id.

    Args:
        cfg: The run config.
        layer: A configured layer name.

    Returns:
        The backend instance, not yet loaded.

    Examples:
        >>> from bga.config import RunConfig
        >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold", "params": {"stride": 10}}})
        >>> make_backend(cfg, "shots").p["stride"]
        10

    """
    entry = cfg.layers[layer]
    return registry.get(layer, entry.backend)(entry.params, entry.weights)


def sequences(cfg: RunConfig, only: list[str] | None = None) -> list[Sequence]:
    """List the sequences the config's shots layer selects, optionally only those named in ``only``.

    Args:
        cfg: The run config.
        only: Sequence names to keep; ``None`` or empty keeps all.

    Returns:
        The selected sequences in the shots backend's order.

    Examples:
        >>> from bga.config import load_config
        >>> [s.name for s in sequences(load_config("configs/baseline.yaml"), ["OSp445e0mXU_s2"])]  # doctest: +SKIP
        ['OSp445e0mXU_s2']

    """
    seqs = make_backend(cfg, "shots").sequences()
    return [s for s in seqs if not only or s.name in only]


def output_path(cfg: RunConfig, runs: Path, seq: Sequence, layer: str) -> Path:
    """Path of one layer's output for one sequence: ``<run_dir>/<sequence>.json``.

    Args:
        cfg: The run config.
        runs: Root of all run outputs.
        seq: The sequence.
        layer: A configured layer name.

    Returns:
        The JSON output path; it may not exist yet.

    Examples:
        >>> from bga.config import RunConfig
        >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold", "params": {"stride": 5}}})
        >>> seq = Sequence(name="clip_s2", clip="clip", video="clip.mp4", shot=2, frames=[0], fps=50.0)
        >>> output_path(cfg, Path("runs"), seq, "shots").as_posix()
        'runs/clip/shots/gold@9e3d5b1a/clip_s2.json'

    """
    return cfg.run_dir(runs, seq.clip, layer) / f"{seq.name}.json"


def load_output(cfg: RunConfig, runs: Path, seq: Sequence, layer: str, arrays: bool = False) -> dict:
    """Read one layer's stored output for one sequence.

    Args:
        cfg: The run config.
        runs: Root of all run outputs.
        seq: The sequence.
        layer: A configured layer name.
        arrays: Also load the arrays the output lists under ``arrays`` (see :func:`write_output`), replacing each
            record with its array; otherwise the records are returned as stored.

    Returns:
        The parsed JSON output.

    Raises:
        FileNotFoundError: When that layer has not run on the sequence.

    Examples:
        >>> import tempfile
        >>> from bga.config import RunConfig
        >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold"}})
        >>> seq = Sequence(name="clip_s2", clip="clip", video="clip.mp4", shot=2, frames=[0], fps=50.0)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = output_path(cfg, Path(tmp), seq, "shots")
        ...     path.parent.mkdir(parents=True)
        ...     _ = path.write_text('{"frames": [0]}')
        ...     load_output(cfg, Path(tmp), seq, "shots")
        {'frames': [0]}

    """
    path = output_path(cfg, runs, seq, layer)
    out = json.loads(path.read_text())
    if arrays and out.get("arrays"):
        out["arrays"] = {name: np.load(path.parent / rec["file"]) for name, rec in out["arrays"].items()}
    return out


def write_output(path: Path, out: dict) -> None:
    """Write one layer's output for one sequence, with any arrays it carries saved next to the JSON.

    Each array in ``out["arrays"]`` goes to ``<sequence>.<name>.npy`` in its own dtype, and the JSON keeps a record
    ``{"file", "shape", "dtype"}`` in its place, so :func:`load_output` with ``arrays=True`` gives it back.

    Args:
        path: The output's JSON path (``output_path``); its directory is created.
        out: The backend's output; it is not modified.

    Examples:
        >>> import tempfile
        >>> from bga.config import RunConfig
        >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold"}})
        >>> seq = Sequence(name="clip_s2", clip="clip", video="clip.mp4", shot=2, frames=[0], fps=50.0)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = output_path(cfg, Path(tmp), seq, "shots")
        ...     write_output(path, {"arrays": {"e": np.ones((2, 3), np.float16)}})
        ...     stored = load_output(cfg, Path(tmp), seq, "shots")
        ...     back = load_output(cfg, Path(tmp), seq, "shots", arrays=True)
        >>> stored["arrays"]
        {'e': {'file': 'clip_s2.e.npy', 'shape': [2, 3], 'dtype': 'float16'}}
        >>> back["arrays"]["e"].dtype, back["arrays"]["e"].shape
        (dtype('float16'), (2, 3))

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = out.get("arrays") or {}
    records = {}
    for name, arr in arrays.items():
        file = f"{path.stem}.{name}.npy"
        np.save(path.parent / file, arr)
        records[name] = {"file": file, "shape": list(arr.shape), "dtype": str(arr.dtype)}
    stored = {**out, "arrays": records} if arrays else out
    path.write_text(json.dumps(stored, separators=(",", ":")))


def run(
    cfg: RunConfig,
    runs: Path = Path("runs"),
    only: list[str] | None = None,
    layers: list[str] | None = None,
    force: list[str] | None = None,
    remote: str | None = None,
    remote_layers: list[str] | None = None,
    remote_gpu: str | None = None,
) -> list[Sequence]:
    """Run every configured layer over the config's sequences, layer by layer, reusing stored outputs.

    The shots layer only selects sequences. A layer runs on the sequences without an output (all of them when
    ``force`` names it) and then writes ``meta.json`` per clip. With ``remote``, CUDA backends and
    ``remote_layers`` run on Modal; without it, CUDA backends are skipped with a warning.

    Args:
        cfg: The run config.
        runs: Root of all run outputs.
        only: Sequence names to run; ``None`` runs all.
        layers: Layers to run; ``None`` runs every configured layer.
        force: Layers to recompute even where outputs exist.
        remote: ``"modal"`` to allow remote execution, else ``None``.
        remote_layers: Layers to run remotely besides the CUDA-only ones.
        remote_gpu: Modal GPU type for ``remote_layers``, or ``None`` for CPU; CUDA backends bring their own.

    Returns:
        The sequences of the run, including those whose outputs were reused.

    Examples:
        >>> from bga.config import load_config
        >>> len(run(load_config("configs/baseline.yaml"), only=["OSp445e0mXU_s2"]))  # doctest: +SKIP
        1

    """
    seqs = sequences(cfg, only)
    for layer in LAYERS:
        if layer == "shots" or layer not in cfg.layers or (layers and layer not in layers):
            continue
        backend = make_backend(cfg, layer)
        here = "modal" if remote and (backend.requires == "cuda" or layer in (remote_layers or ())) else None
        if backend.requires == "cuda" and not here:
            log.warning("%s/%s needs CUDA: run it with --remote modal; skipped", layer, cfg.layers[layer].backend)
            continue
        todo = [s for s in seqs if (force and layer in force) or not output_path(cfg, runs, s, layer).exists()]
        if not todo:
            continue
        log.info("%s/%s on %d sequences%s", layer, cfg.layers[layer].backend, len(todo), " (modal)" if here else "")
        t0 = time.perf_counter()
        if here:
            from bga.remote.modal_runner import run_layer

            ups = [
                {
                    u: load_output(cfg, runs, s, u, arrays=True)
                    for u in LAYERS[layer]
                    if u in cfg.layers and u != "shots"
                }
                for s in todo
            ]
            gpu = getattr(backend, "gpu", None) if backend.requires == "cuda" else remote_gpu
            for seq, out in zip(todo, run_layer(layer, cfg.layers[layer].model_dump(), todo, ups, gpu), strict=True):
                write_output(output_path(cfg, runs, seq, layer), out)
            _write_meta(cfg, runs, layer, todo, t0)
            continue
        backend.load()
        try:
            for seq in todo:
                ups = {
                    u: load_output(cfg, runs, seq, u, arrays=True)
                    for u in LAYERS[layer]
                    if u in cfg.layers and u != "shots"
                }
                read = FrameReader(seq.video)
                try:
                    t = time.perf_counter()
                    out = backend.run(seq, ups, read)
                    out.setdefault("wall_s", round(time.perf_counter() - t, 2))
                finally:
                    read.close()
                write_output(output_path(cfg, runs, seq, layer), out)
                log.info("  %s: %.1f s", seq.name, out["wall_s"])
        finally:
            backend.unload()
        _write_meta(cfg, runs, layer, todo, t0)
    return seqs


def _write_meta(cfg: RunConfig, runs: Path, layer: str, todo: list[Sequence], t0: float) -> None:
    for clip in {s.clip for s in todo}:
        remote = [r for s in todo if s.clip == clip and (r := load_output(cfg, runs, s, layer).get("remote"))]
        meta = {
            "layer": layer,
            "backend": cfg.layers[layer].backend,
            "layer_hash": cfg.layer_hash(layer),
            "entry": cfg.layers[layer].model_dump(),
            "upstream": {u: cfg.layer_hash(u) for u in LAYERS[layer] if u in cfg.layers},
            "config": cfg.name,
            "commit": git_commit(),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        if remote:
            meta["remote"] = {
                "platform": remote[0]["platform"],
                "gpu": remote[0]["gpu"],
                "seconds": round(sum(r["seconds"] for r in remote), 1),
                "usd": round(sum(r["usd"] for r in remote), 4),
            }
        (cfg.run_dir(runs, clip, layer) / "meta.json").write_text(json.dumps(meta, indent=1))
