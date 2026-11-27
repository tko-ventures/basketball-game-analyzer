"""Run a config layer by layer over its sequences (plan section 5).

Each layer runs over every sequence before the next layer starts, so one heavy model is resident at a
time. Outputs go to ``runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json`` with a ``meta.json``;
arrays a backend returns under ``arrays`` go next to it as ``<sequence>.<name>.npy``. An existing output
is reused unless ``force`` names that layer, or its record was made by other code (:func:`code_keys`), so changing
one layer's config reruns only that layer and the layers downstream of it, and changing a backend's code reruns
the layers whose code key changed. Every file is written whole (``bga.paths.write_atomic``), the JSON last, so a
run stopped part way never leaves an output that looks finished.

``meta.json`` describes the directory: the layer entry and hashes, the config, commit and wall time of the last
run that wrote there, and under ``sequences`` one record per sequence written since 2026-11-22 (commit, config,
time written, wall time, code key since 2026-11-27, and platform, GPU, seconds and USD for remote calls). A run
over some of the sequences adds or replaces only their records.

"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from bga import registry
from bga.codehash import source_hash
from bga.config import LAYERS, RunConfig
from bga.paths import write_atomic
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


def code_keys(cfg: RunConfig) -> dict[str, str]:
    """Code key of every configured layer: its backend's source hash, chained with the keys of the layers it reads.

    A layer's key changes when its own backend code changes and when the code of any upstream layer does, since a
    recomputed upstream output changes what the layer reads.

    Args:
        cfg: The run config.

    Returns:
        Layer name to the first 8 hex digits of a SHA-256 over its backend module, source hash and upstream keys.

    Examples:
        >>> from bga.config import RunConfig
        >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold"}})
        >>> keys = code_keys(cfg)
        >>> list(keys), len(keys["shots"])
        (['shots'], 8)

    """
    keys: dict[str, str] = {}
    for layer in LAYERS:  # upstream layers come first
        if layer not in cfg.layers:
            continue
        module = registry.get(layer, cfg.layers[layer].backend).__module__
        ups = {u: keys[u] for u in LAYERS[layer] if u in keys}
        blob = json.dumps({"module": module, "source": source_hash(module), "upstream": ups}, sort_keys=True)
        keys[layer] = hashlib.sha256(blob.encode()).hexdigest()[:8]
    return keys


def stale_sequences(cfg: RunConfig, runs: Path, seqs: list[Sequence], layer: str, code: str) -> list[Sequence]:
    """The sequences in ``seqs`` whose stored ``layer`` output was made by code with another key, or by unknown code.

    An output whose ``meta.json`` record has no code key (written before 2026-11-27) cannot be checked and is kept.
    An output with no record at all is kept only in a directory no run has written since then (its ``meta.json`` has
    no top-level ``code``). Otherwise the run that wrote it stopped before recording it, so it is stale.

    Args:
        cfg: The run config.
        runs: Root of all run outputs.
        seqs: Sequences to check; those without an output are skipped.
        layer: A configured layer name.
        code: The layer's current key from :func:`code_keys`.

    Returns:
        The stale sequences, in the order given.

    Examples:
        >>> import tempfile
        >>> from bga.config import RunConfig
        >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold"}})
        >>> seq = Sequence(name="clip_s2", clip="clip", video="clip.mp4", shot=2, frames=[0], fps=50.0)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = output_path(cfg, Path(tmp), seq, "shots")
        ...     path.parent.mkdir(parents=True)
        ...     _ = path.write_text("{}")
        ...     _ = (path.parent / "meta.json").write_text('{"sequences": {"clip_s2": {"code": "aaaa"}}}')
        ...     [s.name for s in stale_sequences(cfg, Path(tmp), [seq], "shots", "bbbb")]
        ['clip_s2']

    """
    metas: dict[str, dict] = {}
    stale = []
    for seq in seqs:
        if not output_path(cfg, runs, seq, layer).exists():
            continue
        if seq.clip not in metas:
            metas[seq.clip] = _read_meta(cfg.run_dir(runs, seq.clip, layer) / "meta.json")
        meta = metas[seq.clip]
        record = (meta.get("sequences") or {}).get(seq.name)
        if record is None:
            if not meta or "code" in meta:  # written by a run that stopped before recording it
                stale.append(seq)
        elif record.get("code") is not None and record["code"] != code:
            stale.append(seq)
    return stale


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
    Each file is written whole and the JSON last, so an output that exists is complete.

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
        write_atomic(path.parent / file, lambda fh, arr=arr: np.save(fh, arr))
        records[name] = {"file": file, "shape": list(arr.shape), "dtype": str(arr.dtype)}
    stored = {**out, "arrays": records} if arrays else out
    write_atomic(path, json.dumps(stored, separators=(",", ":")))  # last: the JSON marks the output as finished


def _load_upstreams(cfg: RunConfig, runs: Path, seq: Sequence, layer: str) -> dict[str, dict]:
    """Load every configured upstream output of ``layer`` for ``seq``, with arrays."""
    return {u: load_output(cfg, runs, seq, u, arrays=True) for u in LAYERS[layer] if u in cfg.layers and u != "shots"}


def _run_local_sequences(
    cfg: RunConfig, runs: Path, layer: str, backend: Backend, todo: list[Sequence], done: list[Sequence]
) -> None:
    """Run a sequence-scoped backend on every sequence in ``todo``, write its output and append it to ``done``."""
    for seq in todo:
        ups = _load_upstreams(cfg, runs, seq, layer)
        read = FrameReader(seq.video)
        try:
            t = time.perf_counter()
            out = backend.run(seq, ups, read)
            out.setdefault("wall_s", round(time.perf_counter() - t, 2))
        finally:
            read.close()
        write_output(output_path(cfg, runs, seq, layer), out)
        done.append(seq)
        log.info("  %s: %.1f s", seq.name, out["wall_s"])


def _run_local_clips(
    cfg: RunConfig,
    runs: Path,
    layer: str,
    backend: Backend,
    seqs: list[Sequence],
    todo: list[Sequence],
    done: list[Sequence],
) -> None:
    """Run a clip-scoped backend once per clip with a sequence in ``todo``; write and record every shot of it.

    Each shot written is appended to ``done``.

    """
    clips = {s.clip for s in todo}
    by_clip: dict[str, list[Sequence]] = {}
    for seq in seqs:
        if seq.clip in clips:
            by_clip.setdefault(seq.clip, []).append(seq)
    for clip, clip_seqs in by_clip.items():
        clip_ups = {s.name: _load_upstreams(cfg, runs, s, layer) for s in clip_seqs}
        t = time.perf_counter()
        outs = backend.run_clip(clip_seqs, clip_ups)
        wall = round(time.perf_counter() - t, 2)
        for seq in clip_seqs:
            out = outs[seq.name]
            out.setdefault("wall_s", wall)
            write_output(output_path(cfg, runs, seq, layer), out)
            done.append(seq)
        log.info("  %s (%d shots): %.1f s", clip, len(clip_seqs), wall)


def _todo(
    cfg: RunConfig, runs: Path, seqs: list[Sequence], layer: str, keys: dict[str, str], force: list[str] | None
) -> list[Sequence]:
    """The sequences a layer must run on: forced, missing, or made by other code.

    Raises:
        RuntimeError: When an upstream output those sequences read was made by other code and this run did not remake
            it (its layer was skipped or not selected), so the layer would read stale inputs under a current key.

    """
    todo = [s for s in seqs if (force and layer in force) or not output_path(cfg, runs, s, layer).exists()]
    stale = stale_sequences(cfg, runs, [s for s in seqs if s not in todo], layer, keys[layer])
    if stale:
        log.warning("%s: %d cached outputs were made by other code; recomputing", layer, len(stale))
        todo = [s for s in seqs if s in todo or s in stale]
    for up in LAYERS[layer]:
        if up in cfg.layers and up != "shots" and (bad := stale_sequences(cfg, runs, todo, up, keys[up])):
            raise RuntimeError(
                f"{layer} would read {len(bad)} {up} outputs made by other code (e.g. {bad[0].name}); "
                f"rerun {up} first (CUDA backends need --remote modal)"
            )
    return todo


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

    The shots layer only selects sequences. A layer runs on the sequences without an output, or whose output was
    made by code with another key (:func:`stale_sequences`; all of them when ``force`` names it), and then writes
    ``meta.json`` per clip. Backends with ``scope == "clip"`` run once per clip
    across every selected sequence of that clip. With ``remote``, CUDA backends and ``remote_layers`` run on Modal;
    without it, CUDA backends are skipped with a warning.

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
    keys = code_keys(cfg)
    for layer in LAYERS:
        if layer == "shots" or layer not in cfg.layers or (layers and layer not in layers):
            continue
        backend = make_backend(cfg, layer)
        here = "modal" if remote and (backend.requires == "cuda" or layer in (remote_layers or ())) else None
        if backend.requires == "cuda" and not here:
            log.warning("%s/%s needs CUDA: run it with --remote modal; skipped", layer, cfg.layers[layer].backend)
            continue
        todo = _todo(cfg, runs, seqs, layer, keys, force)
        if not todo:
            continue
        log.info("%s/%s on %d sequences%s", layer, cfg.layers[layer].backend, len(todo), " (modal)" if here else "")
        t0 = time.perf_counter()
        if here:
            gpu = getattr(backend, "gpu", None) if backend.requires == "cuda" else remote_gpu
            _run_remote(cfg, runs, layer, todo, gpu, keys[layer])
            continue
        done: list[Sequence] = []
        backend.load()
        try:
            if getattr(backend, "scope", "sequence") == "clip":
                _run_local_clips(cfg, runs, layer, backend, seqs, todo, done)
            else:
                _run_local_sequences(cfg, runs, layer, backend, todo, done)
        finally:
            backend.unload()
            if done:  # also when a sequence fails or the run is stopped, so finished outputs keep their code key
                _write_meta(cfg, runs, layer, done, t0, keys[layer])
    return seqs


def _run_remote(cfg: RunConfig, runs: Path, layer: str, todo: list[Sequence], gpu: str | None, code: str) -> None:
    """Run one layer on Modal, writing each sequence's output as its call finishes, and the meta of those written."""
    from bga.remote.modal_runner import run_layer

    t0 = time.perf_counter()
    reads = [u for u in LAYERS[layer] if u in cfg.layers and u != "shots"]  # shots only selects sequences
    ups = [{u: load_output(cfg, runs, s, u, arrays=True) for u in reads} for s in todo]
    done: list[Sequence] = []

    def keep(i: int, out: dict) -> None:
        write_output(output_path(cfg, runs, todo[i], layer), out)
        done.append(todo[i])

    try:
        run_layer(layer, cfg.layers[layer].model_dump(), todo, ups, gpu, on_result=keep)
    finally:
        if done:
            _write_meta(cfg, runs, layer, done, t0, code)


def sequence_record(out: dict, commit: str, config: str, written: str, code: str | None = None) -> dict:
    """One sequence's record in ``meta.json``: which code and config made its output, when, where and how long.

    Args:
        out: The sequence's stored output (its ``wall_s`` and, for a remote call, its ``remote`` cost record).
        commit: Commit that made it.
        config: Config name of the run.
        written: ISO time the output was written.
        code: The layer's code key (:func:`code_keys`) when the output was made.

    Returns:
        ``{"commit", "config", "written", "wall_s", "platform", "code"}``, with ``gpu``, ``seconds`` and ``usd`` for a
        remote call; ``platform`` is ``local`` otherwise.

    Examples:
        >>> sequence_record({"wall_s": 1.5}, "abc1234", "demo", "2026-11-22T20:00:00")["platform"]
        'local'
        >>> remote = {"platform": "modal", "gpu": "L4", "seconds": 9.0, "usd": 0.002}
        >>> rec = sequence_record({"remote": remote}, "abc1234", "demo", "2026-11-22T20:00:00")
        >>> rec["platform"], rec["gpu"], rec["usd"], rec["wall_s"]
        ('modal', 'L4', 0.002, None)

    """
    rec = {"commit": commit, "config": config, "written": written, "wall_s": out.get("wall_s"), "platform": "local"}
    rec["code"] = code
    remote = out.get("remote")
    if remote:
        rec.update({k: remote.get(k) for k in ("platform", "gpu", "seconds", "usd")})
    return rec


def remote_summary(records: dict[str, dict]) -> dict | None:
    """Total remote time and cost over a directory's sequence records, for the top of ``meta.json``.

    Args:
        records: Sequence name to its :func:`sequence_record`.

    Returns:
        ``{"platform", "gpu", "seconds", "usd"}`` over the sequences made remotely (``gpu`` joins the GPU types used,
        ``cpu`` for none), or ``None`` when every sequence was made locally.

    Examples:
        >>> local = {"platform": "local"}
        >>> l4 = {"platform": "modal", "gpu": "L4", "seconds": 9.0, "usd": 0.002}
        >>> remote_summary({"s1": local, "s2": l4, "s3": l4})
        {'platform': 'modal', 'gpu': 'L4', 'seconds': 18.0, 'usd': 0.004}
        >>> remote_summary({"s1": local}) is None
        True

    """
    remote = [r for r in records.values() if r.get("platform") != "local"]
    if not remote:
        return None
    return {
        "platform": remote[0]["platform"],
        "gpu": ", ".join(sorted({r.get("gpu") or "cpu" for r in remote})),
        "seconds": round(sum(r.get("seconds") or 0.0 for r in remote), 1),
        "usd": round(sum(r.get("usd") or 0.0 for r in remote), 4),
    }


def _read_meta(path: Path) -> dict:
    """An existing ``meta.json``, or ``{}`` when it is absent or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_meta(cfg: RunConfig, runs: Path, layer: str, todo: list[Sequence], t0: float, code: str) -> None:
    """Record a run in each clip's ``meta.json``: the run's own fields, and the records of the sequences it wrote.

    Records of sequences this run did not write are kept.

    """
    commit, written = git_commit(), dt.datetime.now().isoformat(timespec="seconds")
    for clip in {s.clip for s in todo}:
        path = cfg.run_dir(runs, clip, layer) / "meta.json"
        records = dict(_read_meta(path).get("sequences") or {})
        for s in todo:
            if s.clip == clip:
                out = load_output(cfg, runs, s, layer)
                records[s.name] = sequence_record(out, commit, cfg.name, written, code)
        meta = {
            "layer": layer,
            "backend": cfg.layers[layer].backend,
            "layer_hash": cfg.layer_hash(layer),
            "entry": cfg.layers[layer].model_dump(),
            "upstream": {u: cfg.layer_hash(u) for u in LAYERS[layer] if u in cfg.layers},
            "config": cfg.name,
            "commit": commit,
            "code": code,
            "wall_s": round(time.perf_counter() - t0, 1),
            "sequences": dict(sorted(records.items())),
        }
        summary = remote_summary(records)
        if summary:
            meta["remote"] = summary
        write_atomic(path, json.dumps(meta, indent=1))
