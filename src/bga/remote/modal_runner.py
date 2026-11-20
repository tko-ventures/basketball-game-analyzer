"""``bga run --remote modal``: execute one layer's backend on Modal, outputs synced into local ``runs/``.

The backend code ships as the local ``bga`` package; clips go up once to the ``bga-data`` volume under
``clips/`` and stay there. Each sequence is one function call, so ``.map`` runs them in parallel.
Every call reports GPU, wall time and list-price cost; the pipeline writes them into the layer's
``meta.json`` and appends a row to docs/costs.md. Credentials come from ~/.modal.toml, never the repo.

"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict
from pathlib import Path

import modal

#: Modal list price per hour by GPU (None: CPU only), for the cost logged with every job.
USD_PER_HOUR = {
    None: 0.047,
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-80GB": 2.50,
    "H100": 3.95,
}
#: Cost per remote job, relative to the working directory (the repository root under ``bga``).
COSTS_MD = Path("docs/costs.md")

#: The Modal app every remote layer run belongs to.
app = modal.App("bga-runner")
#: Shared volume: clips under ``clips/``, training runs under ``train/``.
vol = modal.Volume.from_name("bga-data", create_if_missing=True)
#: Container image with the Mac's pinned versions and the local ``bga`` package.
image = (
    modal.Image
    .debian_slim(python_version="3.12")
    # Same versions as the Mac's uv.lock, so a layer gives the same output wherever it runs.
    .pip_install(
        "numpy==2.3.5",
        "pydantic==2.11.10",
        "pyyaml==6.0.3",
        "supervision==0.30.9",
        "torch==2.15.0",
        "torchvision==0.29.2",
    )
    # supervision pulls the GUI OpenCV, which needs libGL; the bullseye base is end-of-life, so swap in
    # the headless wheel instead of apt-installing system libraries.
    .run_commands(
        "pip uninstall -y opencv-python opencv-contrib-python",
        "pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93",
    )
    .add_local_python_source("bga")
)


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def run_sequence(layer: str, entry: dict, seq: dict, upstream: dict, gpu: str | None) -> dict:
    """Run one layer's backend on one sequence inside Modal, reading the clip from the volume.

    Args:
        layer: the layer name.
        entry: the layer's config entry (``backend``, ``params``, ``weights``).
        seq: the sequence as a dict (``dataclasses.asdict`` of ``bga.stage.Sequence``).
        upstream: upstream layer outputs for the sequence.
        gpu: the Modal GPU type the call runs on, or None for CPU; prices the run.

    Returns:
        The backend's output with ``remote``: platform, GPU, seconds and list-price USD.

    Examples:
        >>> run_sequence.remote("court", {"backend": "roboflow_keypoints"}, seq, {}, None)  # doctest: +SKIP
        {'frames': {...}, 'units': 'cm', ..., 'remote': {'platform': 'modal', 'gpu': None, ...}}

    """
    from bga import registry
    from bga.stage import FrameReader, Sequence

    t0 = time.perf_counter()
    backend = registry.get(layer, entry["backend"])(entry.get("params", {}), entry.get("weights"))
    backend.load()
    s = Sequence(**{**seq, "video": f"/data/clips/{Path(seq['video']).name}"})
    read = FrameReader(s.video)
    try:
        out = backend.run(s, upstream, read)
    finally:
        read.close()
        backend.unload()
    secs = time.perf_counter() - t0
    out["remote"] = {
        "platform": "modal",
        "gpu": gpu,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR.get(gpu, 0.0), 4),
    }
    return out


def upload_clips(videos: set[str]) -> None:
    """Upload the clips the volume does not have yet to ``clips/<file name>``.

    Args:
        videos: local video paths.

    Examples:
        >>> upload_clips({"data/clips/example.mp4"})  # doctest: +SKIP

    """
    have = {Path(e.path).name for e in vol.listdir("clips")} if _exists("clips") else set()
    missing = [v for v in videos if Path(v).name not in have]
    if missing:
        with vol.batch_upload() as up:
            for v in missing:
                up.put_file(v, f"clips/{Path(v).name}")


def _exists(path: str) -> bool:
    """Whether a path exists on the volume."""
    try:
        vol.listdir(path)
        return True
    except Exception:
        return False


def run_layer(layer: str, entry: dict, seqs: list, upstream: list[dict], gpu: str | None) -> list[dict]:
    """Run one layer for ``seqs`` (with their upstream outputs) on Modal, one parallel call per sequence.

    Args:
        layer: the layer name.
        entry: the layer's config entry (``backend``, ``params``, ``weights``).
        seqs: the sequences (``bga.stage.Sequence``).
        upstream: upstream outputs, one per sequence.
        gpu: Modal GPU type, or None for CPU.

    Returns:
        The outputs in the order of ``seqs``; their costs are appended to docs/costs.md.

    Examples:
        >>> outs = run_layer("court", {"backend": "roboflow_keypoints"}, seqs, [{}] * len(seqs), None)  # doctest: +SKIP

    """
    fn = run_sequence.with_options(gpu=gpu) if gpu else run_sequence
    with modal.enable_output(), app.run():
        upload_clips({s.video for s in seqs})
        args = [(layer, entry, asdict(s), u, gpu) for s, u in zip(seqs, upstream, strict=True)]
        outs = list(fn.starmap(args))
    log_costs(layer, entry["backend"], seqs, outs)
    return outs


def log_costs(layer: str, backend: str, seqs: list, outs: list[dict]) -> None:
    """Add a docs/costs.md row per remote sequence run, after the last table row; nothing when the file is absent.

    Args:
        layer: the layer name.
        backend: the backend name.
        seqs: the sequences (``bga.stage.Sequence``), for their names.
        outs: their outputs, each with a ``remote`` cost record.

    Examples:
        >>> log_costs("court", "roboflow_keypoints", seqs, outs)  # doctest: +SKIP

    """
    append_cost_rows([
        f"| {dt.date.today().isoformat()} | modal | {o['remote']['gpu'] or 'cpu'} "
        f"| bga run {layer}/{backend} {s.name} | {o['remote']['seconds']} | {o['remote']['usd']} |"
        for s, o in zip(seqs, outs, strict=True)
    ])


def append_cost_rows(rows: list[str], path: Path = COSTS_MD) -> None:
    r"""Insert table rows after the last table row of the cost log, above any summary; nothing when it is absent.

    Args:
        rows: Markdown table rows, each starting with ``|``.
        path: The cost log.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     log = Path(tmp) / "costs.md"
        ...     _ = log.write_text("| a |\n|---|\n| 1 |\n\nMonth to date: 1 USD.\n")
        ...     append_cost_rows(["| 2 |"], log)
        ...     log.read_text().split("\n")[2:4]
        ['| 1 |', '| 2 |']

    """
    if not path.exists():
        return
    lines = path.read_text().rstrip("\n").split("\n")
    last_row = max(i for i, line in enumerate(lines) if line.startswith("|"))  # rows go above any summary
    path.write_text("\n".join(lines[: last_row + 1] + rows + lines[last_row + 1 :]) + "\n")


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(USD_PER_HOUR))
