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
from collections.abc import Callable
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
#: Runtime extras installed in the runner image: every layer's backends, re-ID and the Modal client itself.
RUNNER_EXTRAS = ["prefill", "demo", "court", "track", "remote"]
#: Trained weights come from the volume (uploaded once per weight id). Model downloads go to the container's own disk:
#: ``inference_models`` and Hugging Face lock their caches with file locks, which a Modal Volume does not support
#: (the lock timed out on 2026-11-20), so each container downloads its models once on start.
VOLUME_ENV = {"HF_HOME": "/root/cache/hf", "MODEL_CACHE_DIR": "/root/cache/roboflow", "BGA_WEIGHTS": "/data/weights"}
#: facebookresearch/sam2 commit in the runner image (the one the gold prefill used, ``bga.remote.sam2_prefill``).
SAM2_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
#: SAM 2.1 weights baked into the runner image (Apache 2.0) and their checkpoint file.
SAM2_MODEL_ID, SAM2_CKPT = "facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"
#: Container image built from the repository's ``uv.lock``, so every layer runs the Mac's exact versions; the local
#: ``bga`` package and the court geometry file (read by ``court/rfdetr_keypoint`` from the repository root) go in last.
image = (
    # Modal's ``debian_slim`` is bullseye, whose apt archives are gone; bookworm still installs git
    modal.Image
    .from_registry("python:3.12-slim-bookworm")
    .apt_install("git")  # uv fetches the ``sports`` git dependency
    .uv_sync(extras=RUNNER_EXTRAS, frozen=True)
    .env(VOLUME_ENV)
    # SAM 2.1 for track/sam2, at the commit the gold prefill used, built without its CUDA extension; weights baked in
    .uv_pip_install(
        f"sam-2 @ https://github.com/facebookresearch/sam2/archive/{SAM2_SHA}.tar.gz",
        extra_options="--no-build-isolation",
        env={"SAM2_BUILD_CUDA": "0"},
    )
    .run_commands(
        f"python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('{SAM2_MODEL_ID}', '{SAM2_CKPT}')\""
    )
    .add_local_file("tools/labeler/static/fiba_court.json", "/tools/labeler/static/fiba_court.json")
    .add_local_python_source("bga")
)


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("roboflow")], timeout=3600)
def run_sequence(layer: str, entry: dict, seq: dict, upstream: dict, gpu: str | None, job: int = 0) -> dict:
    """Run one layer's backend on one sequence inside Modal, reading the clip from the volume.

    Args:
        layer: the layer name.
        entry: the layer's config entry (``backend``, ``params``, ``weights``).
        seq: the sequence as a dict (``dataclasses.asdict`` of ``bga.stage.Sequence``).
        upstream: upstream layer outputs for the sequence.
        gpu: the Modal GPU type the call runs on, or None for CPU; prices the run.
        job: the call's index in its batch, returned so results can arrive in any order.

    Returns:
        The backend's output with ``remote``: platform, GPU, seconds, list-price USD and ``job``.

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
        "job": job,
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


def upload_weights(weight_id: str | None) -> None:
    """Upload a trained weight id's ``train.json`` and checkpoint files to ``weights/<id>/`` unless the volume has them.

    Pretrained ids (``pretrained:...``) and ``None`` need nothing: their backends download from the source.

    Args:
        weight_id: A config's ``weights`` field.

    Examples:
        >>> upload_weights("detect/rfdetr_torch/plk-r3-poss@c62744fe")  # doctest: +SKIP

    """
    if not weight_id or weight_id.startswith("pretrained:"):
        return
    from bga.paths import weights_dir

    local = weights_dir(weight_id)
    names = ("train.json", "checkpoint_best_total.pth", "weights.safetensors", "reid_config.json")
    files = [f for f in names if (local / f).exists()]
    if not files:
        raise FileNotFoundError(f"no weights under {local}")
    if _exists(f"weights/{weight_id}"):
        have = {Path(e.path).name for e in vol.listdir(f"weights/{weight_id}")}
        files = [f for f in files if f not in have]
    if files:
        with vol.batch_upload() as up:
            for f in files:
                up.put_file(str(local / f), f"weights/{weight_id}/{f}")


def _exists(path: str) -> bool:
    """Whether a path exists on the volume."""
    try:
        vol.listdir(path)
        return True
    except Exception:
        return False


class RemoteCallsFailedError(RuntimeError):
    """Some calls of a remote batch failed; the others finished and were handed over."""


def run_layer(
    layer: str,
    entry: dict | list[dict],
    seqs: list,
    upstream: list[dict],
    gpu: str | None,
    on_result: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Run one layer for ``seqs`` (with their upstream outputs) on Modal, one parallel call per sequence.

    Results are taken as they finish and handed to ``on_result`` at once, so a run stopped part way keeps what had
    finished; the cost rows of the finished calls are logged even then. A failed call does not stop the others.

    Args:
        layer: the layer name.
        entry: the layer's config entry (``backend``, ``params``, ``weights``), or one entry per sequence, so a
            parameter sweep sends every setting and sequence in one parallel batch.
        seqs: the sequences (``bga.stage.Sequence``).
        upstream: upstream outputs, one per sequence.
        gpu: Modal GPU type, or None for CPU.
        on_result: Called with each finished call's index in ``seqs`` and its output, e.g. to write it.

    Returns:
        The outputs in the order of ``seqs``; their costs are appended to docs/costs.md.

    Raises:
        RemoteCallsFailedError: After the batch, when any call failed; the finished ones were handed over.

    Examples:
        >>> outs = run_layer("court", {"backend": "roboflow_keypoints"}, seqs, [{}] * len(seqs), None)  # doctest: +SKIP

    """
    entries = entry if isinstance(entry, list) else [entry] * len(seqs)
    fn = run_sequence.with_options(gpu=gpu) if gpu else run_sequence
    outs: list[dict | None] = [None] * len(seqs)
    try:
        with modal.enable_output(), app.run():
            upload_clips({s.video for s in seqs})
            for w in {e.get("weights") for e in entries}:
                upload_weights(w)
            jobs = zip(entries, seqs, upstream, strict=True)
            args = [(layer, e, asdict(s), u, gpu, i) for i, (e, s, u) in enumerate(jobs)]
            failed = []
            for out in fn.starmap(args, order_outputs=False, return_exceptions=True):
                if isinstance(out, BaseException):  # one failed call must not cancel the batch
                    failed.append(out)
                    continue
                i = out["remote"].pop("job")
                outs[i] = out
                if on_result:
                    on_result(i, out)
    finally:
        done = [i for i, o in enumerate(outs) if o is not None]
        log_costs(layer, entries[0]["backend"], [seqs[i] for i in done], [outs[i] for i in done])
    if failed:
        raise RemoteCallsFailedError(f"{len(failed)} of {len(seqs)} {layer} calls failed; run again to retry them")
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
