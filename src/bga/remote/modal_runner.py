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

USD_PER_HOUR = {
    None: 0.047,
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-80GB": 2.50,
    "H100": 3.95,
}
COSTS_MD = Path("docs/costs.md")

app = modal.App("bga-runner")
vol = modal.Volume.from_name("bga-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
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
    have = {Path(e.path).name for e in vol.listdir("clips")} if _exists("clips") else set()
    missing = [v for v in videos if Path(v).name not in have]
    if missing:
        with vol.batch_upload() as up:
            for v in missing:
                up.put_file(v, f"clips/{Path(v).name}")


def _exists(path: str) -> bool:
    try:
        vol.listdir(path)
        return True
    except Exception:
        return False


def run_layer(layer: str, entry: dict, seqs: list, upstream: list[dict], gpu: str | None) -> list[dict]:
    """Run one layer for ``seqs`` (with their upstream outputs) on Modal; returns outputs in order."""
    fn = run_sequence.with_options(gpu=gpu) if gpu else run_sequence
    with modal.enable_output(), app.run():
        upload_clips({s.video for s in seqs})
        args = [(layer, entry, asdict(s), u, gpu) for s, u in zip(seqs, upstream, strict=True)]
        outs = list(fn.starmap(args))
    log_costs(layer, entry["backend"], seqs, outs)
    return outs


def log_costs(layer: str, backend: str, seqs: list, outs: list[dict]) -> None:
    if not COSTS_MD.exists():
        return
    rows = [
        f"| {dt.date.today().isoformat()} | modal | {o['remote']['gpu'] or 'cpu'} "
        f"| bga run {layer}/{backend} {s.name} | {o['remote']['seconds']} | {o['remote']['usd']} |"
        for s, o in zip(seqs, outs, strict=True)
    ]
    lines = COSTS_MD.read_text().rstrip("\n").split("\n")
    last_row = max(i for i, line in enumerate(lines) if line.startswith("|"))  # rows go above any summary
    COSTS_MD.write_text("\n".join(lines[: last_row + 1] + rows + lines[last_row + 1 :]) + "\n")


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(USD_PER_HOUR))
