"""Run a config layer by layer over its sequences (plan section 5).

Each layer runs over every sequence before the next layer starts, so one heavy model is resident at a
time. Outputs go to ``runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json`` with a ``meta.json``;
an existing output is reused unless ``force`` names that layer, so changing one layer's config reruns
only that layer and the layers downstream of it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from bga import registry
from bga.config import LAYERS, RunConfig
from bga.stage import FrameReader, Sequence

log = logging.getLogger("bga")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def make_backend(cfg: RunConfig, layer: str):
    entry = cfg.layers[layer]
    return registry.get(layer, entry.backend)(entry.params, entry.weights)


def sequences(cfg: RunConfig, only: list[str] | None = None) -> list[Sequence]:
    seqs = make_backend(cfg, "shots").sequences()
    return [s for s in seqs if not only or s.name in only]


def output_path(cfg: RunConfig, runs: Path, seq: Sequence, layer: str) -> Path:
    return cfg.run_dir(runs, seq.clip, layer) / f"{seq.name}.json"


def load_output(cfg: RunConfig, runs: Path, seq: Sequence, layer: str) -> dict:
    return json.loads(output_path(cfg, runs, seq, layer).read_text())


def run(
    cfg: RunConfig,
    runs: Path = Path("runs"),
    only: list[str] | None = None,
    layers: list[str] | None = None,
    force: list[str] | None = None,
    remote: str | None = None,
    remote_layers: list[str] | None = None,
) -> list[Sequence]:
    seqs = sequences(cfg, only)
    for layer in LAYERS:
        if layer == "shots" or layer not in cfg.layers or (layers and layer not in layers):
            continue
        backend = make_backend(cfg, layer)
        here = "modal" if remote and (backend.requires == "cuda" or layer in (remote_layers or ())) else None
        if backend.requires == "cuda" and not here:
            log.warning(
                "%s/%s needs CUDA: run it with --remote modal; skipped", layer, cfg.layers[layer].backend
            )
            continue
        todo = [s for s in seqs if force and layer in force or not output_path(cfg, runs, s, layer).exists()]
        if not todo:
            continue
        log.info(
            "%s/%s on %d sequences%s", layer, cfg.layers[layer].backend, len(todo), " (modal)" if here else ""
        )
        t0 = time.perf_counter()
        if here:
            from bga.remote.modal_runner import run_layer

            ups = [{u: load_output(cfg, runs, s, u) for u in LAYERS[layer] if u in cfg.layers} for s in todo]
            gpu = getattr(backend, "gpu", None) if backend.requires == "cuda" else None
            for seq, out in zip(
                todo, run_layer(layer, cfg.layers[layer].model_dump(), todo, ups, gpu), strict=True
            ):
                path = output_path(cfg, runs, seq, layer)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(out, separators=(",", ":")))
            _write_meta(cfg, runs, layer, todo, t0)
            continue
        backend.load()
        try:
            for seq in todo:
                ups = {
                    u: load_output(cfg, runs, seq, u)
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
                path = output_path(cfg, runs, seq, layer)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(out, separators=(",", ":")))
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
