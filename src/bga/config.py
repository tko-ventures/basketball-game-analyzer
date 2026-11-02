"""Run configs (plan section 5): one backend per layer, validated with pydantic, hashed per layer.

A layer's hash covers its own entry and the hashes of the layers it reads, so swapping the number
reader leaves detection's run directory, and its cached output, untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Pipeline order and the upstream layers each one reads (plan section 5, L0 to L6).
LAYERS: dict[str, tuple[str, ...]] = {
    "shots": (),
    "court": ("shots",),
    "detect": ("shots",),
    "team": ("detect",),
    "track": ("detect", "court"),
    "positions": ("track", "court"),
}


# Shots params that only choose sequences; kept out of the hashes so dev and held-out runs share dirs.
SELECTION_PARAMS = frozenset({"version", "held_out", "root", "labels"})


class LayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str
    params: dict[str, Any] = Field(default_factory=dict)
    weights: str | None = None  # "<layer>/<backend>/<name>@<hash8>" or a declared pretrained id


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    layers: dict[str, LayerConfig]

    def model_post_init(self, _ctx) -> None:
        unknown = set(self.layers) - set(LAYERS)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}; known: {list(LAYERS)}")
        for layer in self.layers:
            missing = [u for u in LAYERS[layer] if u not in self.layers and u != "court"]
            if missing:
                raise ValueError(f"layer {layer!r} needs {missing}")

    def layer_hash(self, layer: str) -> str:
        own = self.layers[layer].model_dump()
        if layer == "shots":  # which sequences run does not change any one sequence's output
            own["params"] = {k: v for k, v in own["params"].items() if k not in SELECTION_PARAMS}
        ups = {u: self.layer_hash(u) for u in LAYERS[layer] if u in self.layers}
        blob = json.dumps({"layer": layer, "own": own, "upstream": ups}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def config_hash(self) -> str:
        """Hash of every layer's hash: the same for dev and held-out runs of one committed config."""
        blob = json.dumps({layer: self.layer_hash(layer) for layer in self.layers}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def run_dir(self, runs: Path, clip: str, layer: str) -> Path:
        return runs / clip / layer / f"{self.layers[layer].backend}@{self.layer_hash(layer)}"


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML config; ``overrides`` are ``layer.params.key=value`` or ``layer.backend=name``."""
    data = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        key, value = ov.split("=", 1)
        *parents, leaf = key.split(".")
        node = data["layers"]
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    if overrides:
        data["name"] = f"{data['name']}[{','.join(overrides)}]"
    return RunConfig(**data)
