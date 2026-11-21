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

#: Pipeline order and the upstream layers each one reads (plan section 5, L0 to L6, with L3b appearance).
LAYERS: dict[str, tuple[str, ...]] = {
    "shots": (),
    "court": ("shots",),
    "detect": ("shots",),
    "team": ("detect",),
    "appearance": ("detect",),
    "track": ("detect", "court"),
    "numbers": ("detect", "track"),
    "positions": ("track", "court"),
}


#: Shots params that only choose sequences; kept out of the hashes so dev and held-out runs share dirs.
SELECTION_PARAMS = frozenset({"version", "held_out", "root", "labels"})


class LayerConfig(BaseModel):
    """One layer's entry in a run config: which backend runs it, with which params and weights.

    Attributes:
        backend: Registered backend name for this layer (see ``bga backends``).
        params: Backend parameters; each backend rejects names it does not declare.
        weights: ``"<layer>/<backend>/<name>@<hash8>"`` or a declared pretrained id, if the backend uses weights.

    Examples:
        >>> LayerConfig(backend="bytetrack").model_dump()
        {'backend': 'bytetrack', 'params': {}, 'weights': None}

    """

    model_config = ConfigDict(extra="forbid")
    backend: str
    params: dict[str, Any] = Field(default_factory=dict)
    weights: str | None = None  # "<layer>/<backend>/<name>@<hash8>" or a declared pretrained id


class RunConfig(BaseModel):
    """A whole run config: a name, a description and one ``LayerConfig`` per layer it runs.

    Attributes:
        name: Config name, extended with any ``--set`` overrides by ``load_config``.
        description: Free text from the YAML file.
        layers: Layer name to its entry; names must be keys of ``LAYERS``.

    Examples:
        >>> layers = {"shots": {"backend": "gold"}, "detect": {"backend": "roboflow_inference"}}
        >>> sorted(RunConfig(name="demo", layers=layers).layers)
        ['detect', 'shots']
        >>> RunConfig(name="bad", layers={"team": {"backend": "siglip_kmeans"}})
        Traceback (most recent call last):
        ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for RunConfig
        ...

    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    layers: dict[str, LayerConfig]

    def model_post_init(self, _ctx: object) -> None:
        """Check that every layer is known and that the layers it reads are configured (court is optional).

        Args:
            _ctx: Pydantic validation context, unused.

        Raises:
            ValueError: On an unknown layer name, or a layer whose upstream layer (other than court) is missing.

        Examples:
            >>> RunConfig(name="demo", layers={"shots": {"backend": "gold"}, "teams": {"backend": "x"}})
            Traceback (most recent call last):
            ...
            pydantic_core._pydantic_core.ValidationError: 1 validation error for RunConfig
              Value error, unknown layers ['teams']; known: [...]
            ...

        """
        unknown = set(self.layers) - set(LAYERS)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}; known: {list(LAYERS)}")
        for layer in self.layers:
            missing = [u for u in LAYERS[layer] if u not in self.layers and u != "court"]
            if missing:
                raise ValueError(f"layer {layer!r} needs {missing}")

    def layer_hash(self, layer: str) -> str:
        """Hash one layer's entry together with the hashes of the configured layers it reads.

        Selection-only shots params (``SELECTION_PARAMS``) are left out, so dev and held-out runs share directories.

        Args:
            layer: A configured layer name.

        Returns:
            The first 8 hex digits of a SHA-256 over the layer, its entry and its upstream hashes.

        Examples:
            >>> cfg = RunConfig(name="demo", layers={"shots": {"backend": "gold", "params": {"stride": 5}}})
            >>> cfg.layer_hash("shots")
            '9e3d5b1a'
            >>> cfg.layers["shots"].params["held_out"] = "only"
            >>> cfg.layer_hash("shots")
            '9e3d5b1a'

        """
        own = self.layers[layer].model_dump()
        if layer == "shots":  # which sequences run does not change any one sequence's output
            own["params"] = {k: v for k, v in own["params"].items() if k not in SELECTION_PARAMS}
        ups = {u: self.layer_hash(u) for u in LAYERS[layer] if u in self.layers}
        blob = json.dumps({"layer": layer, "own": own, "upstream": ups}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def config_hash(self) -> str:
        """Hash of every layer's hash: the same for dev and held-out runs of one committed config.

        Returns:
            The first 8 hex digits of a SHA-256 over all layer hashes.

        Examples:
            >>> shots = {"backend": "gold", "params": {"stride": 5}}
            >>> layers = {"shots": shots, "detect": {"backend": "roboflow_inference"}}
            >>> RunConfig(name="demo", layers=layers).config_hash()
            'fd7c09e9'

        """
        blob = json.dumps({layer: self.layer_hash(layer) for layer in self.layers}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def run_dir(self, runs: Path, clip: str, layer: str) -> Path:
        """Directory holding one layer's outputs for one clip: ``<runs>/<clip>/<layer>/<backend>@<hash8>``.

        Args:
            runs: Root of all run outputs, usually ``runs``.
            clip: Clip stem (``Sequence.clip``).
            layer: A configured layer name.

        Returns:
            The stage output directory; it is not created here.

        Examples:
            >>> shots = {"backend": "gold", "params": {"stride": 5}}
            >>> layers = {"shots": shots, "detect": {"backend": "roboflow_inference"}}
            >>> RunConfig(name="demo", layers=layers).run_dir(Path("runs"), "clip", "detect").as_posix()
            'runs/clip/detect/roboflow_inference@4f1aceae'

        """
        return runs / clip / layer / f"{self.layers[layer].backend}@{self.layer_hash(layer)}"


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML run config, applying ``--set`` style overrides.

    Args:
        path: The YAML file, e.g. ``configs/baseline.yaml``.
        overrides: Items ``layer.params.key=value`` or ``layer.backend=name``; values are parsed as YAML. When
            given, they are appended to the config name in brackets.

    Returns:
        The validated config.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "demo.yaml"
        ...     _ = path.write_text("{name: demo, layers: {shots: {backend: gold}}}")
        ...     cfg = load_config(path, ["shots.params.stride=10"])
        >>> cfg.name, cfg.layers["shots"].params
        ('demo[shots.params.stride=10]', {'stride': 10})

    """
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
