"""Backend registry (plan section 5): implementations register by layer and name.

Adding an approach means a new module with ``@register("<layer>", "<name>")`` and a config entry,
never an ``if`` inside the pipeline. Backend modules are imported lazily by ``load_backends``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

_REGISTRY: dict[tuple[str, str], type] = {}

# Modules holding the built-in backends; importing them runs their @register decorators.
BACKEND_MODULES = (
    "bga.ingest.gold_shots",
    "bga.detect.roboflow_inference",
    "bga.team.siglip_kmeans",
    "bga.track.bytetrack",
    "bga.numbers.smolvlm_ocr",
    "bga.court.roboflow_keypoints",
    "bga.positions.bbox_bottom",
)


def register(layer: str, name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        key = (layer, name)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"backend {layer}/{name} registered twice")
        cls.layer, cls.name = layer, name
        _REGISTRY[key] = cls
        return cls

    return deco


def load_backends() -> None:
    for mod in BACKEND_MODULES:
        importlib.import_module(mod)


def get(layer: str, name: str) -> type:
    if (layer, name) not in _REGISTRY:
        load_backends()
    try:
        return _REGISTRY[(layer, name)]
    except KeyError:
        known = sorted(n for (lay, n) in _REGISTRY if lay == layer)
        raise KeyError(f"no backend {name!r} for layer {layer!r}; known: {known}") from None


def available() -> dict[str, list[str]]:
    load_backends()
    out: dict[str, list[str]] = {}
    for layer, name in sorted(_REGISTRY):
        out.setdefault(layer, []).append(name)
    return out
