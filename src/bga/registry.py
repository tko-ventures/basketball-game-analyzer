"""Backend registry (plan section 5): implementations register by layer and name.

Adding an approach means a new module with ``@register("<layer>", "<name>")`` and a config entry,
never an ``if`` inside the pipeline. Backend modules are imported lazily by ``load_backends``.

"""

from __future__ import annotations

import importlib
from collections.abc import Callable

_REGISTRY: dict[tuple[str, str], type] = {}

#: Modules holding the built-in backends; importing them runs their @register decorators.
BACKEND_MODULES = (
    "bga.ingest.gold_shots",
    "bga.ingest.clip_shots",
    "bga.detect.roboflow_inference",
    "bga.detect.rfdetr_torch",
    "bga.team.siglip_kmeans",
    "bga.appearance.none",
    "bga.appearance.reid_encoder",
    "bga.track.bytetrack",
    "bga.track.roboflow_trackers",
    "bga.track.sam2_video",
    "bga.numbers.smolvlm_ocr",
    "bga.numbers.crop_cls",
    "bga.numbers.parseq",
    "bga.link.greedy",
    "bga.identity.shot",
    "bga.identity.game",
    "bga.court.roboflow_keypoints",
    "bga.court.rfdetr_keypoint",
    "bga.positions.bbox_bottom",
    "bga.positions.ankles_pose",
)


def register(layer: str, name: str) -> Callable[[type], type]:
    """Class decorator that registers a backend under ``(layer, name)`` and sets its ``layer`` and ``name``.

    Registering the same class again is a no-op; a different class under a taken key is an error.

    Args:
        layer: Layer the backend implements, a key of ``bga.config.LAYERS``.
        name: Backend name, as written in a config's ``backend:`` field.

    Returns:
        The decorator, which returns the class unchanged apart from the two attributes.

    Raises:
        ValueError: From the decorator, when another class already holds ``(layer, name)``.

    Examples:
        >>> @register("demo", "noop")
        ... class Noop:
        ...     pass
        >>> Noop.layer, Noop.name, get("demo", "noop") is Noop
        ('demo', 'noop', True)
        >>> register("demo", "noop")(type("Other", (), {}))
        Traceback (most recent call last):
        ...
        ValueError: backend demo/noop registered twice
        >>> _ = _REGISTRY.pop(("demo", "noop"))

    """

    def deco(cls: type) -> type:
        key = (layer, name)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"backend {layer}/{name} registered twice")
        cls.layer, cls.name = layer, name
        _REGISTRY[key] = cls
        return cls

    return deco


def load_backends() -> None:
    """Import every module in ``BACKEND_MODULES`` so the built-in backends are registered.

    Examples:
        >>> load_backends()
        >>> ("track", "bytetrack") in _REGISTRY
        True

    """
    for mod in BACKEND_MODULES:
        importlib.import_module(mod)


def get(layer: str, name: str) -> type:
    """Look up a registered backend class, importing built-in backends first if needed.

    Args:
        layer: Layer name.
        name: Backend name.

    Returns:
        The registered backend class.

    Raises:
        KeyError: When no backend is registered under ``(layer, name)``; lists the ones that are.

    Examples:
        >>> get("track", "bytetrack").__name__
        'ByteTrack'
        >>> get("track", "missing")
        Traceback (most recent call last):
        ...
        KeyError: "unknown backend track/missing; known: ..."

    """
    if (layer, name) not in _REGISTRY:
        load_backends()
    if (layer, name) not in _REGISTRY:
        known = sorted(n for ly, n in _REGISTRY if ly == layer)
        raise KeyError(f"unknown backend {layer}/{name}; known: {known}")
    return _REGISTRY[(layer, name)]


def list_backends() -> dict[str, list[str]]:
    """Every registered backend grouped by layer, sorted.

    Returns:
        Layer name to sorted list of registered backend names.

    Examples:
        >>> "bytetrack" in list_backends()["track"]
        True

    """
    load_backends()
    out: dict[str, list[str]] = {}
    for layer, name in sorted(_REGISTRY):
        out.setdefault(layer, []).append(name)
    return out
