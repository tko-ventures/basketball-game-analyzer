"""Where trained weights live: ``weights/`` in the repository, or ``$BGA_WEIGHTS`` (the Modal volume) when set."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["weights_dir"]

#: Repository root, two levels above ``src/bga/``.
ROOT = Path(__file__).resolve().parents[2]


def weights_dir(weight_id: str) -> Path:
    """Directory of a trained weight id, ``<layer>/<backend>/<name>@<hash8>``.

    Args:
        weight_id: The weight id from a config's ``weights`` field.

    Returns:
        ``$BGA_WEIGHTS/<weight id>`` when the variable is set (remote runs), else ``<repo>/weights/<weight id>``.

    Examples:
        >>> import os
        >>> os.environ["BGA_WEIGHTS"] = "/data/weights"
        >>> str(weights_dir("detect/rfdetr_torch/plk-r3@12345678"))
        '/data/weights/detect/rfdetr_torch/plk-r3@12345678'
        >>> del os.environ["BGA_WEIGHTS"]
        >>> weights_dir("detect/rfdetr_torch/plk-r3@12345678").parts[-4:]
        ('weights', 'detect', 'rfdetr_torch', 'plk-r3@12345678')

    """
    base = os.environ.get("BGA_WEIGHTS")
    return (Path(base) if base else ROOT / "weights") / weight_id
