"""Where trained weights live (``weights/``, or ``$BGA_WEIGHTS`` on the Modal volume), and writing files whole."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

__all__ = ["weights_dir", "write_atomic"]

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


def write_atomic(path: Path, data: str | bytes | Callable[[BinaryIO], object]) -> None:
    """Write a file so that it is either absent or complete: to a sibling temp file first, then renamed over ``path``.

    A process stopped part way leaves only ``<name>.tmp``, which no reader takes for a finished output.

    Args:
        path: The file to write; its directory must exist.
        data: The text or bytes, or a function that writes to the binary file handle it is given (e.g. ``np.save``).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     write_atomic(Path(tmp) / "a.json", "{}")
        ...     write_atomic(Path(tmp) / "b.bin", lambda fh: fh.write(b"xy"))
        ...     sorted(p.name for p in Path(tmp).iterdir()), (Path(tmp) / "b.bin").read_bytes()
        (['a.json', 'b.bin'], b'xy')

    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        if callable(data):
            data(fh)
        else:
            fh.write(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
