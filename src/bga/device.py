"""Torch device choice shared by the backends: one config runs on the Mac (MPS) and on Modal (CUDA) unchanged."""

from __future__ import annotations

__all__ = ["pick_device"]


def pick_device(name: str = "auto", cuda: bool | None = None, mps: bool | None = None) -> str:
    """Resolve a device name to one torch can use here.

    ``auto`` takes CUDA when present, then MPS, then the CPU. ``cuda`` or ``mps`` asked for where it is missing falls
    back the same way, so a config written on the Mac runs on Modal. ``cpu`` stays ``cpu``.

    Args:
        name: ``auto``, ``cuda``, ``mps`` or ``cpu``.
        cuda: Whether CUDA is available; ``None`` asks torch.
        mps: Whether MPS is available; ``None`` asks torch.

    Returns:
        ``cuda``, ``mps`` or ``cpu``.

    Raises:
        ValueError: For any other name.

    Examples:
        >>> pick_device("auto", cuda=True, mps=False)
        'cuda'
        >>> pick_device("mps", cuda=True, mps=False)
        'cuda'
        >>> pick_device("mps", cuda=False, mps=True)
        'mps'
        >>> pick_device("cpu", cuda=True, mps=True)
        'cpu'
        >>> pick_device("auto", cuda=False, mps=False)
        'cpu'

    """
    if name not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"unknown device {name!r}")
    if name == "cpu":
        return "cpu"
    if cuda is None or mps is None:
        import torch

        cuda = torch.cuda.is_available() if cuda is None else cuda
        mps = torch.backends.mps.is_available() if mps is None else mps
    if name == "mps" and mps:
        return "mps"
    if cuda:
        return "cuda"
    return "mps" if mps else "cpu"
