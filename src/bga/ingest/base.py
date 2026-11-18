"""Layer 0 (shots): which sequences a run covers.

Output: ``{"sequences": [Sequence fields]}``.

"""

from bga.stage import Backend as ShotsBackend  # noqa: F401  # re-exported: the shots layer's backend protocol
