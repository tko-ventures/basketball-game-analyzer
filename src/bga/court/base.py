"""Layer 1 (court). Output per sequence:

    {"frames": {"<frame>": [[h00, h01, h02], [h10, ...], [h20, ...]] | null}, "units": "cm"}

Each matrix maps image pixels to the FIBA court plane in centimetres (sports CourtConfiguration).
"""

from bga.stage import Backend as CourtBackend  # noqa: F401
