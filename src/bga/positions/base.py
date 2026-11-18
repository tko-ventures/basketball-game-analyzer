"""Layer 6 (positions).

Output per sequence:
{"tracks": {"<track_id>": [[frame, x_m, y_m], ...]}, "court_m": [28.0, 15.0]}

"""

from bga.stage import Backend as PositionsBackend  # noqa: F401  # re-exported: the positions layer's backend protocol
