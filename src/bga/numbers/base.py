"""Layer 5 (numbers). Output per sequence:

    {"tracks": {"<track_id>": "23" | null}, "reads": int, "ms_per_read": float}

A track's number is confirmed by the backend's own rule (for Skalski's stack, N consecutive equal
reads); null means no confirmed number.
"""

from bga.stage import Backend as NumbersBackend  # noqa: F401
