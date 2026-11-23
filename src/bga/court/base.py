"""Layer 1 (court).

Output per sequence:
    {"frames": {"<frame>": [[h00, h01, h02], [h10, ...], [h20, ...]] | null}, "units": "cm"}

Each matrix maps image pixels to the FIBA court plane in centimetres (sports CourtConfiguration).

"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from bga.stage import Backend as CourtBackend  # noqa: F401  # re-exported: the court layer's backend protocol

#: Canonical FIBA court geometry JSON (33 landmarks in cm, edges, polylines), packaged inside ``bga.court``.
FIBA_COURT_PATH = Path(__file__).with_name("fiba_court.json")


@lru_cache(maxsize=1)
def load_fiba_court() -> dict[str, Any]:
    """Load the packaged FIBA court geometry (vertices in cm, labels, edges, polylines).

    Returns:
        Parsed JSON dict from :data:`FIBA_COURT_PATH`.

    Examples:
        >>> geo = load_fiba_court()
        >>> geo["units"], len(geo["vertices"]), len(geo["labels"])
        ('cm', 33, 33)

    """
    return json.loads(FIBA_COURT_PATH.read_text(encoding="utf-8"))
