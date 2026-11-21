"""Layer 5 (numbers).

Output per sequence:
    {"tracks": {"<track_id>": "23" | null}, "log": [[frame, track_id, "23" | null], ...], "reads": int,
     "ms_per_read": float}

A track's number is confirmed by the backend's own rule (for Skalski's stack, N consecutive equal
reads); null means no confirmed number. ``log`` lists every read in order; outputs written before 2026-11-21
lack it.

"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from bga.stage import Backend as NumbersBackend  # noqa: F401  # re-exported: the numbers layer's backend protocol

#: Folder of the PLK roster files (``scripts/fetch_rosters.py``).
ROSTERS = Path("data/rosters")


def roster_numbers(meta: dict, side: str, rosters: Path = ROSTERS) -> set[str]:
    """Jersey numbers on one side's roster for the clip's season.

    Args:
        meta: A sequence's clip metadata: ``season`` and ``home``/``away`` with a ``roster`` file stem.
        side: ``home`` or ``away``.
        rosters: Folder of roster files.

    Returns:
        The numbers, or an empty set when the clip names no roster or season for that side.

    Examples:
        >>> meta = {"season": "2025/26", "home": {"roster": "dziki_warszawa"}}
        >>> "2" in roster_numbers(meta, "home"), roster_numbers(meta, "away")
        (True, set())

    """
    stem, season = (meta.get(side) or {}).get("roster"), meta.get("season")
    if not stem or not season or not (rosters / f"{stem}.json").exists():
        return set()
    players = json.loads((rosters / f"{stem}.json").read_text())["seasons"].get(season, {}).get("players", [])
    return {p["number"] for p in players}


def confirm_consecutive(values: Iterable[str | None], n: int) -> str | None:
    """The first value read ``n`` times in a row, skipping ``None`` reads (``ConsecutiveValueTracker``'s rule).

    Args:
        values: Reads in time order.
        n: Equal reads in a row needed.

    Returns:
        The confirmed value, or ``None``.

    Examples:
        >>> confirm_consecutive(["2", None, "2", "8", "8", "8", "2"], 3)
        '8'
        >>> confirm_consecutive(["2", "8", "2"], 2) is None
        True

    """
    last, streak = None, 0
    for v in values:
        if v is None:
            continue
        streak = streak + 1 if v == last else 1
        last = v
        if streak >= n:
            return v
    return None
