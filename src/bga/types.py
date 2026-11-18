"""Shared value types.

Layer interfaces (``<layer>/base.py``) land with the registry in M2.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: A box as x1, y1, x2, y2 in pixels.
BBox = tuple[float, float, float, float]
#: What a detection stands for downstream: a player, a jersey number, or anything else.
Role = Literal["player", "number", "other"]

#: The 10-class schema of Skalski's basketball-player-detection-3 dataset (plan section 3).
DETECTION_CLASSES = (
    "ball",
    "ball-in-basket",
    "number",
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
    "rim",
)


def class_name_to_role(class_name: str) -> Role:
    """Map a detector class name to its role: every ``player*`` class is a player.

    Args:
        class_name: Detector class name; case and surrounding spaces are ignored, and ``None`` counts as empty.

    Returns:
        ``"player"``, ``"number"`` or ``"other"``.

    Examples:
        >>> class_name_to_role("player-jump-shot"), class_name_to_role(" Number "), class_name_to_role("referee")
        ('player', 'number', 'other')

    """
    name = (class_name or "").lower().strip()
    if name.startswith("player"):
        return "player"
    if name == "number":
        return "number"
    return "other"


@dataclass(frozen=True)
class Detection:
    """One detector box with its class and confidence.

    Examples:
        >>> Detection(xyxy=(10.0, 20.0, 50.0, 120.0), class_name="player-in-possession", score=0.9).role
        'player'

    """

    xyxy: BBox
    class_name: str
    score: float
    class_id: int = 0

    @property
    def role(self) -> Role:
        """The detection's role, from ``class_name_to_role``."""
        return class_name_to_role(self.class_name)
