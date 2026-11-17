"""Shared value types. Layer interfaces (``<layer>/base.py``) land with the registry in M2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Role = Literal["player", "number", "other"]

# The 10-class schema of Skalski's basketball-player-detection-3 dataset (plan section 3).
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
    name = (class_name or "").lower().strip()
    if name.startswith("player"):
        return "player"
    if name == "number":
        return "number"
    return "other"


@dataclass(frozen=True)
class Detection:
    xyxy: BBox
    class_name: str
    score: float
    class_id: int = 0

    @property
    def role(self) -> Role:
        return class_name_to_role(self.class_name)
