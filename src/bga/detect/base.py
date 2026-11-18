"""Layer 2 (detect).

Output per sequence:
    {"frames": {"<frame>": [[x1, y1, x2, y2, score, class_name], ...]}, "classes": [...]}

Classes follow the 10-class schema in ``bga.types.DETECTION_CLASSES``.

"""

from bga.stage import Backend as DetectBackend  # noqa: F401  # re-exported: the detection layer's backend protocol

#: Detector classes that are people (players in any action, and referees).
PERSON_CLASSES = frozenset({
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
})
