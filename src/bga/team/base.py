"""Layer 3 (team).

Output per sequence, aligned with the detect output's boxes:
    {"frames": {"<frame>": ["home" | "away" | "ref" | "", ...]}}

"" marks boxes that are not people. Team per track is the majority over a track's boxes (eval/export).

"""

from bga.stage import Backend as TeamBackend  # noqa: F401  # re-exported: the team layer's backend protocol

#: Team label emitted by Layer 3 for referee detections.
REF_TEAM = "ref"
