"""Layer 4 (track). Output per sequence:

    {"frames": {"<frame>": [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...]}}

``det_index`` points into the detect output's box list for that frame (-1 when the box is predicted).
"""

from bga.stage import Backend as TrackBackend  # noqa: F401
