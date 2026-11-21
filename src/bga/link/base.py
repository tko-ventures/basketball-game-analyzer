"""Layer 4b (link).

Output per sequence, in the ``track`` layer's format with linked tracklets under one id:
    {"frames": {"<frame>": [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...]},
     "splits": [[tracker_id, frame, new_id, distance], ...],
     "links": [[earlier_id, later_id, cost, gap_s], ...], "ids": {"<piece id>": linked id},
     "numbers": {"<linked id>": "23" | null}}

A tracklet cut at ``frame`` continues under ``new_id``, above every tracker id. A chain of linked pieces takes the id
of its earliest piece. ``numbers`` holds each linked track's jersey number, all null without a numbers layer.
``bga eval`` scores this layer instead of ``track`` when a config has it, numbers included.

"""

from bga.stage import Backend as LinkBackend  # noqa: F401  # re-exported: the link layer's backend protocol
