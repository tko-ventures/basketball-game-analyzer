"""Layer 7 (ball, possession, shots) contract and shared geometry helpers.

Every backend in ``bga.ball`` returns::

    {
        "frames": {
            "<frame>": {
                "x_m": float | None,
                "y_m": float | None,
                "ball_px": [cx, cy] | None,
                "ball_conf": float,
                "holder_track_id": int | None,
                "holder_team": str | None,
                "in_flight": bool,
            }
        },
        "shots": [
            {
                "start_frame": int,
                "rim_frame": int | None,
                "end_frame": int,
                "outcome": "made" | "missed",
                "shot_type": str,
                "shooter_track_id": int | None,
                "shooter_team": str | None,
                "x_m": float | None,
                "y_m": float | None,
                "points": int,
            }
        ],
        "rims": {"<frame>": [x1, y1, x2, y2, conf]},
        "stats": {"held_frames": int, "in_flight_frames": int, "ball_det_frames": int,
                  "shots_made": int, "shots_missed": int},
        "ms_per_frame": float,
    }

"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence as Seq
from dataclasses import dataclass
from typing import Any

import numpy as np

from bga.positions.base import COURT_M, court_scale, project
from bga.stage import Sequence
from bga.team.base import REF_TEAM

#: Left and right FIBA basket centres in metres on the 28 x 15 m court (1.575 m from each baseline).
BASKETS_M: tuple[tuple[float, float], tuple[float, float]] = ((1.575, 7.5), (26.425, 7.5))
#: FIBA three-point arc radius in metres from basket centre.
THREE_PT_ARC_M = 6.75
#: Half-width from court centreline ``y = 7.5`` to the corner 3-point straight line (``7.5 - 0.90 = 6.60 m``).
THREE_PT_CORNER_HALF_W_M = 6.60
#: Distance from each baseline to where the corner straight line meets the FIBA three-point arc.
THREE_PT_CORNER_X_M = 2.99


@dataclass(frozen=True)
class BallTrackState:
    """Per-frame ball, rim, and possession state shared by shot detection and output packing.

    Attributes:
        frames: Ordered source frame indices of the sequence.
        ball_px: Image-space ball centre ``[cx, cy] | None`` per frame.
        ball_conf: Detection confidence of the selected ball per frame (``0.0`` when undetected).
        holders: Ball holder ``track_id | None`` per frame.
        teams: Majority team map ``tid -> label`` (``"home"``, ``"away"``, ``"ref"``).
        rims: Tracked rim box ``[x1, y1, x2, y2, conf]`` per frame.

    """

    frames: list[int]
    ball_px: dict[int, list[float] | None]
    ball_conf: dict[int, float]
    holders: dict[int, int | None]
    teams: dict[int, str]
    rims: dict[int, list[float]]


def effective_fps(seq: Sequence) -> float:
    """Sequence sample rate in frames per second, accounting for frame stride.

    Args:
        seq: Sequence whose ``fps`` and ``frames`` define the sampling cadence.

    Returns:
        Effective sampled frames per second.

    Examples:
        >>> effective_fps(Sequence("s", "c", "v.mp4", 1, [0, 5, 10], 50.0))
        10.0

    """
    stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
    return float(seq.fps) / max(1, stride)


def nearest_basket(x_m: float, y_m: float) -> tuple[float, float]:
    """Return the FIBA basket centre ``(bx, by)`` closest to court point ``(x_m, y_m)``.

    Args:
        x_m: Court x coordinate in metres.
        y_m: Court y coordinate in metres.

    Returns:
        ``(bx, by)`` of the nearer basket in metres.

    Examples:
        >>> nearest_basket(4.0, 7.5)
        (1.575, 7.5)
        >>> nearest_basket(22.0, 7.5)
        (26.425, 7.5)

    """
    return BASKETS_M[0] if x_m <= COURT_M[0] / 2.0 else BASKETS_M[1]


def shot_points(x_m: float | None, y_m: float | None, basket: tuple[float, float] | None = None) -> int:
    """Classify a FIBA shot release point ``(x_m, y_m)`` as ``2`` or ``3`` points.

    Uses the official FIBA 6.75 m arc centred on the target rim plus the 0.90 m corner straight lines
    (``|y - 7.5| >= 6.60 m`` within 2.99 m of the baseline). Returns ``2`` when coordinates are missing.

    Args:
        x_m: Shooter court x in metres, or ``None``.
        y_m: Shooter court y in metres, or ``None``.
        basket: Target basket ``(bx, by)`` in metres; defaults to the basket nearest ``(x_m, y_m)``.

    Returns:
        ``3`` for a three-point attempt, ``2`` otherwise.

    Examples:
        >>> shot_points(3.0, 7.5)
        2
        >>> shot_points(9.0, 7.5)
        3
        >>> shot_points(2.0, 0.7)
        3
        >>> shot_points(None, None)
        2

    """
    if x_m is None or y_m is None or not (math.isfinite(x_m) and math.isfinite(y_m)):
        return 2
    bx, by = basket if basket is not None else nearest_basket(x_m, y_m)
    if math.hypot(x_m - bx, y_m - by) >= THREE_PT_ARC_M:
        return 3
    near_baseline = x_m <= THREE_PT_CORNER_X_M if bx < COURT_M[0] / 2.0 else x_m >= COURT_M[0] - THREE_PT_CORNER_X_M
    return 3 if near_baseline and abs(y_m - by) >= THREE_PT_CORNER_HALF_W_M else 2


def bbox_iou(a: Seq[float], b: Seq[float]) -> float:
    """Intersection-over-Union of two ``[x1, y1, x2, y2]`` boxes.

    Args:
        a: First box ``[x1, y1, x2, y2]``.
        b: Second box ``[x1, y1, x2, y2]``.

    Returns:
        IoU in ``[0.0, 1.0]``.

    Examples:
        >>> bbox_iou([0, 0, 10, 10], [0, 0, 10, 10])
        1.0
        >>> bbox_iou([0, 0, 10, 10], [20, 20, 30, 30])
        0.0

    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0.0 else 0.0


def point_to_box_dist(px: float, py: float, box: Seq[float]) -> float:
    """Euclidean distance from point ``(px, py)`` to the boundary of ``box`` (``0.0`` if inside).

    Args:
        px: Point x in pixels.
        py: Point y in pixels.
        box: Bounding box ``[x1, y1, x2, y2]``.

    Returns:
        Non-negative pixel distance.

    Examples:
        >>> point_to_box_dist(5.0, 5.0, [0.0, 0.0, 10.0, 10.0])
        0.0
        >>> point_to_box_dist(13.0, 14.0, [0.0, 0.0, 10.0, 10.0])
        5.0

    """
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    return float(math.hypot(dx, dy))


def track_teams_map(tracks_out: dict[str, Any], team_out: dict[str, Any] | None) -> dict[int, str]:
    """Majority team label per track id from ``team`` and ``track``/``link`` outputs.

    Args:
        tracks_out: Output dict of ``link`` or ``track`` (``frames`` mapping).
        team_out: Output dict of ``team`` (``frames`` mapping), or ``None``.

    Returns:
        Mapping ``track_id -> team_label`` (``"home"``, ``"away"``, ``"ref"``, etc.).

    Examples:
        >>> t_out = {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}}
        >>> track_teams_map(t_out, {"frames": {"0": ["home"]}})
        {1: 'home'}

    """
    if team_out is None:
        return {}
    votes: dict[int, Counter[str]] = {}
    for f, rows in tracks_out.get("frames", {}).items():
        labels = team_out.get("frames", {}).get(f, [])
        for row in rows:
            det_i = int(row[7]) if len(row) > 7 else -1
            if 0 <= det_i < len(labels) and labels[det_i]:
                votes.setdefault(int(row[0]), Counter())[str(labels[det_i])] += 1
    return {tid: counts.most_common(1)[0][0] for tid, counts in votes.items()}


def frame_track_boxes(tracks_out: dict[str, Any]) -> dict[int, list[tuple[int, list[float], str]]]:
    """Index per-frame active track boxes as ``frame -> [(tid, xyxy, cls), ...]``.

    Args:
        tracks_out: Output dict of ``link`` or ``track``.

    Returns:
        Mapping ``int(frame) -> list of (track_id, [x1, y1, x2, y2], cls)``.

    Examples:
        >>> frame_track_boxes({"frames": {"10": [[3, 1, 2, 11, 22, 0.9, "player", 0]]}})
        {10: [(3, [1.0, 2.0, 11.0, 22.0], 'player')]}

    """
    out: dict[int, list[tuple[int, list[float], str]]] = {}
    for f_str, rows in tracks_out.get("frames", {}).items():
        out[int(f_str)] = [(int(r[0]), [float(r[1]), float(r[2]), float(r[3]), float(r[4])], str(r[6])) for r in rows]
    return out


def extract_track_positions(positions_out: dict[str, Any] | None) -> dict[int, dict[int, tuple[float, float]]]:
    """Index ``positions`` output as ``tid -> {frame -> (x_m, y_m)}`` keeping only finite points.

    Args:
        positions_out: Output dict of the ``positions`` layer, or ``None``.

    Returns:
        Nested mapping of finite 2D court positions in metres.

    Examples:
        >>> extract_track_positions({"tracks": {"2": [[5, 10.0, 7.5]]}})
        {2: {5: (10.0, 7.5)}}

    """
    if positions_out is None:
        return {}
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for tid_s, pts in positions_out.get("tracks", {}).items():
        tid = int(tid_s)
        by_f: dict[int, tuple[float, float]] = {}
        for row in pts:
            x_m, y_m = float(row[1]), float(row[2])
            if math.isfinite(x_m) and math.isfinite(y_m):
                by_f[int(row[0])] = (x_m, y_m)
        if by_f:
            out[tid] = by_f
    return out


def is_referee_track(tid: int, cls: str, teams: dict[int, str]) -> bool:
    """Return ``True`` when a track is a referee by class or majority team label.

    Args:
        tid: Track id.
        cls: Detection class string on this frame.
        teams: Majority team map ``tid -> label`` (referee label is ``"ref"``).

    Returns:
        ``True`` if referee, ``False`` otherwise.

    Examples:
        >>> is_referee_track(1, "referee", {})
        True
        >>> is_referee_track(3, "player", {3: "ref"})
        True
        >>> is_referee_track(2, "player", {2: "home"})
        False

    """
    return cls == "referee" or teams.get(tid) == REF_TEAM


def match_box_to_track(
    box: Seq[float],
    tracks: list[tuple[int, list[float], str]],
    teams: dict[int, str],
    *,
    min_iou: float = 0.25,
    max_center_px: float = 120.0,
) -> int | None:
    """Match a ``player-in-possession`` box to the best non-referee active track.

    Prefers the highest IoU track above ``min_iou``; if none overlaps, falls back to the nearest track centre
    within ``max_center_px``.

    Args:
        box: Detection ``[x1, y1, x2, y2]``.
        tracks: Active tracks ``[(tid, xyxy, cls), ...]`` on this frame.
        teams: Majority team map ``tid -> label``.
        min_iou: Minimum IoU for direct overlap matching.
        max_center_px: Fallback centre-distance cap in pixels.

    Returns:
        Matched ``track_id`` or ``None``.

    Examples:
        >>> tracks = [(1, [0, 0, 50, 100], "player"), (2, [200, 0, 250, 100], "player")]
        >>> match_box_to_track([2, 2, 48, 98], tracks, {1: "home", 2: "away"})
        1

    """
    players = [(tid, xyxy) for tid, xyxy, cls in tracks if not is_referee_track(tid, cls, teams)]
    if not players:
        return None
    best_iou, best_tid = 0.0, None
    for tid, xyxy in players:
        ov = bbox_iou(box, xyxy)
        if ov > best_iou:
            best_iou, best_tid = ov, tid
    if best_tid is not None and best_iou >= min_iou:
        return best_tid
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    best_d, near_tid = max_center_px, None
    for tid, xyxy in players:
        d = math.hypot((xyxy[0] + xyxy[2]) / 2.0 - cx, (xyxy[1] + xyxy[3]) / 2.0 - cy)
        if d < best_d:
            best_d, near_tid = d, tid
    return near_tid


def _project_ball_px(
    px: list[float] | None, frame: int, court_out: dict[str, Any] | None
) -> tuple[float, float] | None:
    """Project a ball pixel point onto the 2D court plane when a valid homography exists."""
    if px is None or court_out is None:
        return None
    entry = court_out.get("frames", {}).get(str(frame))
    homography = entry.get("H") if isinstance(entry, dict) else entry
    if homography is None:
        return None
    pt = project(np.asarray(homography, dtype=float), np.asarray([px], dtype=float))[0] * court_scale(court_out)
    x_m, y_m = float(pt[0]), float(pt[1])
    if not (math.isfinite(x_m) and math.isfinite(y_m)):
        return None
    if not (-2.0 <= x_m <= COURT_M[0] + 2.0 and -2.0 <= y_m <= COURT_M[1] + 2.0):
        return None
    return (x_m, y_m)


def _track_pos_fallback(
    tid: int,
    frame: int,
    pos_by_tid: dict[int, dict[int, tuple[float, float]]],
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]],
    court_out: dict[str, Any] | None,
) -> tuple[float, float] | None:
    """Look up a track's 2D position from ``positions``, falling back to projecting its box bottom-centre."""
    pt = pos_by_tid.get(tid, {}).get(frame)
    if pt is not None:
        return pt
    for row_tid, xyxy, _ in tracks_by_f.get(frame, []):
        if row_tid == tid:
            return _project_ball_px([(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]], frame, court_out)
    return None


def interpolate_ball_2d(
    frames: list[int],
    holders: dict[int, int | None],
    pos_by_tid: dict[int, dict[int, tuple[float, float]]],
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]],
    ball_px: dict[int, list[float] | None],
    court_out: dict[str, Any] | None,
    *,
    shot_anchors: dict[int, tuple[float, float]] | None = None,
    max_interp_frames: int = 125,
    max_blind_frames: int = 25,
) -> dict[int, tuple[float, float] | None]:
    """Compute per-frame 2D ball court coordinates in metres.

    While held, the ball takes the holder's 2D court position. Between release and reception (or through a shot's
    target basket anchor), the 2D position is linearly interpolated across spans up to ``max_interp_frames`` when
    a ball is tracked in the gap (or up to ``max_blind_frames`` for brief blind occlusions). Longer gaps without
    ball evidence and frames outside the first/last anchors fall back only to homography-projected ``ball_px``
    (leaving coordinates ``None`` when no ball was detected).

    Args:
        frames: Ordered source frame indices of the sequence.
        holders: Mapping ``frame -> holder_track_id | None``.
        pos_by_tid: Track 2D positions from ``extract_track_positions``.
        tracks_by_f: Track boxes per frame from ``frame_track_boxes``.
        ball_px: Tracked image-space ball centre ``[cx, cy] | None`` per frame.
        court_out: Output dict of ``court`` layer, or ``None``.
        shot_anchors: Optional ``frame -> (bx, by)`` rim anchors inserted at shot rim-contact frames.
        max_interp_frames: Maximum source-frame span bridged when ball detections exist in the gap.
        max_blind_frames: Maximum source-frame span bridged when no ball detection exists in the gap.

    Returns:
        Mapping ``frame -> (x_m, y_m) | None``.

    Examples:
        >>> pos = {1: {0: (5.0, 7.5)}, 2: {10: (15.0, 7.5)}}
        >>> out = interpolate_ball_2d([0, 5, 10], {0: 1, 5: None, 10: 2}, pos, {}, {}, None)
        >>> out[5]
        (10.0, 7.5)

    """
    anchors: list[tuple[int, tuple[float, float]]] = []
    out: dict[int, tuple[float, float] | None] = dict.fromkeys(frames)
    extra = shot_anchors or {}
    for i, f in enumerate(frames):
        tid = holders.get(f)
        pt = _track_pos_fallback(tid, f, pos_by_tid, tracks_by_f, court_out) if tid is not None else None
        if pt is None and f in extra:
            pt = extra[f]
        if pt is not None:
            out[f] = pt
            anchors.append((i, pt))
    if not anchors:
        return {f: _project_ball_px(ball_px.get(f), f, court_out) for f in frames}
    for (i0, p0), (i1, p1) in itertools.pairwise(anchors):
        span = frames[i1] - frames[i0]
        if span <= 0:
            continue
        has_ball = any(ball_px.get(frames[j]) is not None for j in range(i0 + 1, i1))
        if span <= max_interp_frames and (has_ball or span <= max_blind_frames):
            for j in range(i0 + 1, i1):
                t = (frames[j] - frames[i0]) / span
                out[frames[j]] = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        else:
            for j in range(i0 + 1, i1):
                out[frames[j]] = _project_ball_px(ball_px.get(frames[j]), frames[j], court_out)
    first_i, last_i = anchors[0][0], anchors[-1][0]
    for j in range(first_i):
        out[frames[j]] = _project_ball_px(ball_px.get(frames[j]), frames[j], court_out)
    for j in range(last_i + 1, len(frames)):
        out[frames[j]] = _project_ball_px(ball_px.get(frames[j]), frames[j], court_out)
    return out


def last_holder_before(
    frames: list[int],
    idx: int,
    holders: dict[int, int | None],
    max_lookback_steps: int,
) -> tuple[int | None, int | None]:
    """Find the most recent non-``None`` holder at or before step ``idx``.

    Args:
        frames: Ordered source frame indices.
        idx: Step index in ``frames`` to search backward from.
        holders: Mapping ``frame -> holder_track_id | None``.
        max_lookback_steps: Maximum steps to look backward.

    Returns:
        ``(holder_track_id, frame)`` or ``(None, None)``.

    Examples:
        >>> last_holder_before([0, 5, 10, 15], 3, {0: 4, 5: 4, 10: None, 15: None}, 5)
        (4, 5)

    """
    lo = max(0, idx - max_lookback_steps)
    for j in range(idx, lo - 1, -1):
        tid = holders.get(frames[j])
        if tid is not None:
            return tid, frames[j]
    return None, None


def pack_output(
    state: BallTrackState,
    pos_2d: dict[int, tuple[float, float] | None],
    shots: list[dict[str, Any]],
    ms_per_frame: float,
) -> dict[str, Any]:
    """Pack per-frame ball/possession predictions and shot events into the Layer 7 output contract.

    ``in_flight`` is ``True`` only when no player holds the ball and a tracked ``ball_px`` exists on that frame.

    Args:
        state: Per-frame ball, rim, and possession state.
        pos_2d: 2D court coordinates ``(x_m, y_m) | None`` per frame.
        shots: List of shot-event dicts.
        ms_per_frame: Layer runtime in milliseconds per frame.

    Returns:
        JSON-serialisable Layer 7 output dictionary.

    Examples:
        >>> st = BallTrackState([0], {0: [500.0, 300.0]}, {0: 0.8}, {0: 1}, {1: "home"}, {})
        >>> out = pack_output(st, {0: (10.0, 7.5)}, [], 0.1)
        >>> out["frames"]["0"]["holder_track_id"], out["stats"]["held_frames"]
        (1, 1)

    """
    frame_rows: dict[str, dict[str, Any]] = {}
    held = in_flight_n = det_n = 0
    for f in state.frames:
        tid = state.holders.get(f)
        pt = pos_2d.get(f)
        px = state.ball_px.get(f)
        in_flight = tid is None and px is not None
        if tid is not None:
            held += 1
        elif in_flight:
            in_flight_n += 1
        if px is not None:
            det_n += 1
        frame_rows[str(f)] = {
            "x_m": round(pt[0], 3) if pt is not None else None,
            "y_m": round(pt[1], 3) if pt is not None else None,
            "ball_px": [round(px[0], 1), round(px[1], 1)] if px is not None else None,
            "ball_conf": round(float(state.ball_conf.get(f, 0.0)), 3),
            "holder_track_id": tid,
            "holder_team": state.teams.get(tid) if tid is not None else None,
            "in_flight": in_flight,
        }
    return {
        "frames": frame_rows,
        "shots": shots,
        "rims": {str(f): [round(v, 2) for v in box] for f, box in state.rims.items()},
        "stats": {
            "held_frames": held,
            "in_flight_frames": in_flight_n,
            "ball_det_frames": det_n,
            "shots_made": sum(1 for s in shots if s["outcome"] == "made"),
            "shots_missed": sum(1 for s in shots if s["outcome"] == "missed"),
        },
        "ms_per_frame": round(ms_per_frame, 3),
    }
