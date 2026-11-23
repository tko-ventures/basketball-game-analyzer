"""``positions/ankles_pose``: ankle midpoint from COCO pose with airborne hold, falling back to box bottom-centre.

Implements layer 6 from ``docs/PLAN.md``:
1. Run pretrained COCO 17-keypoint RF-DETR Keypoint (Apache 2.0) per frame and match pose boxes to active track
   boxes (``link`` when configured, else ``track``) by IoU via Hungarian assignment.
2. Take the midpoint of both confident ankles (COCO keypoints 15, 16), the single confident ankle if only one is
   visible, or fall back to box bottom-centre ``((x1 + x2) / 2, y2)`` when neither ankle is visible.
3. Detect brief upward jumps of the lower ankle relative to the straight chord connecting takeoff and landing
   floor frames (invariant to constant-velocity running and smooth camera pans) and hold the last floor court
   position while airborne before ``clean_paths`` smoothing.

"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from bga.device import pick_device
from bga.positions.base import COURT_M, active_tracks, clean_track_paths, court_scale, project
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

#: COCO 17-keypoint indices for left and right ankles.
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix between ``(M, 4)`` and ``(N, 4)`` ``xyxy`` boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0][None], boxes_b[:, 1][None], boxes_b[:, 2][None], boxes_b[:, 3][None]
    inter = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1)) * np.maximum(
        0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1)
    )
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return np.where(denom > 0.0, inter / np.maximum(denom, 1e-9), 0.0)


def match_poses(track_boxes: np.ndarray, pose_boxes: np.ndarray, min_iou: float = 0.3) -> dict[int, int]:
    """Match track boxes to pose person boxes by one-to-one maximum-IoU assignment.

    Args:
        track_boxes: ``(M, 4)`` array of ``xyxy`` track boxes.
        pose_boxes: ``(N, 4)`` array of ``xyxy`` pose detection boxes.
        min_iou: Minimum IoU required to accept a match.

    Returns:
        Mapping ``track_index -> pose_index`` for every matched pair with ``IoU >= min_iou``.

    Examples:
        >>> match_poses(np.array([[0, 0, 10, 20]]), np.array([[1, 0, 11, 20]]))
        {0: 0}
        >>> match_poses(np.array([[0, 0, 10, 20]]), np.array([[50, 50, 60, 70]]))
        {}

    """
    iou = _iou_matrix(np.asarray(track_boxes, dtype=np.float64), np.asarray(pose_boxes, dtype=np.float64))
    if iou.size == 0:
        return {}
    rows, cols = linear_sum_assignment(-iou)
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou}


def ankle_ground_point(
    box: np.ndarray | list[float],
    kp_xy: np.ndarray | None,
    kp_conf: np.ndarray | None,
    ankle_conf: float = 0.4,
) -> tuple[float, float, float | None, str]:
    """Pick a player's image ground point from pose ankles, falling back to box bottom-centre.

    Args:
        box: ``[x1, y1, x2, y2]`` track box.
        kp_xy: ``(17, 2)`` COCO keypoints for the matched pose, or ``None`` when unmatched.
        kp_conf: ``(17,)`` keypoint confidences, or ``None`` when unmatched.
        ankle_conf: Minimum confidence to trust an ankle keypoint.

    Returns:
        Tuple ``(fx, fy, lower_ankle_y, source)`` where ``lower_ankle_y`` is the larger (closer to floor) ankle
        ``y`` when at least one ankle is trusted else ``None``, and ``source`` is ``"both_ankles"``,
        ``"one_ankle"``, or ``"bbox_bottom"``.

    Examples:
        >>> xy = np.zeros((17, 2)); xy[15] = [10, 80]; xy[16] = [20, 90]
        >>> conf = np.zeros(17); conf[15] = 0.9; conf[16] = 0.8
        >>> ankle_ground_point([0, 0, 30, 100], xy, conf)
        (15.0, 85.0, 90.0, 'both_ankles')
        >>> ankle_ground_point([0, 0, 30, 100], None, None)
        (15.0, 100.0, None, 'bbox_bottom')

    """
    if kp_xy is not None and kp_conf is not None and len(kp_xy) > RIGHT_ANKLE:
        ok_l = float(kp_conf[LEFT_ANKLE]) >= ankle_conf and bool(np.all(np.isfinite(kp_xy[LEFT_ANKLE])))
        ok_r = float(kp_conf[RIGHT_ANKLE]) >= ankle_conf and bool(np.all(np.isfinite(kp_xy[RIGHT_ANKLE])))
        if ok_l and ok_r:
            xl, yl = float(kp_xy[LEFT_ANKLE, 0]), float(kp_xy[LEFT_ANKLE, 1])
            xr, yr = float(kp_xy[RIGHT_ANKLE, 0]), float(kp_xy[RIGHT_ANKLE, 1])
            return (xl + xr) / 2.0, (yl + yr) / 2.0, max(yl, yr), "both_ankles"
        if ok_l:
            xl, yl = float(kp_xy[LEFT_ANKLE, 0]), float(kp_xy[LEFT_ANKLE, 1])
            return xl, yl, yl, "one_ankle"
        if ok_r:
            xr, yr = float(kp_xy[RIGHT_ANKLE, 0]), float(kp_xy[RIGHT_ANKLE, 1])
            return xr, yr, yr, "one_ankle"
    return (float(box[0]) + float(box[2])) / 2.0, float(box[3]), None, "bbox_bottom"


def detect_airborne(
    frames: np.ndarray,
    lower_y: np.ndarray,
    box_h: np.ndarray,
    fps: float,
    *,
    airborne_frac: float = 0.15,
    max_airborne_s: float = 1.0,
) -> np.ndarray:
    """Flag frames where a player's lower ankle lifts briefly above the takeoff-to-landing floor chord.

    In image coordinates ``y`` points downward, so floor frames form the upper envelope of ``lower_y(t)`` and a
    jump produces a brief dip (smaller ``y``) between a takeoff frame ``l`` and a landing frame ``r`` within
    ``max_airborne_s``. Linear motion towards or away from the camera produces zero chord residual.

    Args:
        frames: ``(T,)`` source frame numbers.
        lower_y: ``(T,)`` lower-ankle ``y`` in pixels, ``NaN`` where no ankle was observed.
        box_h: ``(T,)`` player box height in pixels.
        fps: Video frame rate.
        airborne_frac: Minimum upward lift relative to ``box_h`` to classify a frame as airborne.
        max_airborne_s: Maximum jump span ``(frames[r] - frames[l]) / fps`` in seconds.

    Returns:
        Boolean array of shape ``(T,)`` marking airborne frames.

    Examples:
        >>> frames = np.array([0, 5, 10, 15, 20, 25])
        >>> y = np.array([200.0, 200.0, 175.0, 165.0, 180.0, 200.0])
        >>> h = np.full(6, 100.0)
        >>> detect_airborne(frames, y, h, fps=50.0).tolist()
        [False, False, True, True, True, False]

    """
    n = len(frames)
    airborne = np.zeros(n, dtype=bool)
    valid = np.where(np.isfinite(lower_y) & np.isfinite(box_h) & (box_h > 0.0))[0]
    if len(valid) < 3 or fps <= 0.0:
        return airborne
    max_df = max_airborne_s * fps
    for idx_pos in range(1, len(valid) - 1):
        i = int(valid[idx_pos])
        fi, yi, hi = float(frames[i]), float(lower_y[i]), float(box_h[i])
        # Pick the best floor-anchor pair (l < i < r) spanning <= max_airborne_s with maximum chord lift at i.
        best_lift = 0.0
        for lp in range(idx_pos - 1, -1, -1):
            l_idx = int(valid[lp])
            fl = float(frames[l_idx])
            if fi - fl > max_df:
                break
            yl = float(lower_y[l_idx])
            for rp in range(idx_pos + 1, len(valid)):
                r_idx = int(valid[rp])
                fr = float(frames[r_idx])
                span = fr - fl
                if span > max_df:
                    break
                chord_y = yl + (float(lower_y[r_idx]) - yl) * ((fi - fl) / span)
                lift = (chord_y - yi) / hi
                if lift > best_lift:
                    best_lift = lift
        if best_lift >= airborne_frac:
            airborne[i] = True
    return airborne


def hold_airborne_positions(xy: np.ndarray, airborne: np.ndarray) -> np.ndarray:
    """Hold each airborne frame at the track's last non-airborne finite court position.

    Args:
        xy: ``(T, 2)`` court coordinates in metres (``NaN`` where unmapped).
        airborne: ``(T,)`` boolean mask of airborne frames.

    Returns:
        Copy of ``xy`` with airborne frames replaced by the last floor position when available.

    Examples:
        >>> xy = np.array([[1.0, 2.0], [1.2, 4.5], [1.4, 2.1]])
        >>> hold_airborne_positions(xy, np.array([False, True, False])).tolist()
        [[1.0, 2.0], [1.0, 2.0], [1.4, 2.1]]

    """
    out = xy.copy()
    last_floor: np.ndarray | None = None
    for i in range(len(out)):
        if airborne[i]:
            if last_floor is not None:
                out[i] = last_floor
        elif np.all(np.isfinite(out[i])):
            last_floor = out[i].copy()
    return out


@register("positions", "ankles_pose")
class AnklesPose(BaseBackend):
    """Positions layer backend: COCO pose ankle midpoint + airborne floor-hold + ``clean_paths`` smoothing.

    Params (``defaults``): ``pose_conf``, ``ankle_conf``, ``min_iou``, ``airborne_frac``, ``max_airborne_s``,
    ``hold_airborne``, ``clean``, ``smooth_window``, ``min_jump_m``, and ``device``.

    Examples:
        >>> AnklesPose({}).p["ankle_conf"], AnklesPose({}).p["hold_airborne"]
        (0.4, True)

    """

    defaults: ClassVar[dict[str, Any]] = {
        "pose_conf": 0.3,
        "ankle_conf": 0.4,
        "min_iou": 0.3,
        "airborne_frac": 0.15,
        "max_airborne_s": 1.0,
        "hold_airborne": True,
        "clean": True,
        "smooth_window": 9,
        "min_jump_m": 0.7,
        "device": "mps",
    }

    def load(self) -> None:
        """Load the pretrained Apache-2.0 COCO RF-DETR Keypoint model on ``device``."""
        from rfdetr import RFDETRKeypointPreview

        self.model = RFDETRKeypointPreview(num_classes=1, device=pick_device(self.p["device"]))

    def unload(self) -> None:
        """Release the pose model between pipeline layers."""
        self.model = None

    def _predict_poses(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run COCO pose on one BGR frame; return ``(boxes_xyxy, kp_xy, kp_conf)``."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kp = self.model.predict(img_rgb, threshold=float(self.p["pose_conf"]))
        if len(kp) == 0:
            return (
                np.empty((0, 4), dtype=np.float64),
                np.empty((0, 17, 2), dtype=np.float64),
                np.empty((0, 17), dtype=np.float64),
            )
        boxes = np.asarray(kp.data["xyxy"], dtype=np.float64)
        conf = np.asarray(kp.keypoint_confidence, dtype=np.float64)
        return boxes, np.asarray(kp.xy, dtype=np.float64), conf

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Estimate 2D court positions from COCO pose ankles with airborne floor-hold for ``seq``.

        Args:
            seq: Sequence to run on.
            upstream: ``link`` (when configured) or ``track``, and ``court`` outputs for ``seq``.
            read: Frame reader for ``seq.video``.

        Returns:
            ``{"tracks", "foot_px", "airborne", "court_m", "paths_left_uncleaned", "stats", "ms_per_frame"}``.

        """
        tracks = active_tracks(upstream)
        homographies = upstream["court"]["frames"]
        scale = court_scale(upstream["court"])
        ids = sorted({int(r[0]) for rows in tracks.values() for r in rows})
        if not ids:
            return {
                "tracks": {},
                "foot_px": {},
                "airborne": {},
                "court_m": list(COURT_M),
                "paths_left_uncleaned": 0,
                "stats": {"both_ankles": 0, "one_ankle": 0, "bbox_bottom": 0, "airborne_frames": 0},
                "ms_per_frame": 0.0,
            }
        col = {tid: j for j, tid in enumerate(ids)}
        n_frames, n_ids = len(seq.frames), len(ids)
        xy = np.full((n_frames, n_ids, 2), np.nan, dtype=np.float64)
        lower_y = np.full((n_frames, n_ids), np.nan, dtype=np.float64)
        box_h = np.full((n_frames, n_ids), np.nan, dtype=np.float64)
        foot_px: dict[str, list[list[float]]] = {str(tid): [] for tid in ids}
        stats = {"both_ankles": 0, "one_ankle": 0, "bbox_bottom": 0, "airborne_frames": 0}
        t0 = time.perf_counter()
        for i, f in enumerate(seq.frames):
            rows = tracks.get(str(f), [])
            if not rows:
                continue
            pose_boxes, pose_xy, pose_conf = self._predict_poses(read(f))
            t_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float64)
            matches = match_poses(t_boxes, pose_boxes, min_iou=float(self.p["min_iou"]))
            pts = np.empty((len(rows), 2), dtype=np.float64)
            for idx, r in enumerate(rows):
                p_idx = matches.get(idx)
                fx, fy, ly, src = ankle_ground_point(
                    t_boxes[idx],
                    pose_xy[p_idx] if p_idx is not None else None,
                    pose_conf[p_idx] if p_idx is not None else None,
                    ankle_conf=float(self.p["ankle_conf"]),
                )
                stats[src] += 1
                pts[idx] = (fx, fy)
                j = col[int(r[0])]
                foot_px[str(int(r[0]))].append([f, round(fx, 1), round(fy, 1)])
                box_h[i, j] = float(r[4] - r[2])
                if ly is not None:
                    lower_y[i, j] = ly
            homography = homographies.get(str(f))
            if homography is None:
                continue
            projected = project(homography, pts) * scale
            for r, p in zip(rows, projected, strict=True):
                xy[i, col[int(r[0])]] = p
        ms_per_frame = round((time.perf_counter() - t0) * 1000.0 / max(len(seq.frames), 1), 1)
        frames_arr = np.asarray(seq.frames, dtype=np.float64)
        airborne_out: dict[str, list[int]] = {}
        for j, tid in enumerate(ids):
            mask = detect_airborne(
                frames_arr,
                lower_y[:, j],
                box_h[:, j],
                seq.fps,
                airborne_frac=float(self.p["airborne_frac"]),
                max_airborne_s=float(self.p["max_airborne_s"]),
            )
            if np.any(mask):
                air_frames = [int(seq.frames[k]) for k in np.where(mask)[0]]
                airborne_out[str(tid)] = air_frames
                stats["airborne_frames"] += len(air_frames)
                if self.p["hold_airborne"]:
                    xy[:, j, :] = hold_airborne_positions(xy[:, j, :], mask)
        tracks_out, uncleaned = clean_track_paths(
            seq,
            xy,
            ids,
            clean=bool(self.p["clean"]),
            smooth_window=int(self.p["smooth_window"]),
            min_jump_m=float(self.p["min_jump_m"]),
        )
        return {
            "tracks": tracks_out,
            "foot_px": {k: v for k, v in foot_px.items() if v},
            "airborne": airborne_out,
            "court_m": list(COURT_M),
            "paths_left_uncleaned": uncleaned,
            "stats": stats,
            "ms_per_frame": ms_per_frame,
        }
