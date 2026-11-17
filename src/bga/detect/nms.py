"""Nested-box suppression for DETR outputs, from the prior repo's ``detect/nms.py``."""

from __future__ import annotations

from collections.abc import Sequence

from bga.types import BBox, Detection


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: BBox, b: BBox) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if inter > 0 and union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area covered by ``outer``."""
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


def nested_box_nms(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = 0.7,
    containment_threshold: float = 0.85,
    same_role_only: bool = True,
) -> list[Detection]:
    """Drop the lower-score box of any pair with IoU >= ``iou_threshold`` or whose smaller box is
    >= ``containment_threshold`` inside the larger. With ``same_role_only`` boxes compete only within
    their role, so a ``number`` inside a ``player`` survives."""
    groups: dict[str, list[Detection]] = {}
    for d in detections:
        groups.setdefault(d.role if same_role_only else "", []).append(d)
    kept: list[Detection] = []
    for group in groups.values():
        selected: list[Detection] = []
        for cand in sorted(group, key=lambda d: d.score, reverse=True):
            a = cand.xyxy
            if not any(
                iou(a, p.xyxy) >= iou_threshold
                or max(containment(a, p.xyxy), containment(p.xyxy, a)) >= containment_threshold
                for p in selected
            ):
                selected.append(cand)
        kept.extend(selected)
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept
