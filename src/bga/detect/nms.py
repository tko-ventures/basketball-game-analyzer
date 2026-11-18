"""Nested-box suppression for DETR outputs, from the prior repo's ``detect/nms.py``."""

from __future__ import annotations

from collections.abc import Sequence

from bga.types import BBox, Detection


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: BBox, b: BBox) -> float:
    """Compute the intersection over union of two xyxy boxes.

    Args:
        a: First box as x1, y1, x2, y2.
        b: Second box as x1, y1, x2, y2.

    Returns:
        IoU in [0, 1]; 0 when the boxes do not overlap or both are empty.

    Examples:
        >>> iou((0, 0, 10, 10), (5, 0, 15, 10))
        0.3333333333333333
        >>> iou((0, 0, 10, 10), (20, 20, 30, 30))
        0.0

    """
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if inter > 0 and union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area covered by ``outer``.

    Args:
        inner: Box whose coverage is measured, as x1, y1, x2, y2.
        outer: Covering box, as x1, y1, x2, y2.

    Returns:
        Covered share of ``inner`` in [0, 1]; 0 when ``inner`` is empty.

    Examples:
        >>> containment((0, 0, 10, 10), (5, 0, 100, 100))
        0.5
        >>> containment((5, 5, 5, 5), (0, 0, 10, 10))
        0.0

    """
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


def nested_box_nms(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = 0.7,
    containment_threshold: float = 0.85,
    same_role_only: bool = True,
) -> list[Detection]:
    """Suppress duplicate and nested boxes, keeping the higher-score box of each clashing pair.

    A box is dropped when it has IoU >= ``iou_threshold`` with a kept box of higher score, or when either of the two
    lies at least ``containment_threshold`` inside the other. With ``same_role_only`` boxes compete only within their
    role, so a ``number`` inside a ``player`` survives.

    Args:
        detections: Boxes to filter.
        iou_threshold: IoU at which two boxes clash.
        containment_threshold: Containment at which two boxes clash.
        same_role_only: Compare boxes only within their role (player, number, other).

    Returns:
        The kept boxes, highest score first.

    Examples:
        >>> player = Detection((0, 0, 100, 200), "player", 0.9)
        >>> nested = Detection((10, 10, 50, 80), "player", 0.8)
        >>> number = Detection((30, 40, 70, 90), "number", 0.85)
        >>> [d.class_name for d in nested_box_nms([nested, number, player])]
        ['player', 'number']
        >>> [d.class_name for d in nested_box_nms([number, player], same_role_only=False)]
        ['player']

    """
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
