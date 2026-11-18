"""Tracking metrics on the gold set: HOTA, CLEAR (MOTA) and Identity (IDF1) from TrackEval.

TrackEval's metric classes are fed directly (no dataset loader): per sequence, the gold boxes at
each gold frame and the tracker's boxes at the same frames, with IoU as similarity. All people
(players and referees) are scored as one class.

"""

from __future__ import annotations

from collections.abc import Sequence as Seq

import numpy as np

for _alias, _type in (("float", float), ("int", int), ("bool", bool)):  # TrackEval predates numpy 2
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)

from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402  # after the numpy aliases TrackEval needs

#: One tracked or gold box: id, x1, y1, x2, y2.
Box = tuple[int, float, float, float, float]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the pairwise IoU of two sets of xyxy boxes.

    Args:
        a: Boxes of shape (N, 4) as x1, y1, x2, y2.
        b: Boxes of shape (M, 4) as x1, y1, x2, y2.

    Returns:
        IoU of shape (N, M); a zero-sized matrix when either set is empty.

    Examples:
        >>> a = np.array([[0.0, 0.0, 10.0, 10.0]])
        >>> b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        >>> iou_matrix(a, b).round(3)
        array([[1.   , 0.333, 0.   ]])
        >>> iou_matrix(a, np.zeros((0, 4))).shape
        (1, 0)

    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def to_trackeval(gt: Seq[list[Box]], pred: Seq[list[Box]]) -> dict:
    """Build TrackEval's per-sequence data dict from per-timestep box lists (ids remapped to 0..N-1).

    Args:
        gt: Gold boxes per timestep.
        pred: Tracker boxes per timestep, as many timesteps as ``gt``.

    Returns:
        The dict TrackEval's ``eval_sequence`` takes: ids and IoU similarity per timestep, and the totals.

    Examples:
        >>> data = to_trackeval([[(7, 0, 0, 10, 10)], []], [[(3, 0, 0, 10, 10)], [(3, 5, 5, 9, 9)]])
        >>> data["num_gt_ids"], data["num_tracker_ids"], data["num_gt_dets"], data["num_tracker_dets"]
        (1, 1, 1, 2)
        >>> data["similarity_scores"][0]
        array([[1.]])

    """
    gid = {i: k for k, i in enumerate(sorted({b[0] for fr in gt for b in fr}))}
    tid = {i: k for k, i in enumerate(sorted({b[0] for fr in pred for b in fr}))}
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for g, p in zip(gt, pred, strict=True):
        data["gt_ids"].append(np.array([gid[b[0]] for b in g], dtype=int))
        data["tracker_ids"].append(np.array([tid[b[0]] for b in p], dtype=int))
        ga = np.array([b[1:] for b in g], dtype=float).reshape(-1, 4)
        pa = np.array([b[1:] for b in p], dtype=float).reshape(-1, 4)
        data["similarity_scores"].append(iou_matrix(ga, pa))
    data.update(
        num_timesteps=len(gt),
        num_gt_ids=len(gid),
        num_tracker_ids=len(tid),
        num_gt_dets=sum(len(g) for g in gt),
        num_tracker_dets=sum(len(p) for p in pred),
    )
    return data


#: TrackEval metric classes run by :func:`evaluate`.
METRICS = (HOTA, CLEAR, Identity)


def evaluate(sequences: dict[str, tuple[list[list[Box]], list[list[Box]]]]) -> dict[str, float]:
    """Combined HOTA, DetA, AssA, MOTA, IDF1, IDSW over sequences {name: (gt, pred)}.

    Args:
        sequences: Per sequence name, the gold and tracker boxes per timestep.

    Returns:
        The six scores, combined over all sequences the way TrackEval combines them.

    Examples:
        >>> gt = [[(1, 0, 0, 10, 20), (2, 50, 0, 60, 20)]] * 4
        >>> r = evaluate({"a": (gt, gt)})
        >>> round(r["HOTA"], 6), r["MOTA"], r["IDF1"], r["IDSW"]
        (1.0, 1.0, 1.0, 0)

    """
    per_metric = {}
    for metric_cls in METRICS:
        m = metric_cls({"PRINT_CONFIG": False}) if metric_cls is not HOTA else metric_cls()
        res = {name: m.eval_sequence(to_trackeval(gt, pred)) for name, (gt, pred) in sequences.items()}
        per_metric[metric_cls.__name__] = m.combine_sequences(res)
    h, c, i = per_metric["HOTA"], per_metric["CLEAR"], per_metric["Identity"]
    return {
        "HOTA": float(np.mean(h["HOTA"])),
        "DetA": float(np.mean(h["DetA"])),
        "AssA": float(np.mean(h["AssA"])),
        "MOTA": float(c["MOTA"]),
        "IDF1": float(i["IDF1"]),
        "IDSW": int(c["IDSW"]),
    }
