"""Per-class detection AP on the box-gold frames (plan section 8): AP50:95 and AP50 for each class and for people.

Box gold v1 (``data/gold/v1/boxes.json``) boxes every object of the 10-class schema by hand, so each class gets its
own COCO average precision. ``people`` merges the six person classes on both sides, so a player found as a referee,
or a possessor found as a plain player, still counts as found; it is comparable with the class-agnostic AP50 of v0.

"""

from __future__ import annotations

import contextlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bga.config import RunConfig
from bga.detect.base import PERSON_CLASSES
from bga.pipeline import load_output
from bga.stage import Sequence
from bga.types import DETECTION_CLASSES

#: Name of the merged person class in the results.
PEOPLE = "people"

#: A gold box: ``(xyxy, class name)``.
GoldBox = tuple[list[float], str]
#: A detection: ``(xyxy, score, class name)``.
DetBox = tuple[list[float], float, str]


@dataclass(frozen=True)
class ClassAP:
    """COCO average precision of one class: over IoU 0.50 to 0.95, at IoU 0.50, and the gold box count."""

    ap: float
    ap50: float
    n: int


def _coco(gold: list[list[GoldBox]], dets: list[list[DetBox]], names: list[str]) -> dict[str, ClassAP]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cat = {n: i + 1 for i, n in enumerate(names)}
    images, anns, res = [], [], []
    counts = dict.fromkeys(names, 0)
    for img_id, (gs, ds) in enumerate(zip(gold, dets, strict=True), 1):
        images.append({"id": img_id, "width": 1920, "height": 1080})
        for (x1, y1, x2, y2), c in gs:
            counts[c] += 1
            anns.append({
                "id": len(anns) + 1,
                "image_id": img_id,
                "category_id": cat[c],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
        res.extend(
            {"image_id": img_id, "category_id": cat[c], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s}
            for (x1, y1, x2, y2), s, c in ds
            if c in cat
        )
    if not res:  # nothing detected: AP 0 where there is gold, undefined elsewhere
        return {n: ClassAP(*(2 * [0.0 if counts[n] else math.nan]), counts[n]) for n in names}
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for n, i in cat.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]  # [IoU thresholds, recall points, classes, area ranges, max detections]
    out = {}
    for n, i in cat.items():
        p = prec[:, :, i - 1, 0, -1]
        valid, valid50 = p[p > -1], p[0][p[0] > -1]
        out[n] = ClassAP(  # rounded: COCO's interpolation leaves 0.9999999... for a perfect class
            round(float(valid.mean()), 6) if valid.size else math.nan,
            round(float(valid50.mean()), 6) if valid50.size else math.nan,
            counts[n],
        )
    return out


def class_ap(gold: list[list[GoldBox]], dets: list[list[DetBox]]) -> dict[str, ClassAP]:
    """COCO AP per class of the 10-class schema, plus ``people`` with the person classes merged.

    Args:
        gold: Per image, the gold boxes as ``(xyxy, class)``.
        dets: Per image, in the same order, the detections as ``(xyxy, score, class)``.

    Returns:
        One :class:`ClassAP` per class name and for ``people``; a class with no gold box has NaN APs.

    Examples:
        >>> gold = [[([0, 0, 50, 100], "player"), ([60, 60, 70, 70], "ball")]]
        >>> dets = [[([0, 0, 50, 100], 0.9, "referee")]]
        >>> r = class_ap(gold, dets)
        >>> r["people"].ap50, r["player"].ap50, r["ball"].ap50, r["ball"].n, math.isnan(r["rim"].ap)
        (1.0, 0.0, 0.0, 1, True)

    """
    out = _coco(gold, dets, list(DETECTION_CLASSES))
    merge = [[(b, PEOPLE) for b, c in gs if c in PERSON_CLASSES] for gs in gold]
    merged = [[(b, s, PEOPLE) for b, s, c in ds if c in PERSON_CLASSES] for ds in dets]
    out[PEOPLE] = _coco(merge, merged, [PEOPLE])[PEOPLE]
    return out


def score(cfg: RunConfig, seqs: list[Sequence], runs: Path, boxes: Path) -> dict:
    """Score a run's ``detect`` output on the box-gold frames that fall inside its sequences.

    Args:
        cfg: Run config whose detections are scored.
        seqs: Sequences of the run; box-gold frames outside them are skipped.
        runs: Root of the run outputs.
        boxes: Box gold file with ``cls`` on every box (gold v1 or later).

    Returns:
        ``{"classes": {name: ClassAP}, "frames": n, "ms_per_frame": mean detect time or NaN}``.

    Raises:
        ValueError: When the box gold has no classes (gold v0 boxes are people only).

    Examples:
        >>> score(cfg, seqs, Path("runs"), Path("data/gold/v1/boxes.json"))["frames"]  # doctest: +SKIP
        76

    """
    frames = json.loads(boxes.read_text())["frames"]
    if frames and "cls" not in frames[0]["boxes"][0]:
        raise ValueError(f"{boxes} has no box classes; per-class AP needs gold v1 or later")
    by_seq = {s.name: s for s in seqs}
    gold, dets, ms = [], [], []
    cache: dict[str, dict] = {}
    for fr in frames:
        name = f"{fr['youtube_id']}_s{fr['shot']}"
        if name not in by_seq:
            continue
        if name not in cache:
            cache[name] = load_output(cfg, runs, by_seq[name], "detect")
            if cache[name].get("ms_per_frame") is not None:
                ms.append(cache[name]["ms_per_frame"])
        gold.append([(b["xyxy"], b["cls"]) for b in fr["boxes"]])
        dets.append([(d[:4], d[4], d[5]) for d in cache[name]["frames"].get(str(fr["frame"]), [])])
    return {
        "classes": class_ap(gold, dets),
        "frames": len(gold),
        "ms_per_frame": float(np.mean(ms)) if ms else math.nan,
    }
