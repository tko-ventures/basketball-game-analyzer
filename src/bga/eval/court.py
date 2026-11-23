"""Court-layer score on the gold court keyframes (plan section 8; M3 done-when).

The gold keyframes are the ``purpose: gold`` entries of data/labels/court/queue.json that Tomasz
labelled ``ok``: 6 per gold game, the held-out game's scored only with ``held_out``. For each one the
config's court backend maps the frame, and its homography is compared with Tomasz's on the court
vertices visible under his mapping: each vertex's image position goes through the predicted mapping
and lands some distance in metres from its true court position.

The backend runs on the shot's frames within ``context`` frames of the keyframe, every ``stride``-th
frame (the stride of a real run), so smoothing has the neighbours it would have in the pipeline.

Reported per arena and overall: the median error per frame (median over frames), the same restricted
to the paint (vertices of the key and free-throw line, where players stand most), the share of
frames whose error is under 0.5 m in the paint, and the share of frames the backend could not map.

"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from bga import registry
from bga.config import RunConfig
from bga.court.base import load_fiba_court
from bga.court.propagate import landmarks
from bga.stage import FrameReader, Sequence

#: Repository root, the base of every data path read here.
ROOT = Path(__file__).resolve().parents[3]
#: FIBA court geometry shared with the labeller (vertices in cm, their labels).
GEO = load_fiba_court()
#: Court vertices in cm, shape (V, 2).
VERTS = np.array(GEO["vertices"], dtype=np.float64)
#: Vertex indices of the paint at both ends, where the 0.5 m target applies: key corners on the baseline (04, 05;
#: 37, 38), free-throw line ends (12, 14; 28, 30) and free-throw line centres (13; 29).
PAINT = {GEO["labels"].index(lb) for lb in ("04", "05", "12", "13", "14", "28", "29", "30", "37", "38")}


def gold_keyframes(held_out: bool) -> list[dict]:
    """List the gold court keyframes Tomasz marked ``ok``, with his homography.

    Args:
        held_out: Take the keyframes of the held-out game instead of the others.

    Returns:
        The queue entries of ``purpose: gold`` keyframes, each with its labelled homography under ``"H"``.

    Examples:
        >>> rows = gold_keyframes(held_out=False)
        >>> all(k["purpose"] == "gold" and "H" in k for k in rows)
        True

    """
    games = {g["clip"]: g for g in json.loads((ROOT / "data/labels/gold.json").read_text())["games"]}
    out = []
    for k in json.loads((ROOT / "data/labels/court/queue.json").read_text())["keyframes"]:
        if k["purpose"] != "gold" or games[k["clip"]]["held_out"] != held_out:
            continue
        labels = json.loads((ROOT / "data/labels/court" / f"{k['clip']}.json").read_text())
        rec = labels["frames"].get(str(k["frame"]))
        if rec and rec.get("status") == "ok" and rec.get("H"):
            out.append({**k, "H": rec["H"]})
    return out


def frame_errors(h_gold: ArrayLike, h_pred: ArrayLike | None, wh: tuple[int, int]) -> tuple[list[float], list[float]]:
    """Measure a predicted image-to-court homography against the gold one on the court vertices in view.

    Each vertex visible under ``h_gold`` is placed in the image by ``h_gold``, mapped back to the court by ``h_pred``,
    and compared with its true court position.

    Args:
        h_gold: Gold 3x3 homography, image pixels to court cm.
        h_pred: Predicted 3x3 homography, or None when the backend could not map the frame.
        wh: Frame width and height in pixels.

    Returns:
        Errors in metres of every visible vertex, and of the visible paint vertices only; both empty when no vertex is
        visible or ``h_pred`` is None.

    Examples:
        >>> gold = np.eye(3)
        >>> shifted = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> allv, paint = frame_errors(gold, shifted, (2800, 1500))
        >>> len(allv), len(paint), round(max(allv), 3)
        (23, 8, 0.5)
        >>> frame_errors(gold, None, (2800, 1500))
        ([], [])

    """
    vis = [(i, x, y) for i, (x, y, v) in enumerate(landmarks(np.asarray(h_gold), VERTS, wh)) if v]
    if not vis or h_pred is None:
        return [], []
    p = np.array([[x, y, 1.0] for _, x, y in vis]) @ np.asarray(h_pred, dtype=np.float64).T
    court = p[:, :2] / p[:, 2:3]
    err = np.linalg.norm(court - VERTS[[i for i, *_ in vis]], axis=1) / 100.0
    return err.tolist(), [float(e) for (i, *_), e in zip(vis, err, strict=True) if i in PAINT]


def score(cfg: RunConfig, held_out: bool = False, context: int = 25, stride: int = 5) -> dict:
    """Score the config's court backend on the gold court keyframes.

    Args:
        cfg: Run config whose ``court`` layer is scored.
        held_out: Score the held-out game instead of the others.
        context: Frames on each side of a keyframe the backend also sees, so smoothing has neighbours; 0 maps the
            keyframe alone with ``homography``.
        stride: Step between those context frames, as in a real run.

    Returns:
        ``{"frames": [...], "summary": ..., "ms_per_frame": float}``: one row per keyframe, the :func:`summarise`
        aggregate, and the backend's mean time per frame.

    Examples:
        >>> from bga.config import load_config  # doctest: +SKIP
        >>> sorted(score(load_config("configs/court-plk-v1.yaml")))  # doctest: +SKIP
        ['frames', 'ms_per_frame', 'summary']

    """
    entry = cfg.layers["court"]
    backend = registry.get("court", entry.backend)(entry.params, entry.weights)
    backend.load()
    rows, t = [], 0.0
    try:
        for k in gold_keyframes(held_out):
            # The backend sees a stretch of the shot around the keyframe, as in a real run, so smoothing
            # can use neighbours; without smoothing the keyframe's mapping is the single-frame one.
            shot = json.loads((ROOT / "data/labels" / f"{k['clip']}.json").read_text())["shots"][k["shot"]]
            frames = list(range(k["frame"] - context, k["frame"] + context + 1, stride))
            frames = [f for f in frames if shot["start"] <= f < shot["end"]]
            seq = Sequence(
                f"{k['youtube_id']}_kf{k['frame']}", k["clip"], str(ROOT / k["video"]), k["shot"], frames, 0.0
            )
            read = FrameReader(seq.video)
            try:
                img = read(k["frame"])
                t0 = time.perf_counter()
                homography = (
                    backend.run(seq, {}, read)["frames"][str(k["frame"])] if context else backend.homography(img)
                )
                t += (time.perf_counter() - t0) / max(1, len(frames) if context else 1)
            finally:
                read.close()
            allv, paint = frame_errors(k["H"], homography, (img.shape[1], img.shape[0]))
            rows.append({
                "hall": k["hall"],
                "clip": k["youtube_id"],
                "frame": k["frame"],
                "mapped": homography is not None,
                "median_m": statistics.median(allv) if allv else None,
                "paint_m": statistics.median(paint) if paint else None,
            })
    finally:
        backend.unload()
    ms = round(1000 * t / max(1, len(rows)), 1)
    return {"frames": rows, "summary": summarise(rows), "ms_per_frame": ms}


def summarise(rows: list[dict]) -> dict:
    """Aggregate keyframe rows of :func:`score` overall and per arena.

    Args:
        rows: Keyframe rows with ``hall``, ``mapped``, ``median_m`` and ``paint_m``.

    Returns:
        ``{"all": agg, "<hall>": agg, ...}`` where each ``agg`` holds the row count, unmapped count, median error,
        median paint error and share of paint errors under 0.5 m (None where nothing was measured).

    Examples:
        >>> rows = [
        ...     {"hall": "A", "mapped": True, "median_m": 0.4, "paint_m": 0.3},
        ...     {"hall": "A", "mapped": True, "median_m": 0.8, "paint_m": 0.7},
        ...     {"hall": "B", "mapped": False, "median_m": None, "paint_m": None},
        ... ]
        >>> summarise(rows)["all"]
        {'n': 3, 'unmapped': 1, 'median_m': 0.6, 'paint_m': 0.5, 'paint_under_0_5': 0.5}
        >>> summarise(rows)["B"]["paint_m"] is None
        True

    """

    def agg(rs: list[dict]) -> dict:
        med = [r["median_m"] for r in rs if r["median_m"] is not None]
        paint = [r["paint_m"] for r in rs if r["paint_m"] is not None]
        return {
            "n": len(rs),
            "unmapped": sum(1 for r in rs if not r["mapped"]),
            "median_m": round(statistics.median(med), 2) if med else None,
            "paint_m": round(statistics.median(paint), 2) if paint else None,
            "paint_under_0_5": round(sum(1 for p in paint if p < 0.5) / len(paint), 2) if paint else None,
        }

    halls = sorted({r["hall"] for r in rows})
    return {"all": agg(rows), **{h: agg([r for r in rows if r["hall"] == h]) for h in halls}}
