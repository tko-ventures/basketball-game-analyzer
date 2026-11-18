"""Court-layer score on the gold court keyframes (plan section 8; M3 done-when).

The gold keyframes are the ``purpose: gold`` entries of data/labels/court/queue.json that Tomasz
labelled ``ok``: 6 per gold game, the held-out game's scored only with ``held_out``. For each one the
config's court backend maps the frame, and its homography is compared with Tomasz's on the court
vertices visible under his mapping: each vertex's image position goes through the predicted mapping
and lands some distance in metres from its true court position.

Reported per arena and overall: the median error per frame (median over frames), the same restricted
to the paint (vertices of the key and free-throw line, where players stand most), the share of
frames whose error is under 0.5 m in the paint, and the share of frames the backend could not map.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from bga import registry
from bga.config import RunConfig
from bga.court.propagate import landmarks

ROOT = Path(__file__).resolve().parents[3]
GEO = json.loads((ROOT / "tools/labeler/static/fiba_court.json").read_text())
VERTS = np.array(GEO["vertices"], dtype=np.float64)
# The paint at both ends, where the 0.5 m target applies: key corners on the baseline (04, 05; 37, 38),
# free-throw line ends (12, 14; 28, 30) and free-throw line centres (13; 29).
PAINT = {GEO["labels"].index(lb) for lb in ("04", "05", "12", "13", "14", "28", "29", "30", "37", "38")}


def gold_keyframes(held_out: bool) -> list[dict]:
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


def frame_errors(H_gold, H_pred, wh) -> tuple[list[float], list[float]]:
    """Per visible vertex error in metres (all, paint) of H_pred against H_gold."""
    vis = [(i, x, y) for i, (x, y, v) in enumerate(landmarks(np.asarray(H_gold), VERTS, wh)) if v]
    if not vis or H_pred is None:
        return [], []
    p = np.array([[x, y, 1.0] for _, x, y in vis]) @ np.asarray(H_pred, dtype=np.float64).T
    court = p[:, :2] / p[:, 2:3]
    err = np.linalg.norm(court - VERTS[[i for i, *_ in vis]], axis=1) / 100.0
    return err.tolist(), [float(e) for (i, *_), e in zip(vis, err, strict=True) if i in PAINT]


def score(cfg: RunConfig, held_out: bool = False) -> dict:
    entry = cfg.layers["court"]
    backend = registry.get("court", entry.backend)(entry.params, entry.weights)
    backend.load()
    rows, t = [], 0.0
    try:
        for k in gold_keyframes(held_out):
            cap = cv2.VideoCapture(str(ROOT / k["video"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, k["frame"])
            ok, img = cap.read()
            cap.release()
            if not ok:
                raise IndexError(f"{k['video']} frame {k['frame']}")
            t0 = time.perf_counter()
            H = backend.homography(img)
            t += time.perf_counter() - t0
            allv, paint = frame_errors(k["H"], H, (img.shape[1], img.shape[0]))
            rows.append(
                {
                    "hall": k["hall"],
                    "clip": k["youtube_id"],
                    "frame": k["frame"],
                    "mapped": H is not None,
                    "median_m": statistics.median(allv) if allv else None,
                    "paint_m": statistics.median(paint) if paint else None,
                }
            )
    finally:
        backend.unload()
    ms = round(1000 * t / max(1, len(rows)), 1)
    return {"frames": rows, "summary": summarise(rows), "ms_per_frame": ms}


def summarise(rows: list[dict]) -> dict:
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
