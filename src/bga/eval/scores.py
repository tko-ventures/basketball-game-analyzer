"""All gold-set scores for one run config (plan section 8): tracking, detection AP, team, speed.

Reads the run's layer outputs and data/gold/<version>; computes nothing a layer computed itself.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bga.config import RunConfig
from bga.detect.base import PERSON_CLASSES
from bga.eval.churn import compute_churn
from bga.eval.mot import evaluate, iou_matrix
from bga.pipeline import load_output, output_path
from bga.stage import Sequence


def read_gt(root: Path, name: str) -> tuple[list[int], dict[int, list[tuple]], dict[str, dict]]:
    grid = [int(x) for x in (root / name / "frames.txt").read_text().split()]
    per = defaultdict(list)
    for line in (root / name / "gt" / "gt.txt").read_text().splitlines():
        k, tid, x, y, w, h, _, cls, vis = line.split(",")
        per[grid[int(k) - 1]].append(
            (int(tid), float(x), float(y), float(x) + float(w), float(y) + float(h), int(cls))
        )
    ids = {}
    with (root / name / "identities.csv").open() as f:
        for row in csv.DictReader(f):
            ids[row["tid"]] = row
    return grid, per, ids


def match(gt: list[tuple], pred: list[list], thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one IoU matching (best first) of gold boxes to tracker rows."""
    if not gt or not pred:
        return []
    m = iou_matrix(np.array([g[1:5] for g in gt]), np.array([p[1:5] for p in pred]))
    pairs, used_g, used_p = [], set(), set()
    for gi, pi in sorted(np.argwhere(m >= thr).tolist(), key=lambda ij: -m[ij[0], ij[1]]):
        if gi not in used_g and pi not in used_p:
            pairs.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)
    return pairs


def box_ap50(cfg: RunConfig, runs: Path, seqs: list[Sequence], gold_root: Path) -> dict:
    """Class-agnostic person AP50 (and AP50:95, unreliable on v0 geometry) on the box-gold frames."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    boxes = json.loads((gold_root / "boxes.json").read_text())["frames"]
    by_seq = {s.name: s for s in seqs}
    images, anns, dets = [], [], []
    for img_id, fr in enumerate(boxes, 1):
        name = f"{fr['youtube_id']}_s{fr['shot']}"
        if name not in by_seq:
            continue
        seq = by_seq[name]
        images.append({"id": img_id, "width": 1920, "height": 1080})
        for b in fr["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        det = load_output(cfg, runs, seq, "detect")["frames"].get(str(fr["frame"]), [])
        for d in det:
            if d[5] in PERSON_CLASSES:
                dets.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
                        "score": d[4],
                    }
                )
    if not images or not dets:
        return {"AP50": float("nan"), "AP50_95": float("nan"), "box_frames": len(images)}
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    gt.createIndex()
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"AP50": float(ev.stats[1]), "AP50_95": float(ev.stats[0]), "box_frames": len(images)}


def score(cfg: RunConfig, seqs: list[Sequence], runs: Path = Path("runs"), gold: str = "v0") -> dict:
    root = Path("data/gold") / gold
    mot, team_ok, team_n, churn_rows, speed = {}, 0, 0, [], defaultdict(list)
    id_n = id_ok = id_confirmed = 0
    for seq in seqs:
        grid, gt_per, ids = read_gt(root, seq.name)
        tracks = load_output(cfg, runs, seq, "track")["frames"]
        teams = load_output(cfg, runs, seq, "team")["frames"] if "team" in cfg.layers else {}
        gt = [[g[:5] for g in gt_per.get(f, [])] for f in grid]
        pred = [[(r[0], *r[1:5]) for r in tracks.get(str(f), [])] for f in grid]
        mot[seq.name] = (gt, pred)
        stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
        short = max(1, round(0.5 * seq.fps / stride))  # tracks living under 0.5 s
        churn_rows.append(
            compute_churn([[r[0] for r in tracks[str(f)]] for f in seq.frames], short_lived_frames=short)
        )
        votes = defaultdict(Counter)  # gold track -> predicted team of its matched detections
        owner = defaultdict(Counter)  # gold track -> predicted tracks it was matched to
        for f in grid:
            g, p = gt_per.get(f, []), tracks.get(str(f), [])
            for gi, pi in match(g, p):
                owner[g[gi][0]][p[pi][0]] += 1
                di = p[pi][7]
                if di >= 0 and str(f) in teams:
                    votes[g[gi][0]][teams[str(f)][di]] += 1
        if "numbers" in cfg.layers:
            nums = load_output(cfg, runs, seq, "numbers")["tracks"]
            for tid, ident in ids.items():
                if not ident.get("number") or ident.get("team") not in ("home", "away"):
                    continue
                id_n += 1
                best = owner[int(tid)].most_common(1)
                got = nums.get(str(best[0][0])) if best else None
                id_confirmed += got is not None
                id_ok += got == ident["number"]
        for tid, v in votes.items():
            truth = ids.get(str(tid), {}).get("team", "")
            if truth in ("home", "away"):
                team_n += 1
                team_ok += v.most_common(1)[0][0] == truth
        for layer in ("court", "detect", "team", "track", "numbers"):
            if layer in cfg.layers and output_path(cfg, runs, seq, layer).exists():
                ms = load_output(cfg, runs, seq, layer).get("ms_per_frame")
                if ms is not None:
                    speed[layer].append(ms)
    out = evaluate(mot)
    out["team_acc"] = team_ok / team_n if team_n else float("nan")
    out["team_tracks"] = team_n
    if "numbers" in cfg.layers:
        out["id_acc"] = id_ok / id_n if id_n else float("nan")  # gold tracks with a legible number
        out["id_confirmed"] = id_confirmed / id_n if id_n else float("nan")
        out["id_wrong"] = id_confirmed - id_ok
        out["id_tracks"] = id_n
    out["churn_short_lived"] = (
        float(np.mean([c["short_lived_id_fraction"] for c in churn_rows])) if churn_rows else float("nan")
    )
    out["pred_ids_per_seq"] = (
        float(np.mean([c["unique_track_ids"] for c in churn_rows])) if churn_rows else float("nan")
    )
    if (root / "boxes.json").exists():
        out.update(box_ap50(cfg, runs, seqs, root))
    out["ms"] = {k: round(float(np.mean(v)), 1) for k, v in speed.items()}
    out["sequences"] = len(seqs)
    return out
