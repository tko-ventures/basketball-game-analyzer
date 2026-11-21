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
    """Read one exported gold sequence back from its MOTChallenge files.

    Args:
        root: Gold version directory (``data/gold/<version>``).
        name: Sequence name.

    Returns:
        The source frame grid, gold boxes ``(tid, x1, y1, x2, y2, class)`` per source frame, and the
        ``identities.csv`` row per track id.

    Examples:
        >>> import tempfile
        >>> from bga.eval.gold import Sequence as GoldSequence, write_sequence
        >>> seq = GoldSequence("s", "c", 0, False, [100, 105], [(2, 4, 1, 0, 10, 20, 1, 1, 1.0)], {"4": {}})
        >>> with tempfile.TemporaryDirectory() as d:
        ...     write_sequence(seq, Path(d), {}, 50.0, 5, (1920, 1080))
        ...     grid, per, ids = read_gt(Path(d), "s")
        >>> grid, dict(per), sorted(ids)
        ([100, 105], {105: [(4, 1.0, 0.0, 11.0, 20.0, 1)]}, ['4'])

    """
    grid = [int(x) for x in (root / name / "frames.txt").read_text().split()]
    per = defaultdict(list)
    for line in (root / name / "gt" / "gt.txt").read_text().splitlines():
        k, tid, x, y, w, h, _, cls, _vis = line.split(",")
        per[grid[int(k) - 1]].append((int(tid), float(x), float(y), float(x) + float(w), float(y) + float(h), int(cls)))
    ids = {}
    with (root / name / "identities.csv").open() as f:
        for row in csv.DictReader(f):
            ids[row["tid"]] = row
    return grid, per, ids


def match(gt: list[tuple], pred: list[list], thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one IoU matching (best first) of gold boxes to tracker rows.

    Args:
        gt: Gold boxes ``(tid, x1, y1, x2, y2, ...)``.
        pred: Tracker rows ``[track_id, x1, y1, x2, y2, ...]``.
        thr: Lowest IoU a pair may have.

    Returns:
        ``(gold index, tracker index)`` pairs, highest IoU first.

    Examples:
        >>> gt = [(1, 0, 0, 10, 10), (2, 20, 0, 30, 10)]
        >>> pred = [[9, 21, 0, 31, 10], [8, 0, 0, 10, 10], [7, 100, 0, 110, 10]]
        >>> match(gt, pred)
        [(0, 1), (1, 0)]
        >>> match(gt, [])
        []

    """
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
    """Class-agnostic person AP50 (and AP50:95, unreliable on v0 geometry) on the box-gold frames.

    Args:
        cfg: Run config whose ``detect`` output is scored.
        runs: Root of the run outputs.
        seqs: Sequences of the run; box-gold frames outside them are skipped.
        gold_root: Gold version directory holding ``boxes.json``.

    Returns:
        ``{"AP50", "AP50_95", "box_frames"}``; the APs are NaN when no frame or no detection is scored.

    Examples:
        >>> box_ap50(cfg, Path("runs"), seqs, Path("data/gold/v0"))  # doctest: +SKIP
        {'AP50': ..., 'AP50_95': ..., 'box_frames': ...}

    """
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
            anns.append({
                "id": len(anns) + 1,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
        det = load_output(cfg, runs, seq, "detect")["frames"].get(str(fr["frame"]), [])
        dets.extend(
            {
                "image_id": img_id,
                "category_id": 1,
                "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
                "score": d[4],
            }
            for d in det
            if d[5] in PERSON_CLASSES
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


def _match_votes(
    grid: list[int], gt_per: dict[int, list[tuple]], tracks: dict, teams: dict
) -> tuple[defaultdict[int, Counter], defaultdict[int, Counter]]:
    """Per gold track, count the predicted teams and predicted tracks of its matched tracker rows."""
    votes = defaultdict(Counter)  # gold track -> predicted team of its matched detections
    owner = defaultdict(Counter)  # gold track -> predicted tracks it was matched to
    for f in grid:
        g, p = gt_per.get(f, []), tracks.get(str(f), [])
        for gi, pi in match(g, p):
            owner[g[gi][0]][p[pi][0]] += 1
            di = p[pi][7]
            if di >= 0 and str(f) in teams:
                votes[g[gi][0]][teams[str(f)][di]] += 1
    return votes, owner


def _number_counts(nums: dict, ids: dict[str, dict], owner: defaultdict[int, Counter]) -> tuple[int, int, int]:
    """Count gold players with a legible number, and how many of their main predicted tracks read it.

    Returns ``(players, right number, any confirmed number)``; a player's main track is the one matched most often.

    """
    id_n = id_ok = id_confirmed = 0
    for tid, ident in ids.items():
        if not ident.get("number") or ident.get("team") not in ("home", "away"):
            continue
        id_n += 1
        best = owner[int(tid)].most_common(1)
        got = nums.get(str(best[0][0])) if best else None
        id_confirmed += got is not None
        id_ok += got == ident["number"]
    return id_n, id_ok, id_confirmed


def _team_counts(votes: defaultdict[int, Counter], ids: dict[str, dict]) -> tuple[int, int]:
    """Count gold player tracks with team votes, and those whose majority vote is their gold team."""
    team_n = team_ok = 0
    for tid, v in votes.items():
        truth = ids.get(str(tid), {}).get("team", "")
        if truth in ("home", "away"):
            team_n += 1
            team_ok += v.most_common(1)[0][0] == truth
    return team_n, team_ok


def _collect_speeds(cfg: RunConfig, runs: Path, seq: Sequence, speed: defaultdict[str, list]) -> None:
    """Append each layer's recorded ``ms_per_frame`` for ``seq`` to ``speed``."""
    for layer in ("court", "detect", "team", "track", "numbers"):
        if layer in cfg.layers and output_path(cfg, runs, seq, layer).exists():
            ms = load_output(cfg, runs, seq, layer).get("ms_per_frame")
            if ms is not None:
                speed[layer].append(ms)


def score(cfg: RunConfig, seqs: list[Sequence], runs: Path = Path("runs"), gold: str = "v0") -> dict:
    """Compute every gold-set score of one run: tracking, team, numbers, churn, detection AP and speed.

    Tracks come from the ``link`` layer when the config has one, else from ``track``.

    Args:
        cfg: Run config whose layer outputs are read.
        seqs: Gold sequences the run covers.
        runs: Root of the run outputs.
        gold: Gold version under ``data/gold``.

    Returns:
        The :func:`~bga.eval.mot.evaluate` scores plus ``team_acc``, ``team_tracks``, the ``id_*`` scores when the
        config has a numbers layer, churn summaries, box AP when ``boxes.json`` exists, ``ms`` per layer and the
        sequence count.

    Raises:
        NotImplementedError: For a config with both ``link`` and ``numbers``.

    Examples:
        >>> sorted(score(cfg, seqs))  # doctest: +SKIP
        ['AP50', 'AP50_95', 'AssA', 'DetA', 'HOTA', 'IDF1', 'IDSW', 'MOTA', ...]

    """
    if "link" in cfg.layers and "numbers" in cfg.layers:
        raise NotImplementedError("number accuracy is keyed by track ids; scoring it after linking is not built")
    root = Path("data/gold") / gold
    mot, team_ok, team_n, churn_rows, speed = {}, 0, 0, [], defaultdict(list)
    id_n = id_ok = id_confirmed = 0
    for seq in seqs:
        grid, gt_per, ids = read_gt(root, seq.name)
        tracks = load_output(cfg, runs, seq, "link" if "link" in cfg.layers else "track")["frames"]
        teams = load_output(cfg, runs, seq, "team")["frames"] if "team" in cfg.layers else {}
        gt = [[g[:5] for g in gt_per.get(f, [])] for f in grid]
        pred = [[(r[0], *r[1:5]) for r in tracks.get(str(f), [])] for f in grid]
        mot[seq.name] = (gt, pred)
        stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
        short = max(1, round(0.5 * seq.fps / stride))  # tracks living under 0.5 s
        churn_rows.append(compute_churn([[r[0] for r in tracks[str(f)]] for f in seq.frames], short_lived_frames=short))
        votes, owner = _match_votes(grid, gt_per, tracks, teams)
        if "numbers" in cfg.layers:
            nums = load_output(cfg, runs, seq, "numbers")["tracks"]
            seq_n, seq_ok, seq_confirmed = _number_counts(nums, ids, owner)
            id_n += seq_n
            id_ok += seq_ok
            id_confirmed += seq_confirmed
        seq_team_n, seq_team_ok = _team_counts(votes, ids)
        team_n += seq_team_n
        team_ok += seq_team_ok
        _collect_speeds(cfg, runs, seq, speed)
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
