"""Ball, possession, and shot-event evaluation on gold shots (plan section 8, M9).

Scores any Layer 7 (``ball``) backend against two gold sources:

1. ``data/gold/v1/boxes.json`` (76 dev / 24 held-out frames):
   - ``ball_f1`` / ``ball_r`` / ``ball_med_px``: tracked ``ball_px`` matched to gold ``ball`` boxes within
     ``max(25 px, box_side)``.
   - ``poss_acc`` / ``holder_f1`` / ``team_acc``: predicted ``holder_track_id`` box matched to gold
     ``player-in-possession`` boxes at ``IoU >= 0.5`` (and exact ``None`` match when no holder is labelled).
2. ``data/gold/v1/events.json`` (optional shot-attempt ground truth):
   - ``shot_f1`` / ``shot_outcome_acc`` / ``shot_shooter_acc``: greedy 1-to-1 matching of predicted shots to
     gold shots within ``rim_tol_frames`` (default 40 source frames = 0.8 s at 50 fps), checking ``outcome``
     (``made`` / ``missed``) and shooter track IoU against ``data/gold/v0/<seq>/gt/gt.txt``.

"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from bga import pipeline
from bga.ball.base import bbox_iou
from bga.config import RunConfig
from bga.eval.scores import read_gt
from bga.stage import Sequence

#: Default paths for gold v1 boxes and shot events.
GOLD_V1_BOXES = Path("data/gold/v1/boxes.json")
GOLD_V1_EVENTS = Path("data/gold/v1/events.json")
GOLD_V0_ROOT = Path("data/gold/v0")
#: Maximum source-frame gap between predicted and gold ``rim_frame`` for a shot match (0.8 s at 50 fps).
DEFAULT_RIM_TOL_FRAMES = 40


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)``, returning ``0.0`` on empty denominators."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _box_for_tid(track_rows: list[list[Any]], tid: int | None) -> list[float] | None:
    """Look up ``[x1, y1, x2, y2]`` for ``tid`` in one frame's track rows."""
    if tid is None:
        return None
    for r in track_rows:
        if int(r[0]) == tid:
            return [float(r[1]), float(r[2]), float(r[3]), float(r[4])]
    return None


def _score_frame_ball(
    gt_balls: list[list[float]],
    pred_b: list[float] | None,
) -> tuple[int, int, int, float | None]:
    """Return ``(tp, fp, fn, dist_px_or_None)`` for one frame's ball prediction."""
    if gt_balls and pred_b is not None:
        gb = gt_balls[0]
        d = math.hypot(pred_b[0] - (gb[0] + gb[2]) / 2.0, pred_b[1] - (gb[1] + gb[3]) / 2.0)
        if d <= max(25.0, gb[2] - gb[0], gb[3] - gb[1]):
            return 1, 0, 0, d
        return 0, 1, 1, None
    if gt_balls:
        return 0, 0, 1, None
    if pred_b is not None:
        return 0, 1, 0, None
    return 0, 0, 0, None


def _score_frame_holder(
    gt_h: list[dict[str, Any]],
    row: dict[str, Any],
    track_rows: list[list[Any]],
) -> tuple[int, int, int, int, int, int]:
    """Return ``(tp, fp, fn, frame_ok, team_ok, team_tot)`` for one frame's possession prediction."""
    pred_tid = row.get("holder_track_id")
    if not gt_h:
        return (0, 0, 0, 1, 0, 0) if pred_tid is None else (0, 1, 0, 0, 0, 0)
    pred_box = _box_for_tid(track_rows, pred_tid)
    team_tot = int(gt_h[0].get("team") in ("home", "away"))
    team_ok = int(team_tot and row.get("holder_team") == gt_h[0]["team"])
    if pred_box is not None and bbox_iou(gt_h[0]["xyxy"], pred_box) >= 0.5:
        return 1, 0, 0, 1, team_ok, team_tot
    if pred_tid is not None:
        return 0, 1, 1, 0, team_ok, team_tot
    return 0, 0, 1, 0, team_ok, team_tot


def score_boxes(
    ball_by_seq: dict[str, dict[str, Any]],
    tracks_by_seq: dict[str, dict[str, list[list[Any]]]],
    gold_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score per-frame ball position and possession against ``data/gold/v1/boxes.json`` frames.

    Args:
        ball_by_seq: Mapping ``seq_name -> Layer 7 output dict``.
        tracks_by_seq: Mapping ``seq_name -> {frame_str: [[tid, x1, y1, x2, y2, ...], ...]}``.
        gold_frames: Filtered list of frame entries from ``boxes.json``.

    Returns:
        Dict of ball-tracking and possession metrics (``NaN`` if ``gold_frames`` is empty).

    Examples:
        >>> ball = {"s1": {"frames": {"0": {"ball_px": [5.0, 5.0], "holder_track_id": 1, "holder_team": "home"}}}}
        >>> tracks = {"s1": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}}
        >>> gf = [{"youtube_id": "s", "shot": 1, "frame": 0, "boxes": [
        ...     {"cls": "ball", "xyxy": [0, 0, 10, 10]},
        ...     {"cls": "player-in-possession", "xyxy": [0, 0, 10, 20], "team": "home"},
        ... ]}]
        >>> m = score_boxes({"s_s1": ball["s1"]}, {"s_s1": tracks["s1"]}, gf)
        >>> (m["ball_f1"], m["holder_f1"], m["poss_acc"], m["team_acc"])
        (1.0, 1.0, 1.0, 1.0)

    """
    b_tp = b_fp = b_fn = 0
    h_tp = h_fp = h_fn = 0
    frame_ok = team_ok = team_tot = n_frames = 0
    b_errs: list[float] = []

    for fr in gold_frames:
        sname = f"{fr['youtube_id']}_s{fr['shot']}"
        if sname not in ball_by_seq:
            continue
        n_frames += 1
        fstr = str(fr["frame"])
        row = ball_by_seq[sname].get("frames", {}).get(fstr, {})
        gt_balls = [b["xyxy"] for b in fr["boxes"] if b["cls"] == "ball"]
        btp, bfp, bfn, err = _score_frame_ball(gt_balls, row.get("ball_px"))
        b_tp, b_fp, b_fn = b_tp + btp, b_fp + bfp, b_fn + bfn
        if err is not None:
            b_errs.append(err)
        gt_h = [b for b in fr["boxes"] if b["cls"] == "player-in-possession"]
        htp, hfp, hfn, fok, tok, ttot = _score_frame_holder(gt_h, row, tracks_by_seq.get(sname, {}).get(fstr, []))
        h_tp, h_fp, h_fn = h_tp + htp, h_fp + hfp, h_fn + hfn
        frame_ok, team_ok, team_tot = frame_ok + fok, team_ok + tok, team_tot + ttot

    bp, br, bf1 = _f1(b_tp, b_fp, b_fn)
    hp, hr, hf1 = _f1(h_tp, h_fp, h_fn)
    return {
        "box_frames": n_frames,
        "ball_p": bp,
        "ball_r": br,
        "ball_f1": bf1,
        "ball_med_px": statistics.median(b_errs) if b_errs else float("nan"),
        "poss_acc": frame_ok / n_frames if n_frames else float("nan"),
        "holder_p": hp,
        "holder_r": hr,
        "holder_f1": hf1,
        "team_acc": team_ok / team_tot if team_tot else float("nan"),
    }


def _match_shots(
    gt_shots: list[dict[str, Any]],
    pred_shots: list[dict[str, Any]],
    rim_tol_frames: int,
) -> list[tuple[int, int]]:
    """Greedy closest-``rim_frame`` 1-to-1 matching between gold and predicted shots in one sequence."""
    pairs: list[tuple[int, int, int]] = []
    for gi, gs in enumerate(gt_shots):
        g_rim = int(gs["rim_frame"])
        for pi, ps in enumerate(pred_shots):
            p_rim = ps.get("rim_frame")
            if p_rim is None:
                p_rim = ps.get("end_frame", ps.get("start_frame", 0))
            dist = abs(int(p_rim) - g_rim)
            if dist <= rim_tol_frames:
                pairs.append((dist, gi, pi))
    pairs.sort()
    matched: list[tuple[int, int]] = []
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            matched.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)
    return matched


def _shooter_matches(
    gs: dict[str, Any],
    ps: dict[str, Any],
    tracks: dict[str, list[list[Any]]],
    gt_by_f: dict[int, list[tuple[Any, ...]]],
) -> bool:
    """Check whether a predicted shooter matches gold ``shooter_gt_tid`` (via IoU) or fallback ``shooter_team``."""
    gt_tid = gs.get("shooter_gt_tid")
    pred_tid = ps.get("shooter_track_id")
    if gt_tid is None:
        return bool(gs.get("shooter_team") and ps.get("shooter_team") == gs.get("shooter_team"))
    if pred_tid is None or not gt_by_f:
        return False
    sf = int(ps.get("start_frame", gs["rim_frame"]))
    pbox = _box_for_tid(tracks.get(str(sf), []), int(pred_tid))
    if pbox is None:
        return False
    near_f = min(gt_by_f, key=lambda f: abs(f - sf))
    best_iou, best_tid = 0.0, None
    for g in gt_by_f[near_f]:
        ov = bbox_iou(pbox, g[1:5])
        if ov > best_iou:
            best_iou, best_tid = ov, int(g[0])
    return best_iou >= 0.4 and best_tid == int(gt_tid)


def score_events(
    ball_by_seq: dict[str, dict[str, Any]],
    tracks_by_seq: dict[str, dict[str, list[list[Any]]]],
    gold_events: list[dict[str, Any]],
    gold_v0_root: Path = GOLD_V0_ROOT,
    *,
    rim_tol_frames: int = DEFAULT_RIM_TOL_FRAMES,
) -> dict[str, Any]:
    """Score predicted shot attempts, made/missed outcomes, and shooters against ``events.json``.

    Args:
        ball_by_seq: Mapping ``seq_name -> Layer 7 output dict``.
        tracks_by_seq: Mapping ``seq_name -> {frame_str: track_rows}``.
        gold_events: List of gold shot dicts (``sequence``, ``rim_frame``, ``outcome``, etc.).
        gold_v0_root: Root directory of MOTChallenge gold tracks (for ``shooter_gt_tid`` IoU check).
        rim_tol_frames: Maximum source-frame difference on ``rim_frame`` for a valid match.

    Returns:
        Dict of shot detection, outcome, and shooter metrics (``NaN`` if no gold events cover the run).

    Examples:
        >>> ball = {"s1": {"shots": [{"start_frame": 10, "rim_frame": 50, "outcome": "made", "shooter_team": "home"}]}}
        >>> gt = [{"sequence": "s1", "rim_frame": 55, "outcome": "made", "shooter_team": "home"}]
        >>> m = score_events(ball, {"s1": {}}, gt)
        >>> (m["shot_f1"], m["shot_outcome_acc"], m["shot_shooter_acc"])
        (1.0, 1.0, 1.0)

    """
    gt_by_seq: dict[str, list[dict[str, Any]]] = {}
    for ev in gold_events:
        if ev["sequence"] in ball_by_seq:
            gt_by_seq.setdefault(ev["sequence"], []).append(ev)
    if not gt_by_seq:
        return {
            "shot_gt": 0,
            "shot_pred": sum(len(v.get("shots", [])) for v in ball_by_seq.values()),
            "shot_p": float("nan"),
            "shot_r": float("nan"),
            "shot_f1": float("nan"),
            "shot_outcome_acc": float("nan"),
            "shot_shooter_acc": float("nan"),
        }

    tp = fp = fn = out_ok = shoot_ok = shoot_tot = 0
    for sname, ball_out in ball_by_seq.items():
        gt_list = gt_by_seq.get(sname, [])
        pred_list = ball_out.get("shots", [])
        matches = _match_shots(gt_list, pred_list, rim_tol_frames)
        tp += len(matches)
        fp += len(pred_list) - len(matches)
        fn += len(gt_list) - len(matches)
        gt_by_f: dict[int, list[tuple[Any, ...]]] = {}
        if matches and (gold_v0_root / sname / "gt" / "gt.txt").exists():
            _, gt_by_f, _ = read_gt(gold_v0_root, sname)
        for gi, pi in matches:
            gs, ps = gt_list[gi], pred_list[pi]
            out_ok += int(ps.get("outcome") == gs.get("outcome"))
            if gs.get("shooter_gt_tid") is not None or gs.get("shooter_team") is not None:
                shoot_tot += 1
                shoot_ok += int(_shooter_matches(gs, ps, tracks_by_seq.get(sname, {}), gt_by_f))

    sp, sr, sf1 = _f1(tp, fp, fn)
    return {
        "shot_gt": tp + fn,
        "shot_pred": tp + fp,
        "shot_p": sp,
        "shot_r": sr,
        "shot_f1": sf1,
        "shot_outcome_acc": out_ok / tp if tp else float("nan"),
        "shot_shooter_acc": shoot_ok / shoot_tot if shoot_tot else float("nan"),
    }


def score(
    cfg: RunConfig,
    runs: Path = Path("runs"),
    *,
    boxes_path: Path = GOLD_V1_BOXES,
    events_path: Path = GOLD_V1_EVENTS,
    gold_v0_root: Path = GOLD_V0_ROOT,
) -> dict[str, Any]:
    """Run ``cfg`` (reusing cached outputs) and score its ``ball`` layer across all selected sequences.

    Args:
        cfg: Run config including ``ball``.
        runs: Root of pipeline run outputs.
        boxes_path: Path to ``data/gold/v1/boxes.json``.
        events_path: Path to ``data/gold/v1/events.json`` (skipped gracefully if absent).
        gold_v0_root: Path to ``data/gold/v0`` MOTChallenge tracks.

    Returns:
        Combined metrics dict covering ball tracking, possession, shot events, and latency.

    """
    seqs: list[Sequence] = pipeline.run(cfg, runs=runs)
    track_layer = "link" if "link" in cfg.layers else "track"
    ball_by_seq = {s.name: pipeline.load_output(cfg, runs, s, "ball") for s in seqs}
    tracks_by_seq = {s.name: pipeline.load_output(cfg, runs, s, track_layer)["frames"] for s in seqs}

    gold_frames = json.loads(boxes_path.read_text(encoding="utf-8"))["frames"] if boxes_path.exists() else []
    gold_events = json.loads(events_path.read_text(encoding="utf-8")).get("events", []) if events_path.exists() else []
    box_m = score_boxes(ball_by_seq, tracks_by_seq, gold_frames)
    ev_m = score_events(ball_by_seq, tracks_by_seq, gold_events, gold_v0_root=gold_v0_root)
    ms_vals = [float(v.get("ms_per_frame", 0.0)) for v in ball_by_seq.values()]
    return {
        "sequences": len(seqs),
        **box_m,
        **ev_m,
        "ms_per_frame": statistics.mean(ms_vals) if ms_vals else 0.0,
    }
