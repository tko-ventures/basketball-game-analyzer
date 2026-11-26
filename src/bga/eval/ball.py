"""Ball, possession, and shot-event evaluation on gold shots (plan section 8, M9).

Scores any Layer 7 (``ball``) backend against two gold sources:

1. ``data/gold/v1/boxes.json`` (76 dev / 24 held-out frames):
   - ``ball_f1`` / ``ball_r`` / ``ball_med_px``: tracked ``ball_px`` matched to gold ``ball`` boxes within
     ``max(25 px, box_side)``.
   - ``poss_acc`` / ``holder_f1`` / ``team_acc``: predicted ``holder_track_id`` box matched to the gold holder box
     at ``IoU >= 0.5`` (and exact ``None`` match when no holder is labelled). The holder comes from
     ``data/gold/v1/holders.json`` (the 2026-11-20 possession definition) when present, else from the frame's
     ``player-in-possession`` box.
2. ``data/gold/v1/events.json`` (verified shot-attempt ground truth):
   - ``shot_f1`` / ``shot_outcome_acc`` (with ``shot_made_r`` / ``shot_missed_r``) / ``shot_shooter_acc``: greedy
     1-to-1 matching of predicted shots to gold shots on sequences listed in ``verified_sequences`` within
     ``rim_tol_frames`` (default 40 source frames = 0.8 s at 50 fps, matching on ``rim_frame`` or ``start_frame``
     vs gold ``release_frame`` when ``rim_frame`` is ``None``), checking ``outcome`` (``made`` / ``missed``) and
     shooter track IoU against ``data/gold/v0/<seq>/gt/gt.txt``.

"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

from bga import pipeline
from bga.ball.base import bbox_iou
from bga.config import RunConfig
from bga.eval.scores import read_gt
from bga.stage import Sequence

log = logging.getLogger(__name__)

#: Default paths for gold v1 boxes, shot events, holders, and exported shot gold.
GOLD_V1_BOXES = Path("data/gold/v1/boxes.json")
GOLD_V1_EVENTS = Path("data/gold/v1/events.json")
GOLD_V1_HOLDERS = Path("data/gold/v1/holders.json")
GOLD_V1_SHOT_GOLD = Path("data/gold/v1/shot_gold.json")
GOLD_V0_ROOT = Path("data/gold/v0")
#: Maximum source-frame gap between predicted and gold shot frame for a match (0.8 s at 50 fps).
DEFAULT_RIM_TOL_FRAMES = 40
#: Shot-matching tolerance in seconds; turned into source frames with each sequence's frame rate when it is known.
DEFAULT_RIM_TOL_S = 0.8
#: Maximum source-frame gap for matching a predicted shooter track to a shot-gold box (0.2 s at 50 fps).
SHOOTER_BOX_TOL_FRAMES = 10
#: Largest gap, in source frames, between a predicted shot's start and a gold ``v0`` frame boxing the shooter: gold
#: tracks often lose the shooter in the air, and the nearest frame then holds no box of theirs (0.5 s at 50 fps).
SHOOTER_GT_GAP_FRAMES = 25


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


def load_holders(path: Path = GOLD_V1_HOLDERS) -> dict[tuple[str, int], int | None] | None:
    """Read holder gold as ``{(youtube_id, frame): box index | None}``, or ``None`` when the file is absent.

    Args:
        path: Path to ``data/gold/v1/holders.json``.

    Returns:
        The holder box index (into the frame's ``boxes.json`` boxes) of each gold frame, ``None`` where unheld.

    Examples:
        >>> load_holders(Path("missing.json")) is None
        True

    """
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["frames"]
    return {(r["youtube_id"], int(r["frame"])): r["holder"] for r in rows}


def load_shot_gold(
    path: Path = GOLD_V1_SHOT_GOLD,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], int | None], list[dict[str, Any]], set[str]]:
    """Read exported shot gold into ``(gold_frames, holders, gold_events, verified_sequences)``.

    Converts each shot-gold frame into the ``boxes.json`` + ``holders.json`` layout consumed by :func:`score_boxes`,
    so both gold sets are scored with the exact same ball and holder rules.

    Args:
        path: Path to ``data/gold/v1/shot_gold.json``.

    Returns:
        Four-tuple ``(frames, holders, events, verified_sequences)``; all empty when ``path`` is absent.

    Examples:
        >>> load_shot_gold(Path("missing.json"))
        ([], {}, [], set())

    """
    if not path.exists():
        return [], {}, [], set()
    doc = json.loads(path.read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []
    holders: dict[tuple[str, int], int | None] = {}
    for fr in doc.get("frames", []):
        boxes: list[dict[str, Any]] = []
        if fr.get("ball") is not None:
            boxes.append({"cls": "ball", "xyxy": fr["ball"]})
        holder = fr.get("holder")
        key = (fr["youtube_id"], int(fr["frame"]))
        if holder is not None:
            holders[key] = len(boxes)
            boxes.append({"cls": "player-in-possession", "xyxy": holder["xyxy"], "team": holder["team"]})
        else:
            holders[key] = None
        frames.append({
            "youtube_id": fr["youtube_id"],
            "shot": int(fr["shot"]),
            "frame": int(fr["frame"]),
            "boxes": boxes,
        })
    return frames, holders, list(doc.get("events", [])), set(doc.get("verified_sequences", []))


def _gold_holder(fr: dict[str, Any], holders: dict[tuple[str, int], int | None] | None) -> list[dict[str, Any]]:
    """Return the frame's gold holder box as a 0- or 1-element list (holder gold first, then box classes)."""
    key = (fr["youtube_id"], int(fr["frame"]))
    if holders is not None and key in holders:
        idx = holders[key]
        return [] if idx is None else [fr["boxes"][idx]]
    return [b for b in fr["boxes"] if b["cls"] == "player-in-possession"]


def score_boxes(
    ball_by_seq: dict[str, dict[str, Any]],
    tracks_by_seq: dict[str, dict[str, list[list[Any]]]],
    gold_frames: list[dict[str, Any]],
    holders: dict[tuple[str, int], int | None] | None = None,
) -> dict[str, Any]:
    """Score per-frame ball position and possession against ``data/gold/v1/boxes.json`` frames.

    Args:
        ball_by_seq: Mapping ``seq_name -> Layer 7 output dict``.
        tracks_by_seq: Mapping ``seq_name -> {frame_str: [[tid, x1, y1, x2, y2, ...], ...]}``.
        gold_frames: Filtered list of frame entries from ``boxes.json``.
        holders: Holder gold from :func:`load_holders`; frames it lacks fall back to ``player-in-possession`` boxes.

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
        >>> score_boxes({"s_s1": ball["s1"]}, {"s_s1": tracks["s1"]}, gf, holders={("s", 0): None})["poss_acc"]
        0.0

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
        gt_h = _gold_holder(fr, holders)
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
    """Greedy closest-frame 1-to-1 matching between gold and predicted shots in one sequence.

    Matches predicted ``rim_frame`` against gold ``rim_frame``; when a backend emits ``rim_frame: None``
    (e.g. a timeout-based miss), falls back to matching ``start_frame`` against gold ``release_frame``.

    """
    pairs: list[tuple[int, int, int]] = []
    for gi, gs in enumerate(gt_shots):
        g_rim = int(gs["rim_frame"])
        g_rel = int(gs["release_frame"]) if gs.get("release_frame") is not None else g_rim
        for pi, ps in enumerate(pred_shots):
            p_rim = ps.get("rim_frame")
            if p_rim is not None:
                dist = abs(int(p_rim) - g_rim)
            else:
                p_start = ps.get("start_frame", ps.get("end_frame", 0))
                dist = abs(int(p_start) - g_rel)
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


def _nearest_track_frame(
    tracks: dict[str, list[list[Any]]],
    tid: int,
    target_f: int,
    max_tol: int,
) -> int | None:
    """Find the frame closest to ``target_f`` (within ``max_tol``) where ``tid`` has a track box."""
    if _box_for_tid(tracks.get(str(target_f), []), tid) is not None:
        return target_f
    best_f: int | None = None
    best_dist = max_tol + 1
    for fstr, rows in tracks.items():
        dist = abs(int(fstr) - target_f)
        if dist < best_dist and _box_for_tid(rows, tid) is not None:
            best_f, best_dist = int(fstr), dist
    return best_f


def _shooter_matches_box(
    gs: dict[str, Any],
    ps: dict[str, Any],
    tracks: dict[str, list[list[Any]]],
) -> bool:
    """Match a predicted shooter track to a shot-gold ``shooter_xyxy`` box at ``release_frame``."""
    pred_tid = ps.get("shooter_track_id")
    gt_box = gs.get("shooter_xyxy")
    if pred_tid is None or gt_box is None or not tracks:
        return False
    rf = int(gs.get("release_frame", gs["rim_frame"]))
    near_f = _nearest_track_frame(tracks, int(pred_tid), rf, SHOOTER_BOX_TOL_FRAMES)
    if near_f is None:
        return False
    best_iou, best_tid = 0.0, None
    for row in tracks.get(str(near_f), []):
        ov = bbox_iou(gt_box, [float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        if ov > best_iou:
            best_iou, best_tid = ov, int(row[0])
    return best_iou >= 0.4 and best_tid == int(pred_tid)


def _shooter_matches(
    gs: dict[str, Any],
    ps: dict[str, Any],
    tracks: dict[str, list[list[Any]]],
    gt_by_f: dict[int, list[tuple[Any, ...]]],
) -> bool:
    """Check whether a predicted shooter matches gold ``shooter_gt_tid``, ``shooter_xyxy``, or fallback team.

    Against a gold ``v0`` track, boxes are compared on the gold frame nearest the predicted start where that track has
    a box (within ``SHOOTER_GT_GAP_FRAMES``); an attempt whose gold shooter has no box that close does not match.

    """
    gt_tid = gs.get("shooter_gt_tid")
    if gt_tid is None:
        if gs.get("shooter_xyxy") is not None:
            return _shooter_matches_box(gs, ps, tracks)
        return bool(gs.get("shooter_team") and ps.get("shooter_team") == gs.get("shooter_team"))
    pred_tid = ps.get("shooter_track_id")
    if pred_tid is None or not gt_by_f:
        return False
    sf = int(ps.get("start_frame", gs["rim_frame"]))
    boxed = [
        f
        for f, rows in gt_by_f.items()
        if abs(f - sf) <= SHOOTER_GT_GAP_FRAMES and any(int(g[0]) == int(gt_tid) for g in rows)
    ]
    if not boxed:
        return False
    near_f = min(boxed, key=lambda f: abs(f - sf))
    track_f = _nearest_track_frame(tracks, int(pred_tid), near_f, SHOOTER_BOX_TOL_FRAMES)
    pbox = _box_for_tid(tracks.get(str(track_f), []), int(pred_tid)) if track_f is not None else None
    if pbox is None:
        return False
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
    verified_sequences: set[str] | None = None,
    fps_by_seq: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score predicted field-goal attempts, made/missed outcomes, and shooters against ``events.json``.

    Free throws (gold attempts worth 1 point, predictions with ``kind == "free_throw"``) are left out here and
    scored by :func:`score_free_throws`.

    ``shot_made_r`` and ``shot_missed_r`` are the outcome recalls of matched attempts whose gold outcome is made or
    missed: with far more made than missed gold attempts, ``shot_outcome_acc`` alone mostly measures made recall.

    Args:
        ball_by_seq: Mapping ``seq_name -> Layer 7 output dict``.
        tracks_by_seq: Mapping ``seq_name -> {frame_str: track_rows}``.
        gold_events: List of gold shot dicts (``sequence``, ``rim_frame``, ``outcome``, etc.).
        gold_v0_root: Root directory of MOTChallenge gold tracks (for ``shooter_gt_tid`` IoU check).
        rim_tol_frames: Maximum source-frame difference for a valid match, for sequences missing from
            ``fps_by_seq``.
        verified_sequences: When provided, restrict both gold events and predictions to these sequence names so
            unverified prefill drafts are never scored as gold.
        fps_by_seq: Frame rate of each sequence; shots then match within ``DEFAULT_RIM_TOL_S`` seconds, so 60 fps
            full games get the same window as 50 fps highlights.

    Returns:
        Dict of shot detection, outcome, and shooter metrics (``NaN`` if no verified gold events cover the run).

    Examples:
        >>> ball = {"s1": {"shots": [{"start_frame": 10, "rim_frame": 50, "outcome": "made", "shooter_team": "home"}]}}
        >>> gt = [{"sequence": "s1", "rim_frame": 55, "outcome": "made", "shooter_team": "home"}]
        >>> m = score_events(ball, {"s1": {}}, gt)
        >>> (m["shot_f1"], m["shot_outcome_acc"], m["shot_shooter_acc"])
        (1.0, 1.0, 1.0)
        >>> (m["shot_made_r"], m["shot_missed_r"])
        (1.0, nan)

    """
    eval_seqs = (
        {s: v for s, v in ball_by_seq.items() if s in verified_sequences}
        if verified_sequences is not None
        else ball_by_seq
    )
    gt_by_seq: dict[str, list[dict[str, Any]]] = {}
    for ev in gold_events:
        if ev["sequence"] in eval_seqs and not is_free_throw(ev):
            gt_by_seq.setdefault(ev["sequence"], []).append(ev)
    if not eval_seqs or (verified_sequences is None and not gt_by_seq):
        return {
            "shot_gt": 0,
            "shot_pred": sum(not is_free_throw(p) for v in eval_seqs.values() for p in v.get("shots", [])),
            "shot_p": float("nan"),
            "shot_r": float("nan"),
            "shot_f1": float("nan"),
            "shot_outcome_acc": float("nan"),
            "shot_made_r": float("nan"),
            "shot_missed_r": float("nan"),
            "shot_shooter_acc": float("nan"),
        }

    tp = fp = fn = out_ok = shoot_ok = shoot_tot = 0
    by_outcome = {o: [0, 0] for o in ("made", "missed")}  # gold outcome -> [right, total] over matched attempts
    for sname, ball_out in eval_seqs.items():
        gt_list = gt_by_seq.get(sname, [])
        pred_list = [p for p in ball_out.get("shots", []) if not is_free_throw(p)]
        fps = (fps_by_seq or {}).get(sname)
        matches = _match_shots(gt_list, pred_list, round(DEFAULT_RIM_TOL_S * fps) if fps else rim_tol_frames)
        tp += len(matches)
        fp += len(pred_list) - len(matches)
        fn += len(gt_list) - len(matches)
        gt_by_f: dict[int, list[tuple[Any, ...]]] = {}
        if matches and (gold_v0_root / sname / "gt" / "gt.txt").exists():
            _, gt_by_f, _ = read_gt(gold_v0_root, sname)
        for gi, pi in matches:
            gs, ps = gt_list[gi], pred_list[pi]
            out_ok += int(ps.get("outcome") == gs.get("outcome"))
            if gs.get("outcome") in by_outcome:
                by_outcome[gs["outcome"]][0] += int(ps.get("outcome") == gs["outcome"])
                by_outcome[gs["outcome"]][1] += 1
            if (
                gs.get("shooter_gt_tid") is not None
                or gs.get("shooter_xyxy") is not None
                or gs.get("shooter_team") is not None
            ):
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
        "shot_made_r": by_outcome["made"][0] / by_outcome["made"][1] if by_outcome["made"][1] else float("nan"),
        "shot_missed_r": by_outcome["missed"][0] / by_outcome["missed"][1] if by_outcome["missed"][1] else float("nan"),
        "shot_shooter_acc": shoot_ok / shoot_tot if shoot_tot else float("nan"),
    }


def is_free_throw(shot: dict[str, Any]) -> bool:
    """Tell a free throw from a field goal: a gold attempt worth 1 point, or a prediction of kind ``free_throw``.

    Examples:
        >>> is_free_throw({"points": 1}), is_free_throw({"kind": "free_throw"}), is_free_throw({"points": 2})
        (True, True, False)

    """
    return shot.get("points") == 1 or shot.get("kind") == "free_throw"


def score_free_throws(
    ball_by_seq: dict[str, dict[str, Any]],
    gold_events: list[dict[str, Any]],
    free_throw_sequences: set[str],
    fps_by_seq: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score predicted free throws and their outcomes on the sequences whose free throws were labelled.

    Args:
        ball_by_seq: Mapping ``seq_name -> Layer 7 output dict``.
        gold_events: Gold shot dicts; free throws are those worth 1 point.
        free_throw_sequences: Sequences whose free throws were labelled (``free_throw_sequences`` in shot gold).
        fps_by_seq: Frame rate of each sequence, for the ``DEFAULT_RIM_TOL_S`` window.

    Returns:
        ``ft_gt``, ``ft_pred``, ``ft_p``, ``ft_r``, ``ft_f1`` and ``ft_outcome_acc`` (``NaN`` without such sequences).

    Examples:
        >>> ball = {"s1": {"shots": [{"rim_frame": 50, "outcome": "made", "kind": "free_throw"}]}}
        >>> gt = [{"sequence": "s1", "rim_frame": 52, "outcome": "missed", "points": 1}]
        >>> m = score_free_throws(ball, gt, {"s1"}, {"s1": 60.0})
        >>> (m["ft_f1"], m["ft_outcome_acc"])
        (1.0, 0.0)

    """
    tp = fp = fn = out_ok = 0
    seqs = [s for s in ball_by_seq if s in free_throw_sequences]
    for sname in seqs:
        gt_list = [e for e in gold_events if e["sequence"] == sname and is_free_throw(e)]
        pred_list = [p for p in ball_by_seq[sname].get("shots", []) if is_free_throw(p)]
        fps = (fps_by_seq or {}).get(sname)
        matches = _match_shots(gt_list, pred_list, round(DEFAULT_RIM_TOL_S * fps) if fps else DEFAULT_RIM_TOL_FRAMES)
        tp, fp, fn = tp + len(matches), fp + len(pred_list) - len(matches), fn + len(gt_list) - len(matches)
        out_ok += sum(pred_list[pi].get("outcome") == gt_list[gi].get("outcome") for gi, pi in matches)
    if not seqs:
        nan = float("nan")
        return {"ft_gt": 0, "ft_pred": 0, "ft_p": nan, "ft_r": nan, "ft_f1": nan, "ft_outcome_acc": nan}
    p, r, f1 = _f1(tp, fp, fn)
    return {
        "ft_gt": tp + fn,
        "ft_pred": tp + fp,
        "ft_p": p,
        "ft_r": r,
        "ft_f1": f1,
        "ft_outcome_acc": out_ok / tp if tp else float("nan"),
    }


def load_free_throw_sequences(path: Path = GOLD_V1_SHOT_GOLD) -> set[str]:
    """Read the shot-gold sequences whose free throws were labelled (empty when absent).

    Examples:
        >>> load_free_throw_sequences(Path("missing.json"))
        set()

    """
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("free_throw_sequences", []))


def _shot_gold_config(cfg: RunConfig, shot_gold_path: Path) -> RunConfig:
    """Clone ``cfg`` to run its pipeline over ``shots/shot_gold`` using the same stride and split."""
    sg = cfg.model_copy(deep=True)
    shots = sg.layers["shots"]
    params: dict[str, Any] = {
        "stride": shots.params.get("stride", 5),
        "held_out": shots.params.get("held_out", "exclude"),
        "queue": str(shot_gold_path),
    }
    if "labels" in shots.params:
        params["labels"] = shots.params["labels"]
    shots.backend = "shot_gold"
    shots.params = params
    return sg


def _collect_run_outputs(
    cfg: RunConfig,
    runs: Path,
    track_layer: str,
    shot_gold_path: Path | None,
) -> tuple[list[Sequence], dict[str, dict[str, Any]], dict[str, dict[str, list[list[Any]]]]]:
    """Run ``cfg`` (and its ``shots/shot_gold`` companion when scoring ``shots/gold``) and load ball/track outputs."""
    seqs: list[Sequence] = pipeline.run(cfg, runs=runs)
    ball_by_seq = {s.name: pipeline.load_output(cfg, runs, s, "ball") for s in seqs}
    tracks_by_seq = {s.name: pipeline.load_output(cfg, runs, s, track_layer)["frames"] for s in seqs}
    if shot_gold_path is not None and shot_gold_path.exists() and cfg.layers["shots"].backend == "gold":
        sg_cfg = _shot_gold_config(cfg, shot_gold_path)
        for s in pipeline.run(sg_cfg, runs=runs):
            if s.name in ball_by_seq:
                continue
            seqs.append(s)
            ball_by_seq[s.name] = pipeline.load_output(sg_cfg, runs, s, "ball")
            tracks_by_seq[s.name] = pipeline.load_output(sg_cfg, runs, s, track_layer)["frames"]
    return seqs, ball_by_seq, tracks_by_seq


def score(
    cfg: RunConfig,
    runs: Path = Path("runs"),
    *,
    boxes_path: Path = GOLD_V1_BOXES,
    events_path: Path = GOLD_V1_EVENTS,
    gold_v0_root: Path = GOLD_V0_ROOT,
    holders_path: Path = GOLD_V1_HOLDERS,
    shot_gold_path: Path | None = GOLD_V1_SHOT_GOLD,
) -> dict[str, Any]:
    """Run ``cfg`` (reusing cached outputs) and score its ``ball`` layer across gold ``v1`` and shot gold.

    Only sequences listed in ``verified_sequences`` in ``events_path`` or ``shot_gold_path`` are included in
    shot-event metrics; any unverified selected sequence is logged and reported in ``unverified_sequences``.

    Args:
        cfg: Run config including ``ball``.
        runs: Root of pipeline run outputs.
        boxes_path: Path to ``data/gold/v1/boxes.json``.
        events_path: Path to ``data/gold/v1/events.json`` (skipped gracefully if absent).
        gold_v0_root: Path to ``data/gold/v0`` MOTChallenge tracks.
        holders_path: Path to ``data/gold/v1/holders.json`` (``player-in-possession`` boxes are used if absent).
        shot_gold_path: Path to ``data/gold/v1/shot_gold.json`` (``None`` or missing to score gold ``v1`` only).

    Returns:
        Combined metrics dict covering ball tracking, possession, shot events, latency, and unverified sequences.

    """
    track_layer = "link" if "link" in cfg.layers else "track"
    seqs, ball_by_seq, tracks_by_seq = _collect_run_outputs(cfg, runs, track_layer, shot_gold_path)

    gold_frames = json.loads(boxes_path.read_text(encoding="utf-8"))["frames"] if boxes_path.exists() else []
    holders = load_holders(holders_path)
    events_doc = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else {}
    gold_events = list(events_doc.get("events", []))
    verified = set(events_doc.get("verified_sequences", []))

    has_shot_gold = shot_gold_path is not None and shot_gold_path.exists()
    if has_shot_gold:
        sg_frames, sg_holders, sg_events, sg_verified = load_shot_gold(shot_gold_path)
        gold_frames = [*gold_frames, *sg_frames]
        holders = {**(holders or {}), **sg_holders}
        gold_events = [*gold_events, *sg_events]
        verified |= sg_verified

    unverified = sorted(s for s in ball_by_seq if s not in verified)
    if (events_path.exists() or has_shot_gold) and unverified:
        log.warning("Skipping shot-event scoring on unverified sequences: %s", ", ".join(unverified))
    box_m = score_boxes(ball_by_seq, tracks_by_seq, gold_frames, holders)
    ev_m = score_events(
        ball_by_seq,
        tracks_by_seq,
        gold_events,
        gold_v0_root=gold_v0_root,
        verified_sequences=verified,
        fps_by_seq={s.name: float(s.fps) for s in seqs},
    )
    ft_seqs = load_free_throw_sequences(shot_gold_path) & verified if has_shot_gold else set()
    ft_m = score_free_throws(ball_by_seq, gold_events, ft_seqs, {s.name: float(s.fps) for s in seqs})
    ms_vals = [float(v.get("ms_per_frame", 0.0)) for v in ball_by_seq.values()]
    return {
        "sequences": len(seqs),
        **box_m,
        **ev_m,
        **ft_m,
        "unverified_sequences": unverified,
        "ms_per_frame": statistics.mean(ms_vals) if ms_vals else 0.0,
    }
