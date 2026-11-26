"""Fused Layer 7 backend: static-FP filtered DAG DP ball tracker, Viterbi possession, and rim-trajectory shots.

Designed for PLK broadcast footage where:
1. Rim-relative false positives (arena fixtures, rim hardware) fire repeatedly at constant offsets from the rim.
2. Ball detections drop out behind players or near the rim during passes and shots.
3. ``player-in-possession`` boxes are high-precision but miss contested holds, while ball-to-player proximity fills
   gaps and Viterbi smoothing removes 1-2 frame flicker.
4. ``detect-plk-r3-poss`` intentionally dropped action classes (``player-jump-shot``, ``player-layup-dunk``,
   ``ball-in-basket``), so shot detection is driven primarily by ball-to-rim approach/contact/exit geometry with
   ``ball-in-basket`` and action detections used as auxiliary cues when present.

"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, ClassVar

from bga.ball.base import (
    BallTrackState,
    bbox_iou,
    effective_fps,
    extract_track_positions,
    frame_track_boxes,
    interpolate_ball_2d,
    is_referee_track,
    last_holder_before,
    match_box_to_track,
    nearest_basket,
    pack_output,
    point_to_box_dist,
    shot_points,
    track_teams_map,
)
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def _rim_center_w(box: list[float]) -> tuple[float, float, float]:
    """Return ``(cx, cy, width)`` of a rim bounding box (width floored at 16 px)."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, max(16.0, box[2] - box[0])


def _track_rims(
    frames: list[int], det_frames: dict[str, list[list[Any]]], rim_conf: float, max_gap: int = 4
) -> dict[int, list[float]]:
    """Select one smooth rim box per frame and linearly bridge short ``<= max_gap`` dropouts."""
    raw: dict[int, list[float]] = {}
    prev: list[float] | None = None
    for f in frames:
        cands = [b for b in det_frames.get(str(f), []) if b[5] == "rim" and float(b[4]) >= rim_conf]
        if not cands:
            continue
        if prev is None:
            best = max(cands, key=lambda b: float(b[4]))
        else:
            pcx, pcy, _ = _rim_center_w(prev)

            def _score(b: list[Any], _pcx: float = pcx, _pcy: float = pcy) -> float:
                cx, cy, _ = _rim_center_w(b)
                return float(b[4]) - math.hypot(cx - _pcx, cy - _pcy) / 600.0

            best = max(cands, key=_score)
        prev = [float(best[0]), float(best[1]), float(best[2]), float(best[3]), float(best[4])]
        raw[f] = prev

    out = dict(raw)
    present = [i for i, f in enumerate(frames) if f in raw]
    for a_i, b_i in itertools.pairwise(present):
        gap = b_i - a_i
        if 1 < gap <= max_gap:
            ra, rb = raw[frames[a_i]], raw[frames[b_i]]
            if math.hypot((ra[0] + ra[2] - rb[0] - rb[2]) / 2.0, (ra[1] + ra[3] - rb[1] - rb[3]) / 2.0) <= 250.0:
                for k in range(1, gap):
                    t = k / gap
                    out[frames[a_i + k]] = [ra[m] + t * (rb[m] - ra[m]) for m in range(5)]
    return out


def _is_player_attached(
    bx: float,
    by: float,
    boxes: list[list[Any]],
    tracks: list[tuple[int, list[float], str]],
    teams: dict[int, str],
) -> bool:
    """Return ``True`` when ``(bx, by)`` sits inside or near a non-referee player or possession box."""
    if any(
        b[5] == "player-in-possession" and float(b[4]) >= 0.15 and point_to_box_dist(bx, by, b[:4]) <= 40.0
        for b in boxes
    ):
        return True
    return any(
        not is_referee_track(tid, cls, teams) and point_to_box_dist(bx, by, xyxy) <= 25.0 for tid, xyxy, cls in tracks
    )


def _cluster_cooccurs(
    hit_frames: set[int],
    raw_balls: dict[int, list[tuple[float, float, float]]],
    rims: dict[int, list[float]],
    cx: float,
    cy: float,
    cooccur_dist: float,
) -> int:
    """Count cluster frames that also contain another distant ball detection."""
    return sum(
        1
        for f in hit_frames
        if any(
            math.hypot((ox - _rim_center_w(rims[f])[0]) - cx, (oy - _rim_center_w(rims[f])[1]) - cy) > cooccur_dist
            for ox, oy, _ in raw_balls.get(f, [])
        )
    )


def _static_rim_clusters(
    frames: list[int],
    raw_balls: dict[int, list[tuple[float, float, float]]],
    rims: dict[int, list[float]],
    min_frames: int,
    cluster_px: float,
    *,
    unattached: dict[int, list[tuple[float, float, float]]] | None = None,
    min_share: float = 0.5,
) -> list[tuple[float, float]]:
    """Identify persistent rim-relative ``(dx, dy)`` clusters caused by static background false positives.

    Excludes player-attached candidates and requires each cluster to either co-occur with another distant ball
    candidate on multiple frames or persist across at least ``min_share`` of rim-visible frames, preventing a
    real ball held still under a static camera from being removed.

    """
    src = raw_balls if unattached is None else unattached
    rel_pts: list[tuple[int, float, float]] = []
    for f in frames:
        rim = rims.get(f)
        if rim is None:
            continue
        rcx, rcy, _ = _rim_center_w(rim)
        for bx, by, _ in src.get(f, []):
            rel_pts.append((f, bx - rcx, by - rcy))
    if len(rel_pts) < min_frames:
        return []

    buckets: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for f, rx, ry in rel_pts:
        buckets.setdefault((int(rx // cluster_px), int(ry // cluster_px)), []).append((f, rx, ry))

    clusters: list[tuple[float, float]] = []
    cooccur_min = max(2, min_frames // 2)
    cooccur_dist = 3.0 * cluster_px
    share_min = max(min_frames, round(min_share * len(rims)))
    for (gx, gy), pts in sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(pts) < max(2, min_frames // 2):
            continue
        seed_x = sum(p[1] for p in pts) / len(pts)
        seed_y = sum(p[2] for p in pts) / len(pts)
        if any(math.hypot(seed_x - cx, seed_y - cy) <= cluster_px for cx, cy in clusters):
            continue
        near = [
            p
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for p in buckets.get((gx + dx, gy + dy), [])
            if math.hypot(p[1] - seed_x, p[2] - seed_y) <= cluster_px
        ]
        hit_frames = {p[0] for p in near}
        if len(hit_frames) < min_frames:
            continue
        cx = sum(p[1] for p in near) / len(near)
        cy = sum(p[2] for p in near) / len(near)
        if (
            _cluster_cooccurs(hit_frames, raw_balls, rims, cx, cy, cooccur_dist) >= cooccur_min
            or len(hit_frames) >= share_min
        ):
            clusters.append((cx, cy))
    return clusters


def _filter_ball_candidates(
    frames: list[int],
    det_frames: dict[str, list[list[Any]]],
    rims: dict[int, list[float]],
    params: dict[str, Any],
    *,
    eff_fps: float = 10.0,
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]] | None = None,
    teams: dict[int, str] | None = None,
) -> dict[int, list[tuple[float, float, float]]]:
    """Extract per-frame ball candidates ``(cx, cy, conf)`` and drop static rim-relative false positives."""
    min_conf = float(params["ball_min_conf"])
    cluster_px = float(params["static_cluster_px"])
    min_frames = max(2, round(float(params["static_min_s"]) * eff_fps))
    min_share = float(params.get("static_min_share", 0.5))
    tr_map = tracks_by_f or {}
    tm_map = teams or {}

    raw: dict[int, list[tuple[float, float, float]]] = {}
    unattached: dict[int, list[tuple[float, float, float]]] = {}
    for f in frames:
        boxes = det_frames.get(str(f), [])
        tracks = tr_map.get(f, [])
        cands = [
            ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0, float(b[4]))
            for b in boxes
            if b[5] == "ball" and float(b[4]) >= min_conf
        ]
        raw[f] = cands
        unattached[f] = [c for c in cands if not _is_player_attached(c[0], c[1], boxes, tracks, tm_map)]

    clusters = _static_rim_clusters(
        frames,
        raw,
        rims,
        min_frames,
        cluster_px,
        unattached=unattached,
        min_share=min_share,
    )
    if not clusters:
        return raw

    out: dict[int, list[tuple[float, float, float]]] = {}
    for f in frames:
        rim = rims.get(f)
        if rim is None:
            out[f] = raw[f]
            continue
        rcx, rcy, _ = _rim_center_w(rim)
        out[f] = [
            (bx, by, conf)
            for bx, by, conf in raw[f]
            if not any(math.hypot((bx - rcx) - cx, (by - rcy) - cy) <= cluster_px for cx, cy in clusters)
        ]
    return out


def _ball_node_score(
    bx: float,
    by: float,
    conf: float,
    boxes: list[list[Any]],
    tracks: list[tuple[int, list[float], str]],
    teams: dict[int, str],
    params: dict[str, Any],
    rim: list[float] | None = None,
) -> float:
    """Unary score for a candidate ball detection: confidence + possession/player proximity + rim context.

    With ``ball_rim_bonus > 0`` a candidate within ``ball_rim_bonus_rw`` rim widths of the rim gains up to that bonus
    (linearly less with distance), so the trajectory through the rim is not outscored by faint detections next to
    players, which collect the possession and player bonuses.

    """
    score = conf - float(params["ball_prior"])
    rim_bonus = float(params.get("ball_rim_bonus", 0.0))
    if rim is not None and rim_bonus > 0.0:
        rcx, rcy, rw = _rim_center_w(rim)
        nd = math.hypot(bx - rcx, by - rcy) / rw
        reach = float(params.get("ball_rim_bonus_rw", 1.5))
        if nd <= reach:
            score += rim_bonus * (1.0 - nd / reach)
    pip_boxes = [b for b in boxes if b[5] == "player-in-possession" and float(b[4]) >= 0.15]
    if pip_boxes:
        d_pip = min(point_to_box_dist(bx, by, b[:4]) for b in pip_boxes)
        if d_pip <= 45.0:
            score += float(params["ball_pip_bonus"]) * (1.0 - d_pip / 45.0)
    player_boxes = [xyxy for tid, xyxy, cls in tracks if not is_referee_track(tid, cls, teams)]
    if player_boxes and min(point_to_box_dist(bx, by, xyxy) for xyxy in player_boxes) <= 25.0:
        score += 0.08
    return score


def _best_ball_predecessor(
    nodes: list[tuple[int, float, float, float, float]],
    dp: list[float],
    i: int,
    eff_fps: float,
    params: dict[str, Any],
    fallback: tuple[float, int],
) -> tuple[float, int]:
    """Scan nodes within ``ball_max_gap_s`` before ``i`` and return the highest-scoring predecessor."""
    best_val, best_j = fallback
    max_gap = max(1, round(float(params["ball_max_gap_s"]) * eff_fps))
    step_s = 1.0 / max(eff_fps, 1e-6)
    t1, x1, y1, _, sc1 = nodes[i]
    max_speed = float(params["ball_max_speed_px_s"])
    smooth_bonus = float(params["ball_smooth_bonus"])
    jump_pen = float(params["ball_jump_penalty"])
    gap_pen_s = float(params["ball_gap_penalty_s"])
    for j in range(i - 1, -1, -1):
        t0, x0, y0, _, _ = nodes[j]
        dt = t1 - t0
        if dt <= 0:
            continue
        if dt > max_gap:
            break
        dt_s = dt * step_s
        lim = max_speed * 0.1 * ((dt_s / 0.1) ** 0.85)
        dist = math.hypot(x1 - x0, y1 - y0)
        trans = (
            smooth_bonus * (1.0 - dist / lim) - gap_pen_s * max(0.0, dt_s - step_s)
            if dist <= lim
            else -jump_pen - gap_pen_s * dt_s
        )
        val = dp[j] + trans + sc1
        if val > best_val:
            best_val, best_j = val, j
    return best_val, best_j


def _track_ball(
    frames: list[int],
    candidates: dict[int, list[tuple[float, float, float]]],
    det_frames: dict[str, list[list[Any]]],
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]],
    teams: dict[int, str],
    params: dict[str, Any],
    *,
    eff_fps: float = 10.0,
    rims: dict[int, list[float]] | None = None,
) -> tuple[dict[int, list[float] | None], dict[int, float]]:
    """Find the maximum-score multi-segment ball trajectory via time-ordered DAG dynamic programming.

    Links detections within ``ball_max_gap_s`` by spatial continuity and allows restarting a fresh segment across
    longer occlusions so every disjoint positive-scoring run survives.

    """
    nodes: list[tuple[int, float, float, float, float]] = []
    for idx, f in enumerate(frames):
        boxes = det_frames.get(str(f), [])
        tracks = tracks_by_f.get(f, [])
        for bx, by, conf in candidates.get(f, []):
            sc = _ball_node_score(bx, by, conf, boxes, tracks, teams, params, (rims or {}).get(f))
            nodes.append((idx, bx, by, conf, sc))

    ball_px: dict[int, list[float] | None] = dict.fromkeys(frames)
    ball_conf: dict[int, float] = dict.fromkeys(frames, 0.0)
    if not nodes:
        return ball_px, ball_conf

    max_gap = max(1, round(float(params["ball_max_gap_s"]) * eff_fps))
    dp = [0.0] * len(nodes)
    prev = [-1] * len(nodes)
    best_past_val = 0.0
    best_past_node = -1
    past_ptr = 0
    for i, (t1, _, _, _, sc1) in enumerate(nodes):
        while past_ptr < i and t1 - nodes[past_ptr][0] > max_gap:
            if dp[past_ptr] > best_past_val:
                best_past_val, best_past_node = dp[past_ptr], past_ptr
            past_ptr += 1
        best_val, best_j = _best_ball_predecessor(nodes, dp, i, eff_fps, params, (best_past_val + sc1, best_past_node))
        dp[i], prev[i] = best_val, best_j

    end_i = max(range(len(nodes)), key=dp.__getitem__)
    if dp[end_i] <= 0.0:
        return ball_px, ball_conf

    cur = end_i
    while cur >= 0:
        t_idx, bx, by, conf, _ = nodes[cur]
        f = frames[t_idx]
        ball_px[f] = [bx, by]
        ball_conf[f] = conf
        cur = prev[cur]
    return ball_px, ball_conf


def _possession_emissions(
    boxes: list[list[Any]],
    tracks: list[tuple[int, list[float], str]],
    teams: dict[int, str],
    bpx: list[float] | None,
    bconf: float,
    params: dict[str, Any],
) -> dict[int | None, float]:
    """Compute per-candidate emission scores (players + ``None`` unheld state) for one frame."""
    scores: dict[int | None, float] = {None: float(params["poss_none_bias"])}
    players = [(tid, xyxy) for tid, xyxy, cls in tracks if not is_referee_track(tid, cls, teams)]
    for tid, _ in players:
        scores[tid] = 0.0

    pip_min = float(params["pip_min_conf"])
    pip_w = float(params["pip_weight"])
    for b in boxes:
        if b[5] != "player-in-possession" or float(b[4]) < pip_min:
            continue
        conf = float(b[4])
        matched = match_box_to_track(b[:4], tracks, teams)
        if matched is not None and matched in scores:
            scores[matched] += pip_w * conf
        for tid, xyxy in players:
            ov = bbox_iou(b[:4], xyxy)
            if ov >= 0.4 and tid != matched:
                scores[tid] += 0.4 * pip_w * conf * ov

    if bpx is not None:
        max_r = float(params["ball_hold_radius_px"])
        ball_w = float(params["ball_prox_weight"])
        w = ball_w * min(1.0, bconf / 0.45)
        far_w = float(params.get("ball_far_penalty", 0.0)) * min(1.0, bconf / 0.45)
        nearest_d = float("inf")
        for tid, xyxy in players:
            d = point_to_box_dist(bpx[0], bpx[1], xyxy)
            nearest_d = min(nearest_d, d)
            if d <= max_r:
                scores[tid] += w * (1.0 - d / max_r)
            elif far_w > 0.0:
                scores[tid] -= far_w * min(1.0, (d - max_r) / max_r)
        if nearest_d > max_r * 1.35:
            scores[None] += float(params["poss_flight_bonus"])
    return scores


def _track_possession(
    frames: list[int],
    det_frames: dict[str, list[list[Any]]],
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]],
    teams: dict[int, str],
    ball_px: dict[int, list[float] | None],
    ball_conf: dict[int, float],
    params: dict[str, Any],
    none_bonus: dict[int, float] | None = None,
) -> dict[int, int | None]:
    """Decode the MAP ball-holder sequence via Viterbi over ``{None} U active_players``.

    ``none_bonus`` adds a per-frame score to the unheld state (see :func:`_loose_ball_bonus`).

    """
    if not frames:
        return {}
    loose = none_bonus or {}
    stay_same = float(params["poss_stay_bonus"])
    stay_none = float(params["poss_none_stay_bonus"])
    switch_pen = float(params["poss_switch_penalty"])
    drop_pen = float(params["poss_drop_penalty"])

    dp_prev: dict[int | None, float] = {}
    back: list[dict[int | None, int | None]] = []

    for i, f in enumerate(frames):
        em = _possession_emissions(
            det_frames.get(str(f), []), tracks_by_f.get(f, []), teams, ball_px.get(f), ball_conf.get(f, 0.0), params
        )
        em[None] += loose.get(f, 0.0)
        if i == 0:
            dp_prev = em
            back.append(dict.fromkeys(em))
            continue
        dp_cur: dict[int | None, float] = {}
        bp_cur: dict[int | None, int | None] = {}
        for s_cur, e_val in em.items():
            best_v, best_p = -1e18, None
            for s_prev, p_val in dp_prev.items():
                if s_cur == s_prev:
                    tr = stay_none if s_cur is None else stay_same
                elif s_prev is None or s_cur is None:
                    tr = -drop_pen
                else:
                    tr = -switch_pen
                v = p_val + tr + e_val
                if v > best_v:
                    best_v, best_p = v, s_prev
            dp_cur[s_cur] = best_v
            bp_cur[s_cur] = best_p
        dp_prev = dp_cur
        back.append(bp_cur)

    state: int | None = max(dp_prev, key=dp_prev.__getitem__)
    out: dict[int, int | None] = {}
    for i in range(len(frames) - 1, -1, -1):
        out[frames[i]] = state
        state = back[i][state]
    return out


def _rim_relative_ball(
    frames: list[int], ball_px: dict[int, list[float] | None], rims: dict[int, list[float]]
) -> list[tuple[float, float, float] | None]:
    """Express tracked ball centre in rim-width units ``(ndx, ndy, ndist)`` for each step."""
    rel: list[tuple[float, float, float] | None] = []
    for f in frames:
        bpx, rim = ball_px.get(f), rims.get(f)
        if bpx is None or rim is None:
            rel.append(None)
            continue
        rcx, rcy, rw = _rim_center_w(rim)
        ndx, ndy = (bpx[0] - rcx) / rw, (bpx[1] - rcy) / rw
        rel.append((ndx, ndy, math.hypot(ndx, ndy)))
    return rel


def _classify_shot_outcome(
    state: BallTrackState,
    rim_idx: int,
    rel: list[tuple[float, float, float] | None],
    det_frames: dict[str, list[list[Any]]],
    params: dict[str, Any],
    eff_fps: float,
) -> tuple[str, int]:
    """Classify a rim-arrival cluster as ``made`` or ``missed`` from net-drop trajectory and ``ball-in-basket``."""
    post_steps = max(2, round(float(params["shot_post_s"]) * eff_fps))
    bib_pre = max(1, round(0.1 * eff_fps))
    bib_post = max(1, round(0.3 * eff_fps))
    bib_hits = sum(
        1
        for j in range(max(0, rim_idx - bib_pre), min(len(state.frames), rim_idx + bib_post + 1))
        if any(
            b[5] == "ball-in-basket" and float(b[4]) >= float(params["bib_conf"])
            for b in det_frames.get(str(state.frames[j]), [])
        )
    )
    post = [
        (j, rel[j]) for j in range(rim_idx + 1, min(len(state.frames), rim_idx + post_steps + 1)) if rel[j] is not None
    ]
    end_f = state.frames[rim_idx]
    if bib_hits >= 1:
        if post:
            end_target = rim_idx + max(1, round(0.2 * eff_fps))
            end_j = next((j for j, _ in post if j >= end_target), post[-1][0])
            end_f = state.frames[end_j]
        return "made", end_f

    # Check if the ball drops straight beneath the hoop through the net cylinder (|ndx| <= 0.85, ndy in [0.45, 3.0])
    # without first bouncing up (ndy < -0.45) or laterally away (|ndx| > 1.25).
    net_hits = 0
    for j, pt in post:
        if pt is None:
            continue
        ndx, ndy, _ = pt
        if ndy < -0.45 or abs(ndx) > 1.25:
            return "missed", state.frames[j]
        if abs(ndx) <= 0.85 and ndy >= 0.45:
            net_hits += 1
            end_f = state.frames[j]
            if net_hits >= 2 or ndy >= 1.1:
                return "made", end_f
    return "missed", end_f


def _rim_hit_clusters(
    rel: list[tuple[float, float, float] | None],
    rim_r: float,
    cooldown: int,
    approach_steps: int,
    arc_steps: int,
    *,
    strict_arc: bool = False,
) -> list[list[int]]:
    """Find separated clusters of step indices where the ball arrives at the rim from outside.

    With ``strict_arc`` the ball must have been at or above rim height in the ``arc_steps`` before the hit; a point
    at rim height reached straight from below (rim hardware, a ball under the basket) no longer counts.

    Examples:
        >>> rel = [(0.0, 4.0, 4.0), (0.0, 4.2, 4.2), (0.4, 0.0, 0.4)]
        >>> _rim_hit_clusters(rel, 0.95, 5, 5, 5), _rim_hit_clusters(rel, 0.95, 5, 5, 5, strict_arc=True)
        ([[2]], [])

    """
    hits: list[int] = []
    for i, pt in enumerate(rel):
        if pt is None or pt[2] > rim_r:
            continue
        _, ndy, _ = pt
        pre = [rel[j] for j in range(max(0, i - approach_steps), i) if rel[j] is not None]
        pre_recent = [rel[j] for j in range(max(0, i - arc_steps), i) if rel[j] is not None]
        from_above = any(p[1] <= 0.15 for p in pre_recent) or (ndy <= 0.15 and not strict_arc)
        if pre and any(p[2] >= 1.25 for p in pre) and from_above:
            hits.append(i)
    clusters: list[list[int]] = []
    for idx in hits:
        if not clusters or idx - clusters[-1][-1] > cooldown:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)
    return clusters


def _shot_rim_clusters(
    rel: list[tuple[float, float, float] | None], params: dict[str, Any], eff_fps: float
) -> list[list[int]]:
    """Rim-arrival clusters of a sequence from the ``shot_*`` params (shared by possession and shot detection)."""
    return _rim_hit_clusters(
        rel,
        float(params["shot_rim_radius_rw"]),
        max(2, round(float(params["shot_cooldown_s"]) * eff_fps)),
        max(2, round(float(params["shot_approach_s"]) * eff_fps)),
        max(1, round(float(params["shot_arc_s"]) * eff_fps)),
        strict_arc=bool(params.get("shot_strict_arc", False)),
    )


def _loose_ball_bonus(
    frames: list[int],
    clusters: list[list[int]],
    rel: list[tuple[float, float, float] | None],
    params: dict[str, Any],
    eff_fps: float,
) -> dict[int, float]:
    """Unheld-state bonus for steps around each rim arrival, when the ball is at the rim or loose under it.

    On PLK dev gold most false holds fall between just before a rim hit and the rebound or inbound pickup, where
    the ball overlaps the boxes of players under the basket. ``poss_loose_bonus`` is added to the unheld state from
    ``poss_loose_pre_s`` before to ``poss_loose_post_s`` after each rim frame; ``0`` disables it (the v2 behaviour).

    Examples:
        >>> rel = [(0.0, -3.0, 3.0), (0.0, 0.0, 0.0), None, None]
        >>> p = {"poss_loose_bonus": 1.0, "poss_loose_pre_s": 0.0, "poss_loose_post_s": 0.2}
        >>> _loose_ball_bonus([0, 5, 10, 15], [[1]], rel, p, 10.0)
        {5: 1.0, 10: 1.0, 15: 1.0}

    """
    bonus = float(params.get("poss_loose_bonus", 0.0))
    if bonus <= 0.0 or not clusters:
        return {}
    pre = round(float(params.get("poss_loose_pre_s", 0.0)) * eff_fps)
    post = round(float(params.get("poss_loose_post_s", 0.0)) * eff_fps)
    out: dict[int, float] = {}
    for cl in clusters:
        rim_idx = min(cl, key=lambda j: rel[j][2] if rel[j] is not None else 1e9)
        for j in range(max(0, rim_idx - pre), min(len(frames), rim_idx + post + 1)):
            out[frames[j]] = bonus
    return out


def _approach_start(rel: list[tuple[float, float, float] | None], rim_idx: int, excl_rw: float, max_steps: int) -> int:
    """Step index where the ball was last seen outside ``excl_rw`` rim widths before arriving at ``rim_idx``.

    Holders assigned while the ball is at the rim are players standing under it, not the shooter, so the release is
    searched for before the ball entered the rim zone. ``excl_rw <= 0`` keeps ``rim_idx`` (the M9 behaviour).

    Examples:
        >>> rel = [(0.0, -3.0, 3.0), (0.0, -1.0, 1.0), None, (0.0, 0.0, 0.0)]
        >>> _approach_start(rel, 3, 1.5, 10), _approach_start(rel, 3, 0.0, 10)
        (0, 3)

    """
    if excl_rw <= 0.0:
        return rim_idx
    lo = max(0, rim_idx - max_steps)
    j = rim_idx
    while j > lo and (rel[j] is None or rel[j][2] <= excl_rw):
        j -= 1
    return j


def _possession_team(state: BallTrackState, idx: int, window: int) -> str | None:
    """Return the team that held the ball on most steps of ``window`` steps up to ``idx``, or ``None``.

    Examples:
        >>> st = BallTrackState([0, 1, 2, 3], {}, {}, {0: 1, 1: 1, 2: 2, 3: None}, {1: "home", 2: "away"}, {})
        >>> _possession_team(st, 3, 4), _possession_team(st, 3, 0)
        ('home', None)

    """
    if window <= 0:
        return None
    counts: dict[str, int] = {}
    for j in range(max(0, idx - window + 1), idx + 1):
        tid = state.holders.get(state.frames[j])
        team = state.teams.get(tid) if tid is not None else None
        if team in ("home", "away"):
            counts[team] = counts.get(team, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


def _find_shooter(
    state: BallTrackState,
    rim_idx: int,
    rel: list[tuple[float, float, float] | None],
    params: dict[str, Any],
    eff_fps: float,
) -> tuple[int | None, int | None]:
    """Return ``(shooter_track_id, release_frame)``: the last holder before the ball's approach to the rim.

    With ``shot_release_rim_excl_rw > 0`` the search starts where the ball was last outside the rim zone. With
    ``shot_team_window_s > 0`` the shooter is the last holder from the team that held the ball most in that window
    before the release (a defender whose box overlaps the shooter at release loses to the team in possession).

    """
    lookback = max(2, round(float(params["shot_release_lookback_s"]) * eff_fps))
    start = _approach_start(rel, rim_idx, float(params.get("shot_release_rim_excl_rw", 0.0)), lookback)
    shooter, release = last_holder_before(state.frames, start, state.holders, lookback)
    window = round(float(params.get("shot_team_window_s", 0.0)) * eff_fps)
    team = _possession_team(state, start, window)
    if shooter is None or team is None or state.teams.get(shooter) == team:
        return shooter, release
    lo = max(0, start - lookback)
    for j in range(start, lo - 1, -1):
        tid = state.holders.get(state.frames[j])
        if tid is not None and state.teams.get(tid) == team:
            return tid, state.frames[j]
    return shooter, release


def _detect_shots(
    seq: Sequence,
    state: BallTrackState,
    det_frames: dict[str, list[list[Any]]],
    pos_by_tid: dict[int, dict[int, tuple[float, float]]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    """Detect shot attempts, outcomes, shooters, and 2/3-point values from rim-arrival trajectories."""
    eff_fps = effective_fps(seq)
    rel = _rim_relative_ball(state.frames, state.ball_px, state.rims)
    clusters = _shot_rim_clusters(rel, params, eff_fps)
    fallback_pre = max(1, round(0.3 * eff_fps))
    shots: list[dict[str, Any]] = []
    anchors: dict[int, tuple[float, float]] = {}

    for cl in clusters:
        rim_idx = min(cl, key=lambda j: rel[j][2] if rel[j] is not None else 1e9)
        rim_f = state.frames[rim_idx]
        outcome, end_f = _classify_shot_outcome(state, rim_idx, rel, det_frames, params, eff_fps)
        shooter_tid, release_f = _find_shooter(state, rim_idx, rel, params, eff_fps)
        start_f = release_f if release_f is not None else state.frames[max(0, cl[0] - fallback_pre)]
        pt = (
            pos_by_tid.get(shooter_tid, {}).get(release_f)
            if shooter_tid is not None and release_f is not None
            else None
        )
        basket = nearest_basket(pt[0], pt[1]) if pt is not None else None
        if basket is not None:
            anchors[rim_f] = basket
        pts = shot_points(pt[0], pt[1], basket) if pt is not None else 2
        shot_type = (
            "layup"
            if pt is not None and basket is not None and math.hypot(pt[0] - basket[0], pt[1] - basket[1]) < 2.2
            else "jump"
        )
        shots.append({
            "start_frame": start_f,
            "rim_frame": rim_f,
            "end_frame": end_f,
            "outcome": outcome,
            "shot_type": shot_type,
            "shooter_track_id": shooter_tid,
            "shooter_team": state.teams.get(shooter_tid) if shooter_tid is not None else None,
            "x_m": round(pt[0], 3) if pt is not None else None,
            "y_m": round(pt[1], 3) if pt is not None else None,
            "points": pts,
        })
    return shots, anchors


@register("ball", "fused")
class FusedBall(BaseBackend):
    """Fused Layer 7 backend: static-FP filtered DAG DP ball tracker, Viterbi possession, and rim-trajectory shots.

    Examples:
        >>> FusedBall({}).p["ball_max_gap_s"]
        0.6

    """

    defaults: ClassVar[dict[str, Any]] = {
        "rim_conf": 0.25,
        "rim_max_gap_s": 0.4,
        "ball_min_conf": 0.05,
        "ball_prior": 0.22,
        "ball_pip_bonus": 0.28,
        "ball_max_gap_s": 0.6,
        "ball_max_speed_px_s": 2600.0,
        "ball_smooth_bonus": 0.35,
        "ball_jump_penalty": 0.65,
        "ball_gap_penalty_s": 0.4,
        "static_min_s": 0.8,
        "static_min_share": 0.5,
        "static_cluster_px": 14.0,
        "pip_min_conf": 0.15,
        "pip_weight": 2.4,
        "ball_hold_radius_px": 65.0,
        "ball_prox_weight": 1.35,
        "poss_none_bias": 0.45,
        "poss_flight_bonus": 0.45,
        "poss_stay_bonus": 0.55,
        "poss_none_stay_bonus": 0.25,
        "poss_switch_penalty": 0.85,
        "poss_drop_penalty": 0.35,
        "ball_far_penalty": 0.0,
        "ball_rim_bonus": 0.0,
        "ball_rim_bonus_rw": 1.5,
        "poss_loose_bonus": 0.0,
        "poss_loose_pre_s": 0.0,
        "poss_loose_post_s": 0.0,
        "shot_rim_radius_rw": 0.95,
        "shot_approach_s": 1.4,
        "shot_arc_s": 0.8,
        "shot_cooldown_s": 1.2,
        "shot_post_s": 0.9,
        "shot_release_lookback_s": 2.0,
        "shot_strict_arc": False,
        "shot_release_rim_excl_rw": 0.0,
        "shot_team_window_s": 0.0,
        "bib_conf": 0.20,
    }

    def run(self, seq: Sequence, upstream: dict[str, Any], read: FrameReader) -> dict[str, Any]:
        """Track the ball, smooth possession via Viterbi, detect rim-trajectory shots, and map the ball to 2D.

        Args:
            seq: Sequence to process.
            upstream: Upstream layer outputs (``detect``, ``link`` or ``track``, optional ``team``, ``positions``,
                ``court``).
            read: Frame reader (unused).

        Returns:
            Layer 7 output dictionary conforming to :mod:`bga.ball.base`.

        """
        _ = read
        t0 = time.perf_counter()
        eff_fps = effective_fps(seq)
        det_frames = upstream["detect"]["frames"]
        tracks_out = upstream.get("link") or upstream["track"]
        team_out = upstream.get("team")
        positions_out = upstream.get("positions")
        court_out = upstream.get("court")

        teams = track_teams_map(tracks_out, team_out)
        tracks_by_f = frame_track_boxes(tracks_out)
        pos_by_tid = extract_track_positions(positions_out)

        rim_max_gap = max(1, round(float(self.p["rim_max_gap_s"]) * eff_fps))
        rims = _track_rims(seq.frames, det_frames, float(self.p["rim_conf"]), max_gap=rim_max_gap)
        cands = _filter_ball_candidates(
            seq.frames,
            det_frames,
            rims,
            self.p,
            eff_fps=eff_fps,
            tracks_by_f=tracks_by_f,
            teams=teams,
        )
        ball_px, ball_conf = _track_ball(
            seq.frames, cands, det_frames, tracks_by_f, teams, self.p, eff_fps=eff_fps, rims=rims
        )
        rel = _rim_relative_ball(seq.frames, ball_px, rims)
        loose = _loose_ball_bonus(seq.frames, _shot_rim_clusters(rel, self.p, eff_fps), rel, self.p, eff_fps)
        holders = _track_possession(seq.frames, det_frames, tracks_by_f, teams, ball_px, ball_conf, self.p, loose)
        state = BallTrackState(seq.frames, ball_px, ball_conf, holders, teams, rims)
        shots, anchors = _detect_shots(seq, state, det_frames, pos_by_tid, self.p)
        pos_2d = interpolate_ball_2d(
            seq.frames,
            holders,
            pos_by_tid,
            tracks_by_f,
            ball_px,
            court_out,
            shot_anchors=anchors,
            max_interp_frames=max(1, round(2.5 * seq.fps)),
            max_blind_frames=max(1, round(0.5 * seq.fps)),
        )
        ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(seq.frames))
        return pack_output(state, pos_2d, shots, ms)
