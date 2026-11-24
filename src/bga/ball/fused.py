"""Fused Layer 7 backend: temporal ball tracking, Viterbi possession, and rim-trajectory shots.

Designed for PLK broadcast footage where ``detect-plk-r3-poss`` detects ``ball`` (91.4% recall at
``conf >= 0.05``), ``rim`` (F1 0.915), and ``player-in-possession``, while the retired single-frame
action classes (``player-jump-shot``, ``player-layup-dunk``, ``ball-in-basket``) are sparse.

"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, ClassVar

from bga.ball.base import (
    bbox_iou,
    extract_track_positions,
    frame_track_boxes,
    interpolate_ball_2d,
    is_referee_track,
    last_holder_before,
    nearest_basket,
    pack_output,
    point_to_box_dist,
    shot_points,
    track_teams_map,
)
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def _effective_fps(seq: Sequence) -> float:
    """Sequence sample rate in frames per second, accounting for frame stride."""
    stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
    return float(seq.fps) / max(1, stride)


def _track_rims(
    frames: list[int],
    det_frames: dict[str, list[list[Any]]],
    rim_conf: float,
    max_gap: int = 4,
) -> dict[int, list[float]]:
    """Select the strongest rim box per frame and linearly bridge short dropouts."""
    raw: dict[int, list[float]] = {}
    for f in frames:
        cands = [b for b in det_frames.get(str(f), []) if b[5] == "rim" and float(b[4]) >= rim_conf]
        if cands:
            best = max(cands, key=lambda b: float(b[4]))
            raw[f] = [float(best[0]), float(best[1]), float(best[2]), float(best[3]), float(best[4])]
    present = [i for i, f in enumerate(frames) if f in raw]
    out = dict(raw)
    for i0, i1 in itertools.pairwise(present):
        if not (1 < i1 - i0 <= max_gap + 1):
            continue
        b0, b1 = raw[frames[i0]], raw[frames[i1]]
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        if math.hypot(c1[0] - c0[0], c1[1] - c0[1]) > 250.0:
            continue
        for j in range(i0 + 1, i1):
            t = (j - i0) / (i1 - i0)
            out[frames[j]] = [b0[k] + t * (b1[k] - b0[k]) for k in range(4)] + [0.5 * (b0[4] + b1[4])]
    return out


def _rim_center_w(rim: list[float]) -> tuple[float, float, float]:
    """Return ``(cx, cy, width)`` for a rim box ``[x1, y1, x2, y2, conf]``."""
    return (rim[0] + rim[2]) / 2.0, (rim[1] + rim[3]) / 2.0, max(25.0, rim[2] - rim[0])


def _static_rim_clusters(
    frames: list[int],
    raw_balls: dict[int, list[tuple[float, float, float]]],
    rims: dict[int, list[float]],
    min_frames: int,
    cluster_px: float,
) -> list[tuple[float, float]]:
    """Find rim-relative ``(dx, dy)`` offsets where false-positive balls persist across many frames."""
    offsets: list[tuple[int, float, float]] = []
    for f in frames:
        rim = rims.get(f)
        if rim is None:
            continue
        rcx, rcy, _ = _rim_center_w(rim)
        for bx, by, _ in raw_balls.get(f, []):
            dx, dy = bx - rcx, by - rcy
            if math.hypot(dx, dy) > 10.0:
                offsets.append((f, dx, dy))
    clusters: list[tuple[float, float]] = []
    for _, dx, dy in offsets:
        if any(math.hypot(dx - cx, dy - cy) <= cluster_px for cx, cy in clusters):
            continue
        hit_frames = {f for f, ox, oy in offsets if math.hypot(ox - dx, oy - dy) <= cluster_px}
        if len(hit_frames) >= min_frames:
            xs = [ox for _, ox, oy in offsets if math.hypot(ox - dx, oy - dy) <= cluster_px]
            ys = [oy for _, ox, oy in offsets if math.hypot(ox - dx, oy - dy) <= cluster_px]
            clusters.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return clusters


def _filter_ball_candidates(
    frames: list[int],
    det_frames: dict[str, list[list[Any]]],
    rims: dict[int, list[float]],
    params: dict[str, Any],
) -> dict[int, list[tuple[float, float, float]]]:
    """Extract ``ball`` detections above ``ball_min_conf`` and drop static arena/rim-hardware FPs."""
    min_conf = float(params["ball_min_conf"])
    raw: dict[int, list[tuple[float, float, float]]] = {}
    for f in frames:
        raw[f] = [
            ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0, float(b[4]))
            for b in det_frames.get(str(f), [])
            if b[5] == "ball" and float(b[4]) >= min_conf
        ]
    cluster_px = float(params["static_cluster_px"])
    clusters = _static_rim_clusters(frames, raw, rims, int(params["static_min_frames"]), cluster_px)
    if not clusters:
        return raw
    filtered: dict[int, list[tuple[float, float, float]]] = {}
    for f in frames:
        rim = rims.get(f)
        if rim is None:
            filtered[f] = raw[f]
            continue
        rcx, rcy, _ = _rim_center_w(rim)
        filtered[f] = [
            b
            for b in raw[f]
            if not any(math.hypot((b[0] - rcx) - cx, (b[1] - rcy) - cy) <= cluster_px for cx, cy in clusters)
        ]
    return filtered


def _ball_node_score(
    cand: tuple[float, float, float],
    boxes: list[list[Any]],
    tracks: list[tuple[int, list[float], str]],
    params: dict[str, Any],
) -> float:
    """Unary quality score for one candidate ball detection."""
    bx, by, conf = cand
    score = conf - float(params["ball_prior"])
    pip_boxes = [b[:4] for b in boxes if b[5] == "player-in-possession" and float(b[4]) >= 0.25]
    if any(point_to_box_dist(bx, by, pb) <= 55.0 for pb in pip_boxes):
        score += 0.14
    elif any(point_to_box_dist(bx, by, xyxy) <= 35.0 for _, xyxy, cls in tracks if cls != "referee"):
        score += 0.06
    return score


def _track_ball(
    frames: list[int],
    cands_by_f: dict[int, list[tuple[float, float, float]]],
    det_frames: dict[str, list[list[Any]]],
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]],
    params: dict[str, Any],
) -> tuple[dict[int, list[float] | None], dict[int, float]]:
    """Globally optimal single-ball path via DAG dynamic programming over filtered detections."""
    nodes: list[tuple[int, float, float, float, float]] = []
    for idx, f in enumerate(frames):
        boxes = det_frames.get(str(f), [])
        tracks = tracks_by_f.get(f, [])
        for bx, by, conf in cands_by_f.get(f, []):
            nodes.append((idx, bx, by, conf, _ball_node_score((bx, by, conf), boxes, tracks, params)))
    if not nodes:
        return dict.fromkeys(frames), dict.fromkeys(frames, 0.0)

    max_gap = int(params["ball_max_gap"])
    max_step = float(params["ball_max_step_px"])
    smooth_bonus = float(params["ball_smooth_bonus"])
    jump_pen = float(params["ball_jump_penalty"])
    gap_pen = float(params["ball_gap_penalty"])

    dp = [n[4] for n in nodes]
    prev = [-1] * len(nodes)
    for i, (t1, x1, y1, _, sc1) in enumerate(nodes):
        best_val, best_j = sc1, -1
        for j in range(i - 1, -1, -1):
            t0, x0, y0, _, _ = nodes[j]
            dt = t1 - t0
            if dt <= 0:
                continue
            if dt > max_gap:
                break
            d = math.hypot(x1 - x0, y1 - y0)
            lim = max_step * (dt**0.85)
            trans = (smooth_bonus * (1.0 - d / lim) if d <= lim else -jump_pen) - gap_pen * (dt - 1)
            val = dp[j] + trans + sc1
            if val > best_val:
                best_val, best_j = val, j
        dp[i], prev[i] = best_val, best_j

    end_i = max(range(len(nodes)), key=dp.__getitem__)
    ball_px: dict[int, list[float] | None] = dict.fromkeys(frames)
    ball_conf: dict[int, float] = dict.fromkeys(frames, 0.0)
    if dp[end_i] <= 0.0:
        return ball_px, ball_conf
    cur = end_i
    while cur != -1:
        t_idx, bx, by, conf, _ = nodes[cur]
        f = frames[t_idx]
        ball_px[f] = [bx, by]
        ball_conf[f] = conf
        cur = prev[cur]
    return ball_px, ball_conf


def _frame_possession_emissions(
    tracks: list[tuple[int, list[float], str]],
    teams: dict[int, str],
    boxes: list[list[Any]],
    bpx: list[float] | None,
    bconf: float,
    params: dict[str, Any],
) -> dict[int | None, float]:
    """Compute per-state unary possession scores for ``None`` (in-flight) and active players."""
    players = [(tid, xyxy) for tid, xyxy, cls in tracks if not is_referee_track(tid, cls, teams)]
    pip_dets = [b for b in boxes if b[5] == "player-in-possession" and float(b[4]) >= 0.05]
    w_pip = float(params["w_pip"])
    w_ball = float(params["w_ball"])
    prox_norm = float(params["ball_prox_norm"])
    free_norm = float(params["free_ball_norm"])
    ball_use_conf = float(params["poss_ball_min_conf"])

    scores: dict[int | None, float] = {None: float(params["none_base"])}
    min_d_norm = math.inf
    min_top = min((xyxy[1] for _, xyxy in players), default=0.0)

    for tid, xyxy in players:
        pip_s = max((float(b[4]) for b in pip_dets if bbox_iou(b[:4], xyxy) >= 0.30), default=0.0)
        ball_s = 0.0
        if bpx is not None and bconf >= ball_use_conf:
            h = max(40.0, xyxy[3] - xyxy[1])
            d_norm = point_to_box_dist(bpx[0], bpx[1], xyxy) / h
            min_d_norm = min(min_d_norm, d_norm)
            high_above = bpx[1] < xyxy[1] - 0.15 * h and d_norm > 0.15
            if not high_above and d_norm <= prox_norm:
                ball_s = math.exp(-d_norm / 0.12) * min(1.0, bconf / 0.55)
        scores[tid] = w_pip * pip_s + w_ball * ball_s

    if bpx is not None and bconf >= ball_use_conf and players and (min_d_norm > free_norm or bpx[1] < min_top - 25.0):
        scores[None] += float(params["free_ball_boost"])
    return scores


def _track_possession(
    frames: list[int],
    det_frames: dict[str, list[list[Any]]],
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]],
    teams: dict[int, str],
    ball_px: dict[int, list[float] | None],
    ball_conf: dict[int, float],
    params: dict[str, Any],
) -> dict[int, int | None]:
    """Decode smooth per-frame ball holder via Viterbi over ``{player_tids..., None}``."""
    stay_bonus = float(params["stay_bonus"])
    none_stay = float(params["none_stay_bonus"])
    switch_pen = float(params["switch_penalty"])

    prev_dp: dict[int | None, float] = {}
    back: list[dict[int | None, int | None]] = []
    for f in frames:
        em = _frame_possession_emissions(
            tracks_by_f.get(f, []), teams, det_frames.get(str(f), []), ball_px.get(f), ball_conf.get(f, 0.0), params
        )
        if not prev_dp:
            prev_dp = em
            back.append(dict.fromkeys(em))
            continue
        cur_dp: dict[int | None, float] = {}
        cur_bk: dict[int | None, int | None] = {}
        for st, esc in em.items():
            best_v, best_p = -math.inf, None
            for pst, pval in prev_dp.items():
                if st == pst:
                    tr = none_stay if st is None else stay_bonus
                elif st is not None and pst is not None:
                    tr = -switch_pen
                else:
                    tr = 0.0
                v = pval + tr + esc
                if v > best_v:
                    best_v, best_p = v, pst
            cur_dp[st], cur_bk[st] = best_v, best_p
        prev_dp = cur_dp
        back.append(cur_bk)

    holders: dict[int, int | None] = {}
    st = max(prev_dp, key=prev_dp.__getitem__) if prev_dp else None
    for idx in range(len(frames) - 1, -1, -1):
        holders[frames[idx]] = st
        st = back[idx].get(st)
    return holders


def _rim_rel_series(
    frames: list[int],
    ball_px: dict[int, list[float] | None],
    rims: dict[int, list[float]],
) -> list[tuple[float, float, float] | None]:
    """Per-step rim-normalised ball offsets ``(ndx, ndy, dist_rw)`` or ``None``."""
    out: list[tuple[float, float, float] | None] = []
    for f in frames:
        bpx, rim = ball_px.get(f), rims.get(f)
        if bpx is None or rim is None:
            out.append(None)
            continue
        rcx, rcy, rw = _rim_center_w(rim)
        ndx, ndy = (bpx[0] - rcx) / rw, (bpx[1] - rcy) / rw
        out.append((ndx, ndy, math.hypot(ndx, ndy)))
    return out


def _classify_shot_outcome(
    rim_idx: int,
    rel: list[tuple[float, float, float] | None],
    frames: list[int],
    ball_conf: dict[int, float],
    det_frames: dict[str, list[list[Any]]],
    post_steps: int,
    params: dict[str, Any],
) -> tuple[str, int]:
    """Classify a rim arrival as ``made`` or ``missed`` from post-rim descent geometry and ``ball-in-basket``."""
    hi = min(len(frames), rim_idx + post_steps + 1)
    bib_thr = float(params["bib_confirm_conf"])
    has_bib = any(
        b[5] == "ball-in-basket" and float(b[4]) >= bib_thr
        for j in range(max(0, rim_idx - 1), min(len(frames), rim_idx + 4))
        for b in det_frames.get(str(frames[j]), [])
    )
    min_post_conf = float(params["shot_post_min_conf"])
    start_j = rim_idx if (rel[rim_idx] is not None and rel[rim_idx][1] >= 0.45) else rim_idx + 1
    post = [
        (j, rel[j])
        for j in range(start_j, hi)
        if rel[j] is not None and rel[j][2] <= 3.0 and ball_conf.get(frames[j], 0.0) >= min_post_conf
    ]
    if not post:
        return ("made" if has_bib else "missed"), frames[rim_idx]

    made_x_lim = float(params["made_x_mult"])
    first_drop = next(((j, pt) for j, pt in post if pt is not None and pt[1] >= 0.45), None)
    max_ndy = max(pt[1] for _, pt in post if pt is not None)
    end_f = frames[post[min(2, len(post) - 1)][0]]
    if first_drop is None:
        return ("made" if has_bib else "missed"), end_f
    drop_idx, drop_pt = first_drop
    pre_drop_wide = any(
        pt is not None and abs(pt[0]) > made_x_lim * 1.45 and pt[1] < 0.45 for j, pt in post if j < drop_idx
    )
    in_net_cyl = abs(drop_pt[0]) <= made_x_lim and max_ndy >= 0.90 and not pre_drop_wide
    return ("made" if (in_net_cyl or has_bib) else "missed"), frames[drop_idx]


def _rim_hit_clusters(
    frames: list[int],
    rel: list[tuple[float, float, float] | None],
    holders: dict[int, int | None],
    x_lim: float,
    cooldown: int,
) -> list[list[int]]:
    """Group airborne rim-entry step indices separated by at least ``cooldown`` steps."""
    hits: list[int] = []
    for i, pt in enumerate(rel):
        if pt is None or holders.get(frames[i]) is not None:
            continue
        ndx, ndy, _ = pt
        if abs(ndx) > x_lim or not (-1.15 <= ndy <= 0.85):
            continue
        pre = [rel[j] for j in range(max(0, i - 14), i) if rel[j] is not None]
        if pre and any(p[2] >= 1.25 for p in pre) and (any(p[1] <= 0.15 for p in pre[-8:]) or ndy <= 0.15):
            hits.append(i)
    clusters: list[list[int]] = []
    for i in hits:
        if clusters and i - clusters[-1][-1] <= cooldown:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _detect_shots(
    seq: Sequence,
    ball_px: dict[int, list[float] | None],
    ball_conf: dict[int, float],
    rims: dict[int, list[float]],
    det_frames: dict[str, list[list[Any]]],
    holders: dict[int, int | None],
    teams: dict[int, str],
    pos_by_tid: dict[int, dict[int, tuple[float, float]]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    """Detect shot attempts and made/missed outcomes from tracked ball-to-rim trajectories."""
    frames = seq.frames
    eff_fps = _effective_fps(seq)
    rel = _rim_rel_series(frames, ball_px, rims)
    cooldown = max(1, round(float(params["shot_cooldown_s"]) * eff_fps))
    post_steps = max(2, round(float(params["shot_post_s"]) * eff_fps))
    lookback = max(2, round(float(params["shooter_lookback_s"]) * eff_fps))
    clusters = _rim_hit_clusters(frames, rel, holders, float(params["rim_x_mult"]), cooldown)

    shots: list[dict[str, Any]] = []
    anchors: dict[int, tuple[float, float]] = {}
    for cl in clusters:
        rim_idx = min(cl, key=lambda j: rel[j][2] if rel[j] is not None else math.inf)
        outcome, end_f = _classify_shot_outcome(rim_idx, rel, frames, ball_conf, det_frames, post_steps, params)
        shooter_tid, release_f = last_holder_before(frames, max(0, cl[0] - 1), holders, lookback)
        start_f = release_f if release_f is not None else frames[max(0, cl[0] - 3)]
        rel_pt = (
            pos_by_tid.get(shooter_tid, {}).get(release_f)
            if shooter_tid is not None and release_f is not None
            else None
        )
        basket = nearest_basket(rel_pt[0], rel_pt[1]) if rel_pt is not None else None
        if basket is not None:
            anchors[frames[rim_idx]] = basket
        near_rim = (
            rel_pt is not None and basket is not None and math.hypot(rel_pt[0] - basket[0], rel_pt[1] - basket[1]) < 2.6
        )
        shots.append({
            "start_frame": start_f,
            "rim_frame": frames[rim_idx],
            "end_frame": end_f,
            "outcome": outcome,
            "shot_type": "layup" if near_rim else "jump",
            "shooter_track_id": shooter_tid,
            "shooter_team": teams.get(shooter_tid) if shooter_tid is not None else None,
            "x_m": round(rel_pt[0], 3) if rel_pt is not None else None,
            "y_m": round(rel_pt[1], 3) if rel_pt is not None else None,
            "points": shot_points(rel_pt[0], rel_pt[1], basket) if rel_pt is not None else 2,
        })
    return shots, anchors


@register("ball", "fused")
class FusedBall(BaseBackend):
    """Temporal ball tracker + Viterbi fused possession + rim-trajectory shot detection.

    Examples:
        >>> "ball_min_conf" in FusedBall.defaults and "w_pip" in FusedBall.defaults
        True

    """

    defaults: ClassVar[dict[str, Any]] = {
        "rim_conf": 0.25,
        "ball_min_conf": 0.05,
        "ball_prior": 0.22,
        "ball_max_gap": 6,
        "ball_max_step_px": 260.0,
        "ball_smooth_bonus": 0.35,
        "ball_jump_penalty": 0.65,
        "ball_gap_penalty": 0.04,
        "static_min_frames": 8,
        "static_cluster_px": 14.0,
        "w_pip": 1.15,
        "w_ball": 0.75,
        "poss_ball_min_conf": 0.20,
        "ball_prox_norm": 0.25,
        "free_ball_norm": 0.30,
        "none_base": 0.56,
        "free_ball_boost": 0.65,
        "stay_bonus": 0.30,
        "none_stay_bonus": 0.14,
        "switch_penalty": 0.25,
        "rim_x_mult": 1.10,
        "made_x_mult": 0.70,
        "bib_confirm_conf": 0.45,
        "shot_cooldown_s": 1.2,
        "shot_post_s": 1.0,
        "shot_post_min_conf": 0.15,
        "shooter_lookback_s": 2.2,
    }

    def run(self, seq: Sequence, upstream: dict[str, Any], read: FrameReader) -> dict[str, Any]:
        """Track the ball, decode fused player possession, and detect made/missed shots.

        Args:
            seq: Sequence to process.
            upstream: Outputs of ``detect``, ``track``, and optionally ``link``, ``team``, ``court``, ``positions``.
            read: Frame reader (unused).

        Returns:
            Layer 7 output dict conforming to ``bga.ball.base``.

        """
        t0 = time.perf_counter()
        det_frames = upstream["detect"]["frames"]
        tracks_out = upstream.get("link") or upstream["track"]
        tracks_by_f = frame_track_boxes(tracks_out)
        teams = track_teams_map(tracks_out, upstream.get("team"))
        pos_by_tid = extract_track_positions(upstream.get("positions"))

        rims = _track_rims(seq.frames, det_frames, float(self.p["rim_conf"]))
        cands = _filter_ball_candidates(seq.frames, det_frames, rims, self.p)
        ball_px, ball_conf = _track_ball(seq.frames, cands, det_frames, tracks_by_f, self.p)
        holders = _track_possession(seq.frames, det_frames, tracks_by_f, teams, ball_px, ball_conf, self.p)
        shots, anchors = _detect_shots(seq, ball_px, ball_conf, rims, det_frames, holders, teams, pos_by_tid, self.p)
        pos_2d = interpolate_ball_2d(
            seq.frames, holders, pos_by_tid, tracks_by_f, ball_px, upstream.get("court"), shot_anchors=anchors
        )
        ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(seq.frames))
        return pack_output(seq.frames, pos_2d, ball_px, ball_conf, holders, teams, rims, shots, ms)
