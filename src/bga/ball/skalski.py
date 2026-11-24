"""Skalski reference Layer 7 backend: ``player-in-possession`` matching and ``ShotEventTracker``.

Implements the baseline design from ``docs/PLAN.md`` section 5:

- Possession: highest-confidence ``player-in-possession`` box matched to the best active player track.
- Ball 2D position: holder's 2D court coordinates while held, linearly interpolated between release and reception
  in flight.
- Shot events: ``sports.basketball.ShotEventTracker`` driven by ``player-jump-shot``, ``player-layup-dunk``, and
  ``ball-in-basket`` detections, scaled to the sequence's effective frame rate.

"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from sports.basketball import ShotEventTracker

from bga.ball.base import (
    extract_track_positions,
    frame_track_boxes,
    interpolate_ball_2d,
    last_holder_before,
    match_box_to_track,
    nearest_basket,
    pack_output,
    shot_points,
    track_teams_map,
)
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


def _effective_fps(seq: Sequence) -> float:
    """Sequence sample rate in frames per second, accounting for frame stride."""
    stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
    return float(seq.fps) / max(1, stride)


def _best_box(boxes: list[list[Any]], cls_name: str, min_conf: float) -> list[Any] | None:
    """Highest-confidence detection of ``cls_name`` above ``min_conf``, or ``None``."""
    cands = [b for b in boxes if b[5] == cls_name and float(b[4]) >= min_conf]
    return max(cands, key=lambda b: float(b[4])) if cands else None


def _run_shot_tracker(
    seq: Sequence,
    det_frames: dict[str, list[list[Any]]],
    holders: dict[int, int | None],
    teams: dict[int, str],
    pos_by_tid: dict[int, dict[int, tuple[float, float]]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    """Run ``sports.basketball.ShotEventTracker`` over the sequence and attach shooter & 2D release coords."""
    eff_fps = _effective_fps(seq)
    tracker = ShotEventTracker(
        reset_time_frames=max(1, round(float(params["reset_s"]) * eff_fps)),
        minimum_frames_between_starts=max(1, round(float(params["min_between_starts_s"]) * eff_fps)),
        cooldown_frames_after_made=max(1, round(float(params["cooldown_after_made_s"]) * eff_fps)),
    )
    act_conf = float(params["action_conf"])
    lookback = max(1, round(float(params["shooter_lookback_s"]) * eff_fps))
    shots: list[dict[str, Any]] = []
    anchors: dict[int, tuple[float, float]] = {}
    pending_start: int | None = None
    for idx, f in enumerate(seq.frames):
        boxes = det_frames.get(str(f), [])
        has_js = any(b[5] == "player-jump-shot" and float(b[4]) >= act_conf for b in boxes)
        has_ld = any(b[5] == "player-layup-dunk" and float(b[4]) >= act_conf for b in boxes)
        has_bib = any(b[5] == "ball-in-basket" and float(b[4]) >= act_conf for b in boxes)
        for ev in tracker.update(idx, has_js, has_ld, has_bib):
            ev_name = str(ev["event"])
            ev_idx = int(ev["frame"])
            if ev_name == "START":
                pending_start = ev_idx
                continue
            if ev_name not in ("MADE", "MISSED"):
                continue
            s_idx = pending_start if pending_start is not None else ev_idx
            pending_start = None
            shooter_tid, release_f = last_holder_before(seq.frames, s_idx, holders, lookback)
            rel_pt = (
                pos_by_tid.get(shooter_tid, {}).get(release_f)
                if shooter_tid is not None and release_f is not None
                else None
            )
            end_f = seq.frames[ev_idx]
            if rel_pt is not None:
                anchors[end_f] = nearest_basket(rel_pt[0], rel_pt[1])
            shots.append({
                "start_frame": seq.frames[s_idx],
                "rim_frame": end_f,
                "end_frame": end_f,
                "outcome": "made" if ev_name == "MADE" else "missed",
                "shot_type": str(ev["type"]).lower(),
                "shooter_track_id": shooter_tid,
                "shooter_team": teams.get(shooter_tid) if shooter_tid is not None else None,
                "x_m": round(rel_pt[0], 3) if rel_pt is not None else None,
                "y_m": round(rel_pt[1], 3) if rel_pt is not None else None,
                "points": shot_points(rel_pt[0], rel_pt[1]) if rel_pt is not None else 2,
            })
    return shots, anchors


@register("ball", "skalski")
class SkalskiBall(BaseBackend):
    """Reference Layer 7 backend: nearest ``player-in-possession`` box + ``ShotEventTracker``.

    Examples:
        >>> 'pip_conf' in SkalskiBall.defaults and 'rim_conf' in SkalskiBall.defaults
        True

    """

    defaults: ClassVar[dict[str, Any]] = {
        "pip_conf": 0.35,
        "ball_conf": 0.25,
        "rim_conf": 0.25,
        "action_conf": 0.25,
        "reset_s": 1.7,
        "min_between_starts_s": 0.5,
        "cooldown_after_made_s": 0.5,
        "shooter_lookback_s": 1.5,
    }

    def run(self, seq: Sequence, upstream: dict[str, Any], read: FrameReader) -> dict[str, Any]:
        """Assign frame-level possession from ``player-in-possession`` boxes and detect shots via ``ShotEventTracker``.

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
        pip_conf = float(self.p["pip_conf"])
        ball_conf_thr = float(self.p["ball_conf"])
        rim_conf_thr = float(self.p["rim_conf"])

        holders: dict[int, int | None] = {}
        ball_px: dict[int, list[float] | None] = {}
        ball_conf: dict[int, float] = {}
        rims: dict[int, list[float]] = {}
        for f in seq.frames:
            boxes = det_frames.get(str(f), [])
            pip = _best_box(boxes, "player-in-possession", pip_conf)
            holders[f] = match_box_to_track(pip[:4], tracks_by_f.get(f, []), teams) if pip is not None else None
            ball = _best_box(boxes, "ball", ball_conf_thr)
            ball_px[f] = (
                [(float(ball[0]) + float(ball[2])) / 2.0, (float(ball[1]) + float(ball[3])) / 2.0] if ball else None
            )
            ball_conf[f] = float(ball[4]) if ball else 0.0
            rim = _best_box(boxes, "rim", rim_conf_thr)
            if rim is not None:
                rims[f] = [float(rim[0]), float(rim[1]), float(rim[2]), float(rim[3]), float(rim[4])]

        shots, anchors = _run_shot_tracker(seq, det_frames, holders, teams, pos_by_tid, self.p)
        pos_2d = interpolate_ball_2d(
            seq.frames, holders, pos_by_tid, tracks_by_f, ball_px, upstream.get("court"), shot_anchors=anchors
        )
        ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(seq.frames))
        return pack_output(seq.frames, pos_2d, ball_px, ball_conf, holders, teams, rims, shots, ms)
