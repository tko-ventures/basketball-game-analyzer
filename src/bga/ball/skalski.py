"""Skalski reference Layer 7 backend: ``player-in-possession`` matching and action-based shot tracker.

Implements the baseline architecture from ``docs/PLAN.md`` §3 & §5:

- Possession: highest-confidence ``player-in-possession`` box matched to the nearest non-referee player track.
- Ball image position: highest-confidence ``ball`` detection per frame (no static-FP filtering or DP tracking).
- 2D court ball: holder's 2D position while held, linear interpolation in flight.
- Shot events: stride-invariant equivalent of Roboflow ``sports.basketball.ShotEventTracker`` driven by
  ``player-jump-shot``, ``player-layup-dunk``, and ``ball-in-basket`` detections, with all frame windows and
  consecutive-frame thresholds scaled to the sequence's effective FPS (no top-level ``sports`` import).

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from bga.ball.base import (
    BallTrackState,
    effective_fps,
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


@dataclass
class _ShotStateMachine:
    """Stride-invariant state machine matching ``sports.basketball.ShotEventTracker`` semantics."""

    reset_time_frames: int
    min_frames_between_starts: int
    cooldown_frames_after_made: int
    min_start_frames: int
    min_bib_frames: int

    shot_in_progress: bool = False
    shot_type: str = "jump"
    shot_start_idx: int | None = None
    shot_deadline_idx: int | None = None
    frames_since_start: int = 0
    consec_js: int = 0
    consec_ld: int = 0
    consec_bib: int = 0
    last_made_idx: int | None = None

    def _reset_shot(self) -> None:
        self.shot_in_progress = False
        self.shot_start_idx = None
        self.shot_deadline_idx = None
        self.frames_since_start = 0
        self.consec_js = self.consec_ld = self.consec_bib = 0

    def update(self, idx: int, has_js: bool, has_ld: bool, has_bib: bool) -> list[tuple[str, int, str]]:
        """Advance one sampled step and return ``(event, step_idx, shot_type)`` transitions."""
        events: list[tuple[str, int, str]] = []
        self.consec_js = self.consec_js + 1 if has_js else 0
        self.consec_ld = self.consec_ld + 1 if has_ld else 0
        self.consec_bib = self.consec_bib + 1 if has_bib else 0

        reached_js = self.consec_js == self.min_start_frames
        reached_ld = self.consec_ld == self.min_start_frames
        should_start = reached_js or reached_ld
        if (
            should_start
            and self.last_made_idx is not None
            and (idx - self.last_made_idx) < self.cooldown_frames_after_made
        ):
            should_start = False

        if should_start:
            if self.shot_in_progress:
                if self.frames_since_start >= self.min_frames_between_starts:
                    events.append(("MISSED", idx, self.shot_type))
                    self._reset_shot()
                else:
                    should_start = False
            if should_start:
                self.shot_in_progress = True
                self.shot_type = "jump" if reached_js else "layup"
                self.shot_start_idx = idx
                self.shot_deadline_idx = idx + self.reset_time_frames
                self.frames_since_start = 0
                self.consec_bib = 0
                events.append(("START", idx, self.shot_type))

        if self.shot_in_progress:
            self.frames_since_start += 1
            if self.consec_bib >= self.min_bib_frames:
                events.append(("MADE", idx, self.shot_type))
                self.last_made_idx = idx
                self._reset_shot()
            elif self.shot_deadline_idx is not None and idx >= self.shot_deadline_idx:
                events.append(("MISSED", idx, self.shot_type))
                self._reset_shot()
        return events


def _best_box(boxes: list[list[Any]], cls: str, min_conf: float) -> list[Any] | None:
    """Return the highest-confidence box of ``cls`` scoring at least ``min_conf``."""
    cands = [b for b in boxes if b[5] == cls and float(b[4]) >= min_conf]
    return max(cands, key=lambda b: float(b[4])) if cands else None


def _run_shot_tracker(
    seq: Sequence,
    state: BallTrackState,
    det_frames: dict[str, list[list[Any]]],
    pos_by_tid: dict[int, dict[int, tuple[float, float]]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    """Run the stride-scaled action shot tracker over the sequence and attach shooter & 2D release coords."""
    eff_fps = effective_fps(seq)
    tracker = _ShotStateMachine(
        reset_time_frames=max(1, round(float(params["reset_s"]) * eff_fps)),
        min_frames_between_starts=max(1, round(float(params["min_between_starts_s"]) * eff_fps)),
        cooldown_frames_after_made=max(1, round(float(params["cooldown_after_made_s"]) * eff_fps)),
        min_start_frames=max(1, round(float(params["min_start_s"]) * eff_fps)),
        min_bib_frames=max(1, round(float(params["min_bib_s"]) * eff_fps)),
    )
    act_conf = float(params["action_conf"])
    bib_conf = float(params["bib_conf"])
    lookback = max(1, round(float(params["shooter_lookback_s"]) * eff_fps))

    shots: list[dict[str, Any]] = []
    anchors: dict[int, tuple[float, float]] = {}
    open_start: tuple[int, str] | None = None

    for idx, f in enumerate(state.frames):
        boxes = det_frames.get(str(f), [])
        has_js = any(b[5] == "player-jump-shot" and float(b[4]) >= act_conf for b in boxes)
        has_ld = any(b[5] == "player-layup-dunk" and float(b[4]) >= act_conf for b in boxes)
        has_bib = any(b[5] == "ball-in-basket" and float(b[4]) >= bib_conf for b in boxes)
        for ev_name, ev_idx, ev_type in tracker.update(idx, has_js, has_ld, has_bib):
            if ev_name == "START":
                open_start = (ev_idx, ev_type)
            elif ev_name in {"MADE", "MISSED"}:
                start_idx, stype = open_start if open_start is not None else (ev_idx, ev_type)
                open_start = None
                start_f, end_f = state.frames[start_idx], state.frames[ev_idx]
                rim_f = end_f if ev_name == "MADE" else None
                shooter_tid, release_f = last_holder_before(state.frames, start_idx, state.holders, lookback)
                pt = (
                    pos_by_tid.get(shooter_tid, {}).get(release_f)
                    if shooter_tid is not None and release_f is not None
                    else None
                )
                basket = nearest_basket(pt[0], pt[1]) if pt is not None else None
                if basket is not None and rim_f is not None:
                    anchors[rim_f] = basket
                shots.append({
                    "start_frame": start_f,
                    "rim_frame": rim_f,
                    "end_frame": end_f,
                    "outcome": "made" if ev_name == "MADE" else "missed",
                    "shot_type": stype,
                    "shooter_track_id": shooter_tid,
                    "shooter_team": state.teams.get(shooter_tid) if shooter_tid is not None else None,
                    "x_m": round(pt[0], 3) if pt is not None else None,
                    "y_m": round(pt[1], 3) if pt is not None else None,
                    "points": shot_points(pt[0], pt[1], basket) if pt is not None else 2,
                    "kind": "field_goal",
                })
    return shots, anchors


@register("ball", "skalski")
class SkalskiBall(BaseBackend):
    """Reference Layer 7 backend: nearest ``player-in-possession`` box + action-based shot tracker.

    Examples:
        >>> SkalskiBall({}).p["pip_conf"]
        0.25

    """

    defaults: ClassVar[dict[str, Any]] = {
        "pip_conf": 0.25,
        "ball_conf": 0.25,
        "rim_conf": 0.25,
        "action_conf": 0.25,
        "bib_conf": 0.25,
        "reset_s": 1.7,
        "min_between_starts_s": 0.5,
        "cooldown_after_made_s": 0.5,
        "min_start_s": 0.10,
        "min_bib_s": 0.07,
        "shooter_lookback_s": 1.5,
    }

    def run(self, seq: Sequence, upstream: dict[str, Any], read: FrameReader) -> dict[str, Any]:
        """Assign frame-level possession from ``player-in-possession`` boxes and detect shots from action boxes.

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
        det_frames = upstream["detect"]["frames"]
        tracks_out = upstream.get("link") or upstream["track"]
        team_out = upstream.get("team")
        positions_out = upstream.get("positions")
        court_out = upstream.get("court")

        teams = track_teams_map(tracks_out, team_out)
        tracks_by_f = frame_track_boxes(tracks_out)
        pos_by_tid = extract_track_positions(positions_out)

        pip_conf = float(self.p["pip_conf"])
        ball_min = float(self.p["ball_conf"])
        rim_min = float(self.p["rim_conf"])

        ball_px: dict[int, list[float] | None] = {}
        ball_conf: dict[int, float] = {}
        holders: dict[int, int | None] = {}
        rims: dict[int, list[float]] = {}

        for f in seq.frames:
            boxes = det_frames.get(str(f), [])
            b_box = _best_box(boxes, "ball", ball_min)
            if b_box is not None:
                ball_px[f] = [(float(b_box[0]) + float(b_box[2])) / 2.0, (float(b_box[1]) + float(b_box[3])) / 2.0]
                ball_conf[f] = float(b_box[4])
            else:
                ball_px[f] = None
                ball_conf[f] = 0.0

            r_box = _best_box(boxes, "rim", rim_min)
            if r_box is not None:
                rims[f] = [float(r_box[0]), float(r_box[1]), float(r_box[2]), float(r_box[3]), float(r_box[4])]

            p_box = _best_box(boxes, "player-in-possession", pip_conf)
            holders[f] = match_box_to_track(p_box[:4], tracks_by_f.get(f, []), teams) if p_box is not None else None

        state = BallTrackState(seq.frames, ball_px, ball_conf, holders, teams, rims)
        shots, anchors = _run_shot_tracker(seq, state, det_frames, pos_by_tid, self.p)
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
