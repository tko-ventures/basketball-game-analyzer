"""Radar video per sequence (plan section 9, M2/M8/M9): broadcast frame above, FIBA court with players below.

Consumes ``link`` when configured (falling back to ``track``), ``team``, ``identity`` (or ``numbers``),
``positions``, and optionally ``ball``. Draws team-coloured player boxes, foot ground points, confirmed jersey
numbers, and (when ``ball`` is configured) the ball dot, holder highlight ring, and made/missed shot markers on
both the broadcast frame and the 2D FIBA court. Written to ``runs/<clip>/radar/<config>@<hash8>/<sequence>.mp4``.

"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bga.config import RunConfig
from bga.pipeline import load_output
from bga.stage import FrameReader, Sequence

#: Dot and box colour per team label in BGR; ``""`` (unknown) is magenta.
TEAM_BGR = {"home": (240, 240, 240), "away": (45, 55, 220), "ref": (130, 130, 130), "": (200, 0, 200)}
#: High-contrast text colour inside each team's radar circle.
TEAM_TXT_BGR = {"home": (20, 20, 20), "away": (255, 255, 255), "ref": (255, 255, 255), "": (255, 255, 255)}
#: Ball and holder-ring highlight colour in BGR (vivid basketball orange).
BALL_BGR = (0, 145, 255)


def track_teams(tracks: dict, teams: dict) -> dict[int, str]:
    """Give each track the majority team label of its detected boxes.

    Args:
        tracks: Track output frames, ``{"<frame>": [[track_id, ..., det_index], ...]}``.
        teams: Team output frames, ``{"<frame>": [label per detect box]}``.

    Returns:
        Team label per track id; tracks with no detected box in a labelled frame are absent.

    Examples:
        >>> tracks = {"0": [[1, 0, 0, 1, 1, 0.9, "player", 0], [2, 0, 0, 1, 1, 0.9, "player", -1]],
        ...           "5": [[1, 0, 0, 1, 1, 0.9, "player", 1]], "10": [[1, 0, 0, 1, 1, 0.9, "player", 0]]}
        >>> teams = {"0": ["home"], "5": ["away", "home"], "10": ["home"]}
        >>> track_teams(tracks, teams)
        {1: 'home'}

    """
    votes = defaultdict(Counter)
    for f, rows in tracks.items():
        for r in rows:
            if r[7] >= 0 and f in teams:
                votes[int(r[0])][teams[f][r[7]]] += 1
    return {t: v.most_common(1)[0][0] for t, v in votes.items()}


def _load_numbers(cfg: RunConfig, runs: Path, seq: Sequence) -> dict[int, str | None]:
    """Load confirmed jersey numbers per active track id from ``identity`` or ``numbers`` when configured."""
    if "identity" in cfg.layers:
        return {int(k): v for k, v in load_output(cfg, runs, seq, "identity").get("numbers", {}).items()}
    if "numbers" in cfg.layers:
        raw = {int(k): v for k, v in load_output(cfg, runs, seq, "numbers").get("tracks", {}).items()}
        if "link" in cfg.layers:
            id_map = {int(k): int(v) for k, v in load_output(cfg, runs, seq, "link").get("ids", {}).items()}
            out: dict[int, str | None] = {}
            for rtid, val in raw.items():
                ltid = id_map.get(rtid, rtid)
                if val is not None or ltid not in out:
                    out[ltid] = val
            return out
        return raw
    return {}


def _label_for(tid: int, jersey: str | None) -> str:
    """Short display label for a track: ``#<jersey>`` when confirmed, else ``t<tid>``."""
    return f"#{jersey}" if jersey else f"t{tid}"


def _court_px(x_m: float, y_m: float, scale: float, pad: int, shape: tuple[int, ...]) -> tuple[int, int]:
    """Convert court metres ``(x_m, y_m)`` to clamped radar canvas pixels."""
    h, w = shape[:2]
    px = max(12, min(w - 13, int(x_m * 100.0 * scale) + pad))
    py = max(12, min(h - 13, int(y_m * 100.0 * scale) + pad))
    return px, py


def _draw_broadcast_overlays(
    top: np.ndarray,
    rows: list[list[Any]],
    sx: float,
    sy: float,
    team_of: dict[int, str],
    num_of: dict[int, str | None],
    foot_frame: dict[int, tuple[float, float]],
    airborne_frame: set[int],
    holder_tid: int | None = None,
    ball_px: list[float] | None = None,
) -> None:
    """Draw team-coloured boxes, foot points, jersey labels, holder highlight, and ball dot on broadcast."""
    for r in rows:
        tid = int(r[0])
        x1, y1, x2, y2 = int(r[1] * sx), int(r[2] * sy), int(r[3] * sx), int(r[4] * sy)
        team = team_of.get(tid, "")
        color = TEAM_BGR.get(team, TEAM_BGR[""])
        is_holder = holder_tid is not None and tid == holder_tid
        if is_holder:
            cv2.rectangle(top, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), BALL_BGR, 3)
        cv2.rectangle(top, (x1, y1), (x2, y2), color, 2)
        fx, fy = foot_frame.get(tid, ((r[1] + r[3]) / 2.0, r[4]))
        foot_xy = (int(fx * sx), int(fy * sy))
        foot_color = (0, 220, 255) if tid in airborne_frame else (0, 230, 80)
        cv2.circle(top, foot_xy, 4, foot_color, -1)
        cv2.circle(top, foot_xy, 4, (0, 0, 0), 1)
        label = _label_for(tid, num_of.get(tid))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1 - 2)
        cv2.rectangle(top, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), BALL_BGR if is_holder else color, -1)
        cv2.putText(
            top,
            label,
            (x1 + 3, ty - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 15, 15) if is_holder else TEAM_TXT_BGR.get(team, (0, 0, 0)),
            1,
            cv2.LINE_AA,
        )
    if ball_px is not None:
        bx, by = int(ball_px[0] * sx), int(ball_px[1] * sy)
        cv2.circle(top, (bx, by), 7, BALL_BGR, -1, cv2.LINE_AA)
        cv2.circle(top, (bx, by), 7, (15, 15, 15), 2, cv2.LINE_AA)


def _draw_radar_dots(
    rad: np.ndarray,
    frame_pos: dict[int, tuple[float, float]],
    scale: float,
    pad: int,
    team_of: dict[int, str],
    num_of: dict[int, str | None],
    airborne_frame: set[int],
    holder_tid: int | None = None,
) -> None:
    """Draw 2D court player dots with jersey labels and holder halo on ``rad``."""
    for tid, (x, y) in frame_pos.items():
        px, py = _court_px(x, y, scale, pad, rad.shape)
        team = team_of.get(tid, "")
        fill = TEAM_BGR.get(team, TEAM_BGR[""])
        if holder_tid is not None and tid == holder_tid:
            cv2.circle(rad, (px, py), 16, BALL_BGR, 3, cv2.LINE_AA)
        ring = (0, 210, 255) if tid in airborne_frame else (15, 15, 15)
        cv2.circle(rad, (px, py), 12, fill, -1, cv2.LINE_AA)
        cv2.circle(rad, (px, py), 12, ring, 2, cv2.LINE_AA)
        label = num_of.get(tid) or f"t{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(
            rad,
            label,
            (px - tw // 2, py + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEAM_TXT_BGR.get(team, (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )


def _draw_radar_ball_and_shots(
    rad: np.ndarray,
    frame_idx: int,
    ball_row: dict[str, Any] | None,
    shots: list[dict[str, Any]],
    scale: float,
    pad: int,
) -> None:
    """Draw cumulative made/missed shot markers and the current 2D ball dot on ``rad``."""
    for sh in shots:
        rf = sh.get("rim_frame") or sh.get("end_frame") or sh.get("start_frame")
        if rf is None or int(rf) > frame_idx or sh.get("x_m") is None or sh.get("y_m") is None:
            continue
        sx, sy = _court_px(float(sh["x_m"]), float(sh["y_m"]), scale, pad, rad.shape)
        if sh.get("outcome") == "made":
            cv2.circle(rad, (sx, sy), 9, (40, 200, 60), 3, cv2.LINE_AA)
        else:
            cv2.line(rad, (sx - 7, sy - 7), (sx + 7, sy + 7), (50, 50, 235), 3, cv2.LINE_AA)
            cv2.line(rad, (sx - 7, sy + 7), (sx + 7, sy - 7), (50, 50, 235), 3, cv2.LINE_AA)
    if ball_row and ball_row.get("x_m") is not None and ball_row.get("y_m") is not None:
        bx, by = _court_px(float(ball_row["x_m"]), float(ball_row["y_m"]), scale, pad, rad.shape)
        cv2.circle(rad, (bx, by), 7, BALL_BGR, -1, cv2.LINE_AA)
        cv2.circle(rad, (bx, by), 7, (15, 15, 15), 2, cv2.LINE_AA)


def _draw_header(top: np.ndarray, seq: Sequence, frame_idx: int, backend: str) -> None:
    """Draw a compact translucent header bar with sequence name, timestamp, and team colour legend."""
    overlay = top.copy()
    cv2.rectangle(overlay, (0, 0), (top.shape[1], 32), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, top, 0.35, 0.0, top)
    t_s = frame_idx / seq.fps if seq.fps > 0 else 0.0
    home = (seq.meta.get("home") or {}).get("team", "home")
    away = (seq.meta.get("away") or {}).get("team", "away")
    title = f"{seq.name}  f{frame_idx} ({t_s:.2f}s)  [{backend}]"
    cv2.putText(top, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    legend = f"W:{home}  R:{away}"
    (lw, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        top,
        legend,
        (max(10, top.shape[1] - lw - 12), 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def _index_positions(
    pos_out: dict[str, Any],
) -> tuple[dict[int, dict[int, tuple[float, float]]], dict[int, dict[int, tuple[float, float]]], dict[int, set[int]]]:
    """Index ``positions`` output by frame for fast radar rendering."""
    pos: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for tid, pts in pos_out["tracks"].items():
        for f, x, y in pts:
            pos[int(f)][int(tid)] = (float(x), float(y))
    foot: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for tid, pts in pos_out.get("foot_px", {}).items():
        for f, fx, fy in pts:
            foot[int(f)][int(tid)] = (float(fx), float(fy))
    airborne: dict[int, set[int]] = defaultdict(set)
    for tid, air_frames in pos_out.get("airborne", {}).items():
        for f in air_frames:
            airborne[int(f)].add(int(tid))
    return pos, foot, airborne


def render(cfg: RunConfig, seq: Sequence, runs: Path = Path("runs"), width: int = 960) -> Path:
    """Write the radar video of one sequence: annotated broadcast frame above, FIBA 2D court below.

    Args:
        cfg: Run config whose ``link``/``track``, ``team`` (optional), ``identity``/``numbers`` (optional),
            ``positions``, and ``ball`` (optional) outputs are drawn.
        seq: Sequence to render.
        runs: Root of the run outputs.
        width: Video width in pixels.

    Returns:
        Path of the written ``.mp4``.

    Examples:
        >>> render(cfg, seq)  # doctest: +SKIP
        PosixPath('runs/<clip>/radar/<config>@<hash8>/<sequence>.mp4')

    """
    from sports.basketball import CourtConfiguration, League, draw_court

    court_cfg = CourtConfiguration(league=League.FIBA)
    scale = width / (court_cfg.court_length + 200)  # px per cm, 1 m margin each side
    pad = int(100 * scale)
    court = draw_court(court_cfg, scale=scale, padding=pad, line_thickness=2)
    track_layer = "link" if "link" in cfg.layers else "track"
    tracks = load_output(cfg, runs, seq, track_layer)["frames"]
    teams = load_output(cfg, runs, seq, "team")["frames"] if "team" in cfg.layers else {}
    team_of = track_teams(tracks, teams)
    num_of = _load_numbers(cfg, runs, seq)
    pos, foot, airborne_by_frame = _index_positions(load_output(cfg, runs, seq, "positions"))
    ball_out = load_output(cfg, runs, seq, "ball") if "ball" in cfg.layers else None
    ball_frames = ball_out.get("frames", {}) if ball_out else {}
    ball_shots = ball_out.get("shots", []) if ball_out else []

    out = runs / seq.clip / "radar" / f"{cfg.name.split('[')[0]}@{cfg.config_hash()}" / f"{seq.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
    read = FrameReader(seq.video)
    h_top = round(width * 9 / 16)
    backend_tag = cfg.layers["ball"].backend if "ball" in cfg.layers else cfg.layers["positions"].backend
    writer = None
    for f in seq.frames:
        raw = read(f)
        sx, sy = width / raw.shape[1], h_top / raw.shape[0]
        top = cv2.resize(raw, (width, h_top))
        b_row = ball_frames.get(str(f))
        holder_tid = b_row.get("holder_track_id") if b_row else None
        _draw_broadcast_overlays(
            top,
            tracks.get(str(f), []),
            sx,
            sy,
            team_of,
            num_of,
            foot.get(f, {}),
            airborne_by_frame.get(f, set()),
            holder_tid=holder_tid,
            ball_px=b_row.get("ball_px") if b_row else None,
        )
        _draw_header(top, seq, f, backend_tag)
        rad = court.copy()
        _draw_radar_dots(rad, pos.get(f, {}), scale, pad, team_of, num_of, airborne_by_frame.get(f, set()), holder_tid)
        if ball_out is not None:
            _draw_radar_ball_and_shots(rad, f, b_row, ball_shots, scale, pad)
        rad_resized = cv2.resize(rad, (width, round(rad.shape[0] * width / rad.shape[1])))
        frame = np.vstack([top, rad_resized])
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), seq.fps / stride, frame.shape[1::-1])
        writer.write(frame)
    read.close()
    if writer is not None:
        writer.release()
    return out
