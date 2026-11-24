"""Parquet export per sequence (plan section 5, Export contract).

Player table schema (per frame per player):
    ``game_id, clip, frame_idx, t_video_s, shot_id, camera, game_clock, track_id, player_uid,
    team, jersey_number, player_name, bbox_xyxy_px, foot_px, x_m, y_m, confidence``

Ball table schema (per frame, exported alongside players when ``ball`` is configured):
    ``frame_idx, x_m, y_m, holder_track_id, in_flight``

Court coordinates are in metres on the FIBA court (``CourtConfiguration(League.FIBA, CENTIMETERS) / 100``).
Written to ``runs/<clip>/export/<config>@<hash8>/<sequence>.parquet`` (and ``<sequence>.ball.parquet``).

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from bga.config import RunConfig
from bga.export.radar import track_teams
from bga.numbers.base import ROSTERS
from bga.pipeline import load_output
from bga.stage import Sequence

#: Exact per-frame per-player parquet schema from ``docs/PLAN.md`` §5 Export contract.
PLAYER_SCHEMA = pa.schema([
    pa.field("game_id", pa.string(), nullable=False),
    pa.field("clip", pa.string(), nullable=False),
    pa.field("frame_idx", pa.int32(), nullable=False),
    pa.field("t_video_s", pa.float64(), nullable=False),
    pa.field("shot_id", pa.string(), nullable=False),
    pa.field("camera", pa.string(), nullable=False),
    pa.field("game_clock", pa.string(), nullable=True),
    pa.field("track_id", pa.int32(), nullable=False),
    pa.field("player_uid", pa.string(), nullable=True),
    pa.field("team", pa.string(), nullable=True),
    pa.field("jersey_number", pa.string(), nullable=True),
    pa.field("player_name", pa.string(), nullable=True),
    pa.field("bbox_xyxy_px", pa.list_(pa.float32()), nullable=False),
    pa.field("foot_px", pa.list_(pa.float32()), nullable=True),
    pa.field("x_m", pa.float32(), nullable=True),
    pa.field("y_m", pa.float32(), nullable=True),
    pa.field("confidence", pa.float32(), nullable=False),
])

#: Exact per-frame ball parquet schema from ``docs/PLAN.md`` §5 Export contract.
BALL_SCHEMA = pa.schema([
    pa.field("frame_idx", pa.int32(), nullable=False),
    pa.field("x_m", pa.float32(), nullable=True),
    pa.field("y_m", pa.float32(), nullable=True),
    pa.field("holder_track_id", pa.int32(), nullable=True),
    pa.field("in_flight", pa.bool_(), nullable=False),
])

_YT_ID_RE = re.compile(r"^\d{8}_([A-Za-z0-9_-]{11})_")


def game_id_for(seq: Sequence) -> str:
    """Derive a stable ``game_id`` for ``seq`` (YouTube id from meta/clip, else sequence prefix or clip stem).

    Args:
        seq: Sequence being exported.

    Returns:
        Stable game identifier string.

    Examples:
        >>> s = Sequence("OSp445e0mXU_s2", "20260328_OSp445e0mXU_SKROT", "v.mp4", 2, [0], 50.0)
        >>> game_id_for(s)
        'OSp445e0mXU'

    """
    src = (seq.meta or {}).get("source", "")
    if src:
        vid = parse_qs(urlparse(src).query).get("v")
        if vid and vid[0]:
            return vid[0]
    m = _YT_ID_RE.match(seq.clip)
    if m:
        return m.group(1)
    if "_s" in seq.name:
        return seq.name.rsplit("_s", 1)[0]
    return seq.clip


def roster_players(meta: dict[str, Any], side: str, rosters: Path = ROSTERS) -> dict[str, dict[str, Any]]:
    """Map jersey number string to roster player entry for one team side and season.

    Args:
        meta: Clip metadata containing ``season`` and ``<side>.roster``.
        side: ``"home"`` or ``"away"``.
        rosters: Directory containing ``<roster>.json`` files.

    Returns:
        Dict ``{number_str: player_dict}``, empty when the roster or season is missing.

    Examples:
        >>> p = roster_players({"season": "2025/26", "home": {"roster": "dziki_warszawa"}}, "home")
        >>> bool(p) and all("name" in v for v in p.values())
        True

    """
    stem, season = (meta.get(side) or {}).get("roster"), meta.get("season")
    if not stem or not season:
        return {}
    path = rosters / f"{stem}.json"
    if not path.exists():
        return {}
    players = json.loads(path.read_text(encoding="utf-8"))["seasons"].get(season, {}).get("players", [])
    return {str(p["number"]): p for p in players if p.get("number") is not None}


def _track_numbers(cfg: RunConfig, runs: Path, seq: Sequence) -> tuple[dict[int, str | None], dict[int, str]]:
    """Load confirmed jersey numbers and optional cluster ids per active track id."""
    if "identity" in cfg.layers:
        ident = load_output(cfg, runs, seq, "identity")
        nums = {int(k): v for k, v in ident.get("numbers", {}).items()}
        clusters = {int(k): str(v) for k, v in ident.get("clusters", {}).items()}
        return nums, clusters
    if "numbers" in cfg.layers:
        num_out = load_output(cfg, runs, seq, "numbers")
        raw_nums = {int(k): v for k, v in num_out.get("tracks", {}).items()}
        if "link" in cfg.layers:
            link_out = load_output(cfg, runs, seq, "link")
            id_map = {int(k): int(v) for k, v in link_out.get("ids", {}).items()}
            mapped: dict[int, str | None] = {}
            for rtid, val in raw_nums.items():
                ltid = id_map.get(rtid, rtid)
                if val is not None or ltid not in mapped:
                    mapped[ltid] = val
            return mapped, {}
        return raw_nums, {}
    return {}, {}


def _index_track_points(mapping: dict[str, list[list[float]]]) -> dict[tuple[int, int], tuple[float, float]]:
    """Index ``{tid_str: [[frame, x, y], ...]}`` into ``{(frame, tid): (x, y)}``."""
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for tid_s, pts in mapping.items():
        tid = int(tid_s)
        for row in pts:
            out[(int(row[0]), tid)] = (float(row[1]), float(row[2]))
    return out


def _player_identity_fields(
    game_id: str,
    team: str | None,
    jersey: str | None,
    cluster: str | None,
    rosters_by_side: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str | None, str | None]:
    """Resolve ``(player_uid, player_name)`` from roster metadata and optional game cluster."""
    player = rosters_by_side.get(team or "", {}).get(jersey or "") if jersey else None
    if player is not None:
        id_plk = player.get("id_plk")
        uid = f"plk:{id_plk}" if id_plk is not None else (f"{game_id}:{cluster}" if cluster else None)
        return uid, str(player["name"]) if player.get("name") else None
    return (f"{game_id}:{cluster}" if cluster else None), None


def sequence_rows(
    cfg: RunConfig,
    seq: Sequence,
    runs: Path = Path("runs"),
    rosters: Path = ROSTERS,
    camera: str = "central",
) -> list[dict[str, Any]]:
    """Build export-contract rows for every tracked person on every processed frame of ``seq``.

    Args:
        cfg: Run config whose outputs exist under ``runs``.
        seq: Sequence to export.
        runs: Root of pipeline run outputs.
        rosters: Directory of roster JSON files.
        camera: Camera label for the shot (``"central"`` for gold central-camera shots).

    Returns:
        List of row dicts matching :data:`PLAYER_SCHEMA`, ordered by ``(frame_idx, track_id)``.

    """
    track_layer = "link" if "link" in cfg.layers else "track"
    tracks = load_output(cfg, runs, seq, track_layer)["frames"]
    teams = load_output(cfg, runs, seq, "team")["frames"] if "team" in cfg.layers else {}
    team_of = track_teams(tracks, teams)
    num_of, cluster_of = _track_numbers(cfg, runs, seq)
    pos_out = load_output(cfg, runs, seq, "positions") if "positions" in cfg.layers else {}
    pos_idx = _index_track_points(pos_out.get("tracks", {}))
    foot_idx = _index_track_points(pos_out.get("foot_px", {}))
    gid = game_id_for(seq)
    rosters_by_side = {side: roster_players(seq.meta, side, rosters) for side in ("home", "away")}

    rows: list[dict[str, Any]] = []
    for f in seq.frames:
        t_s = round(float(f) / float(seq.fps), 4) if seq.fps > 0 else 0.0
        for r in sorted(tracks.get(str(f), []), key=lambda item: int(item[0])):
            tid = int(r[0])
            x1, y1, x2, y2 = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            conf = float(r[5])
            team = team_of.get(tid) or None
            jersey = num_of.get(tid)
            uid, name = _player_identity_fields(gid, team, jersey, cluster_of.get(tid), rosters_by_side)
            foot = foot_idx.get((f, tid), ((x1 + x2) / 2.0, y2))
            xy = pos_idx.get((f, tid))
            rows.append({
                "game_id": gid,
                "clip": seq.clip,
                "frame_idx": int(f),
                "t_video_s": t_s,
                "shot_id": seq.name,
                "camera": camera,
                "game_clock": None,
                "track_id": tid,
                "player_uid": uid,
                "team": team,
                "jersey_number": jersey,
                "player_name": name,
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "foot_px": [float(foot[0]), float(foot[1])],
                "x_m": float(xy[0]) if xy is not None else None,
                "y_m": float(xy[1]) if xy is not None else None,
                "confidence": conf,
            })
    return rows


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert row dicts from :func:`sequence_rows` into a typed :class:`pyarrow.Table`.

    Args:
        rows: List of export row dicts.

    Returns:
        Arrow table strictly adhering to :data:`PLAYER_SCHEMA`.

    Examples:
        >>> tbl = rows_to_table([])
        >>> tbl.num_rows, tbl.column_names[:4]
        (0, ['game_id', 'clip', 'frame_idx', 't_video_s'])

    """
    return pa.Table.from_pylist(rows, schema=PLAYER_SCHEMA)


def ball_rows(cfg: RunConfig, seq: Sequence, runs: Path = Path("runs")) -> list[dict[str, Any]]:
    """Build per-frame ball export rows matching :data:`BALL_SCHEMA` for one sequence.

    Args:
        cfg: Run config including the ``ball`` layer.
        seq: Sequence to export.
        runs: Root of pipeline run outputs.

    Returns:
        List of row dicts ordered by ``frame_idx``.

    """
    ball_out = load_output(cfg, runs, seq, "ball")
    frames_map = ball_out.get("frames", {})
    rows: list[dict[str, Any]] = []
    for f in seq.frames:
        fr = frames_map.get(str(f), {})
        tid = fr.get("holder_track_id")
        rows.append({
            "frame_idx": int(f),
            "x_m": float(fr["x_m"]) if fr.get("x_m") is not None else None,
            "y_m": float(fr["y_m"]) if fr.get("y_m") is not None else None,
            "holder_track_id": int(tid) if tid is not None else None,
            "in_flight": bool(fr.get("in_flight", tid is None)),
        })
    return rows


def ball_rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert row dicts from :func:`ball_rows` into a typed :class:`pyarrow.Table`.

    Args:
        rows: List of per-frame ball row dicts.

    Returns:
        Arrow table strictly adhering to :data:`BALL_SCHEMA`.

    Examples:
        >>> row = {"frame_idx": 0, "x_m": 10.0, "y_m": 7.5, "holder_track_id": 1, "in_flight": False}
        >>> tbl = ball_rows_to_table([row])
        >>> tbl.num_rows, tbl.column_names
        (1, ['frame_idx', 'x_m', 'y_m', 'holder_track_id', 'in_flight'])

    """
    return pa.Table.from_pylist(rows, schema=BALL_SCHEMA)


def export_path(cfg: RunConfig, seq: Sequence, runs: Path = Path("runs")) -> Path:
    """Default parquet destination for one sequence: ``runs/<clip>/export/<config>@<hash8>/<sequence>.parquet``."""
    return runs / seq.clip / "export" / f"{cfg.name.split('[')[0]}@{cfg.config_hash()}" / f"{seq.name}.parquet"


def ball_export_path(cfg: RunConfig, seq: Sequence, runs: Path = Path("runs")) -> Path:
    """Default ball parquet destination: ``runs/<clip>/export/<config>@<hash8>/<sequence>.ball.parquet``."""
    return runs / seq.clip / "export" / f"{cfg.name.split('[')[0]}@{cfg.config_hash()}" / f"{seq.name}.ball.parquet"


def export_sequence(
    cfg: RunConfig,
    seq: Sequence,
    runs: Path = Path("runs"),
    out: Path | None = None,
    rosters: Path = ROSTERS,
) -> Path:
    """Write one sequence's per-frame player table to parquet and return its path.

    Args:
        cfg: Run config whose outputs are already on disk under ``runs``.
        seq: Sequence to export.
        runs: Root of pipeline run outputs.
        out: Optional explicit destination path; defaults to :func:`export_path`.
        rosters: Directory of roster JSON files.

    Returns:
        Path of the written ``.parquet`` file.

    """
    dest = out or export_path(cfg, seq, runs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = rows_to_table(sequence_rows(cfg, seq, runs=runs, rosters=rosters))
    pq.write_table(table, dest)
    return dest


def export_ball_sequence(
    cfg: RunConfig,
    seq: Sequence,
    runs: Path = Path("runs"),
    out: Path | None = None,
) -> Path:
    """Write one sequence's per-frame ball table to parquet and return its path.

    Args:
        cfg: Run config including ``ball``, whose outputs are already on disk under ``runs``.
        seq: Sequence to export.
        runs: Root of pipeline run outputs.
        out: Optional explicit destination path; defaults to :func:`ball_export_path`.

    Returns:
        Path of the written ``.ball.parquet`` file.

    """
    dest = out or ball_export_path(cfg, seq, runs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = ball_rows_to_table(ball_rows(cfg, seq, runs=runs))
    pq.write_table(table, dest)
    return dest
