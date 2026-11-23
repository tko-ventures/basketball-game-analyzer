"""Tests for M8 parquet export contract and upgraded radar renderer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from bga.config import RunConfig
from bga.export.parquet import PLAYER_SCHEMA, export_sequence, game_id_for
from bga.export.radar import render
from bga.stage import Sequence


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _sample_cfg() -> RunConfig:
    return RunConfig(
        name="demo-m8",
        layers={
            "shots": {"backend": "gold"},
            "court": {"backend": "rfdetr_keypoint"},
            "detect": {"backend": "rfdetr_torch"},
            "team": {"backend": "siglip_kmeans"},
            "track": {"backend": "bytetrack"},
            "link": {"backend": "greedy"},
            "numbers": {"backend": "parseq"},
            "identity": {"backend": "game"},
            "positions": {"backend": "ankles_pose"},
        },
    )


def _seed_outputs(cfg: RunConfig, seq: Sequence, runs: Path) -> None:
    _write_json(
        cfg.run_dir(runs, seq.clip, "link") / f"{seq.name}.json",
        {
            "frames": {
                "0": [
                    [10, 100.0, 50.0, 140.0, 150.0, 0.92, "player", 0],
                    [20, 200.0, 60.0, 240.0, 160.0, 0.85, "player", 1],
                ],
                "5": [
                    [10, 105.0, 50.0, 145.0, 150.0, 0.90, "player", 0],
                ],
            },
            "ids": {"1": 10, "2": 20},
        },
    )
    _write_json(
        cfg.run_dir(runs, seq.clip, "team") / f"{seq.name}.json",
        {"frames": {"0": ["home", "away"], "5": ["home"]}},
    )
    _write_json(
        cfg.run_dir(runs, seq.clip, "identity") / f"{seq.name}.json",
        {"numbers": {"10": "2", "20": None}, "clusters": {"10": "home:0", "20": "away:3"}},
    )
    _write_json(
        cfg.run_dir(runs, seq.clip, "positions") / f"{seq.name}.json",
        {
            "tracks": {"10": [[0, 14.0, 7.5], [5, 14.5, 7.6]], "20": [[0, 30.5, -2.0]]},
            "foot_px": {"10": [[0, 121.0, 148.0], [5, 126.0, 147.0]], "20": [[0, 220.0, 159.0]]},
            "airborne": {"10": [5]},
            "court_m": [28.0, 15.0],
        },
    )


def test_export_schema_matches_plan_contract() -> None:
    expected = [
        "game_id",
        "clip",
        "frame_idx",
        "t_video_s",
        "shot_id",
        "camera",
        "game_clock",
        "track_id",
        "player_uid",
        "team",
        "jersey_number",
        "player_name",
        "bbox_xyxy_px",
        "foot_px",
        "x_m",
        "y_m",
        "confidence",
    ]
    assert PLAYER_SCHEMA.names == expected


def test_game_id_for_sources() -> None:
    s1 = Sequence("s", "c", "v.mp4", 0, [0], 50.0, meta={"source": "https://www.youtube.com/watch?v=OSp445e0mXU"})
    assert game_id_for(s1) == "OSp445e0mXU"
    s2 = Sequence("x_s4", "20250213_4XMD9FWmQws_PGE_Start", "v.mp4", 4, [0], 50.0)
    assert game_id_for(s2) == "4XMD9FWmQws"
    s3 = Sequence("custom_s1", "plain_clip", "v.mp4", 1, [0], 50.0)
    assert game_id_for(s3) == "custom"


def test_parquet_export_roundtrip(tmp_path: Path) -> None:
    cfg = _sample_cfg()
    rosters = tmp_path / "rosters"
    _write_json(
        rosters / "home_club.json",
        {"seasons": {"2025/26": {"players": [{"number": 2, "name": "Jan Kowalski", "id_plk": 12345}]}}},
    )
    seq = Sequence(
        name="OSp445e0mXU_s2",
        clip="20260328_OSp445e0mXU_SKROT",
        video="unused.mp4",
        shot=2,
        frames=[0, 5],
        fps=50.0,
        meta={"season": "2025/26", "home": {"team": "Home", "roster": "home_club"}, "away": {"team": "Away"}},
    )
    runs = tmp_path / "runs"
    _seed_outputs(cfg, seq, runs)
    out = export_sequence(cfg, seq, runs=runs, rosters=rosters)
    assert out.is_file()

    table = pq.read_table(out)
    assert table.schema == PLAYER_SCHEMA
    rows = table.to_pylist()
    assert len(rows) == 3

    r0 = rows[0]
    assert r0["game_id"] == "OSp445e0mXU"
    assert r0["frame_idx"] == 0
    assert r0["t_video_s"] == 0.0
    assert r0["shot_id"] == "OSp445e0mXU_s2"
    assert r0["camera"] == "central"
    assert r0["game_clock"] is None
    assert r0["track_id"] == 10
    assert r0["player_uid"] == "plk:12345"
    assert r0["team"] == "home"
    assert r0["jersey_number"] == "2"
    assert r0["player_name"] == "Jan Kowalski"
    assert np.allclose(r0["foot_px"], [121.0, 148.0])
    assert np.isclose(r0["x_m"], 14.0)
    assert np.isclose(r0["y_m"], 7.5)

    # Track 20 has no jersey number -> falls back to game cluster uid and None player_name
    r1 = rows[1]
    assert r1["track_id"] == 20
    assert r1["player_uid"] == "OSp445e0mXU:away:3"
    assert r1["jersey_number"] is None
    assert r1["player_name"] is None


def test_radar_render_with_link_and_identity(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (320, 180))
    for i in range(6):
        frame = np.full((180, 320, 3), 40 + i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    cfg = _sample_cfg()
    seq = Sequence(
        name="OSp445e0mXU_s2",
        clip="20260328_OSp445e0mXU_SKROT",
        video=str(video),
        shot=2,
        frames=[0, 5],
        fps=50.0,
        meta={"home": {"team": "Dziki"}, "away": {"team": "Legia"}},
    )
    runs = tmp_path / "runs"
    _seed_outputs(cfg, seq, runs)
    mp4 = render(cfg, seq, runs=runs, width=480)
    assert mp4.is_file()
    assert mp4.stat().st_size > 0
    cap = cv2.VideoCapture(str(mp4))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    cap.release()
