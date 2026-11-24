"""Unit tests for M9 Layer 7 (ball, possession, shots), evaluation, parquet export, and radar overlays."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from bga.ball.base import (
    BASKETS_M,
    interpolate_ball_2d,
    last_holder_before,
    nearest_basket,
    shot_points,
)
from bga.ball.fused import FusedBall, _filter_ball_candidates
from bga.ball.skalski import SkalskiBall
from bga.config import LAYERS, OPTIONAL_BY_LAYER, LayerConfig, RunConfig
from bga.eval.ball import score_boxes, score_events
from bga.export.parquet import BALL_SCHEMA, export_ball_sequence
from bga.export.radar import render
from bga.stage import Sequence


def _dummy_read(_frame: int) -> np.ndarray:
    raise AssertionError("ball backends must not read video frames")


def _static_fp_inputs() -> tuple[list[int], dict[int, list[float]], dict[str, list[list]]]:
    """Build a 12-frame sequence with one static rim-relative FP and one moving ball."""
    frames = list(range(0, 60, 5))
    rims = {f: [900.0, 200.0, 960.0, 225.0, 0.9] for f in frames}
    det_frames: dict[str, list[list]] = {}
    for idx, f in enumerate(frames):
        det_frames[str(f)] = [
            [1020.0, 165.0, 1040.0, 185.0, 0.45, "ball"],
            [500.0 + idx * 35.0, 400.0 - idx * 10.0, 520.0 + idx * 35.0, 420.0 - idx * 10.0, 0.70, "ball"],
        ]
    return frames, rims, det_frames


def _synthetic_shot_upstream(made: bool) -> tuple[Sequence, dict]:
    """Build a synthetic 50 fps stride-5 sequence with holder -> 3pt release -> rim arrival -> made/missed."""
    frames = list(range(0, 110, 5))
    seq = Sequence("synth_s1", "synth_clip", "unused.mp4", 1, frames, 50.0)
    rim = [900.0, 200.0, 960.0, 225.0, 0.92, "rim"]
    det_frames: dict[str, list[list]] = {}
    track_frames: dict[str, list[list]] = {}
    team_frames: dict[str, list[str]] = {}
    pos_pts: list[list[float]] = []

    for i, f in enumerate(frames):
        player_box = [370.0, 460.0, 430.0, 620.0]
        track_frames[str(f)] = [[1, *player_box, 0.95, "player", 0]]
        team_frames[str(f)] = ["home"]
        pos_pts.append([float(f), 9.0, 7.5])
        boxes = [[*player_box, 0.95, "player"], rim]
        if i <= 5:
            if i != 3:
                boxes.append([*player_box, 0.85, "player-in-possession"])
            bx, by = 405.0, 500.0
        elif i <= 10:
            t = (i - 5) / 5.0
            bx = 405.0 + t * (930.0 - 405.0)
            by = 500.0 + t * (212.0 - 500.0) - 180.0 * t * (1.0 - t)
        elif made:
            bx = 932.0
            by = 212.5 + (i - 10) * 32.0
        else:
            bx = 930.0 + (i - 10) * 55.0
            by = 190.0 + max(0, i - 12) * 30.0
        if i <= 16:
            boxes.append([bx - 12.0, by - 12.0, bx + 12.0, by + 12.0, 0.85, "ball"])
        det_frames[str(f)] = boxes

    upstream = {
        "detect": {"frames": det_frames},
        "track": {"frames": track_frames},
        "team": {"frames": team_frames},
        "positions": {"tracks": {"1": pos_pts}},
    }
    return seq, upstream


def _synthetic_skalski_upstream() -> tuple[Sequence, dict]:
    """Build a synthetic sequence with ``player-jump-shot`` and ``ball-in-basket`` detections."""
    frames = list(range(0, 80, 5))
    seq = Sequence("sk_s1", "sk_clip", "unused.mp4", 1, frames, 50.0)
    det_frames: dict[str, list[list]] = {}
    track_frames: dict[str, list[list]] = {}
    team_frames: dict[str, list[str]] = {}
    pos_pts: list[list[float]] = []
    for i, f in enumerate(frames):
        pbox = [300.0, 400.0, 360.0, 580.0]
        track_frames[str(f)] = [[7, *pbox, 0.9, "player", 0]]
        team_frames[str(f)] = ["away"]
        pos_pts.append([float(f), 5.0, 7.5])
        boxes = [[*pbox, 0.9, "player"], [900.0, 200.0, 960.0, 225.0, 0.9, "rim"]]
        if i <= 3:
            boxes.append([*pbox, 0.85, "player-in-possession"])
        if 3 <= i <= 6:
            boxes.append([*pbox, 0.85, "player-jump-shot"])
        if 8 <= i <= 10:
            boxes.append([910.0, 200.0, 950.0, 235.0, 0.80, "ball-in-basket"])
        det_frames[str(f)] = boxes
    upstream = {
        "detect": {"frames": det_frames},
        "track": {"frames": track_frames},
        "team": {"frames": team_frames},
        "positions": {"tracks": {"7": pos_pts}},
    }
    return seq, upstream


def _make_blank_video(path: Path, n_frames: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (320, 180))
    for _ in range(n_frames):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()


def _write_layer(cfg: RunConfig, runs: Path, seq: Sequence, layer: str, data: dict) -> None:
    d = cfg.run_dir(runs, seq.clip, layer)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{seq.name}.json").write_text(json.dumps(data), encoding="utf-8")


class TestFibaGeometry:
    """FIBA 2-point / 3-point arc and corner geometry."""

    @pytest.mark.parametrize(
        ("x_m", "y_m", "basket", "expected"),
        [
            pytest.param(3.5, 7.5, BASKETS_M[0], 2, id="left-paint-2pt"),
            pytest.param(8.6, 7.5, BASKETS_M[0], 3, id="left-wing-arc-3pt"),
            pytest.param(1.2, 0.5, BASKETS_M[0], 3, id="left-top-corner-3pt"),
            pytest.param(1.2, 14.5, BASKETS_M[0], 3, id="left-bottom-corner-3pt"),
            pytest.param(1.2, 1.5, BASKETS_M[0], 2, id="left-short-corner-2pt"),
            pytest.param(26.8, 0.6, BASKETS_M[1], 3, id="right-corner-3pt"),
            pytest.param(24.0, 7.5, BASKETS_M[1], 2, id="right-paint-2pt"),
        ],
    )
    def test_shot_points_classifies_fiba_zones(
        self, x_m: float, y_m: float, basket: tuple[float, float], expected: int
    ) -> None:
        assert shot_points(x_m, y_m, basket) == expected

    def test_nearest_basket_selects_right_hoop(self) -> None:
        assert nearest_basket(25.0, 7.5) == BASKETS_M[1]


class TestBallHelpers:
    """2D ball interpolation, holder lookback, and static-FP filtering."""

    def test_interpolate_ball_2d_respects_shot_anchor(self) -> None:
        frames = [0, 5, 10, 15, 20]
        holders = {0: 1, 5: None, 10: None, 15: None, 20: 2}
        pos_by_tid = {1: {0: (10.0, 7.5)}, 2: {20: (2.0, 7.5)}}
        pts = interpolate_ball_2d(
            frames, holders, pos_by_tid, {}, dict.fromkeys(frames), None, shot_anchors={10: (1.575, 7.5)}
        )
        assert pts[0] == (10.0, 7.5)
        assert pts[5] == pytest.approx((5.7875, 7.5))
        assert pts[10] == pytest.approx((1.575, 7.5))
        assert pts[15] == pytest.approx((1.7875, 7.5))
        assert pts[20] == (2.0, 7.5)
        assert last_holder_before(frames, 3, holders, 5) == (1, 0)

    def test_static_rim_fp_filter_drops_persistent_artifact(self) -> None:
        frames, rims, det_frames = _static_fp_inputs()
        filtered = _filter_ball_candidates(
            frames,
            det_frames,
            rims,
            {"ball_min_conf": 0.05, "static_min_frames": 6, "static_cluster_px": 14.0},
        )
        assert all(len(filtered[f]) == 1 and filtered[f][0][0] < 1000.0 for f in frames)


class TestFusedBall:
    """Fused Layer 7 backend: Viterbi possession smoothing and rim-trajectory shots."""

    @pytest.mark.parametrize(
        ("made", "expected_outcome"),
        [
            pytest.param(True, "made", id="made-3pt"),
            pytest.param(False, "missed", id="missed-3pt"),
        ],
    )
    def test_detects_shot_outcome_and_smooths_possession(self, made: bool, expected_outcome: str) -> None:
        seq, upstream = _synthetic_shot_upstream(made=made)
        out = FusedBall({}).run(seq, upstream, _dummy_read)
        assert out["frames"]["15"]["holder_track_id"] == 1
        assert out["frames"]["50"]["holder_track_id"] is None
        assert len(out["shots"]) == 1
        sh = out["shots"][0]
        assert sh["outcome"] == expected_outcome
        assert sh["shooter_track_id"] == 1
        assert sh["shooter_team"] == "home"
        assert sh["points"] == 3


class TestSkalskiBall:
    """Reference Skalski baseline backend."""

    def test_detects_action_based_made_shot(self) -> None:
        seq, upstream = _synthetic_skalski_upstream()
        out = SkalskiBall({}).run(seq, upstream, _dummy_read)
        assert out["frames"]["0"]["holder_track_id"] == 7
        assert out["frames"]["0"]["holder_team"] == "away"
        assert len(out["shots"]) == 1
        assert out["shots"][0]["outcome"] == "made"
        assert out["shots"][0]["shooter_track_id"] == 7
        assert out["shots"][0]["points"] == 2


class TestEvalBall:
    """Gold evaluation against v1 boxes and events."""

    def test_scores_boxes_and_events(self) -> None:
        ball_by_seq = {
            "vid_s1": {
                "frames": {
                    "0": {"ball_px": [105.0, 105.0], "holder_track_id": 1, "holder_team": "home"},
                    "5": {"ball_px": None, "holder_track_id": None, "holder_team": None},
                },
                "shots": [
                    {
                        "start_frame": 0,
                        "rim_frame": 50,
                        "outcome": "made",
                        "shooter_track_id": 1,
                        "shooter_team": "home",
                    },
                ],
            }
        }
        tracks_by_seq = {"vid_s1": {"0": [[1, 10.0, 10.0, 60.0, 110.0, 0.9, "player", 0]], "5": []}}
        gold_frames = [
            {
                "youtube_id": "vid",
                "shot": 1,
                "frame": 0,
                "boxes": [
                    {"cls": "ball", "xyxy": [95.0, 95.0, 115.0, 115.0]},
                    {"cls": "player-in-possession", "xyxy": [10.0, 10.0, 60.0, 110.0], "team": "home"},
                ],
            },
            {"youtube_id": "vid", "shot": 1, "frame": 5, "boxes": []},
        ]
        bm = score_boxes(ball_by_seq, tracks_by_seq, gold_frames)
        assert (bm["ball_f1"], bm["poss_acc"], bm["holder_f1"], bm["team_acc"]) == (1.0, 1.0, 1.0, 1.0)

        gt_events = [
            {"sequence": "vid_s1", "rim_frame": 55, "outcome": "made", "shooter_team": "home"},
            {"sequence": "vid_s1", "rim_frame": 200, "outcome": "missed", "shooter_team": "away"},
        ]
        em = score_events(ball_by_seq, tracks_by_seq, gt_events)
        assert (em["shot_gt"], em["shot_pred"], em["shot_p"], em["shot_r"]) == (2, 1, 1.0, 0.5)
        assert (em["shot_outcome_acc"], em["shot_shooter_acc"]) == (1.0, 1.0)

    def test_shooter_match_requires_best_overlapping_gt_box(self, tmp_path: Path) -> None:
        # Predicted box overlaps defender (gt#2) at IoU ~0.85 and true shooter (gt#1) at IoU ~0.48 -> must NOT match.
        gt_dir = tmp_path / "vid_s1" / "gt"
        gt_dir.mkdir(parents=True)
        (tmp_path / "vid_s1" / "frames.txt").write_text("0\n", encoding="utf-8")
        (tmp_path / "vid_s1" / "identities.csv").write_text(
            "tid,team,number,name,team_name,note\n1,home,10,Shooter,Home,\n2,away,5,Defender,Away,\n",
            encoding="utf-8",
        )
        (gt_dir / "gt.txt").write_text(
            "1,1,100,100,60,120,1,1,1\n1,2,115,100,60,120,1,2,1\n",
            encoding="utf-8",
        )
        ball_by_seq = {
            "vid_s1": {
                "frames": {},
                "shots": [
                    {
                        "start_frame": 0,
                        "rim_frame": 50,
                        "outcome": "made",
                        "shooter_track_id": 9,
                        "shooter_team": "away",
                    },
                ],
            }
        }
        tracks_by_seq = {"vid_s1": {"0": [[9, 118.0, 100.0, 178.0, 220.0, 0.9, "player", 0]]}}
        gt_events = [{"sequence": "vid_s1", "rim_frame": 50, "outcome": "made", "shooter_gt_tid": 1}]
        em = score_events(ball_by_seq, tracks_by_seq, gt_events, gold_v0_root=tmp_path)
        assert em["shot_shooter_acc"] == 0.0


class TestBallExportAndRadar:
    """Parquet ball export schema and radar overlay rendering."""

    def test_writes_ball_parquet_and_renders_radar_overlay(self, tmp_path: Path) -> None:
        assert "ball" in LAYERS
        assert "positions" in OPTIONAL_BY_LAYER["ball"]
        cfg = RunConfig(
            name="demo-m9",
            layers={
                "shots": LayerConfig(backend="gold"),
                "detect": LayerConfig(backend="rfdetr_torch"),
                "track": LayerConfig(backend="bytetrack"),
                "team": LayerConfig(backend="siglip_kmeans"),
                "positions": LayerConfig(backend="bbox_bottom"),
                "ball": LayerConfig(backend="fused"),
            },
        )
        video = tmp_path / "clip.mp4"
        _make_blank_video(video)
        seq = Sequence("demo_s1", "demo_clip", str(video), 1, [0, 5], 50.0)
        runs = tmp_path / "runs"

        _write_layer(
            cfg,
            runs,
            seq,
            "track",
            {
                "frames": {
                    "0": [[1, 50, 40, 90, 130, 0.9, "player", 0]],
                    "5": [[1, 55, 40, 95, 130, 0.9, "player", 0]],
                }
            },
        )
        _write_layer(cfg, runs, seq, "team", {"frames": {"0": ["home"], "5": ["home"]}})
        _write_layer(cfg, runs, seq, "positions", {"tracks": {"1": [[0, 10.0, 7.5], [5, 10.5, 7.5]]}})
        _write_layer(
            cfg,
            runs,
            seq,
            "ball",
            {
                "frames": {
                    "0": {
                        "x_m": 10.0,
                        "y_m": 7.5,
                        "ball_px": [70.0, 80.0],
                        "holder_track_id": 1,
                        "holder_team": "home",
                        "in_flight": False,
                    },
                    "5": {
                        "x_m": 6.0,
                        "y_m": 7.5,
                        "ball_px": [120.0, 50.0],
                        "holder_track_id": None,
                        "holder_team": None,
                        "in_flight": True,
                    },
                },
                "shots": [
                    {
                        "start_frame": 0,
                        "rim_frame": 5,
                        "end_frame": 5,
                        "outcome": "made",
                        "x_m": 10.0,
                        "y_m": 7.5,
                        "points": 3,
                    }
                ],
                "rims": {},
                "stats": {},
                "ms_per_frame": 0.2,
            },
        )

        pq_path = tmp_path / "demo_s1.ball.parquet"
        out_pq = export_ball_sequence(cfg, seq, runs=runs, out=pq_path)
        table = pq.read_table(pq_path)
        assert out_pq == pq_path
        assert table.num_rows == 2
        assert table.schema.equals(BALL_SCHEMA)
        assert table.column("holder_track_id").to_pylist() == [1, None]
        assert table.column("in_flight").to_pylist() == [False, True]

        mp4_path = render(cfg, seq, runs=runs)
        assert mp4_path.exists()
        assert mp4_path.stat().st_size > 0
