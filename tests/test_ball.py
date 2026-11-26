"""Unit tests for M9 Layer 7 (ball, possession, shots), evaluation, parquet export, and radar overlays."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from bga.ball.base import (
    BASKETS_M,
    BallTrackState,
    interpolate_ball_2d,
    is_referee_track,
    last_holder_before,
    match_box_to_track,
    nearest_basket,
    pack_output,
    shot_points,
)
from bga.ball.fused import (
    FusedBall,
    _ball_node_score,
    _classify_shot_outcome,
    _filter_ball_candidates,
    _find_shooter,
    _loose_ball_bonus,
    _possession_emissions,
    _track_ball,
    _track_possession,
)
from bga.ball.skalski import SkalskiBall
from bga.config import LAYERS, OPTIONAL_BY_LAYER, LayerConfig, RunConfig
from bga.eval.ball import load_holders, load_shot_gold, score, score_boxes, score_events
from bga.export.parquet import BALL_SCHEMA, export_ball_sequence
from bga.export.radar import render
from bga.stage import Sequence


def _dummy_read(_frame: int) -> np.ndarray:
    raise AssertionError("ball backends must not read video frames")


def _static_fp_inputs() -> tuple[list[int], dict[int, list[float]], dict[str, list[list[Any]]]]:
    """Build a 12-frame sequence with one static rim-relative FP and one moving ball."""
    frames = list(range(0, 60, 5))
    rims = {f: [900.0, 200.0, 960.0, 225.0, 0.9] for f in frames}
    det_frames: dict[str, list[list[Any]]] = {}
    for idx, f in enumerate(frames):
        det_frames[str(f)] = [
            [1020.0, 165.0, 1040.0, 185.0, 0.45, "ball"],
            [500.0 + idx * 35.0, 400.0 - idx * 10.0, 520.0 + idx * 35.0, 420.0 - idx * 10.0, 0.70, "ball"],
        ]
    return frames, rims, det_frames


def _free_throw_hold_inputs(
    *, with_player: bool
) -> tuple[
    list[int], dict[int, list[float]], dict[str, list[list[Any]]], dict[int, list[tuple[int, list[float], str]]]
]:
    """Build a 30-step sequence: ball held still with 2-3 px jitter for 13 steps, then a shot arc."""
    frames = list(range(0, 150, 5))
    rims = {f: [900.0, 200.0, 960.0, 225.0, 0.9] for f in frames}
    det_frames: dict[str, list[list[Any]]] = {}
    tracks_by_f: dict[int, list[tuple[int, list[float], str]]] = {}
    pbox = [380.0, 440.0, 450.0, 620.0]
    for i, f in enumerate(frames):
        if i <= 12:
            bx, by = 415.0 + (i % 3) - 1.0, 490.0 + ((i + 1) % 3) - 1.0
        else:
            t = (i - 12) / 17.0
            bx = 415.0 + t * (930.0 - 415.0)
            by = 490.0 + t * (212.0 - 490.0) - 140.0 * t * (1.0 - t)
        boxes: list[list[Any]] = [[bx - 10.0, by - 10.0, bx + 10.0, by + 10.0, 0.85, "ball"]]
        if with_player:
            boxes.append([*pbox, 0.95, "player"])
            if i <= 12:
                boxes.append([*pbox, 0.85, "player-in-possession"])
            tracks_by_f[f] = [(1, pbox, "player")]
        else:
            tracks_by_f[f] = []
        det_frames[str(f)] = boxes
    return frames, rims, det_frames, tracks_by_f


def _synthetic_shot_upstream(made: bool, *, stride: int = 5) -> tuple[Sequence, dict[str, Any]]:
    """Build a synthetic 50 fps sequence with holder -> 3pt release -> rim arrival -> made/missed."""
    frames = list(range(0, 110, stride))
    seq = Sequence("synth_s1", "synth_clip", "unused.mp4", 1, frames, 50.0)
    rim = [900.0, 200.0, 960.0, 225.0, 0.92, "rim"]
    det_frames: dict[str, list[list[Any]]] = {}
    track_frames: dict[str, list[list[Any]]] = {}
    team_frames: dict[str, list[str]] = {}
    pos_pts: list[list[float]] = []

    for f in frames:
        t_s = f / 50.0
        player_box = [370.0, 460.0, 430.0, 620.0]
        track_frames[str(f)] = [[1, *player_box, 0.95, "player", 0]]
        team_frames[str(f)] = ["home"]
        pos_pts.append([float(f), 9.0, 7.5])
        boxes: list[list[Any]] = [[*player_box, 0.95, "player"], rim]
        if t_s <= 0.50:
            if not (0.28 <= t_s <= 0.32):
                boxes.append([*player_box, 0.85, "player-in-possession"])
            bx, by = 405.0, 500.0
        elif t_s <= 1.00:
            u = (t_s - 0.50) / 0.50
            bx = 405.0 + u * (930.0 - 405.0)
            by = 500.0 + u * (212.0 - 500.0) - 180.0 * u * (1.0 - u)
        elif made:
            bx = 932.0
            by = 212.5 + (t_s - 1.00) * 320.0
        else:
            bx = 930.0 + (t_s - 1.00) * 550.0
            by = 190.0 + max(0.0, t_s - 1.20) * 300.0
        if t_s <= 1.60:
            boxes.append([bx - 12.0, by - 12.0, bx + 12.0, by + 12.0, 0.85, "ball"])
        det_frames[str(f)] = boxes

    upstream = {
        "detect": {"frames": det_frames},
        "track": {"frames": track_frames},
        "team": {"frames": team_frames},
        "positions": {"tracks": {"1": pos_pts}},
    }
    return seq, upstream


def _synthetic_skalski_upstream(*, made: bool = True, stride: int = 5) -> tuple[Sequence, dict[str, Any]]:
    """Build a synthetic 50 fps sequence with ``player-jump-shot`` and optional ``ball-in-basket``."""
    frames = list(range(0, 140, stride))
    seq = Sequence("sk_s1", "sk_clip", "unused.mp4", 1, frames, 50.0)
    det_frames: dict[str, list[list[Any]]] = {}
    track_frames: dict[str, list[list[Any]]] = {}
    team_frames: dict[str, list[str]] = {}
    pos_pts: list[list[float]] = []
    for f in frames:
        t_s = f / 50.0
        pbox = [300.0, 400.0, 360.0, 580.0]
        track_frames[str(f)] = [[7, *pbox, 0.9, "player", 0]]
        team_frames[str(f)] = ["away"]
        pos_pts.append([float(f), 5.0, 7.5])
        boxes: list[list[Any]] = [[*pbox, 0.9, "player"], [900.0, 200.0, 960.0, 225.0, 0.9, "rim"]]
        if t_s <= 0.30:
            boxes.append([*pbox, 0.85, "player-in-possession"])
        if 0.30 <= t_s <= 0.60:
            boxes.append([*pbox, 0.85, "player-jump-shot"])
        if made and 0.80 <= t_s <= 1.00:
            boxes.append([910.0, 200.0, 950.0, 235.0, 0.80, "ball-in-basket"])
        det_frames[str(f)] = boxes
    upstream = {
        "detect": {"frames": det_frames},
        "track": {"frames": track_frames},
        "team": {"frames": team_frames},
        "positions": {"tracks": {"7": pos_pts}},
    }
    return seq, upstream


def _long_dropout_candidates() -> tuple[
    list[int], dict[int, list[tuple[float, float, float]]], dict[str, list[list[Any]]]
]:
    """Build 60 steps of conf-0.8 ball detections with a 10-step dropout at steps 20..29."""
    frames = list(range(0, 300, 5))
    cands: dict[int, list[tuple[float, float, float]]] = {}
    det_frames: dict[str, list[list[Any]]] = {}
    for i, f in enumerate(frames):
        det_frames[str(f)] = []
        if 20 <= i <= 29:
            cands[f] = []
        else:
            cands[f] = [(300.0 + i * 10.0, 400.0, 0.80)]
    return frames, cands, det_frames


def _rim_approach_state(rim_holder_team: str) -> tuple[BallTrackState, list[tuple[float, float, float] | None]]:
    """Ten steps at 10 fps: home player 1 holds, releases at step 5, the ball reaches the rim at step 9.

    From step 6 on, the ball is within 2 rim widths and player 2 (``rim_holder_team``) stands under it holding.

    """
    frames = list(range(0, 50, 5))
    holders = {f: (1 if f <= 25 else 2) for f in frames}
    rel: list[tuple[float, float, float] | None] = [(0.0, -6.0 + 0.5 * i, 6.0 - 0.5 * i) for i in range(6)]
    rel += [(0.0, -1.5, 1.5), (0.0, -1.0, 1.0), (0.0, -0.5, 0.5), (0.0, 0.0, 0.0)]
    state = BallTrackState(frames, {}, {}, holders, {1: "home", 2: rim_holder_team}, {})
    return state, rel


def _make_blank_video(path: Path, n_frames: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (320, 180))
    for _ in range(n_frames):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()


def _write_layer(cfg: RunConfig, runs: Path, seq: Sequence, layer: str, data: dict[str, Any]) -> None:
    d = cfg.run_dir(runs, seq.clip, layer)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{seq.name}.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def eval_sample() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[list[Any]]]]]:
    """Minimal synthetic Layer 7 and track outputs shared across box/event eval tests."""
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
    return ball_by_seq, tracks_by_seq


@pytest.fixture
def export_env(tmp_path: Path) -> tuple[RunConfig, Path, Sequence, Path]:
    """Populate a temporary run directory with track, team, positions, and ball outputs."""
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
        {"frames": {"0": [[1, 50, 40, 90, 130, 0.9, "player", 0]], "5": [[1, 55, 40, 95, 130, 0.9, "player", 0]]}},
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
    return cfg, runs, seq, tmp_path


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
    """2D ball interpolation, holder lookback, referee exclusion, and output packing."""

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

    def test_interpolate_ball_2d_leaves_long_blind_gap_null(self) -> None:
        frames = [0, 50, 100, 150]
        holders = {0: 1, 50: None, 100: None, 150: 2}
        pos_by_tid = {1: {0: (5.0, 7.5)}, 2: {150: (20.0, 7.5)}}
        pts = interpolate_ball_2d(frames, holders, pos_by_tid, {}, dict.fromkeys(frames), None)
        assert pts[0] == (5.0, 7.5)
        assert pts[50] is None
        assert pts[100] is None
        assert pts[150] == (20.0, 7.5)

    def test_last_holder_before_finds_prior_holder(self) -> None:
        frames = [0, 5, 10, 15, 20]
        holders = {0: 1, 5: None, 10: None, 15: None, 20: 2}
        assert last_holder_before(frames, 3, holders, 5) == (1, 0)

    def test_referee_team_label_excluded_from_possession_matching(self) -> None:
        teams = {1: "ref", 2: "home"}
        tracks = [(1, [0.0, 0.0, 50.0, 100.0], "player"), (2, [300.0, 0.0, 350.0, 100.0], "player")]
        assert is_referee_track(1, "player", teams) is True
        assert match_box_to_track([2.0, 2.0, 48.0, 98.0], tracks, teams) is None

    def test_pack_output_sets_in_flight_false_when_ball_undetected(self) -> None:
        state = BallTrackState(
            frames=[0, 5],
            ball_px={0: None, 5: [500.0, 300.0]},
            ball_conf={0: 0.0, 5: 0.8},
            holders={0: None, 5: None},
            teams={},
            rims={},
        )
        out = pack_output(state, {0: None, 5: (10.0, 7.5)}, [], 0.1)
        assert out["frames"]["0"]["in_flight"] is False
        assert out["frames"]["5"]["in_flight"] is True
        assert out["stats"]["in_flight_frames"] == 1


class TestFusedBall:
    """Fused Layer 7 backend: static-FP filtering, multi-segment DP, possession, and rim-trajectory shots."""

    def test_static_rim_fp_filter_drops_persistent_artifact(self) -> None:
        frames, rims, det_frames = _static_fp_inputs()
        filtered = _filter_ball_candidates(
            frames,
            det_frames,
            rims,
            {"ball_min_conf": 0.05, "static_min_s": 0.6, "static_cluster_px": 14.0},
            eff_fps=10.0,
        )
        assert all(len(filtered[f]) == 1 and filtered[f][0][0] < 1000.0 for f in frames)

    @pytest.mark.parametrize(
        "with_player",
        [
            pytest.param(True, id="with-player-box"),
            pytest.param(False, id="without-player-box"),
        ],
    )
    def test_static_rim_fp_filter_preserves_held_ball(self, with_player: bool) -> None:
        frames, rims, det_frames, tracks_by_f = _free_throw_hold_inputs(with_player=with_player)
        filtered = _filter_ball_candidates(
            frames,
            det_frames,
            rims,
            FusedBall.defaults,
            eff_fps=10.0,
            tracks_by_f=tracks_by_f,
            teams={1: "home"},
        )
        assert all(len(filtered[f]) == 1 for f in frames[:13])

    def test_track_ball_recovers_both_segments_across_long_dropout(self) -> None:
        frames, cands, det_frames = _long_dropout_candidates()
        ball_px, _ = _track_ball(frames, cands, det_frames, {}, {}, FusedBall.defaults, eff_fps=10.0)
        assert all(ball_px[f] is not None for f in frames[:20])
        assert all(ball_px[f] is None for f in frames[20:30])
        assert all(ball_px[f] is not None for f in frames[30:])

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

    def test_detects_made_shot_at_stride_1(self) -> None:
        seq, upstream = _synthetic_shot_upstream(made=True, stride=1)
        out = FusedBall({}).run(seq, upstream, _dummy_read)
        assert len(out["shots"]) == 1
        assert out["shots"][0]["outcome"] == "made"
        assert out["shots"][0]["shooter_track_id"] == 1

    @pytest.mark.parametrize(
        ("excl_rw", "expected"),
        [
            pytest.param(0.0, (2, 45), id="m9-last-holder-at-rim"),
            pytest.param(2.0, (1, 25), id="search-before-rim-zone"),
        ],
    )
    def test_shooter_search_skips_holders_in_rim_zone(self, excl_rw: float, expected: tuple[int, int]) -> None:
        state, rel = _rim_approach_state("away")
        params = {**FusedBall.defaults, "shot_release_rim_excl_rw": excl_rw}
        assert _find_shooter(state, 9, rel, params, 10.0) == expected

    @pytest.mark.parametrize(
        ("window_s", "expected"),
        [
            pytest.param(0.0, (2, 25), id="last-holder"),
            pytest.param(2.0, (1, 20), id="team-in-possession"),
        ],
    )
    def test_team_window_prefers_team_in_possession(self, window_s: float, expected: tuple[int, int]) -> None:
        state, rel = _rim_approach_state("away")
        state.holders[25] = 2  # a defender's box overlaps the shooter at release
        params = {**FusedBall.defaults, "shot_release_rim_excl_rw": 2.0, "shot_team_window_s": window_s}
        assert _find_shooter(state, 9, rel, params, 10.0) == expected

    @pytest.mark.parametrize(
        ("penalty", "expected"),
        [
            pytest.param(0.0, 0.0, id="off"),
            pytest.param(1.5, -1.5, id="ball-two-radii-away"),
        ],
    )
    def test_far_ball_penalty_scores_against_far_player(self, penalty: float, expected: float) -> None:
        tracks = [(1, [0.0, 0.0, 50.0, 100.0], "player")]
        params = {**FusedBall.defaults, "ball_far_penalty": penalty}
        em = _possession_emissions([], tracks, {1: "home"}, [180.0, 50.0], 0.9, params)
        assert em[1] == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("local_rw", "expected"),
        [
            pytest.param(0.0, "missed", id="off-far-jump-counts-as-miss"),
            pytest.param(4.0, "made", id="far-jump-ignored-net-drop-counts"),
        ],
    )
    def test_outcome_ignores_far_tracker_jump(self, local_rw: float, expected: str) -> None:
        frames = [0, 5, 10, 15]
        state = BallTrackState(frames, {}, {}, {}, {}, {})
        rel = [(0.0, 0.0, 0.0), (-5.0, 4.0, 6.4), (0.1, 1.2, 1.2), None]
        params = {**FusedBall.defaults, "outcome_local_rw": local_rw}
        assert _classify_shot_outcome(state, 0, rel, {}, params, 10.0)[0] == expected

    @pytest.mark.parametrize(
        ("bonus", "ball_x", "expected"),
        [
            pytest.param(0.0, 930.0, 0.0, id="off"),
            pytest.param(0.6, 930.0, 0.6, id="at-rim-centre"),
            pytest.param(0.6, 960.0, 0.3, id="three-quarter-rim-width-away"),
            pytest.param(0.6, 1100.0, 0.0, id="outside-reach"),
        ],
    )
    def test_rim_bonus_lifts_candidates_near_rim(self, bonus: float, ball_x: float, expected: float) -> None:
        rim = [910.0, 200.0, 950.0, 210.0, 0.9]
        params = {**FusedBall.defaults, "ball_rim_bonus": bonus, "ball_rim_bonus_rw": 1.5}
        base = _ball_node_score(ball_x, 205.0, 0.5, [], [], {}, params)
        assert _ball_node_score(ball_x, 205.0, 0.5, [], [], {}, params, rim) - base == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("bonus", "expected"),
        [
            pytest.param(None, 1, id="off-ball-in-box-is-held"),
            pytest.param(2.0, None, id="loose-window-ball-in-box-is-unheld"),
        ],
    )
    def test_loose_bonus_releases_ball_overlapping_player(self, bonus: float | None, expected: int | None) -> None:
        frames = [0, 5, 10]
        tracks = {f: [(1, [0.0, 0.0, 50.0, 100.0], "player")] for f in frames}
        ball_px: dict[int, list[float] | None] = {f: [25.0, 50.0] for f in frames}
        none_bonus = None if bonus is None else dict.fromkeys(frames, bonus)
        holders = _track_possession(
            frames, {}, tracks, {1: "home"}, ball_px, dict.fromkeys(frames, 0.9), FusedBall.defaults, none_bonus
        )
        assert holders[5] == expected

    def test_loose_bonus_covers_window_around_closest_rim_step(self) -> None:
        rel = [None, (0.0, -3.0, 3.0), (0.0, -0.5, 0.5), (0.0, 0.1, 0.1), (0.0, 1.0, 1.0), None, None]
        params = {"poss_loose_bonus": 1.0, "poss_loose_pre_s": 0.1, "poss_loose_post_s": 0.2}
        frames = [0, 5, 10, 15, 20, 25, 30]
        assert _loose_ball_bonus(frames, [[2, 3]], rel, params, 10.0) == {10: 1.0, 15: 1.0, 20: 1.0, 25: 1.0}

    def test_loose_bonus_keeps_shooter_released_before_rim(self) -> None:
        seq, upstream = _synthetic_shot_upstream(made=True)
        params = {"poss_loose_bonus": 1.0, "poss_loose_pre_s": 0.3, "poss_loose_post_s": 2.5}
        out = FusedBall(params).run(seq, upstream, _dummy_read)
        assert [(s["shooter_track_id"], s["outcome"]) for s in out["shots"]] == [(1, "made")]


class TestSkalskiBall:
    """Reference Skalski baseline backend."""

    def test_detects_action_based_made_shot(self) -> None:
        seq, upstream = _synthetic_skalski_upstream(made=True)
        out = SkalskiBall({}).run(seq, upstream, _dummy_read)
        assert out["frames"]["0"]["holder_track_id"] == 7
        assert out["frames"]["0"]["holder_team"] == "away"
        assert len(out["shots"]) == 1
        assert out["shots"][0]["outcome"] == "made"
        assert out["shots"][0]["rim_frame"] is not None
        assert out["shots"][0]["shooter_track_id"] == 7
        assert out["shots"][0]["points"] == 2

    def test_emits_null_rim_frame_for_missed_shot(self) -> None:
        seq, upstream = _synthetic_skalski_upstream(made=False)
        out = SkalskiBall({}).run(seq, upstream, _dummy_read)
        assert len(out["shots"]) == 1
        assert out["shots"][0]["outcome"] == "missed"
        assert out["shots"][0]["rim_frame"] is None
        assert out["shots"][0]["start_frame"] == 15

    def test_detects_made_shot_at_stride_1(self) -> None:
        seq, upstream = _synthetic_skalski_upstream(made=True, stride=1)
        out = SkalskiBall({}).run(seq, upstream, _dummy_read)
        assert len(out["shots"]) == 1
        assert out["shots"][0]["outcome"] == "made"

    def test_importing_skalski_does_not_load_sports(self) -> None:
        res = subprocess.run(
            [sys.executable, "-c", "import sys, bga.ball.skalski; assert 'sports' not in sys.modules"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, res.stderr


class TestEvalBall:
    """Gold evaluation against v1 boxes and verified events."""

    def test_scores_boxes(self, eval_sample: tuple[dict[str, dict[str, Any]], dict[str, Any]]) -> None:
        ball_by_seq, tracks_by_seq = eval_sample
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

    @pytest.mark.parametrize(
        ("holder", "expected"),
        [
            pytest.param(1, (1.0, 1.0), id="holder-gold-agrees"),
            pytest.param(None, (0.5, 0.0), id="holder-gold-says-unheld"),
        ],
    )
    def test_scores_boxes_against_holder_gold(
        self,
        eval_sample: tuple[dict[str, dict[str, Any]], dict[str, Any]],
        tmp_path: Path,
        holder: int | None,
        expected: tuple[float, float],
    ) -> None:
        ball_by_seq, tracks_by_seq = eval_sample
        gold_frames = [
            {
                "youtube_id": "vid",
                "shot": 1,
                "frame": 0,
                "boxes": [
                    {"cls": "ball", "xyxy": [95.0, 95.0, 115.0, 115.0]},
                    {"cls": "player-jump-shot", "xyxy": [10.0, 10.0, 60.0, 110.0], "team": "home"},
                ],
            },
            {"youtube_id": "vid", "shot": 1, "frame": 5, "boxes": []},
        ]
        rows = [{"youtube_id": "vid", "shot": 1, "frame": 0, "holder": holder, "status": "implementer"}]
        path = tmp_path / "holders.json"
        path.write_text(json.dumps({"version": "v1", "frames": rows}), encoding="utf-8")
        bm = score_boxes(ball_by_seq, tracks_by_seq, gold_frames, load_holders(path))
        assert (bm["poss_acc"], bm["holder_f1"]) == expected

    def test_scores_events(self, eval_sample: tuple[dict[str, dict[str, Any]], dict[str, Any]]) -> None:
        ball_by_seq, tracks_by_seq = eval_sample
        gt_events = [
            {"sequence": "vid_s1", "rim_frame": 55, "outcome": "made", "shooter_team": "home"},
            {"sequence": "vid_s1", "rim_frame": 200, "outcome": "missed", "shooter_team": "away"},
        ]
        em = score_events(ball_by_seq, tracks_by_seq, gt_events)
        assert (em["shot_gt"], em["shot_pred"], em["shot_p"], em["shot_r"]) == (2, 1, 1.0, 0.5)
        assert (em["shot_outcome_acc"], em["shot_shooter_acc"]) == (1.0, 1.0)

    def test_outcome_recall_splits_by_gold_outcome(self) -> None:
        shots = [
            {"start_frame": 0, "rim_frame": 50, "outcome": "made"},
            {"start_frame": 150, "rim_frame": 200, "outcome": "made"},
            {"start_frame": 300, "rim_frame": 350, "outcome": "made"},
        ]
        gt_events = [
            {"sequence": "vid_s1", "rim_frame": 50, "outcome": "made"},
            {"sequence": "vid_s1", "rim_frame": 200, "outcome": "missed"},
            {"sequence": "vid_s1", "rim_frame": 350, "outcome": "missed"},
        ]
        em = score_events({"vid_s1": {"frames": {}, "shots": shots}}, {"vid_s1": {}}, gt_events)
        assert (em["shot_outcome_acc"], em["shot_made_r"], em["shot_missed_r"]) == (pytest.approx(1 / 3), 1.0, 0.0)

    def test_matches_null_rim_frame_against_gold_release_frame(self) -> None:
        ball_by_seq = {
            "vid_s1": {
                "frames": {},
                "shots": [
                    {
                        "start_frame": 2360,
                        "rim_frame": None,
                        "end_frame": 2445,
                        "outcome": "missed",
                        "shooter_team": "home",
                    }
                ],
            }
        }
        gt_events = [
            {
                "sequence": "vid_s1",
                "release_frame": 2360,
                "rim_frame": 2395,
                "outcome": "made",
                "shooter_team": "home",
            }
        ]
        em = score_events(ball_by_seq, {"vid_s1": {}}, gt_events)
        assert (em["shot_f1"], em["shot_outcome_acc"], em["shot_shooter_acc"]) == (1.0, 0.0, 1.0)

    def test_score_events_excludes_unverified_sequences(self) -> None:
        ball_by_seq = {
            "vid_s1": {"frames": {}, "shots": [{"start_frame": 10, "rim_frame": 50, "outcome": "made"}]},
            "vid_s2": {"frames": {}, "shots": [{"start_frame": 10, "rim_frame": 50, "outcome": "made"}]},
        }
        gt_events = [
            {"sequence": "vid_s1", "rim_frame": 50, "outcome": "made"},
            {"sequence": "vid_s2", "rim_frame": 50, "outcome": "made"},
        ]
        em = score_events(ball_by_seq, {"vid_s1": {}, "vid_s2": {}}, gt_events, verified_sequences={"vid_s1"})
        assert (em["shot_gt"], em["shot_pred"], em["shot_f1"]) == (1, 1, 1.0)

    def test_score_warns_and_skips_unverified_sequences(
        self,
        export_env: tuple[RunConfig, Path, Sequence, Path],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cfg, runs, seq, tmp_path = export_env
        monkeypatch.setattr("bga.eval.ball.pipeline.run", lambda _cfg, runs=runs: [seq])
        events_path = tmp_path / "events.json"
        events_path.write_text(
            json.dumps({
                "version": 1,
                "verified_sequences": [],
                "events": [{"sequence": seq.name, "rim_frame": 5, "outcome": "made"}],
            }),
            encoding="utf-8",
        )
        res = score(cfg, runs=runs, boxes_path=tmp_path / "missing.json", events_path=events_path)
        assert res["unverified_sequences"] == [seq.name]
        assert math.isnan(res["shot_f1"])
        assert "unverified sequences" in caplog.text

    @pytest.mark.parametrize(
        ("shooter_frame", "expected"),
        [
            pytest.param(1, 1.0, id="gold-box-0.4s-before-start"),
            pytest.param(None, 0.0, id="no-gold-box-near-start"),
        ],
    )
    def test_shooter_match_uses_nearest_gold_frame_boxing_the_shooter(
        self, tmp_path: Path, shooter_frame: int | None, expected: float
    ) -> None:
        gt_dir = tmp_path / "vid_s1" / "gt"
        gt_dir.mkdir(parents=True)
        (tmp_path / "vid_s1" / "frames.txt").write_text("0\n20\n", encoding="utf-8")
        (tmp_path / "vid_s1" / "identities.csv").write_text(
            "tid,team,number,name,team_name,note\n1,home,10,Shooter,Home,\n2,away,5,Defender,Away,\n",
            encoding="utf-8",
        )
        shooter_rows = "" if shooter_frame is None else f"{shooter_frame},1,100,100,60,120,1,1,1\n"
        (gt_dir / "gt.txt").write_text(shooter_rows + "2,2,400,100,60,120,1,2,1\n", encoding="utf-8")
        box = [9, 100.0, 100.0, 160.0, 220.0, 0.9, "player", 0]
        ball_by_seq = {
            "vid_s1": {
                "frames": {},
                "shots": [{"start_frame": 20, "rim_frame": 60, "outcome": "made", "shooter_track_id": 9}],
            }
        }
        tracks_by_seq = {"vid_s1": {"0": [box], "20": [box]}}
        gt_events = [{"sequence": "vid_s1", "rim_frame": 60, "outcome": "made", "shooter_gt_tid": 1}]
        em = score_events(ball_by_seq, tracks_by_seq, gt_events, gold_v0_root=tmp_path)
        assert em["shot_shooter_acc"] == expected

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

    @pytest.mark.parametrize(
        ("pred_tid", "track_frame", "expected"),
        [
            pytest.param(1, "20", 1.0, id="exact-release-frame-match"),
            pytest.param(1, "25", 1.0, id="nearby-frame-fallback"),
            pytest.param(2, "20", 0.0, id="defender-overlaps-less-than-shooter"),
            pytest.param(None, "20", 0.0, id="missing-shooter-tid"),
        ],
    )
    def test_shooter_match_by_shot_gold_box(self, pred_tid: int | None, track_frame: str, expected: float) -> None:
        ball_by_seq = {
            "vid_s1": {
                "frames": {},
                "shots": [
                    {
                        "start_frame": 15,
                        "rim_frame": 60,
                        "outcome": "made",
                        "shooter_track_id": pred_tid,
                        "shooter_team": "home",
                    }
                ],
            }
        }
        tracks_by_seq = {
            "vid_s1": {
                track_frame: [
                    [1, 100.0, 100.0, 160.0, 220.0, 0.9, "player", 0],
                    [2, 118.0, 100.0, 178.0, 220.0, 0.9, "player", 0],
                ]
            }
        }
        gt_events = [
            {
                "sequence": "vid_s1",
                "release_frame": 20,
                "rim_frame": 60,
                "outcome": "made",
                "shooter_xyxy": [100.0, 100.0, 160.0, 220.0],
                "shooter_team": "home",
            }
        ]
        em = score_events(ball_by_seq, tracks_by_seq, gt_events)
        assert em["shot_shooter_acc"] == expected

    def test_load_shot_gold_converts_frames_and_events(self, tmp_path: Path) -> None:
        sg_path = tmp_path / "shot_gold.json"
        sg_path.write_text(
            json.dumps({
                "version": "v1",
                "verified_sequences": ["vid_s2"],
                "shots": [],
                "frames": [
                    {
                        "sequence": "vid_s2",
                        "youtube_id": "vid",
                        "shot": 2,
                        "frame": 10,
                        "ball": [95.0, 95.0, 115.0, 115.0],
                        "holder": {"xyxy": [10.0, 10.0, 60.0, 110.0], "team": "home"},
                    },
                    {
                        "sequence": "vid_s2",
                        "youtube_id": "vid",
                        "shot": 2,
                        "frame": 60,
                        "ball": None,
                        "holder": None,
                    },
                ],
                "events": [
                    {
                        "sequence": "vid_s2",
                        "release_frame": 10,
                        "rim_frame": 50,
                        "outcome": "made",
                        "shooter_xyxy": [10.0, 10.0, 60.0, 110.0],
                        "shooter_team": "home",
                    }
                ],
            }),
            encoding="utf-8",
        )
        frames, holders, events, verified = load_shot_gold(sg_path)
        assert verified == {"vid_s2"}
        assert holders == {("vid", 10): 1, ("vid", 60): None}
        assert len(frames) == 2
        assert len(events) == 1

    def test_score_combines_gold_and_shot_gold(
        self,
        export_env: tuple[RunConfig, Path, Sequence, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg, runs, seq, tmp_path = export_env
        sg_seq = Sequence("demo_s2", "demo_clip", seq.video, 2, [0, 5], 50.0)
        sg_path = tmp_path / "shot_gold.json"
        sg_path.write_text(
            json.dumps({
                "version": "v1",
                "verified_sequences": ["demo_s2"],
                "shots": [],
                "frames": [
                    {
                        "sequence": "demo_s2",
                        "youtube_id": "demo",
                        "shot": 2,
                        "frame": 0,
                        "ball": [60.0, 70.0, 80.0, 90.0],
                        "holder": {"xyxy": [50.0, 40.0, 90.0, 130.0], "team": "home"},
                    }
                ],
                "events": [
                    {
                        "sequence": "demo_s2",
                        "release_frame": 0,
                        "rim_frame": 5,
                        "outcome": "made",
                        "shooter_xyxy": [50.0, 40.0, 90.0, 130.0],
                        "shooter_team": "home",
                    }
                ],
            }),
            encoding="utf-8",
        )
        sg_cfg = cfg.model_copy(deep=True)
        sg_cfg.layers["shots"].backend = "shot_gold"
        sg_cfg.layers["shots"].params = {"stride": 5, "held_out": "exclude", "queue": str(sg_path)}
        _write_layer(
            sg_cfg,
            runs,
            sg_seq,
            "track",
            {"frames": {"0": [[1, 50, 40, 90, 130, 0.9, "player", 0]], "5": []}},
        )
        _write_layer(
            sg_cfg,
            runs,
            sg_seq,
            "ball",
            {
                "frames": {"0": {"ball_px": [70.0, 80.0], "holder_track_id": 1, "holder_team": "home"}},
                "shots": [
                    {"start_frame": 0, "rim_frame": 5, "outcome": "made", "shooter_track_id": 1, "shooter_team": "home"}
                ],
                "ms_per_frame": 0.2,
            },
        )
        monkeypatch.setattr(
            "bga.eval.ball.pipeline.run",
            lambda run_cfg, runs=runs: [sg_seq] if run_cfg.layers["shots"].backend == "shot_gold" else [seq],
        )
        events_path = tmp_path / "events.json"
        events_path.write_text(
            json.dumps({
                "version": 1,
                "verified_sequences": [seq.name],
                "events": [{"sequence": seq.name, "rim_frame": 5, "outcome": "made"}],
            }),
            encoding="utf-8",
        )
        res = score(
            cfg,
            runs=runs,
            boxes_path=tmp_path / "missing.json",
            events_path=events_path,
            holders_path=tmp_path / "missing_holders.json",
            shot_gold_path=sg_path,
        )
        assert (res["sequences"], res["box_frames"], res["shot_gt"], res["shot_f1"]) == (2, 1, 2, 1.0)
        assert (res["ball_f1"], res["poss_acc"], res["shot_shooter_acc"]) == (1.0, 1.0, 1.0)


class TestBallExportAndRadar:
    """Layer registration, Parquet ball export schema, and radar overlay rendering."""

    def test_ball_layer_registered_in_config(self) -> None:
        assert "ball" in LAYERS
        assert "positions" in OPTIONAL_BY_LAYER["ball"]

    def test_writes_ball_parquet(self, export_env: tuple[RunConfig, Path, Sequence, Path]) -> None:
        cfg, runs, seq, tmp_path = export_env
        pq_path = tmp_path / "demo_s1.ball.parquet"
        out_pq = export_ball_sequence(cfg, seq, runs=runs, out=pq_path)
        table = pq.read_table(pq_path)
        assert out_pq == pq_path
        assert table.num_rows == 2
        assert table.schema.equals(BALL_SCHEMA)
        assert table.column("holder_track_id").to_pylist() == [1, None]
        assert table.column("in_flight").to_pylist() == [False, True]

    def test_renders_radar_overlay(self, export_env: tuple[RunConfig, Path, Sequence, Path]) -> None:
        cfg, runs, seq, _ = export_env
        mp4_path = render(cfg, seq, runs=runs)
        assert mp4_path.exists()
        assert mp4_path.stat().st_size > 0
