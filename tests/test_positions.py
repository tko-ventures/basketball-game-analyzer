"""Tests for layer 6 (positions), projective singularity guards, COCO pose ankle ground points, and package court
data."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from bga.config import RunConfig
from bga.court.base import FIBA_COURT_PATH, load_fiba_court
from bga.positions.ankles_pose import (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    AnklesPose,
    ankle_ground_point,
    detect_airborne,
    hold_airborne_positions,
    match_poses,
)
from bga.positions.base import active_tracks, project
from bga.positions.bbox_bottom import BBoxBottom
from bga.stage import Sequence


class TestProjectSingularityGuard:
    """``project`` maps normal points accurately and yields NaN instead of Inf/flipped coords on ``w <= 0``."""

    def test_valid_and_empty_points(self) -> None:
        h = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(project(h, pts), [[12.0, 26.0], [16.0, 32.0]])
        assert project(h, np.empty((0, 2))).shape == (0, 2)

    def test_zero_and_negative_w_yield_nan(self) -> None:
        # w = y - 100: y=100 is on horizon (w=0), y=50 is behind camera (w=-50), y=200 is valid (w=100)
        h = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 1.0, -100.0]]
        pts = np.array([[10.0, 100.0], [10.0, 50.0], [10.0, 200.0]])
        out = project(h, pts)
        assert np.all(np.isnan(out[0]))
        assert np.all(np.isnan(out[1]))
        np.testing.assert_allclose(out[2], [10.0, 200.0])


class TestPositionsConfigWiring:
    """``positions`` optionally depends on ``link`` while ``identity`` still requires ``link``."""

    def test_positions_without_and_with_link(self) -> None:
        base = {
            "shots": {"backend": "gold"},
            "court": {"backend": "rfdetr_keypoint"},
            "detect": {"backend": "rfdetr_torch"},
            "track": {"backend": "bytetrack"},
            "positions": {"backend": "bbox_bottom"},
        }
        cfg_no_link = RunConfig(name="no-link", layers=base)
        cfg_with_link = RunConfig(
            name="with-link",
            layers={**base, "team": {"backend": "siglip_kmeans"}, "link": {"backend": "greedy"}},
        )
        assert cfg_no_link.layer_hash("positions") != cfg_with_link.layer_hash("positions")

    def test_identity_still_requires_link(self) -> None:
        with pytest.raises(ValidationError, match="needs \\['link'\\]"):
            RunConfig(
                name="bad",
                layers={
                    "shots": {"backend": "gold"},
                    "detect": {"backend": "rfdetr_torch"},
                    "team": {"backend": "siglip_kmeans"},
                    "track": {"backend": "bytetrack"},
                    "identity": {"backend": "shot"},
                },
            )


class TestBBoxBottom:
    """``positions/bbox_bottom`` consumes ``link`` when present, skips singular frames, and emits ``foot_px``."""

    def test_prefers_link_over_track_and_interpolates_singular_frame(self) -> None:
        seq = Sequence("s", "c", "v.mp4", 0, [0, 5, 10], 50.0)
        up = {
            "track": {"frames": {"0": [[99, 0, 0, 10, 10, 0.9, "player", 0]]}},
            "link": {
                "frames": {
                    "0": [[7, 0, 0, 20, 40, 0.9, "player", 0]],
                    "5": [[7, 10, 0, 30, 50, 0.9, "player", 0]],
                    "10": [[7, 20, 0, 40, 60, 0.9, "player", 0]],
                }
            },
            "court": {
                "units": "cm",
                "frames": {
                    "0": [[10, 0, 0], [0, 10, 0], [0, 0, 1]],
                    "5": [[10, 0, 0], [0, 10, 0], [0, 0, 0]],  # singular w=0
                    "10": [[10, 0, 0], [0, 10, 0], [0, 0, 1]],
                },
            },
        }
        assert active_tracks(up) is up["link"]["frames"]
        out = BBoxBottom({"clean": False}).run(seq, up, None)
        assert list(out["tracks"]) == ["7"]
        # Frame 0 -> (1.0, 4.0), Frame 10 -> (3.0, 6.0), singular frame 5 interpolated to (2.0, 5.0)
        assert out["tracks"]["7"] == [[0, 1.0, 4.0], [5, 2.0, 5.0], [10, 3.0, 6.0]]
        assert out["foot_px"]["7"] == [[0, 10.0, 40.0], [5, 20.0, 50.0], [10, 30.0, 60.0]]


class TestAnklesPoseHelpers:
    """Pure-function tests for pose matching, ankle midpoint selection, and airborne hold."""

    def test_match_poses_one_to_one_with_threshold(self) -> None:
        tracks = np.array([[0, 0, 10, 20], [100, 0, 110, 20]])
        poses = np.array([[101, 0, 111, 20], [1, 0, 11, 20], [500, 500, 510, 520]])
        assert match_poses(tracks, poses, min_iou=0.3) == {0: 1, 1: 0}
        assert match_poses(tracks, np.array([[500, 500, 510, 520]]), min_iou=0.3) == {}

    def test_ankle_ground_point_modes(self) -> None:
        box = [0.0, 0.0, 40.0, 100.0]
        xy = np.zeros((17, 2))
        xy[LEFT_ANKLE] = [12.0, 88.0]
        xy[RIGHT_ANKLE] = [28.0, 92.0]
        conf = np.zeros(17)
        conf[LEFT_ANKLE] = 0.9
        conf[RIGHT_ANKLE] = 0.8
        assert ankle_ground_point(box, xy, conf, ankle_conf=0.4) == (20.0, 90.0, 92.0, "both_ankles")

        conf[RIGHT_ANKLE] = 0.1
        assert ankle_ground_point(box, xy, conf, ankle_conf=0.4) == (12.0, 88.0, 88.0, "one_ankle")

        conf[LEFT_ANKLE] = 0.1
        assert ankle_ground_point(box, xy, conf, ankle_conf=0.4) == (20.0, 100.0, None, "bbox_bottom")

    def test_detect_airborne_catches_jump_and_ignores_running(self) -> None:
        frames = np.arange(0, 50, 5)  # 0..45 at 50 fps (0.1 s step)
        box_h = np.full(len(frames), 100.0)
        # Running smoothly away from camera: linear y decrease -> zero chord lift
        running_y = np.linspace(300.0, 240.0, len(frames))
        assert not np.any(detect_airborne(frames, running_y, box_h, fps=50.0))

        # Jump peaking 30 px (0.3 * box_h) above floor for 0.3 s between frames 10 and 30
        jump_y = np.array([200.0, 200.0, 200.0, 178.0, 168.0, 180.0, 200.0, 200.0, 200.0, 200.0])
        mask = detect_airborne(frames, jump_y, box_h, fps=50.0, airborne_frac=0.15, max_airborne_s=1.0)
        assert mask.tolist() == [False, False, False, True, True, True, False, False, False, False]

    def test_hold_airborne_positions(self) -> None:
        xy = np.array([[5.0, 6.0], [5.2, 8.1], [5.4, 8.9], [5.6, 6.1]])
        held = hold_airborne_positions(xy, np.array([False, True, True, False]))
        np.testing.assert_allclose(held, [[5.0, 6.0], [5.0, 6.0], [5.0, 6.0], [5.6, 6.1]])

    def test_ankles_pose_run_with_mocked_predictor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seq = Sequence("s", "c", "v.mp4", 0, [0, 5, 10, 15, 20], 50.0)
        # Track jumps upward on frame 10 (ankles lift 30px inside 100px box)
        ankle_ys = [200.0, 200.0, 170.0, 200.0, 200.0]
        call_idx = {"i": 0}

        def fake_predict(_self: AnklesPose, _img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            i = call_idx["i"]
            call_idx["i"] += 1
            boxes = np.array([[10.0, 100.0, 50.0, 200.0]])
            kp_xy = np.zeros((1, 17, 2))
            kp_xy[0, LEFT_ANKLE] = [25.0, ankle_ys[i]]
            kp_xy[0, RIGHT_ANKLE] = [35.0, ankle_ys[i]]
            kp_conf = np.zeros((1, 17))
            kp_conf[0, LEFT_ANKLE] = 0.95
            kp_conf[0, RIGHT_ANKLE] = 0.95
            return boxes, kp_xy, kp_conf

        monkeypatch.setattr(AnklesPose, "_predict_poses", fake_predict)
        up = {
            "link": {"frames": {str(f): [[1, 10.0, 100.0, 50.0, 200.0, 0.9, "player", 0]] for f in seq.frames}},
            "court": {"units": "cm", "frames": {str(f): [[10, 0, 0], [0, 10, 0], [0, 0, 1]] for f in seq.frames}},
        }
        backend = AnklesPose({"clean": False, "hold_airborne": True})
        out = backend.run(seq, up, lambda _f: np.zeros((240, 320, 3), np.uint8))
        assert out["stats"]["both_ankles"] == 5
        assert out["airborne"] == {"1": [10]}
        # Frame 10 held at takeoff floor y = 20.0 m instead of jumping to 17.0 m
        assert out["tracks"]["1"][2] == [10, 3.0, 20.0]


def test_fiba_court_package_data() -> None:
    """``fiba_court.json`` lives inside ``bga.court`` package data and loads cleanly."""
    assert FIBA_COURT_PATH.is_file()
    geo = load_fiba_court()
    assert len(geo["vertices"]) == 33
    assert (geo["length"], geo["width"]) == (2800.0, 1500.0)


def test_clean_paths_enabled_end_to_end() -> None:
    """Exercise ``sports.common.path.clean_paths`` through ``BBoxBottom`` with ``clean=True``."""
    frames = list(range(0, 60, 5))
    seq = Sequence("s", "c", "v.mp4", 0, frames, 50.0)
    up = {
        "track": {"frames": {str(f): [[1, 100.0 + f, 100.0, 140.0 + f, 200.0, 0.9, "player", 0]] for f in frames}},
        "court": {"units": "cm", "frames": {str(f): [[10, 0, 0], [0, 10, 0], [0, 0, 1]] for f in frames}},
    }
    out = BBoxBottom({"clean": True, "smooth_window": 5}).run(seq, up, None)
    assert out["paths_left_uncleaned"] == 0
    assert len(out["tracks"]["1"]) == len(frames)
