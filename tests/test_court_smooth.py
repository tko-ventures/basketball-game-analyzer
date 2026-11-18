"""Median smoothing of court mappings within a shot (bga.court.smooth)."""

import cv2
import numpy as np
import pytest

from bga.court.smooth import smooth_homographies

WH = (1920, 1080)


def _H(shift_cm: float = 0.0) -> list:
    """Image->court for a broadcast-like view, optionally shifted along the court by shift_cm."""
    court = np.float64([[0, 0], [2800, 0], [2800, 1500], [0, 1500]])
    image = np.float64([[700, 300], [1220, 300], [1900, 900], [20, 900]])
    H, _ = cv2.findHomography(image, court)
    T = np.array([[1, 0, shift_cm], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return (T @ H).tolist()


def _court_of(H, pt=(960, 700)) -> np.ndarray:
    p = np.asarray(H) @ [pt[0], pt[1], 1.0]
    return p[:2] / p[2]


FRAMES = list(range(0, 50, 5))


@pytest.fixture
def outlier_and_gap() -> dict:
    """Smoothed mappings of a still camera with a 2 m jump at frame 20 and no mapping at frame 30."""
    Hs = {str(f): _H() for f in FRAMES}
    Hs["20"] = _H(200.0)  # a 2 m jump on one frame
    Hs["30"] = None  # a frame the model could not map
    return smooth_homographies(FRAMES, Hs, WH, window=2)


class TestOutlierAndGapFollowNeighbours:
    def test_outlier_replaced(self, outlier_and_gap):
        assert np.linalg.norm(_court_of(outlier_and_gap["20"]) - _court_of(_H())) < 5.0  # cm

    def test_gap_filled(self, outlier_and_gap):
        assert outlier_and_gap["30"] is not None

    def test_gap_follows_neighbours(self, outlier_and_gap):
        assert np.linalg.norm(_court_of(outlier_and_gap["30"]) - _court_of(_H())) < 5.0  # cm


@pytest.mark.parametrize("frame", [pytest.param(f, id=f"frame-{f}") for f in FRAMES[2:-2]])
def test_steady_pan_is_not_lagged(frame):
    Hs = {str(f): _H(10.0 * i) for i, f in enumerate(FRAMES)}  # 10 cm per frame, linear
    out = smooth_homographies(FRAMES, Hs, WH, window=2)
    assert np.linalg.norm(_court_of(out[str(frame)]) - _court_of(Hs[str(frame)])) < 2.0


def test_window_zero_is_identity():
    Hs = {"0": _H(), "5": None}
    assert smooth_homographies([0, 5], Hs, WH, window=0) is Hs
