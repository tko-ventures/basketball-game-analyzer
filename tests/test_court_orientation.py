"""Court projection must not depend on the overall sign of the homography (2026-11-17 bug)."""

import cv2
import numpy as np
import pytest

from bga.court.propagate import COURT_CM, court_mask, court_to_image, landmarks

VERTS = np.array([[0, 0], [COURT_CM[0], 0], [COURT_CM[0], COURT_CM[1]], [0, COURT_CM[1]], [1400, 750]], float)


def _broadcast_H() -> np.ndarray:
    """Image->court for a camera whose horizon crosses the frame: the top-left pixel lies above it."""
    court_corners = np.float64([[0, 0], [2800, 0], [2800, 1500], [0, 1500]])
    image_corners = np.float64([[700, 300], [1220, 300], [1900, 900], [20, 900]])
    H, _ = cv2.findHomography(image_corners, court_corners)
    return H


def test_sign_does_not_change_projection():
    H = _broadcast_H()
    wh = (1920, 1080)
    a, b = landmarks(H, VERTS, wh), landmarks(-H, VERTS, wh)
    assert a == b
    assert sum(v for *_, v in a) == 2 * len(VERTS)  # all five points visible
    assert (court_mask(H, (1080, 1920)) == court_mask(-H, (1080, 1920))).all()
    assert court_mask(H, (1080, 1920)).any()


#: The same mapping under the overall rescalings a homography fit may return.
SCALINGS = [
    pytest.param(lambda h: h, id="as-fitted"),
    pytest.param(lambda h: -h, id="negated"),
    pytest.param(lambda h: h / h[2, 2], id="normalised"),
]


@pytest.fixture
def low_camera_H() -> np.ndarray:
    """Image->court from a low camera near the sideline: the far end's near corner is behind it."""
    img = np.float64([[1068, 378], [1010, 403], [455, 549], [25, 667], [228, 806], [1186, 556]])
    court = np.float64([[0, 0], [0, 90], [0, 995], [0, 1410], [330, 1410], [580, 505]])
    H, _ = cv2.findHomography(img, court)
    return H


@pytest.mark.parametrize("sign", [pytest.param(1, id="as-fitted"), pytest.param(-1, id="negated")])
def test_court_centre_in_front(sign):
    assert (court_to_image(sign * _broadcast_H()) @ [1400, 750, 1])[2] > 0


@pytest.mark.parametrize("scale", SCALINGS)
class TestCourtPartlyBehindALowCamera:
    """From a low camera near the sideline the far end's near corner is behind it; the rest still projects."""

    def test_far_corner_behind_the_camera(self, low_camera_H, scale):
        assert (court_to_image(scale(low_camera_H)) @ [2800, 1500, 1])[2] < 0

    @pytest.mark.parametrize(
        "point", [pytest.param([0, 0, 1], id="corner"), pytest.param([580, 505, 1], id="interior")]
    )
    def test_near_points_in_front(self, low_camera_H, scale, point):
        assert (court_to_image(scale(low_camera_H)) @ point)[2] > 0

    def test_near_corner_in_view(self, low_camera_H, scale):
        kp = landmarks(scale(low_camera_H), VERTS, (1920, 1080))
        assert kp[0][2] == 2  # corner (0, 0) in view
