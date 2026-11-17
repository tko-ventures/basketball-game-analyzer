import numpy as np

from bga.detect.nms import nested_box_nms
from bga.detect.onnx_rfdetr import ROBOFLOW_EXPORT_CLASSES, decode
from bga.types import Detection, class_name_to_role


def test_role_map():
    assert class_name_to_role("player-in-possession") == "player"
    assert class_name_to_role("number") == "number"
    assert class_name_to_role("referee") == "other"
    assert class_name_to_role("ball") == "other"


def test_nested_child_is_suppressed():
    parent = Detection((0, 0, 100, 200), "player", 0.9)
    child = Detection((10, 10, 50, 80), "player", 0.8)
    assert nested_box_nms([parent, child]) == [parent]


def test_number_inside_player_is_kept():
    player = Detection((0, 0, 100, 200), "player", 0.9)
    number = Detection((30, 40, 70, 90), "number", 0.85)
    assert {d.role for d in nested_box_nms([player, number])} == {"player", "number"}


def test_decode_drops_background_and_scales_boxes():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    logits = np.full((2, len(ROBOFLOW_EXPORT_CLASSES)), -10.0, dtype=np.float32)
    logits[0, 4] = 5.0  # "player" (index 3 after dropping background)
    logits[1, 0] = 5.0  # background only
    dets = decode(boxes, logits, ROBOFLOW_EXPORT_CLASSES, (1000, 500), confidence=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "player" and d.class_id == 3
    assert np.allclose(d.xyxy, (400, 150, 600, 350))
