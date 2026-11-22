"""Track/sam2 without a GPU: prompt boxes from detections, and mask boxes turned into track rows."""

import pytest

from bga.track.sam2_video import mask_rows, prompt_boxes


class TestPromptBoxes:
    @pytest.mark.parametrize(
        ("det", "kept"),
        [
            pytest.param([0, 0, 10, 30, 0.9, "player"], 1, id="upright-player"),
            pytest.param([0, 0, 10, 30, 0.2, "player"], 0, id="low-score"),
            pytest.param([0, 0, 30, 10, 0.9, "player"], 0, id="wider-than-tall"),
            pytest.param([0, 0, 10, 30, 0.9, "number"], 0, id="not-a-person"),
        ],
    )
    def test_only_confident_upright_people_prompt(self, det, kept):
        assert len(prompt_boxes([det], 0.4)) == kept


class TestMaskRows:
    def test_matched_mask_takes_the_detection_class_and_index(self):
        rows = mask_rows({4: [0, 0, 10, 20]}, [[50, 50, 60, 70, 0.9, "player"], [0, 0, 10, 20, 0.7, "referee"]], 0.5)
        assert rows[0][5:] == [0.7, "referee", 1]

    def test_unmatched_mask_is_a_player_without_detection(self):
        assert mask_rows({4: [0, 0, 10, 20]}, [], 0.5)[0][5:] == [0.5, "player", -1]

    def test_object_without_mask_has_no_row(self):
        assert mask_rows({4: None}, [], 0.5) == []
