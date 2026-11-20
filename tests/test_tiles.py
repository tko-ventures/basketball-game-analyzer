import pytest

from bga.detect.tiles import crop_box, merge, square_tiles
from bga.train.detect_data import category_map


class TestSquareTiles:
    @pytest.mark.parametrize(
        ("w", "h", "n", "tiles"),
        [
            pytest.param(1920, 1080, 2, [(0, 0, 1080, 1080), (840, 0, 1920, 1080)], id="landscape-two"),
            pytest.param(1080, 1920, 2, [(0, 0, 1080, 1080), (0, 840, 1080, 1920)], id="portrait-two"),
            pytest.param(1920, 1080, 3, [(0, 0, 1080, 1080), (420, 0, 1500, 1080), (840, 0, 1920, 1080)], id="three"),
        ],
    )
    def test_tiles_cover_the_frame(self, w, h, n, tiles):
        assert square_tiles(w, h, n) == tiles


class TestCropBox:
    @pytest.mark.parametrize(
        ("box", "expected"),
        [
            pytest.param((1040, 0, 1120, 100), (0, 0, 40, 100), id="half-inside-kept"),
            pytest.param((1040, 0, 1100, 100), None, id="third-inside-dropped"),
            pytest.param((10, 10, 10, 50), None, id="empty-dropped"),
        ],
    )
    def test_keeps_boxes_mostly_inside(self, box, expected):
        assert crop_box(box, (1080, 0, 2160, 1080)) == expected


class TestMerge:
    def test_keeps_different_classes_on_the_same_spot(self):
        tiles = square_tiles(1920, 1080)
        dets = [[[100, 100, 140, 140, 0.9, "ball"], [100, 100, 140, 140, 0.8, "number"]], []]
        assert len(merge(dets, tiles)) == 2

    def test_single_tile_is_unchanged(self):
        dets = [[[10, 10, 50, 90, 0.7, "player"]]]
        assert merge(dets, square_tiles(1920, 1080, 1)) == [[10, 10, 50, 90, 0.7, "player"]]


class TestCategoryMap:
    def test_matches_names_case_insensitively(self):
        assert category_map([{"id": 3, "name": "Number"}, {"id": 0, "name": "basketball"}]) == {3: 3}
