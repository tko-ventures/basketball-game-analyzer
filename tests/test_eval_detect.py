import math

import pytest

from bga.eval.detect import PEOPLE, class_ap

pytest.importorskip("pycocotools")

PLAYER = ([100, 100, 160, 260], "player")
BALL = ([300, 300, 320, 320], "ball")
NUMBER = ([120, 140, 135, 160], "number")


class TestClassAp:
    def test_perfect_detections_score_one(self):
        gold = [[PLAYER, BALL, NUMBER]]
        dets = [[(b, 0.9, c) for b, c in gold[0]]]
        r = class_ap(gold, dets)
        assert (r["player"].ap, r["ball"].ap, r["number"].ap, r[PEOPLE].ap) == (1.0, 1.0, 1.0, 1.0)

    def test_missed_class_scores_zero(self):
        r = class_ap([[PLAYER, BALL]], [[(PLAYER[0], 0.9, "player")]])
        assert (r["ball"].ap50, r["ball"].n) == (0.0, 1)

    def test_class_without_gold_is_undefined(self):
        r = class_ap([[PLAYER]], [[(PLAYER[0], 0.9, "player")]])
        assert (math.isnan(r["rim"].ap), r["rim"].n) == (True, 0)

    def test_people_ignore_the_action_class(self):
        r = class_ap([[([100, 100, 160, 260], "player-in-possession")]], [[(PLAYER[0], 0.9, "player")]])
        assert (r[PEOPLE].ap50, r["player-in-possession"].ap50) == (1.0, 0.0)

    @pytest.mark.parametrize(
        ("shift", "ap50", "ap"),
        [
            pytest.param(0, 1.0, 1.0, id="exact"),
            pytest.param(40, 1.0, 0.7, id="iou-0.8-passes-7-of-10-thresholds"),
        ],
    )
    def test_geometry_drives_ap50_95(self, shift, ap50, ap):
        x1, y1, x2, y2 = PLAYER[0]
        r = class_ap([[PLAYER]], [[([x1, y1, x2, y2 + shift], 0.9, "player")]])
        assert (r["player"].ap50, round(r["player"].ap, 1)) == (ap50, ap)

    def test_nothing_detected_scores_zero_where_gold_exists(self):
        r = class_ap([[PLAYER]], [[]])
        assert (r["player"].ap, math.isnan(r["ball"].ap)) == (0.0, True)
