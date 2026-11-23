import pytest

from bga.eval.numbers import GoldCrop, Tally, ece
from bga.numbers.base import UNREADABLE, CropRead, roster_choice


def _crop(label):
    return GoldCrop("n001", "s", 0, (0.0, 0.0, 10.0, 10.0), label, "home")


class TestRosterChoice:
    @pytest.mark.parametrize(
        ("read", "roster", "expected"),
        [
            pytest.param(CropRead("8"), {"8", "3"}, ("8", None), id="decoded-on-roster"),
            pytest.param(CropRead("8"), {"3"}, (None, None), id="decoded-off-roster"),
            pytest.param(CropRead(None), {"3"}, (None, None), id="decoded-none"),
            pytest.param(CropRead("8"), set(), ("8", None), id="no-roster"),
            pytest.param(CropRead("8", {"8": 0.6, "3": 0.2, UNREADABLE: 0.2}), {"3"}, (None, 0.5), id="tie-unreadable"),
            pytest.param(
                CropRead("8", {"8": 0.7, "3": 0.2, UNREADABLE: 0.1}), {"3"}, ("3", 0.666667), id="renormalised"
            ),
        ],
    )
    def test_answer(self, read, roster, expected):
        assert roster_choice(read, roster) == expected


class TestTally:
    def test_scores_legible_crops_free_and_under_the_roster(self):
        t = Tally()
        t.add(_crop("23"), CropRead("28"), ("23", None))
        t.add(_crop("23"), CropRead("28"), ("28", None))
        s = t.summary()
        assert (s["top1_free"], s["top1_roster"], s["answered"]) == (0.0, 0.5, 1.0)

    def test_a_declined_illegible_crop_is_right(self):
        t = Tally()
        t.add(_crop("illegible"), CropRead("8"), (None, 0.9))
        assert (t.summary()["refused"], t.right) == (1.0, [True])

    def test_lists_wrong_answers(self):
        t = Tally()
        t.add(_crop("not_number"), CropRead("8"), ("8", None))
        assert t.summary()["wrong"] == [("n001", "not_number", "8")]


class TestEce:
    @pytest.mark.parametrize(
        ("conf", "right", "expected"),
        [
            pytest.param([0.95, 0.95], [True, True], 0.05, id="slightly-underconfident"),
            pytest.param([0.15, 0.95], [False, False], 0.55, id="two-bins"),
        ],
    )
    def test_value(self, conf, right, expected):
        assert ece(conf, right) == pytest.approx(expected)
