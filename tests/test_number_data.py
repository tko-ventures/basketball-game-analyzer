import pytest

from bga.numbers.base import NUMBER_CLASSES, UNREADABLE
from bga.train.cls_trainer import sample_weights
from bga.train.number_data import class_of


class TestClassOf:
    @pytest.mark.parametrize(
        ("label", "illegible_as_unreadable", "expected"),
        [
            pytest.param("23", False, "23", id="number"),
            pytest.param("0", False, "0", id="zero"),
            pytest.param("not_number", False, UNREADABLE, id="sponsor"),
            pytest.param("illegible", True, UNREADABLE, id="illegible-trains-unreadable"),
            pytest.param("illegible", False, None, id="illegible-left-out"),
            pytest.param("mixed", True, None, id="mixed-left-out"),
            pytest.param(None, True, None, id="unread-left-out"),
            pytest.param("123", False, None, id="not-a-class"),
        ],
    )
    def test_class(self, label, illegible_as_unreadable, expected):
        assert class_of(label, illegible_as_unreadable) == expected


def test_classes_are_unreadable_then_0_to_99():
    assert (len(NUMBER_CLASSES), NUMBER_CLASSES[0], NUMBER_CLASSES[1], NUMBER_CLASSES[-1]) == (
        101,
        UNREADABLE,
        "0",
        "99",
    )


def test_rare_classes_are_drawn_more_often_per_crop():
    w = sample_weights([0] * 9 + [1])
    assert w[-1] / w[0] == pytest.approx(3.0)
