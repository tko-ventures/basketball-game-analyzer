import math

import numpy as np
import pytest
import torch

from bga.numbers.base import NUMBER_CLASSES, UNREADABLE
from bga.numbers.crop_cls import Head, class_probs, create_model, num_outputs
from bga.train.cls_trainer import head_loss, score


def factorised_logits(unreadable: float, tens: int, ones: int) -> torch.Tensor:
    """One row of factorised outputs that favours a tens group index and a ones digit."""
    logits = torch.zeros(1, 21)
    logits[0, 0] = unreadable
    logits[0, 1 + tens] = 12.0
    logits[0, 11 + ones] = 12.0
    return logits


class TestClassProbs:
    @pytest.mark.parametrize(
        "head", [pytest.param(Head.FLAT, id="flat"), pytest.param(Head.FACTORISED, id="factorised")]
    )
    def test_rows_are_distributions_over_the_101_classes(self, head):
        probs = class_probs(torch.randn(5, num_outputs(head)), head)

        assert probs.shape == (5, len(NUMBER_CLASSES))
        assert torch.allclose(probs.sum(1), torch.ones(5), atol=1e-5)

    @pytest.mark.parametrize(
        ("tens", "ones", "expected"),
        [
            pytest.param(0, 7, "7", id="single-digit-has-no-tens"),
            pytest.param(1, 0, "10", id="ten"),
            pytest.param(2, 3, "23", id="two-digits"),
            pytest.param(9, 9, "99", id="last-class"),
        ],
    )
    def test_factorised_number_is_tens_then_ones(self, tens, ones, expected):
        probs = class_probs(factorised_logits(-12.0, tens, ones), Head.FACTORISED)

        assert NUMBER_CLASSES[int(probs.argmax())] == expected

    def test_factorised_unreadable_wins_when_its_output_is_high(self):
        probs = class_probs(factorised_logits(12.0, 2, 3), Head.FACTORISED)

        assert NUMBER_CLASSES[int(probs.argmax())] == UNREADABLE

    def test_factorised_number_probability_is_the_product_of_its_parts(self):
        logits = torch.randn(1, 21)

        probs = class_probs(logits, Head.FACTORISED)

        readable = 1 - torch.sigmoid(logits[0, 0])
        tens, ones = torch.softmax(logits[0, 1:11], 0), torch.softmax(logits[0, 11:21], 0)
        assert float(probs[0, NUMBER_CLASSES.index("47")]) == pytest.approx(float(readable * tens[4] * ones[7]))


class TestHeadLoss:
    def test_flat_is_smoothed_cross_entropy(self):
        logits, y = torch.randn(4, 101), torch.tensor([0, 1, 24, 100])

        loss = head_loss(logits, y, Head.FLAT)

        assert float(loss) == pytest.approx(float(torch.nn.functional.cross_entropy(logits, y, label_smoothing=0.1)))

    def test_factorised_right_answer_costs_less_than_a_wrong_digit(self):
        y = torch.tensor([NUMBER_CLASSES.index("23")])

        right = head_loss(factorised_logits(-12.0, 2, 3), y, Head.FACTORISED)
        wrong = head_loss(factorised_logits(-12.0, 2, 8), y, Head.FACTORISED)

        assert float(right) < float(wrong)

    def test_factorised_unreadable_crops_do_not_train_the_digits(self):
        y = torch.tensor([NUMBER_CLASSES.index(UNREADABLE)])
        logits = factorised_logits(3.0, 2, 3).requires_grad_()

        head_loss(logits, y, Head.FACTORISED).backward()

        assert float(logits.grad[0, 1:].abs().sum()) == 0.0
        assert float(logits.grad[0, 0]) != 0.0

    def test_factorised_gradient_reaches_every_group_for_a_number(self):
        y = torch.tensor([NUMBER_CLASSES.index("23")])
        logits = torch.zeros(1, 21, requires_grad=True)

        head_loss(logits, y, Head.FACTORISED).backward()

        groups = (logits.grad[0, :1], logits.grad[0, 1:11], logits.grad[0, 11:21])
        assert all(float(g.abs().sum()) > 0 for g in groups)


class TestScore:
    def test_numbers_split_by_whether_training_has_them(self):
        classes = [NUMBER_CLASSES.index(c) for c in (UNREADABLE, "23", "12", "7")]
        y = np.array(classes)
        pred = np.array([classes[0], classes[1], classes[1], classes[3]])

        got = score(pred, y, seen={classes[0], classes[1], classes[2]})

        keys = ("val/top1", "val/refused", "val/top1_seen", "val/top1_unseen")
        assert tuple(got[k] for k in keys) == (0.75, 1.0, 0.5, 1.0)

    def test_empty_split_scores_nan(self):
        got = score(np.zeros(0, np.int64), np.zeros(0, np.int64), seen=set())

        assert all(math.isnan(v) for v in got.values())


def test_factorised_model_has_21_outputs():
    model = create_model("resnet18.tv_in1k", num_outputs(Head.FACTORISED), 64, pretrained=False)

    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 21)
