import numpy as np
import pytest

from bga.numbers.base import NUMBER_CLASSES, UNREADABLE
from bga.numbers.parseq import EOS, HPARAMS, PRETRAINED, ParseqReader, build_system, number_probs, to_input


def tokens_for(text: str, positions: int = 26, confidence: float = 0.9) -> np.ndarray:
    """Token probabilities of one crop whose most probable string is ``text`` (digits only), then the end token."""
    tokens = np.full((1, positions, 95), (1 - confidence) / 94)
    ids = [int(ch) + 1 for ch in text] + [EOS]
    tokens[0, np.arange(len(ids)), ids] = confidence
    return tokens


class TestNumberProbs:
    def test_rows_are_distributions_over_the_101_classes(self):
        tokens = np.random.default_rng(0).dirichlet(np.ones(95), size=(4, 26))

        probs = number_probs(tokens)

        assert probs.shape == (4, len(NUMBER_CLASSES))
        assert np.allclose(probs.sum(1), 1.0)

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("7", id="one-digit"),
            pytest.param("0", id="zero"),
            pytest.param("10", id="ten"),
            pytest.param("23", id="two-digits"),
            pytest.param("99", id="last-class"),
        ],
    )
    def test_most_probable_class_is_the_decoded_number(self, text):
        probs = number_probs(tokens_for(text))

        assert NUMBER_CLASSES[int(probs.argmax())] == text

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("", id="empty-string"),
            pytest.param("123", id="three-digits"),
            pytest.param("07", id="leading-zero"),
        ],
    )
    def test_strings_that_are_not_a_number_class_are_unreadable(self, text):
        probs = number_probs(tokens_for(text))

        assert NUMBER_CLASSES[int(probs.argmax())] == UNREADABLE

    def test_letters_are_unreadable(self):
        tokens = np.zeros((1, 26, 95))
        tokens[0, 0, 11], tokens[0, 1, EOS] = 1.0, 1.0  # the letter a, then the end

        probs = number_probs(tokens)

        assert probs[0, NUMBER_CLASSES.index(UNREADABLE)] == pytest.approx(1.0)

    def test_two_digit_probability_is_digits_times_end(self):
        tokens = tokens_for("23", confidence=0.8)

        probs = number_probs(tokens)

        assert probs[0, NUMBER_CLASSES.index("23")] == pytest.approx(0.8**3)


def test_input_is_stretched_to_32_by_128_rgb():
    crop = np.zeros((40, 30, 3), np.uint8)
    crop[..., 0] = 255  # blue in BGR

    out = to_input(crop)

    assert out.shape == (32, 128, 3)
    assert tuple(out[0, 0]) == (0, 0, 255)


def test_other_pretrained_ids_are_refused():
    reader = ParseqReader({}, "pretrained:github/baudm/parseq/v1.0.0/other")

    with pytest.raises(ValueError, match="numbers/parseq reads"):
        reader.load()


def test_released_model_has_95_outputs_per_position():
    pytest.importorskip("strhub", reason="needs the parseq extra")
    import torch

    system = build_system(HPARAMS).eval()

    with torch.no_grad():
        out = system(torch.zeros(1, 3, 32, 128))

    assert out.shape == (1, 26, 95)
    assert PRETRAINED.endswith("parseq-bb5792a6")


class TestPlmLoss:
    @pytest.fixture(scope="class")
    def system(self):
        pytest.importorskip("strhub", reason="needs the parseq extra")
        import torch

        torch.manual_seed(0)
        return build_system(HPARAMS).train()

    @pytest.mark.parametrize(
        "labels",
        [
            pytest.param(["23", "7", ""], id="numbers-and-unreadable"),
            pytest.param(["7", "0"], id="one-digit-only"),
            pytest.param(["", ""], id="all-unreadable"),
        ],
    )
    def test_loss_is_finite_and_positive(self, system, labels):
        import torch

        from bga.train.parseq_trainer import plm_loss

        loss = plm_loss(system, torch.zeros(len(labels), 3, 32, 128), labels)

        assert torch.isfinite(loss)
        assert float(loss) > 0

    def test_gradient_reaches_encoder_and_head(self, system):
        import torch

        from bga.train.parseq_trainer import plm_loss

        system.zero_grad()

        plm_loss(system, torch.randn(2, 3, 32, 128), ["23", ""]).backward()  # zeros would leave the patch weights

        assert float(system.model.head.weight.grad.abs().sum()) > 0
        assert float(system.model.encoder.patch_embed.proj.weight.grad.abs().sum()) > 0


def test_unreadable_is_trained_as_the_empty_string():
    from bga.train.parseq_trainer import text_of

    assert (text_of(NUMBER_CLASSES.index(UNREADABLE)), text_of(NUMBER_CLASSES.index("0"))) == ("", "0")
