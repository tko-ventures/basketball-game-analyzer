"""Layer 3b: the appearance output format, its array storage, the encoder's weights check and the gold-box scores."""

from pathlib import Path

import numpy as np
import pytest

from bga.appearance.none import NoAppearance
from bga.appearance.reid_encoder import check_weights
from bga.config import RunConfig
from bga.eval.appearance import GoldBoxes, Ranks, numbered_means, rank_across, reassociate
from bga.pipeline import load_output, output_path, write_output
from bga.stage import Sequence


def gold(step: list[int], tid: list[int], cls: list[int], number: dict | None = None) -> GoldBoxes:
    """Gold boxes with zero geometry: the scores read only steps, ids, classes and embeddings."""
    n = len(step)
    return GoldBoxes(
        "s", "game", np.array(step), np.array(tid), np.array(cls), np.zeros((n, 4)), np.zeros(n), number or {}
    )


@pytest.fixture
def stored(tmp_path: Path) -> tuple[RunConfig, Sequence, Path]:
    cfg = RunConfig(name="t", layers={"shots": {"backend": "gold"}})
    seq = Sequence(name="clip_s2", clip="clip", video="clip.mp4", shot=2, frames=[0], fps=50.0)
    return cfg, seq, tmp_path


class TestWriteOutput:
    def test_array_comes_back_unchanged(self, stored):
        cfg, seq, runs = stored
        emb = np.arange(6, dtype=np.float16).reshape(2, 3)
        write_output(output_path(cfg, runs, seq, "shots"), {"boxes": [[0, 1], [0, 2]], "arrays": {"embeddings": emb}})
        back = load_output(cfg, runs, seq, "shots", arrays=True)
        np.testing.assert_array_equal(back["arrays"]["embeddings"], emb)

    def test_json_keeps_only_the_record(self, stored):
        cfg, seq, runs = stored
        write_output(output_path(cfg, runs, seq, "shots"), {"arrays": {"embeddings": np.zeros((4, 2), np.float16)}})
        assert load_output(cfg, runs, seq, "shots")["arrays"] == {
            "embeddings": {"file": "clip_s2.embeddings.npy", "shape": [4, 2], "dtype": "float16"}
        }

    def test_output_without_arrays_is_written_as_given(self, stored):
        cfg, seq, runs = stored
        write_output(output_path(cfg, runs, seq, "shots"), {"frames": {"0": []}})
        assert load_output(cfg, runs, seq, "shots", arrays=True) == {"frames": {"0": []}}


class TestCheckWeights:
    @pytest.mark.parametrize(
        ("weights", "error"),
        [
            pytest.param(None, ValueError, id="missing"),
            pytest.param("pretrained:roboflow/reid/osnet_x1_0_msmt17", ValueError, id="other-encoder"),
            pytest.param("appearance/reid/plk@00000000", NotImplementedError, id="trained"),
        ],
    )
    def test_refuses(self, weights, error):
        with pytest.raises(error):
            check_weights("osnet_x1_0_sportsmot", weights)

    def test_accepts_the_declared_encoder(self):
        assert check_weights("osnet_x1_0_sportsmot", "pretrained:roboflow/reid/osnet_x1_0_sportsmot") is None


class TestLayerConfig:
    def test_appearance_needs_detect(self):
        with pytest.raises(ValueError, match="needs"):
            RunConfig(name="t", layers={"shots": {"backend": "gold"}, "appearance": {"backend": "none"}})

    def test_none_backend_embeds_nothing(self):
        assert NoAppearance({}).run(None, {}, None)["boxes"] == []


class TestRanks:
    def test_empty_summary_is_zero(self):
        assert Ranks().summary().n == 0

    def test_tie_counts_in_favour(self):
        r = Ranks()
        r.add(np.array([0.2, 0.2]), 1)
        assert r.summary().rank1 == 1.0

    def test_second_place_halves_the_reciprocal_rank(self):
        r = Ranks()
        r.add(np.array([0.1, 0.5, 0.9]), 1)
        assert (r.summary().rank1, r.summary().mrr) == (0.0, 0.5)


#: Two home players at steps 0 and 1 and one away player: the away player has no teammate to rank against.
TWO_HOME = gold(step=[0, 0, 0, 1, 1, 1], tid=[1, 2, 3, 1, 2, 3], cls=[1, 1, 2, 1, 1, 2])


class TestReassociate:
    @pytest.mark.parametrize(
        ("emb", "rank1"),
        [
            pytest.param([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], 1.0, id="same-looks"),
            pytest.param([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0]], 0.0, id="swapped-looks"),
        ],
    )
    def test_rank1_follows_the_embeddings(self, emb, rank1):
        r = Ranks()
        reassociate(TWO_HOME, np.array(emb, dtype=float), gap=1, every=1, ranks=r)
        assert (r.summary().n, r.summary().rank1) == (2, rank1)

    def test_player_without_teammates_is_not_queried(self):
        g = gold(step=[0, 1], tid=[3, 3], cls=[2, 2])
        r = Ranks()
        reassociate(g, np.eye(2), gap=1, every=1, ranks=r)
        assert r.summary().n == 0

    def test_queries_only_every_nth_step(self):
        g = gold(step=[1, 1, 2, 2], tid=[1, 2, 1, 2], cls=[1, 1, 1, 1])
        r = Ranks()
        reassociate(g, np.eye(4), gap=1, every=2, ranks=r)
        assert r.summary().n == 0


class TestNumberedMeans:
    def test_player_with_too_few_boxes_is_dropped(self):
        g = gold(step=[0, 1, 2], tid=[7, 7, 7], cls=[1, 1, 1], number={7: "23"})
        assert numbered_means(g, np.ones((3, 2))) == {}

    def test_unnumbered_player_is_dropped(self):
        g = gold(step=list(range(5)), tid=[7] * 5, cls=[1] * 5)
        assert numbered_means(g, np.ones((5, 2))) == {}


class TestRankAcross:
    def test_player_missing_from_the_other_shot_is_not_queried(self):
        r = Ranks()
        gallery = {(1, "5"): np.array([1.0, 0.0]), (1, "7"): np.array([0.0, 1.0])}
        rank_across({(1, "9"): np.array([1.0, 0.0])}, gallery, r)
        assert r.summary().n == 0

    def test_other_team_is_not_a_candidate(self):
        r = Ranks()
        gallery = {(1, "5"): np.array([0.0, 1.0]), (2, "7"): np.array([1.0, 0.0]), (1, "8"): np.array([0.6, 0.8])}
        rank_across({(1, "5"): np.array([0.0, 1.0])}, gallery, r)
        assert (r.summary().candidates, r.summary().rank1) == (2.0, 1.0)
