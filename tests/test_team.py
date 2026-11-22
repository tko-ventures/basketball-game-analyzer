"""Team backend: clips without kits are refused, and SigLIP is loaded once for all sequences."""

import numpy as np
import pytest

from bga.stage import Sequence
from bga.team.siglip_kmeans import SiglipKMeans, light_side

KITS = {"home": {"kit": "light"}, "away": {"kit": "dark"}}
#: Upstream outputs: one frame with four player boxes, enough for the backend to cluster.
UPSTREAM = {"detect": {"frames": {"0": [[10 + 40 * i, 10, 40 + 40 * i, 90, 0.9, "player"] for i in range(4)]}}}


class FakeTeamClassifier:
    """Stands in for ``sports.TeamClassifier`` and its SigLIP download: counts constructions and remembers its fits."""

    built = 0

    def __init__(self, device="cpu"):
        type(self).built += 1
        self.reducers = []

    def fit(self, crops):
        self.reducers.append(self.reducer)
        self.reducer.embedding_ = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0], [5.1, 5.0, 5.0]])
        self.cluster_model.labels_ = np.array([0, 0, 1, 1])

    def predict(self, crops):
        return np.array([0, 0, 1, 1])


@pytest.fixture
def backend(monkeypatch):
    """The team backend, loaded with the fake classifier in place of SigLIP."""
    import sports

    FakeTeamClassifier.built = 0
    monkeypatch.setattr(sports, "TeamClassifier", FakeTeamClassifier)
    team = SiglipKMeans({"device": "cpu", "seed": 0})
    team.load()
    return team


def sequence(meta: dict) -> Sequence:
    """A one-frame sequence with the given clip meta."""
    return Sequence(name="clip_s1", clip="clip", video="clip.mp4", shot=1, frames=[0], fps=50.0, meta=meta)


def read(_frame: int) -> np.ndarray:
    """A frame bright on the left half and dark on the right."""
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 220
    return img


class TestLightSide:
    @pytest.mark.parametrize(
        ("meta", "side"),
        [
            pytest.param({"home": {"kit": "light"}, "away": {"kit": "dark"}}, "home", id="home-light"),
            pytest.param({"home": {"kit": "dark"}, "away": {"kit": "light"}}, "away", id="away-light"),
        ],
    )
    def test_names_the_side_marked_light(self, meta, side):
        assert light_side(meta) == side

    @pytest.mark.parametrize(
        "meta",
        [
            pytest.param({}, id="no-meta"),
            pytest.param({"home": {"kit": ""}, "away": {"kit": ""}}, id="empty-kits"),
            pytest.param({"home": {"kit": "dark"}, "away": {"kit": "dark"}}, id="both-dark"),
            pytest.param({"home": {"kit": "light"}, "away": {"kit": "light"}}, id="both-light"),
            pytest.param({"home": {"team": "Dziki Warszawa"}, "away": {"kit": "dark"}}, id="one-missing"),
        ],
    )
    def test_refuses_meta_without_one_light_side(self, meta):
        with pytest.raises(ValueError, match="one side's kit light"):
            light_side(meta)


class TestSiglipKMeans:
    def test_clip_without_kits_is_refused_before_any_frame_is_read(self):
        with pytest.raises(ValueError, match="one side's kit light"):
            SiglipKMeans({}).run(sequence({}), UPSTREAM, None)

    def test_classifier_is_built_once_for_all_sequences(self, backend):
        backend.run(sequence(KITS), UPSTREAM, read)
        backend.run(sequence(KITS), UPSTREAM, read)
        assert FakeTeamClassifier.built == 1

    def test_each_sequence_is_fitted_with_its_own_reducer(self, backend):
        backend.run(sequence(KITS), UPSTREAM, read)
        backend.run(sequence(KITS), UPSTREAM, read)
        first, second = backend.classifier.reducers
        assert first is not second

    def test_brighter_cluster_takes_the_light_side(self, backend):
        out = backend.run(sequence({"home": {"kit": "dark"}, "away": {"kit": "light"}}), UPSTREAM, read)
        assert out["frames"]["0"] == ["away", "away", "home", "home"]
