import pytest

from bga.config import RunConfig


def _cfg(**shots):
    return RunConfig(
        name="t",
        layers={
            "shots": {"backend": "gold", "params": {"stride": 5, **shots}},
            "detect": {"backend": "d", "params": {"confidence": 0.4}},
            "track": {"backend": "t"},
        },
    )


@pytest.fixture
def track_changed():
    """The base config and a copy that differs only in the track layer's params."""
    a = _cfg()
    b = RunConfig(**{
        **a.model_dump(),
        "layers": {**a.model_dump()["layers"], "track": {"backend": "t", "params": {"x": 1}}},
    })
    return a, b


class TestLayerHash:
    @pytest.mark.parametrize(
        "layer",
        [pytest.param("shots", id="shots"), pytest.param("detect", id="detect"), pytest.param("track", id="track")],
    )
    def test_selection_params_do_not_change_it(self, layer):
        a, b = _cfg(held_out="exclude", version="v0"), _cfg(held_out="only", version="v1")
        assert a.layer_hash(layer) == b.layer_hash(layer)

    def test_stride_changes_downstream(self):
        assert _cfg().layer_hash("track") != _cfg(stride=1).layer_hash("track")

    def test_upstream_unchanged_by_downstream_change(self, track_changed):
        a, b = track_changed
        assert a.layer_hash("detect") == b.layer_hash("detect")

    def test_changed_layer_rehashed(self, track_changed):
        a, b = track_changed
        assert a.layer_hash("track") != b.layer_hash("track")
