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


def test_selection_params_do_not_change_hashes_but_stride_does():
    a, b = _cfg(held_out="exclude", version="v0"), _cfg(held_out="only", version="v1")
    assert all(a.layer_hash(x) == b.layer_hash(x) for x in ("shots", "detect", "track"))
    assert _cfg().layer_hash("track") != RunConfig(
        name="t",
        layers={
            "shots": {"backend": "gold", "params": {"stride": 1}},
            "detect": {"backend": "d", "params": {"confidence": 0.4}},
            "track": {"backend": "t"},
        },
    ).layer_hash("track")


def test_downstream_hash_follows_upstream_change_only():
    a = _cfg()
    b = RunConfig(
        **{
            **a.model_dump(),
            "layers": {**a.model_dump()["layers"], "track": {"backend": "t", "params": {"x": 1}}},
        }
    )
    assert a.layer_hash("detect") == b.layer_hash("detect") and a.layer_hash("track") != b.layer_hash("track")
