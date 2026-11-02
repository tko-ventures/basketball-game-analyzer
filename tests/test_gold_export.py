from bga.eval.gold import build_sequence, interpolate


def _box(x):
    return {"x1": x, "y1": 0, "x2": x + 10, "y2": 20}


def test_interpolate_fills_short_gaps_only():
    grid = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    obs = {0: _box(0), 15: _box(30), 40: _box(80)}
    out = interpolate(obs, grid, max_gap=2)
    assert sorted(out) == [0, 5, 10, 15, 40]
    assert out[5] == ({"x1": 10, "y1": 0, "x2": 20, "y2": 20}, True)
    assert out[15][1] is False


def test_sequence_grid_classes_and_visibility():
    labels = {
        "frames": {
            "100": {"boxes": [{"tid": 1, **_box(0)}, {"tid": 2, **_box(50), "occluded": True}]},
            "110": {"boxes": [{"tid": 1, **_box(20)}]},
        },
        "identities": {"1": {"team": "home", "number": "4"}, "2": {"team": "ref"}},
    }
    game = {"youtube_id": "abc", "clip": "c", "held_out": True}
    seq = build_sequence(labels, game, {"shot": 3, "start": 98, "end": 112}, stride=5, max_gap=3)
    assert seq.name == "abc_s3" and seq.frames == [100, 105, 110] and seq.held_out
    assert seq.rows == [
        (1, 1, 0, 0, 10, 20, 1, 1, 1.0),
        (1, 2, 50, 0, 10, 20, 1, 3, 0.5),
        (2, 1, 10.0, 0.0, 10.0, 20.0, 1, 1, 0.5),
        (3, 1, 20, 0, 10, 20, 1, 1, 1.0),
    ]
    assert set(seq.identities) == {"1", "2"}
