"""``shots/video``: raw video to central sequences through registered cut and camera backends, cached per video."""

import json
from typing import ClassVar

import pytest

from bga import registry
from bga.ingest import video
from bga.ingest.camera import CameraBackend
from bga.ingest.cuts import CutsBackend

CLIP = "20990101_FAKEyoutube_Game"
#: Shots the fake cut detector finds at 10 fps: 5 s, 2 s and 4 s long.
CUTS = [{"start": 0, "end": 49}, {"start": 50, "end": 69}, {"start": 70, "end": 109}]
#: Camera scores the fake classifier gives those shots: central, central (too short), other.
SCORES = [0.9, 1.0, 0.2]


class FakeCuts(CutsBackend):
    """Cut detector returning ``CUTS``; counts its calls."""

    calls = 0
    defaults: ClassVar[dict] = {"x": 0}

    def shots(self, video_path, progress=None):
        FakeCuts.calls += 1
        return [dict(s) for s in CUTS]


class FakeCamera(CameraBackend):
    """Camera classifier returning ``SCORES``."""

    def score_shots(self, shots):
        return SCORES[: len(shots)]


@pytest.fixture
def game(tmp_path, monkeypatch):
    """A label file whose own shots disagree with the fake cuts, and the fake backends registered.

    The video does not exist, so the camera backend gets no frames; the fake ignores them.

    """
    labels = tmp_path / "labels"
    labels.mkdir()
    label = {
        "clip": str(tmp_path / "missing.mp4"),
        "fps": 10.0,
        "meta": {"home": {"team": "A"}},
        "shots": [{"start": 0, "end": 109, "camera": "central"}],
    }
    (labels / f"{CLIP}.json").write_text(json.dumps(label))
    FakeCuts.calls = 0
    for kind, cls in (("cuts", FakeCuts), ("camera", FakeCamera)):
        cls.name = f"_fake_{kind}"
        monkeypatch.setitem(registry._REGISTRY, (kind, cls.name), cls)
    return {
        "clips": [CLIP],
        "cuts": {"backend": "_fake_cuts", "params": {}},
        "camera": {"backend": "_fake_camera", "params": {}},
        "labels": str(labels),
        "cache": str(tmp_path / "runs"),
        "min_s": 3.0,
        "stride": 10,
    }


class TestVideoShots:
    def test_keeps_central_shots_long_enough(self, game):
        seqs = video.VideoShots(game).sequences()
        assert [(s.name, s.frames[0], s.frames[-1]) for s in seqs] == [("FAKEyoutube_v0", 0, 40)]

    def test_carries_the_games_meta(self, game):
        seqs = video.VideoShots(game).sequences()
        assert seqs[0].meta == {"home": {"team": "A"}}

    def test_cuts_are_cached_per_video(self, game):
        video.VideoShots(game).sequences()
        video.VideoShots(game).sequences()
        assert FakeCuts.calls == 1

    def test_other_cut_params_recompute(self, game):
        video.VideoShots(game).sequences()
        video.VideoShots({**game, "cuts": {"backend": "_fake_cuts", "params": {"x": 1}}}).sequences()
        assert FakeCuts.calls == 2
