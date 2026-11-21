"""Shots of non-gold clips: shot length filter, grid frames, names, and the gold refusal."""

import json
from pathlib import Path

import pytest

from bga.config import RunConfig
from bga.ingest.clip_shots import ClipShots

#: A non-gold label stem (its YouTube id is not a gold game's).
CLIP = "20251113_kW4HG-5MLNM_SKROT_Dziki_-_King_8._kolejka"


@pytest.fixture
def labels(tmp_path: Path) -> Path:
    shots = [{"start": 0, "end": 99}, {"start": 100, "end": 102}, {"start": 103, "end": 302}]
    (tmp_path / f"{CLIP}.json").write_text(
        json.dumps({"clip": "v.mp4", "fps": 50.0, "shots": shots, "meta": {"season": "2025/26"}})
    )
    return tmp_path


class TestClipShots:
    def test_short_shots_are_left_out(self, labels):
        seqs = ClipShots({"clips": [CLIP], "labels": str(labels), "min_s": 1.0}).sequences()
        assert [s.name for s in seqs] == ["kW4HG-5MLNM_s0", "kW4HG-5MLNM_s2"]

    def test_frames_sit_on_the_stride_grid(self, labels):
        seqs = ClipShots({"clips": [CLIP], "labels": str(labels), "min_s": 1.0}).sequences()
        assert seqs[1].frames[:2] == [105, 110]

    def test_gold_game_is_refused(self, labels):
        with pytest.raises(ValueError, match="gold"):
            ClipShots({"clips": ["20250213_4XMD9FWmQws_PGE_Start_Dziki_Skrot_meczu"]}).sequences()


class TestSelectionHash:
    def test_clip_list_does_not_change_the_hash(self):
        def cfg(clips):
            return RunConfig(name="t", layers={"shots": {"backend": "clips", "params": {"clips": clips}}})

        assert cfg([CLIP]).layer_hash("shots") == cfg([]).layer_hash("shots")
