"""Tests for layer 5b (identity), crop probability logging, and clip-scoped pipeline dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from bga import pipeline, registry
from bga.config import RunConfig
from bga.eval.scores import _track_numbers
from bga.identity.base import (
    TrackEvidence,
    assign_covisible,
    assign_owner_rule,
    cluster_team_tracks,
    number_margins,
    roster_logprob_scores,
    vote_consecutive,
    vote_roster_logprob,
)
from bga.identity.game import GameIdentity
from bga.identity.shot import ConsecutiveIdentity, CovisibleAssignIdentity, RosterLogprobIdentity
from bga.numbers.base import NUMBER_CLASSES, UNREADABLE, CropRead, TrackNumbers, roster_numbers
from bga.stage import BaseBackend, Sequence


def _prob_vec(dist: dict[str, float]) -> np.ndarray:
    """Build a full ``NUMBER_CLASSES`` probability vector from sparse class weights."""
    idx = {c: i for i, c in enumerate(NUMBER_CLASSES)}
    vec = np.zeros(len(NUMBER_CLASSES), dtype=np.float32)
    for k, v in dist.items():
        vec[idx[k]] = v
    return vec


def _seed_empty_upstreams(cfg: RunConfig, runs: Path, seqs: tuple[Sequence, ...], layers: tuple[str, ...]) -> None:
    """Write minimal empty layer outputs, recorded under the current code keys, so a downstream test layer can run."""
    keys = pipeline.code_keys(cfg)
    for layer in layers:
        for seq in seqs:
            pipeline.write_output(pipeline.output_path(cfg, runs, seq, layer), {"frames": {}, "numbers": {}})
        pipeline._write_meta(cfg, runs, layer, list(seqs), 0.0, keys[layer])


class _StubProbReader(TrackNumbers):
    """Test numbers backend returning fixed probability distributions per crop."""

    layer, name = "numbers", "_stub_probs"

    def read_crops(self, crops: list[np.ndarray], rosters: list[set[str]]) -> list[CropRead]:
        del rosters
        return [CropRead("7", {"7": 0.85, "1": 0.1, UNREADABLE: 0.05}) for _ in crops]


class _StubTextReader(TrackNumbers):
    """Test numbers backend returning decoded text without probability distributions."""

    layer, name = "numbers", "_stub_text"

    def read_crop(self, crop: np.ndarray) -> str | None:
        del crop
        return "7"


class TestNumbersProbabilityLogging:
    """``TrackNumbers`` logs aligned probability arrays and ``roster_numbers`` ignores missing jersey numbers."""

    def test_roster_numbers_skips_null_entries(self, tmp_path: Path) -> None:
        roster = {"seasons": {"2025/26": {"players": [{"number": "7"}, {"number": None}, {"number": "23"}]}}}
        (tmp_path / "team_a.json").write_text(json.dumps(roster))
        meta = {"season": "2025/26", "home": {"roster": "team_a"}}
        assert roster_numbers(meta, "home", rosters=tmp_path) == {"7", "23"}

    def test_prob_reader_writes_aligned_probs_array(self) -> None:
        seq = Sequence("c_s1", "c", "v.mp4", 1, [0, 5], 50.0)
        det = [[10, 10, 20, 20, 0.9, "number"]]
        trk = [[1, 0, 0, 40, 40, 0.9, "player", 0]]
        ups = {"detect": {"frames": {"0": det, "5": det}}, "track": {"frames": {"0": trk, "5": trk}}}
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        out = _StubProbReader({"n_consecutive": 2, "stop_on_confirm": False}).run(seq, ups, lambda _f: img)
        assert out["reads"] == 2
        assert out["classes"] == list(NUMBER_CLASSES)
        probs = out["arrays"]["probs"]
        assert probs.shape == (2, len(NUMBER_CLASSES))
        assert probs.dtype == np.float32
        assert probs[0, NUMBER_CLASSES.index("7")] == pytest.approx(0.85)

    def test_text_only_reader_omits_probs_array(self) -> None:
        seq = Sequence("c_s1", "c", "v.mp4", 1, [0], 50.0)
        ups = {
            "detect": {"frames": {"0": [[10, 10, 20, 20, 0.9, "number"]]}},
            "track": {"frames": {"0": [[1, 0, 0, 40, 40, 0.9, "player", 0]]}},
        }
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        out = _StubTextReader({}).run(seq, ups, lambda _f: img)
        assert "arrays" not in out
        assert "classes" not in out


class TestRosterLogprobVoting:
    """Summed renormalised log-probabilities over a team's roster."""

    @pytest.mark.parametrize(
        ("reads", "roster", "expected"),
        [
            ([(0, "8"), (5, "99"), (10, "8"), (15, "8")], {"8", "23"}, "8"),
            ([(0, "8"), (5, "23"), (10, "8")], {"8", "23"}, None),
            ([(0, "99"), (5, "99"), (10, "99")], {"8", "23"}, None),
        ],
    )
    def test_vote_consecutive(
        self, reads: list[tuple[int, str | None]], roster: set[str], expected: str | None
    ) -> None:
        assert vote_consecutive(reads, roster, n_consecutive=3) == expected

    def test_renormalises_over_roster_and_skips_unreadable(self) -> None:
        p_legible = _prob_vec({"99": 0.7, "7": 0.25, "12": 0.02, UNREADABLE: 0.03})
        p_illegible = _prob_vec({UNREADABLE: 0.9, "7": 0.05, "12": 0.05})
        scores = roster_logprob_scores([p_legible, p_illegible, p_legible], {"7", "12"})
        margins = number_margins(scores)
        assert margins["7"] > 4.0
        assert margins["12"] < 0.0

    def test_abstains_below_margin(self) -> None:
        p = [_prob_vec({"7": 0.55, "12": 0.35, UNREADABLE: 0.10})]
        confirmed, _ = vote_roster_logprob(p, {"7", "12"}, margin=4.0)
        assert confirmed is None

    def test_empty_roster_falls_back_to_all_numbers(self) -> None:
        p = [_prob_vec({"42": 0.9, "12": 0.05, UNREADABLE: 0.05})] * 3
        confirmed, _ = vote_roster_logprob(p, set(), margin=6.0)
        assert confirmed == "42"


class TestCovisibleAssignment:
    """``covisible_assign`` uses scores instead of box counts and flags conflicts without silent reassignment."""

    def test_owner_rule_prefers_longer_track_while_covisible_prefers_higher_score(self) -> None:
        long_wrong = TrackEvidence("s1", 1, "home", boxes=100, frames={0, 5, 10})
        short_right = TrackEvidence("s1", 2, "home", boxes=60, frames={0, 5, 10})
        owner = assign_owner_rule([long_wrong, short_right], {1: "10", 2: "10"})
        assert owner == {1: "10", 2: None}

        margins = {
            1: {"10": 30.0, "20": -30.0},
            2: {"10": 80.0, "20": -80.0},
        }
        assigned, conflicts = assign_covisible(
            [long_wrong, short_right], margins, {"home": {"10", "20"}, "away": set()}, threshold=8.0
        )
        assert assigned == {1: None, 2: "10"}
        assert len(conflicts) == 1
        assert conflicts[0]["winner"] == 2
        assert conflicts[0]["tracks"] == [1, 2]

    def test_non_covisible_tracks_do_not_block_each_other(self) -> None:
        early = TrackEvidence("s1", 1, "home", boxes=30, frames={0, 5})
        late = TrackEvidence("s1", 2, "home", boxes=25, frames={100, 105})
        margins = {1: {"2": 40.0, "8": -40.0}, 2: {"2": 35.0, "8": -35.0}}
        assigned, conflicts = assign_covisible([early, late], margins, {"home": {"2", "8"}, "away": set()}, 8.0)
        assert assigned == {1: "2", 2: "2"}
        assert conflicts == []


class TestGameClusteringAndPooling:
    """``cluster_team_tracks`` and ``GameIdentity`` respect co-visibility, split conflicts, and pool reads."""

    def test_never_merges_covisible_or_conflicting_numbers(self) -> None:
        e_same = np.array([1.0, 0.0], dtype=np.float32)
        t1 = TrackEvidence("s1", 1, "home", boxes=10, frames={0, 5}, embedding=e_same)
        t2_covis = TrackEvidence("s1", 2, "home", boxes=10, frames={5, 10}, embedding=e_same)
        t3_other_shot_same = TrackEvidence("s2", 3, "home", boxes=10, frames={0, 5}, embedding=e_same)
        t4_conflict = TrackEvidence("s3", 4, "home", boxes=10, frames={0, 5}, embedding=e_same)
        clusters, splits = cluster_team_tracks(
            [t1, t2_covis, t3_other_shot_same, t4_conflict],
            confirmed=["7", None, None, "11"],
            max_dist=0.2,
        )
        assert clusters == [[0, 2], [1, 3]]
        assert any(sp["kind"] == "cluster_split_conflict" for sp in splits)

    def test_game_identity_pools_votes_across_shots(self) -> None:
        seq1 = Sequence("g_s1", "g", "v.mp4", 1, [0, 5], 50.0)
        seq2 = Sequence("g_s2", "g", "v.mp4", 2, [0, 5], 50.0)
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        p7 = np.stack([_prob_vec({"7": 0.92, "11": 0.04, UNREADABLE: 0.04})] * 3)
        ups = {
            "g_s1": {
                "link": {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}, "ids": {"1": 1}},
                "track": {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}},
                "team": {"frames": {"0": ["home"]}},
                "numbers": {
                    "log": [[0, 1, "7"], [0, 1, "7"], [0, 1, "7"]],
                    "classes": list(NUMBER_CLASSES),
                    "arrays": {"probs": p7},
                },
                "appearance": {"boxes": [[0, 0]], "arrays": {"embeddings": emb}},
            },
            "g_s2": {
                "link": {"frames": {"0": [[9, 0, 0, 10, 20, 0.9, "player", 0]]}, "ids": {"9": 9}},
                "track": {"frames": {"0": [[9, 0, 0, 10, 20, 0.9, "player", 0]]}},
                "team": {"frames": {"0": ["home"]}},
                "numbers": {"log": [], "classes": list(NUMBER_CLASSES)},
                "appearance": {"boxes": [[0, 0]], "arrays": {"embeddings": emb}},
            },
        }
        outs = GameIdentity({"roster": False, "margin": 6.0, "max_dist": 0.2}).run_clip([seq1, seq2], ups)
        assert outs["g_s1"]["numbers"] == {"1": "7"}
        assert outs["g_s2"]["numbers"] == {"9": "7"}
        assert outs["g_s1"]["clusters"]["1"] == outs["g_s2"]["clusters"]["9"]


class _DummyClipIdentity(BaseBackend):
    """Minimal clip-scoped backend verifying ``pipeline.run`` passes all shots of a clip together."""

    scope: ClassVar[str] = "clip"

    def run_clip(self, seqs: list[Sequence], upstreams: dict[str, dict[str, dict]]) -> dict[str, dict]:
        del upstreams
        names = sorted(s.name for s in seqs)
        return {s.name: {"numbers": {"1": "7"}, "seen": names} for s in seqs}


class TestPipelineAndEvalIntegration:
    """Clip-scoped pipeline dispatch, shot backends, and ``_track_numbers`` precedence."""

    def test_shot_backends_end_to_end(self) -> None:
        seq = Sequence("c_s1", "c", "v.mp4", 1, [0], 50.0)
        p7 = np.stack([_prob_vec({"7": 0.95, "11": 0.02, UNREADABLE: 0.03})] * 3)
        ups = {
            "link": {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}, "ids": {"1": 1}},
            "track": {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}},
            "team": {"frames": {"0": ["home"]}},
            "numbers": {
                "log": [[0, 1, "7"], [0, 1, "7"], [0, 1, "7"]],
                "classes": list(NUMBER_CLASSES),
                "arrays": {"probs": p7},
            },
        }
        assert ConsecutiveIdentity({"roster": False}).run(seq, ups, None)["numbers"] == {"1": "7"}
        assert RosterLogprobIdentity({"roster": False, "margin": 6.0}).run(seq, ups, None)["numbers"] == {"1": "7"}
        assert CovisibleAssignIdentity({"roster": False, "margin": 6.0}).run(seq, ups, None)["numbers"] == {"1": "7"}

    def test_clip_scope_pipeline_and_eval_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        s1 = Sequence("clipA_s1", "clipA", "v.mp4", 1, [0], 50.0)
        s2 = Sequence("clipA_s2", "clipA", "v.mp4", 2, [0], 50.0)

        class _TwoShots(BaseBackend):
            def sequences(self) -> list[Sequence]:
                return [s1, s2]

        monkeypatch.setitem(registry._REGISTRY, ("shots", "_two"), _TwoShots)
        monkeypatch.setitem(registry._REGISTRY, ("identity", "_dummy_clip"), _DummyClipIdentity)
        cfg = RunConfig(
            name="clip-test",
            layers={
                "shots": {"backend": "_two"},
                "detect": {"backend": "roboflow_inference"},
                "team": {"backend": "siglip_kmeans"},
                "track": {"backend": "bytetrack"},
                "link": {"backend": "greedy"},
                "identity": {"backend": "_dummy_clip"},
            },
        )
        _seed_empty_upstreams(cfg, tmp_path, (s1, s2), ("track", "team", "link"))
        pipeline.run(cfg, runs=tmp_path, layers=["identity"])
        out1 = pipeline.load_output(cfg, tmp_path, s1, "identity")
        out2 = pipeline.load_output(cfg, tmp_path, s2, "identity")
        assert out1["seen"] == ["clipA_s1", "clipA_s2"]
        assert out2["numbers"] == {"1": "7"}
        assert _track_numbers(cfg, tmp_path, s1, {"numbers": {"1": "99"}}) == {"1": "7"}
