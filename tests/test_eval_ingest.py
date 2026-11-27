"""Layer 0 cut scoring: gold windows and matching a detector's cuts to them."""

import pytest

from bga.eval.ingest import CutCounts, Transition, gold_transitions, match_cuts

#: A hard cut at 100, a dissolve blending 198..205 (named by two candidates), and a rejected candidate at 300.
GAME = {
    "transitions": [
        {"frame": 100, "verdict": "cut", "reviewed": False},
        {"frame": 200, "verdict": "gradual", "first": 198, "last": 205, "reviewed": True},
        {"frame": 206, "verdict": "gradual", "first": 198, "last": 205, "reviewed": True},
        {"frame": 300, "verdict": "no", "reviewed": True},
    ]
}
SPANS = [[0, 500]]


@pytest.fixture
def gold():
    return gold_transitions(GAME, tol=2)


class TestGoldTransitions:
    def test_one_window_per_transition(self, gold):
        assert [(t.lo, t.hi, t.real) for t in gold] == [(98, 102, True), (196, 208, True), (298, 302, False)]

    def test_gradual_window_ends_on_the_frame_after_the_blend(self):
        game = {"transitions": [{"frame": 10, "verdict": "gradual", "first": 8, "last": 12, "reviewed": True}]}
        assert gold_transitions(game, tol=0)[0].hi == 13


class TestMatchCuts:
    @pytest.mark.parametrize(
        ("cuts", "expected"),
        [
            pytest.param([102], CutCounts(tp=1, fp=0, fn=1, dup=0, unseen=0), id="hard-cut-at-tolerance"),
            pytest.param([103], CutCounts(tp=0, fp=1, fn=2, dup=0, unseen=1), id="hard-cut-past-tolerance"),
            pytest.param([197, 207], CutCounts(tp=1, fp=0, fn=1, dup=1, unseen=0), id="two-cuts-on-one-dissolve"),
            pytest.param([300], CutCounts(tp=0, fp=1, fn=2, dup=0, unseen=0), id="rejected-candidate"),
            pytest.param([999], CutCounts(tp=0, fp=0, fn=2, dup=0, unseen=0), id="outside-game-time"),
        ],
    )
    def test_counts(self, gold, cuts, expected):
        assert match_cuts(cuts, gold, SPANS) == expected

    def test_every_real_transition_found(self, gold):
        assert match_cuts([100, 205], gold, SPANS).recall == 1.0


class TestCutCounts:
    def test_add_sums_each_count(self):
        total = CutCounts(1, 2, 3, 4, 5) + CutCounts(10, 20, 30, 40, 50)
        assert total == CutCounts(11, 22, 33, 44, 55)

    def test_empty_counts_score_zero(self):
        assert (CutCounts(0, 0, 0, 0, 0).precision, CutCounts(0, 0, 0, 0, 0).recall) == (0.0, 0.0)


class TestTransition:
    def test_holds_its_window_ends(self):
        t = Transition(lo=5, hi=9, real=True, reviewed=True)
        assert (t.holds(5), t.holds(9), t.holds(10)) == (True, True, False)
