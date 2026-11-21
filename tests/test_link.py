"""Layer 4b: splitting tracklets where their look changes, the join gates, numbers, and the greedy linking pass."""

import numpy as np
import pytest

from bga.config import RunConfig
from bga.link.greedy import (
    GreedyLink,
    LinkParams,
    Tracklet,
    chain_ids,
    chain_numbers,
    greedy_links,
    join_cost,
    number_of,
    roster_number,
    split_points,
)
from bga.numbers.base import confirm_consecutive, roster_numbers
from bga.stage import Sequence

#: Gates wide enough that only the gate under test refuses: 10 frames a second.
LP = LinkParams(max_gap_s=2.0, max_cost=0.5, max_speed_mps=8.0, court_slack_m=1.0, step_s=0.1)
X, Y = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def tracklet(tid: int, first: int, last: int, team: str = "home", **kw) -> Tracklet:
    return Tracklet(tid, first, last, team, **kw)


class TestSplitPoints:
    @pytest.mark.parametrize(
        ("looks", "cuts"),
        [
            pytest.param([X] * 6, [], id="one-look"),
            pytest.param([X] * 3 + [Y] * 3, [3], id="two-looks"),
            pytest.param([X] * 3 + [Y] * 3 + [X] * 3, [3, 6], id="swap-and-back"),
        ],
    )
    def test_cuts_where_the_look_changes(self, looks, cuts):
        assert [i for i, _ in split_points(np.stack(looks), 2, 0.5)] == cuts

    def test_short_tracklet_is_never_cut(self):
        assert split_points(np.stack([X, Y, X]), 2, 0.1) == []


class TestJoinCost:
    @pytest.mark.parametrize(
        ("b", "lp"),
        [
            pytest.param(tracklet(2, 5, 20), LP, id="overlap"),
            pytest.param(tracklet(2, 12, 20, team="away"), LP, id="other-team"),
            pytest.param(tracklet(2, 40, 50), LP, id="gap-too-long"),
            pytest.param(tracklet(2, 12, 20, head=Y), LP, id="cost-too-high"),
            pytest.param(tracklet(2, 12, 20, head_m=np.array([20.0, 0.0])), LP, id="too-fast-on-the-floor"),
        ],
    )
    def test_gate_refuses(self, b, lp):
        a = tracklet(1, 0, 9, tail=X, tail_m=np.array([0.0, 0.0]))
        assert join_cost(a, b, lp) is None

    def test_without_embeddings_cost_is_the_gap_share(self):
        assert join_cost(tracklet(1, 0, 9), tracklet(2, 19, 30), LP) == pytest.approx(0.5)

    def test_speed_gate_is_skipped_without_court_positions(self):
        assert join_cost(tracklet(1, 0, 9, tail=X), tracklet(2, 12, 20, head=X), LP) == pytest.approx(0.0)

    def test_unknown_team_is_never_joined(self):
        assert join_cost(tracklet(1, 0, 9, team=""), tracklet(2, 12, 20, team=""), LP) is None

    def test_different_numbers_are_never_joined(self):
        assert join_cost(tracklet(1, 0, 9, tail=X, number="5"), tracklet(2, 12, 20, head=X, number="7"), LP) is None

    def test_same_number_joins_past_the_appearance_gate(self):
        assert join_cost(tracklet(1, 0, 9, tail=X, number="5"), tracklet(2, 12, 20, head=Y, number="5"), LP) == 0.0

    def test_same_number_still_needs_the_same_team(self):
        a, b = tracklet(1, 0, 9, number="5"), tracklet(2, 12, 20, team="away", number="5")
        assert join_cost(a, b, LP) is None


class TestGreedyLinks:
    def test_each_tracklet_keeps_one_successor(self):
        ts = [tracklet(1, 0, 9, tail=X), tracklet(2, 12, 20, head=X), tracklet(3, 12, 20, head=X)]
        assert len(greedy_links(ts, LP)) == 1

    def test_cheaper_join_wins(self):
        ts = [tracklet(1, 0, 9, tail=X), tracklet(2, 12, 20, head=np.array([0.8, 0.6])), tracklet(3, 12, 20, head=X)]
        assert greedy_links(ts, LP)[0][:2] == (1, 3)

    def test_chains_with_different_numbers_stay_apart(self):
        ts = [tracklet(1, 0, 9, number="5"), tracklet(2, 12, 20), tracklet(3, 25, 30, number="7")]
        assert [link[:2] for link in greedy_links(ts, LP)] == [(1, 2)]

    def test_chain_takes_the_earliest_id(self):
        assert chain_ids([(2, 3, 0.1), (1, 2, 0.1)]) == {2: 1, 3: 1, 1: 1}


class TestNumberOf:
    @pytest.mark.parametrize(
        ("log", "whole", "number"),
        [
            pytest.param([], True, "23", id="whole-track"),
            pytest.param([[0, 4, "23"]], False, "23", id="piece-with-a-read"),
            pytest.param([[50, 4, "23"]], False, None, id="piece-without-a-read"),
            pytest.param([[0, 9, "23"]], False, None, id="read-of-another-track"),
        ],
    )
    def test_number_follows_the_reads(self, log, whole, number):
        rows = [(0, 0, [4, 0, 0, 1, 1, 0.9, "player", 0])]
        assert number_of(4, rows, {"tracks": {"4": "23"}, "log": log}, whole) == number


class TestConfirmConsecutive:
    @pytest.mark.parametrize(
        ("reads", "number"),
        [
            pytest.param(["5", "5", "5"], "5", id="three-in-a-row"),
            pytest.param(["5", None, "5", None, "5"], "5", id="empty-reads-are-skipped"),
            pytest.param(["5", "5", "8", "5"], None, id="streak-broken"),
        ],
    )
    def test_confirms_after_three_equal_reads(self, reads, number):
        assert confirm_consecutive(reads, 3) == number


#: One tracker id's rows on four frames.
FOUR_ROWS = tuple((k, f, [4, 0, 0, 1, 1, 0.9, "player", 0]) for k, f in enumerate((0, 5, 10, 15)))


class TestRosterNumber:
    def test_off_roster_reads_are_dropped(self):
        log = [[0, 4, "8"], [5, 4, "2"], [10, 4, "2"], [15, 4, "2"]]
        assert roster_number(4, list(FOUR_ROWS), {"log": log}, {"2"}, 3) == "2"

    def test_reads_outside_the_piece_are_ignored(self):
        log = [[0, 4, "2"], [5, 4, "2"], [99, 4, "2"]]
        assert roster_number(4, list(FOUR_ROWS), {"log": log}, {"2"}, 3) is None


class TestRosterNumbers:
    def test_unknown_roster_is_empty(self):
        assert roster_numbers({"season": "2025/26", "home": {"roster": "no_such_club"}}, "home") == set()


class TestChainNumbers:
    def test_chain_carries_the_number_of_any_piece(self):
        ts = [tracklet(1, 0, 9), tracklet(2, 12, 20, number="5")]
        assert chain_numbers(ts, {1: 1, 2: 1}, one_owner=True) == {1: "5"}

    def test_one_owner_keeps_the_longer_chain(self):
        ts = [tracklet(1, 0, 9, number="5", rows=[(0, [])]), tracklet(2, 0, 9, number="5", rows=[(0, [])] * 2)]
        assert chain_numbers(ts, {}, one_owner=True) == {1: None, 2: "5"}

    def test_same_number_on_both_teams_is_kept(self):
        ts = [tracklet(1, 0, 9, number="5"), tracklet(2, 0, 9, team="away", number="5")]
        assert chain_numbers(ts, {}, one_owner=True) == {1: "5", 2: "5"}


def row(tid: int, det: int) -> list:
    return [tid, 0.0, 0.0, 10.0, 20.0, 0.9, "player", det]


@pytest.fixture
def upstream() -> dict:
    """Track 1 on frames 0 to 3, track 2 on frames 5 to 8, one home player with one look throughout."""
    frames = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    track = {str(f): [row(1 if f < 4 else 2, 0)] if f != 4 else [] for f in frames}
    team = {str(f): ["home"] for f in frames}
    boxes = [[f, 0] for f in frames if f != 4]
    emb = np.tile(X, (len(boxes), 1)).astype(np.float16)
    return {
        "track": {"frames": track},
        "team": {"frames": team},
        "appearance": {"boxes": boxes, "dim": 2, "arrays": {"embeddings": emb}},
    }


@pytest.fixture
def seq() -> Sequence:
    return Sequence(name="s", clip="c", video="v.mp4", shot=1, frames=list(range(9)), fps=10.0)


class TestGreedyLink:
    def test_split_tracker_ids_become_one(self, seq, upstream):
        out = GreedyLink({}).run(seq, upstream, None)
        assert {r[0] for rows in out["frames"].values() for r in rows} == {1}

    def test_row_order_within_a_frame_is_kept(self, seq, upstream):
        upstream["track"]["frames"]["2"] = [row(3, 1), row(1, 0)]
        upstream["team"]["frames"]["2"] = ["home", "home"]
        out = GreedyLink({}).run(seq, upstream, None)
        assert [r[0] for r in out["frames"]["2"]] == [3, 1]

    def test_rows_are_kept(self, seq, upstream):
        out = GreedyLink({}).run(seq, upstream, None)
        assert sum(map(len, out["frames"].values())) == 8

    def test_link_is_reported_with_its_gap(self, seq, upstream):
        assert GreedyLink({}).run(seq, upstream, None)["links"] == [[1, 2, 0.0, 0.2]]

    def test_different_numbers_keep_the_tracks_apart(self, seq, upstream):
        upstream["numbers"] = {"tracks": {"1": "5", "2": "7"}, "log": []}
        assert GreedyLink({}).run(seq, upstream, None)["numbers"] == {"1": "5", "2": "7"}

    def test_linked_track_gets_its_number(self, seq, upstream):
        upstream["numbers"] = {"tracks": {"1": None, "2": "7"}, "log": []}
        assert GreedyLink({}).run(seq, upstream, None)["numbers"] == {"1": "7"}


class TestLinkConfig:
    def test_appearance_court_and_numbers_are_optional(self):
        layers = {
            "shots": {"backend": "gold"},
            "detect": {"backend": "rfdetr_torch"},
            "team": {"backend": "siglip_kmeans"},
            "track": {"backend": "trackers"},
            "link": {"backend": "greedy"},
        }
        assert "link" in RunConfig(name="t", layers=layers).layers
