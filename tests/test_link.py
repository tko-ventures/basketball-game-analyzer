"""Layer 4b: splitting tracklets where their look changes, the join gates, and the greedy linking pass."""

import numpy as np
import pytest

from bga.config import RunConfig
from bga.link.greedy import GreedyLink, LinkParams, Tracklet, chain_ids, greedy_links, join_cost, split_points
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


class TestGreedyLinks:
    def test_each_tracklet_keeps_one_successor(self):
        ts = [tracklet(1, 0, 9, tail=X), tracklet(2, 12, 20, head=X), tracklet(3, 12, 20, head=X)]
        assert len(greedy_links(ts, LP)) == 1

    def test_cheaper_join_wins(self):
        ts = [tracklet(1, 0, 9, tail=X), tracklet(2, 12, 20, head=np.array([0.8, 0.6])), tracklet(3, 12, 20, head=X)]
        assert greedy_links(ts, LP)[0][:2] == (1, 3)

    def test_chain_takes_the_earliest_id(self):
        assert chain_ids([(2, 3, 0.1), (1, 2, 0.1)]) == {2: 1, 3: 1, 1: 1}


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

    def test_rows_are_kept(self, seq, upstream):
        out = GreedyLink({}).run(seq, upstream, None)
        assert sum(map(len, out["frames"].values())) == 8

    def test_link_is_reported_with_its_gap(self, seq, upstream):
        assert GreedyLink({}).run(seq, upstream, None)["links"] == [[1, 2, 0.0, 0.2]]


class TestLinkConfig:
    def test_appearance_and_court_are_optional(self):
        layers = {
            "shots": {"backend": "gold"},
            "detect": {"backend": "rfdetr_torch"},
            "team": {"backend": "siglip_kmeans"},
            "track": {"backend": "trackers"},
            "link": {"backend": "greedy"},
        }
        assert "link" in RunConfig(name="t", layers=layers).layers
