"""``link/greedy``: split tracklets where their appearance changes, then join them cheapest first (plan 5, layer 4).

**Split.** A tracker sometimes moves a track from one player to another. With embeddings, a tracklet is cut where the
cosine distance between the mean embedding of its ``split_window`` boxes before a point and that of the
``split_window`` boxes after it peaks above ``split_cost``, and each piece is searched again. ``split_cost: null``
turns splitting off. Pieces after the first get new ids above the tracker's.

**Join.** A tracklet may be joined to one later tracklet of the same team that starts after it ends, within
``max_gap_s``. The cost of a join is the cosine distance between the mean appearance embedding of the earlier
tracklet's last ``ends`` boxes and that of the later one's first ``ends`` boxes (layer 3b); without embeddings
(``appearance: none`` or no appearance layer) it is the gap as a share of ``max_gap_s``. Joins above ``max_cost`` are
refused. With court homographies at both ends, a join that needs a speed above ``max_speed_mps`` on the floor is
refused too. Joins are taken cheapest first, each tracklet having at most one successor and one predecessor.

**Numbers.** With a ``numbers`` layer, a tracklet carries its tracker id's confirmed jersey number (a split piece
only when it holds a read of that number). Two tracklets with different numbers are never joined, nor are two
chains whose numbers differ; two with the same number join at cost 0, past the appearance gate. With ``one_owner``,
a (team, number) left on more than one chain stays only on the chain with the most boxes. With ``roster``, a
tracklet's number is instead confirmed again from the logged reads inside it: reads not on its team's roster for the
clip's season are dropped, and ``confirm_reads`` equal reads in a row confirm.

"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from bga.appearance.base import box_rows
from bga.numbers.base import confirm_consecutive, roster_numbers
from bga.positions.bbox_bottom import project
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@dataclass(slots=True)
class Tracklet:
    """One tracker id within a shot, with what the linker compares at its two ends.

    Attributes:
        tid: Tracker id.
        first: First frame index on the sequence grid.
        last: Last frame index on the sequence grid.
        team: Majority team label of its detected boxes, ``""`` when it has none.
        head: Unit mean embedding of its first boxes, or ``None``.
        tail: Unit mean embedding of its last boxes, or ``None``.
        head_m: Court position in metres of its first box, or ``None`` without a homography.
        tail_m: Court position in metres of its last box, or ``None``.
        rows: Its track rows with their source frames, in order.
        number: Confirmed jersey number, or ``None``.

    """

    tid: int
    first: int
    last: int
    team: str = ""
    head: np.ndarray | None = None
    tail: np.ndarray | None = None
    head_m: np.ndarray | None = None
    tail_m: np.ndarray | None = None
    rows: list[tuple[int, list]] = field(default_factory=list)
    number: str | None = None


def _unit_mean(vectors: list[np.ndarray]) -> np.ndarray | None:
    """Unit-length mean of some embeddings, or ``None`` when there are none."""
    if not vectors:
        return None
    m = np.mean(vectors, axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else None


def _court_m(homography: list | None, row: list) -> np.ndarray | None:
    """Court position in metres of a track row's bottom centre, or ``None`` without a homography."""
    if homography is None:
        return None
    return project(homography, np.array([[(row[1] + row[3]) / 2, row[4]]]))[0] / 100.0


#: A track row with its sequence-grid index and source frame: ``(k, frame, row)``.
Row = tuple[int, int, list]


def collect_rows(frames: list[int], tracks: dict) -> dict[int, list[Row]]:
    """Group the track rows of a sequence by tracker id, in frame order.

    Args:
        frames: The sequence's frames, in order.
        tracks: ``track`` output frames.

    Returns:
        Tracker id to its rows.

    Examples:
        >>> collect_rows([0, 5], {"0": [[1, 0, 0, 1, 1, 0.9, "player", 0]], "5": []})
        {1: [(0, 0, [1, 0, 0, 1, 1, 0.9, 'player', 0])]}

    """
    out: dict[int, list[Row]] = {}
    for k, f in enumerate(frames):
        for r in tracks.get(str(f), []):
            out.setdefault(r[0], []).append((k, f, r))
    return out


def split_points(vecs: np.ndarray, window: int, max_cost: float) -> list[tuple[int, float]]:
    """Find where a sequence of embeddings changes look: the peaks of the before/after mean distance above a cost.

    The best point of a span is taken when its distance exceeds ``max_cost`` and both sides keep ``window``
    vectors; each side is then searched again.

    Args:
        vecs: Unit embeddings in time order, ``(n, d)``.
        window: Vectors averaged on each side of a point.
        max_cost: Distance a change must exceed.

    Returns:
        ``(index, distance)`` of each cut, sorted; a cut at ``i`` starts a new piece at ``vecs[i]``.

    Examples:
        >>> v = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)
        >>> split_points(v, 2, 0.5)
        [(4, 1.0)]
        >>> split_points(v[:4], 2, 0.5)
        []

    """
    n = len(vecs)
    if n < 2 * window:
        return []
    c = np.cumsum(np.vstack([np.zeros((1, vecs.shape[1])), vecs]), axis=0)
    idx = np.arange(window, n - window + 1)
    before, after = c[idx] - c[idx - window], c[idx + window] - c[idx]
    before /= np.linalg.norm(before, axis=1, keepdims=True).clip(min=1e-9)
    after /= np.linalg.norm(after, axis=1, keepdims=True).clip(min=1e-9)
    d = 1.0 - (before * after).sum(axis=1)
    best = int(np.argmax(d))
    if d[best] <= max_cost:
        return []
    i = int(idx[best])
    left = split_points(vecs[:i], window, max_cost)
    right = [(j + i, dj) for j, dj in split_points(vecs[i:], window, max_cost)]
    return [*left, (i, round(float(d[best]), 4)), *right]


def describe(tid: int, rows: list[Row], upstream: dict, vec_of: dict, ends: int) -> Tracklet:
    """Describe one tracklet: its span, majority team, end embeddings and end court positions.

    Args:
        tid: Its id.
        rows: Its rows, in order.
        upstream: ``team`` and ``court`` outputs where configured.
        vec_of: ``(frame, det_index)`` to unit embedding.
        ends: Boxes averaged at each end.

    Returns:
        The tracklet.

    Examples:
        >>> rows = [(0, 0, [1, 0, 0, 10, 20, 0.9, "player", 0]), (1, 5, [1, 0, 0, 10, 20, 0.9, "player", 0])]
        >>> t = describe(1, rows, {"team": {"frames": {"0": ["home"], "5": ["home"]}}}, {}, 3)
        >>> t.tid, t.first, t.last, t.team, t.head
        (1, 0, 1, 'home', None)

    """
    teams = (upstream.get("team") or {}).get("frames", {})
    court = (upstream.get("court") or {}).get("frames", {})
    votes = Counter(teams[str(f)][r[7]] for _, f, r in rows if r[7] >= 0 and str(f) in teams)
    votes.pop("", None)
    vecs = [vec_of[(f, r[7])] for _, f, r in rows if (f, r[7]) in vec_of]
    (k0, f0, r0), (k1, f1, r1) = rows[0], rows[-1]
    return Tracklet(
        tid=tid,
        first=k0,
        last=k1,
        team=votes.most_common(1)[0][0] if votes else "",
        head=_unit_mean(vecs[:ends]),
        tail=_unit_mean(vecs[-ends:]),
        head_m=_court_m(court.get(str(f0)), r0),
        tail_m=_court_m(court.get(str(f1)), r1),
        rows=[(f, r) for _, f, r in rows],
    )


def embeddings_by_box(upstream: dict) -> dict[tuple[int, int], np.ndarray]:
    """Unit embedding of each embedded box, keyed by ``(frame, det_index)``; empty without embeddings.

    Args:
        upstream: Upstream outputs; ``appearance`` with its arrays loaded, if configured.

    Returns:
        Box to float32 embedding.

    Examples:
        >>> embeddings_by_box({}), embeddings_by_box({"appearance": {"boxes": [], "dim": 0}})
        ({}, {})

    """
    app = upstream.get("appearance") or {}
    emb = (app.get("arrays") or {}).get("embeddings")
    if emb is None or not len(emb):
        return {}
    return {box: emb[row].astype(np.float32) for box, row in box_rows(app).items()}


def split_rows(rows: list[Row], vec_of: dict, window: int, max_cost: float | None) -> list[tuple[list[Row], float]]:
    """Cut one tracker id's rows where its appearance changes (:func:`split_points` over its embedded rows).

    Args:
        rows: The rows, in order.
        vec_of: ``(frame, det_index)`` to unit embedding.
        window: Embedded boxes averaged on each side of a cut.
        max_cost: Distance a change must exceed; ``None`` never cuts.

    Returns:
        The pieces, each with the distance at its cut (0 for the first piece).

    Examples:
        >>> rows = [(k, k, [1, 0, 0, 1, 1, 0.9, "player", k]) for k in range(6)]
        >>> vec_of = {(k, k): np.array([1.0, 0.0] if k < 3 else [0.0, 1.0]) for k in range(6)}
        >>> [(len(p), d) for p, d in split_rows(rows, vec_of, 2, 0.5)]
        [(3, 0.0), (3, 1.0)]

    """
    at = [j for j, (_, f, r) in enumerate(rows) if (f, r[7]) in vec_of]
    if max_cost is None or not at:
        return [(rows, 0.0)]
    cuts = split_points(np.stack([vec_of[(rows[j][1], rows[j][2][7])] for j in at]), window, max_cost)
    bounds = [0, *[at[i] for i, _ in cuts], len(rows)]
    dists = [0.0, *[d for _, d in cuts]]
    return [(rows[a:b], d) for a, b, d in zip(bounds[:-1], bounds[1:], dists, strict=True)]


def number_of(tid: int, rows: list[Row], numbers: dict, whole: bool) -> str | None:
    """Confirmed number of one piece of a tracker id: the id's number, if the piece is the whole track or holds a read.

    Args:
        tid: The tracker id the piece was cut from.
        rows: The piece's rows.
        numbers: ``numbers`` output; ``{}`` without a numbers layer.
        whole: Whether the piece is the tracker id's only piece.

    Returns:
        The number, or ``None``.

    Examples:
        >>> rows = [(0, 10, [4, 0, 0, 1, 1, 0.9, "player", 0])]
        >>> nums = {"tracks": {"4": "23"}, "log": [[10, 4, "23"], [30, 4, "23"]]}
        >>> number_of(4, rows, nums, whole=False), number_of(4, rows, {"tracks": {"4": "23"}}, whole=False)
        ('23', None)

    """
    value = (numbers.get("tracks") or {}).get(str(tid))
    if value is None or whole:
        return value
    frames = {f for _, f, _ in rows}
    return value if any(f in frames and t == tid and v == value for f, t, v in numbers.get("log", [])) else None


def roster_number(tid: int, rows: list[Row], numbers: dict, allowed: set[str], n: int) -> str | None:
    """Number of one piece of a tracker id confirmed from its own logged reads, keeping only roster numbers.

    Args:
        tid: The tracker id the piece was cut from.
        rows: The piece's rows.
        numbers: ``numbers`` output with its ``log``.
        allowed: The roster numbers of the piece's team.
        n: Equal reads in a row needed.

    Returns:
        The number, or ``None``.

    Examples:
        >>> rows = [(0, f, [4, 0, 0, 1, 1, 0.9, "player", 0]) for f in (0, 5, 10, 15)]
        >>> log = [[0, 4, "8"], [5, 4, "2"], [10, 4, "8"], [15, 4, "2"]]
        >>> roster_number(4, rows, {"log": log}, {"2", "23"}, 2), roster_number(4, rows, {"log": log}, {"2", "8"}, 2)
        ('2', None)

    """
    frames = {f for _, f, _ in rows}
    reads = [v if v in allowed else None for f, t, v in numbers.get("log", []) if t == tid and f in frames]
    return confirm_consecutive(reads, n)


@dataclass(frozen=True)
class NumberRule:
    """How a tracklet gets its number from the ``numbers`` output.

    Attributes:
        rosters: Team to its roster numbers, or ``None`` to take the numbers layer's confirmations.
        confirm_reads: Equal roster reads in a row needed when ``rosters`` is set.

    """

    rosters: dict[str, set[str]] | None = None
    confirm_reads: int = 3


def build_tracklets(
    frames: list[int],
    upstream: dict,
    ends: int,
    window: int = 5,
    split_cost: float | None = None,
    rule: NumberRule | None = None,
) -> tuple[list[Tracklet], list[list]]:
    """Collect each tracker id's rows, cut them where the appearance changes, and describe every piece.

    Args:
        frames: The sequence's frames, in order.
        upstream: ``track`` output, and ``team``, ``appearance``, ``court`` and ``numbers`` outputs where configured.
        ends: Boxes averaged at each end for the head and tail embeddings.
        window: Embedded boxes averaged on each side of a cut.
        split_cost: Distance a change must exceed to cut; ``None`` never cuts.
        rule: How pieces get their numbers; the numbers layer's confirmations by default.

    Returns:
        The pieces in order of first appearance, and the cuts as ``[tracker id, frame, new id, distance]``.

    Examples:
        >>> track = {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]], "5": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}}
        >>> ts, cuts = build_tracklets([0, 5], {"track": track}, 3)
        >>> [(t.tid, t.first, t.last) for t in ts], cuts
        ([(1, 0, 1)], [])

    """
    vec_of = embeddings_by_box(upstream)
    numbers = upstream.get("numbers") or {}
    by_id = collect_rows(frames, upstream["track"]["frames"])
    next_id = max(by_id, default=0) + 1
    out, cuts = [], []
    for tid, rows in by_id.items():
        pieces = split_rows(rows, vec_of, window, split_cost)
        for j, (piece, d) in enumerate(pieces):
            pid = tid
            if j:
                pid, next_id = next_id, next_id + 1
                cuts.append([tid, piece[0][1], pid, d])
            t = describe(pid, piece, upstream, vec_of, ends)
            if rule and rule.rosters is not None:
                allowed = rule.rosters.get(t.team, set())
                t.number = roster_number(tid, piece, numbers, allowed, rule.confirm_reads)
            else:
                t.number = number_of(tid, piece, numbers, whole=len(pieces) == 1)
            out.append(t)
    return sorted(out, key=lambda t: t.first), cuts


@dataclass(frozen=True)
class LinkParams:
    """Gates and cost of a join; see the module docstring.

    Attributes:
        max_gap_s: Longest gap between two tracklets that may be joined.
        max_cost: Highest join cost accepted.
        max_speed_mps: Floor speed a join may imply, checked where both ends have a court position; ``None`` skips.
        court_slack_m: Distance added to the speed gate, for homography and foot-point error.
        step_s: Seconds between two frames of the sequence.

    """

    max_gap_s: float
    max_cost: float
    max_speed_mps: float | None
    court_slack_m: float
    step_s: float


def join_cost(a: Tracklet, b: Tracklet, lp: LinkParams) -> float | None:
    """Cost of continuing tracklet ``a`` with ``b``, or ``None`` when a gate refuses the join.

    Args:
        a: The earlier tracklet.
        b: The later tracklet.
        lp: Gates and cost settings.

    Returns:
        The join cost, at most ``max_cost``, or ``None``; 0 for two tracklets with the same number.

    Examples:
        >>> lp = LinkParams(max_gap_s=2.0, max_cost=0.3, max_speed_mps=None, court_slack_m=1.0, step_s=0.1)
        >>> a = Tracklet(1, 0, 9, "home", tail=np.array([1.0, 0.0]))
        >>> b = Tracklet(2, 15, 30, "home", head=np.array([0.8, 0.6]))
        >>> round(join_cost(a, b, lp), 3)
        0.2
        >>> join_cost(a, Tracklet(3, 15, 30, "away", head=np.array([1.0, 0.0])), lp) is None
        True
        >>> join_cost(Tracklet(1, 0, 9, "home", number="5"), Tracklet(2, 15, 30, "home", number="7"), lp) is None
        True

    """
    gap_s = (b.first - a.last) * lp.step_s
    if b.first <= a.last or gap_s > lp.max_gap_s or not a.team or a.team != b.team:
        return None
    court = lp.max_speed_mps is not None and a.tail_m is not None and b.head_m is not None
    if court and np.linalg.norm(b.head_m - a.tail_m) > lp.max_speed_mps * gap_s + lp.court_slack_m:
        return None
    if a.number and b.number:
        return 0.0 if a.number == b.number else None
    cost = 1.0 - float(a.tail @ b.head) if a.tail is not None and b.head is not None else gap_s / lp.max_gap_s
    return cost if cost <= lp.max_cost else None


def greedy_links(tracklets: list[Tracklet], lp: LinkParams) -> list[tuple[int, int, float]]:
    """Choose joins cheapest first, each tracklet keeping at most one successor and one predecessor.

    A join is skipped when the two chains it would connect carry different numbers.

    Args:
        tracklets: The shot's tracklets.
        lp: Gates and cost settings.

    Returns:
        The chosen joins as ``(earlier id, later id, cost)``, cheapest first.

    Examples:
        >>> lp = LinkParams(max_gap_s=2.0, max_cost=0.5, max_speed_mps=None, court_slack_m=1.0, step_s=0.1)
        >>> ts = [Tracklet(1, 0, 9, "home"), Tracklet(2, 12, 20, "home"), Tracklet(3, 15, 25, "home")]
        >>> greedy_links(ts, lp)
        [(1, 2, 0.15)]
        >>> five, seven = Tracklet(1, 0, 9, "home", number="5"), Tracklet(3, 25, 30, "home", number="7")
        >>> greedy_links([five, Tracklet(2, 12, 20, "home"), seven], lp)
        [(1, 2, 0.15)]

    """
    pairs = [(c, a.tid, b.tid) for a in tracklets for b in tracklets if (c := join_cost(a, b, lp)) is not None]
    chain = {t.tid: t.tid for t in tracklets}
    number = {t.tid: t.number for t in tracklets}
    has_next, has_prev, out = set(), set(), []
    for c, a, b in sorted(pairs):
        ca, cb = chain[a], chain[b]
        if a in has_next or b in has_prev or (number[ca] and number[cb] and number[ca] != number[cb]):
            continue
        has_next.add(a)
        has_prev.add(b)
        out.append((a, b, round(c, 4)))
        number[ca] = number[ca] or number[cb]
        chain.update({t: ca for t, k in chain.items() if k == cb})
    return out


def chain_ids(links: list[tuple[int, int, float]]) -> dict[int, int]:
    """Map every joined tracklet id to the id of its chain's earliest tracklet.

    Args:
        links: Joins as ``(earlier id, later id, cost)``.

    Returns:
        Tracklet id to chain id, for the tracklets in a join.

    Examples:
        >>> chain_ids([(5, 9, 0.1), (1, 5, 0.2), (2, 3, 0.1)])
        {5: 1, 9: 1, 1: 1, 2: 2, 3: 2}

    """
    prev = {b: a for a, b, _ in links}
    out = {}
    for a, b, _ in links:
        for t in (a, b):
            root = t
            while root in prev:
                root = prev[root]
            out[t] = root
    return out


def chain_numbers(tracklets: list[Tracklet], ids: dict[int, int], one_owner: bool) -> dict[int, str | None]:
    """Number of each final track: the number its tracklets carry, kept by one chain per (team, number) if asked.

    Args:
        tracklets: The shot's tracklets.
        ids: Tracklet id to chain id (:func:`chain_ids`); tracklets missing from it keep their own id.
        one_owner: Keep a (team, number) only on the chain with the most boxes.

    Returns:
        Final track id to its number or ``None``.

    Examples:
        >>> ts = [Tracklet(1, 0, 9, "home", number="5", rows=[(0, [])] * 3), Tracklet(2, 0, 9, "home", number="5")]
        >>> chain_numbers(ts, {}, one_owner=False), chain_numbers(ts, {}, one_owner=True)
        ({1: '5', 2: '5'}, {1: '5', 2: None})

    """
    out: dict[int, str | None] = {}
    boxes: Counter = Counter()
    team: dict[int, str] = {}
    for t in tracklets:
        cid = ids.get(t.tid, t.tid)
        out[cid] = out.get(cid) or t.number
        boxes[cid] += len(t.rows)
        team.setdefault(cid, t.team)
    if one_owner:
        owner: dict[tuple[str, str], int] = {}
        for cid in sorted(out, key=lambda k: (-boxes[k], k)):
            if out[cid] and owner.setdefault((team[cid], out[cid]), cid) != cid:
                out[cid] = None
    return out


@register("link", "greedy")
class GreedyLink(BaseBackend):
    """Link backend: cheapest-first joins of same-team tracklets by appearance, gap and court speed.

    Params (``defaults``): ``max_gap_s``, ``max_cost``, ``ends`` (boxes averaged at each end), ``max_speed_mps``
    (``None`` turns the court gate off), ``court_slack_m``, ``split_window``, ``split_cost`` (``None`` turns
    splitting off), ``one_owner`` (one chain per team and number; off by default), and ``roster`` with
    ``confirm_reads`` (confirm numbers again from roster reads only; off by default).

    Examples:
        >>> GreedyLink({"max_gap_s": 5.0}).run(seq, upstream, read)["links"]  # doctest: +SKIP
        [[12, 31, 0.08, 1.4], ...]

    """

    defaults: ClassVar[dict[str, Any]] = {
        "max_gap_s": 10.0,
        "max_cost": 0.3,
        "ends": 5,
        "max_speed_mps": 8.0,
        "court_slack_m": 2.0,
        "split_window": 5,
        "split_cost": None,
        "one_owner": False,
        "roster": False,
        "confirm_reads": 3,
    }

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Split the shot's tracklets where their appearance changes, join the pieces, and relabel the track rows.

        Args:
            seq: The sequence.
            upstream: ``track`` and ``team`` outputs, with ``appearance``, ``court`` and ``numbers`` where configured.
            read: Unused.

        Returns:
            The link layer output (``bga.link.base``).

        """
        stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
        lp = LinkParams(
            max_gap_s=self.p["max_gap_s"],
            max_cost=self.p["max_cost"],
            max_speed_mps=self.p["max_speed_mps"],
            court_slack_m=self.p["court_slack_m"],
            step_s=stride / seq.fps,
        )
        rosters = {side: roster_numbers(seq.meta, side) for side in ("home", "away")} if self.p["roster"] else None
        rule = NumberRule(rosters=rosters, confirm_reads=self.p["confirm_reads"])
        tracklets, cuts = build_tracklets(
            seq.frames, upstream, self.p["ends"], self.p["split_window"], self.p["split_cost"], rule
        )
        by_id = {t.tid: t for t in tracklets}
        links = greedy_links(tracklets, lp)
        ids = chain_ids(links)
        frames = {str(f): [] for f in seq.frames}
        for t in tracklets:
            for f, r in t.rows:
                frames[str(f)].append([ids.get(t.tid, t.tid), *r[1:]])
        return {
            "frames": frames,
            "splits": cuts,
            "links": [[a, b, c, round((by_id[b].first - by_id[a].last) * lp.step_s, 2)] for a, b, c in links],
            "ids": {str(k): v for k, v in ids.items()},
            "numbers": {str(k): v for k, v in chain_numbers(tracklets, ids, self.p["one_owner"]).items()},
        }
