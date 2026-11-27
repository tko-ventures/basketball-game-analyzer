"""Shared building blocks for layer 5b (identity).

Output per sequence:
    {"numbers": {"<track_id>": "23" | null}, "conflicts": [...], "clusters": {"<track_id>": "<cluster_id>"}}

``numbers`` maps every linked track id of the sequence to its confirmed jersey number or ``null``. ``conflicts``
lists co-visible or cross-shot number conflicts detected during assignment or clustering. ``clusters`` maps each
track id to a game-level player cluster id (for clip-scoped backends; omitted or trivial for shot-level backends).

"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence as Seq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from bga.numbers.base import NUMBER_CLASSES, UNREADABLE, confirm_consecutive, roster_numbers
from bga.stage import Sequence


@dataclass
class TrackEvidence:
    """Everything upstream layers know about one linked track in one sequence.

    Attributes:
        seq: Sequence name.
        tid: Linked track id.
        team: Majority team label (``"home"``, ``"away"``, ``"referee"`` or ``""``).
        boxes: Number of frames the track appears in.
        frames: Source frame indices where the track appears.
        reads: ``(frame, decoded_value)`` pairs from ``numbers["log"]`` mapped onto this track.
        probs: Crop probability rows aligned with ``reads`` (each over ``classes``), empty when the reader logged
            no probabilities.
        embedding: L2-normalised mean layer-3b embedding over the track's boxes, or ``None`` when absent.

    Examples:
        >>> ev = TrackEvidence(seq="c_s1", tid=1, team="home", boxes=5, frames={0, 5})
        >>> ev.team, ev.boxes
        ('home', 5)

    """

    seq: str
    tid: int
    team: str
    boxes: int
    frames: set[int] = field(default_factory=set)
    reads: list[tuple[int, str | None]] = field(default_factory=list)
    probs: list[np.ndarray] = field(default_factory=list)
    embedding: np.ndarray | None = None


def _embedding_lookup(appearance: dict | None) -> dict[tuple[int, int], np.ndarray]:
    """Map ``(frame, det_idx)`` to its L2-normalised appearance vector, or empty when absent."""
    if not appearance:
        return {}
    arrays = appearance.get("arrays") or {}
    emb = arrays.get("embeddings")
    boxes = appearance.get("boxes") or []
    if emb is None or len(boxes) == 0:
        return {}
    arr = np.asarray(emb, dtype=np.float32)
    return {(int(f), int(di)): arr[i] for i, (f, di) in enumerate(boxes)}


def _mean_unit_vector(vecs: list[np.ndarray]) -> np.ndarray | None:
    """L2-normalised mean of ``vecs``, or ``None`` when empty or zero."""
    if not vecs:
        return None
    mean = np.mean(vecs, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else None


def collect_track_evidence(seq: Sequence, upstream: dict[str, dict]) -> tuple[list[TrackEvidence], tuple[str, ...]]:
    """Join ``link``, ``track``, ``team``, ``numbers`` and optional ``appearance`` into one record per linked track.

    Args:
        seq: Sequence being processed.
        upstream: Upstream layer outputs for ``seq`` (with arrays loaded).

    Returns:
        Sorted list of :class:`TrackEvidence` (by track id) and the class tuple for ``probs``.

    Examples:
        >>> seq = Sequence("c_s1", "c", "v.mp4", 1, [0], 50.0)
        >>> ups = {
        ...     "link": {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}, "ids": {"1": 1}},
        ...     "track": {"frames": {"0": [[1, 0, 0, 10, 20, 0.9, "player", 0]]}},
        ...     "team": {"frames": {"0": ["home"]}},
        ...     "numbers": {"log": [[0, 1, "7"]], "tracks": {"1": None}},
        ... }
        >>> evs, _ = collect_track_evidence(seq, ups)
        >>> [(e.tid, e.team, e.reads) for e in evs]
        [(1, 'home', [(0, '7')])]

    """
    link_out, track_out, team_out = upstream["link"], upstream["track"], upstream["team"]
    num_out = upstream.get("numbers") or {}
    emb_by_box = _embedding_lookup(upstream.get("appearance"))

    raw_to_det = {(int(f_str), int(r[0])): int(r[7]) for f_str, rows in track_out["frames"].items() for r in rows}
    det_to_linked: dict[tuple[int, int], int] = {}
    frames_by: dict[int, set[int]] = defaultdict(set)
    boxes_by: Counter[int] = Counter()
    team_votes: dict[int, Counter[str]] = defaultdict(Counter)
    embs_by: dict[int, list[np.ndarray]] = defaultdict(list)

    for f_str, rows in link_out["frames"].items():
        f = int(f_str)
        tf = team_out["frames"].get(f_str, [])
        for r in rows:
            ltid, di = int(r[0]), int(r[7])
            frames_by[ltid].add(f)
            boxes_by[ltid] += 1
            if di >= 0:
                det_to_linked[(f, di)] = ltid
                if di < len(tf):
                    team_votes[ltid][tf[di]] += 1
                if (f, di) in emb_by_box:
                    embs_by[ltid].append(emb_by_box[(f, di)])

    id_remap = {int(k): int(v) for k, v in (link_out.get("ids") or {}).items()}
    reads_by: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    probs_by: dict[int, list[np.ndarray]] = defaultdict(list)
    probs_arr = (num_out.get("arrays") or {}).get("probs")
    classes = tuple(num_out.get("classes") or NUMBER_CLASSES)

    for i, entry in enumerate(num_out.get("log") or []):
        f, raw_tid, val = int(entry[0]), int(entry[1]), entry[2]
        di = raw_to_det.get((f, raw_tid), -1)
        ltid = det_to_linked.get((f, di)) if di >= 0 else None
        if ltid is None:
            ltid = id_remap.get(raw_tid, raw_tid)
        if ltid not in boxes_by:
            continue
        reads_by[ltid].append((f, val))
        if probs_arr is not None:
            probs_by[ltid].append(probs_arr[i])

    out = [
        TrackEvidence(
            seq=seq.name,
            tid=ltid,
            team=team_votes[ltid].most_common(1)[0][0] if team_votes[ltid] else "",
            boxes=boxes_by[ltid],
            frames=frames_by[ltid],
            reads=reads_by[ltid],
            probs=probs_by[ltid],
            embedding=_mean_unit_vector(embs_by[ltid]),
        )
        for ltid in sorted(boxes_by)
    ]
    return out, classes


def sequence_rosters(seq: Sequence, use_roster: bool = True) -> dict[str, set[str]]:
    """Return ``{"home": ..., "away": ...}`` roster sets for ``seq``, empty when ``use_roster`` is off.

    Args:
        seq: Sequence carrying clip metadata.
        use_roster: Whether to load roster numbers from ``data/rosters``.

    Returns:
        Mapping of team side to its set of valid jersey numbers.

    Examples:
        >>> seq = Sequence("c_s1", "c", "v.mp4", 1, [0], 50.0, meta={})
        >>> sequence_rosters(seq, use_roster=False)
        {'home': set(), 'away': set()}

    """
    if not use_roster:
        return {"home": set(), "away": set()}
    valid = set(NUMBER_CLASSES) - {UNREADABLE}
    return {side: roster_numbers(seq.meta, side) & valid for side in ("home", "away")}


def vote_consecutive(reads: Seq[tuple[int, str | None]], roster: set[str], n_consecutive: int = 3) -> str | None:
    """Confirm the first jersey number read ``n_consecutive`` times in a row, optionally filtered to ``roster``.

    Args:
        reads: ``(frame, value)`` pairs for one track.
        roster: Allowed jersey numbers; empty leaves all numbers allowed.
        n_consecutive: Consecutive equal reads required to confirm.

    Returns:
        The confirmed number, or ``None``.

    Examples:
        >>> vote_consecutive([(0, "8"), (5, "99"), (10, "8"), (15, "8")], {"8", "23"}, 3)
        '8'
        >>> vote_consecutive([(0, "8"), (5, "23"), (10, "8")], {"8", "23"}, 2) is None
        True

    """
    ordered = sorted(reads, key=lambda x: x[0])
    values = [v if (not roster or v in roster) else None for _, v in ordered]
    return confirm_consecutive(values, n_consecutive)


def roster_logprob_scores(
    probs: Seq[np.ndarray],
    roster: set[str],
    classes: tuple[str, ...] = NUMBER_CLASSES,
    skip_unreadable: bool = True,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Sum renormalised per-crop log-probabilities over ``roster`` plus :data:`UNREADABLE`.

    Each crop's distribution is restricted to ``roster U {UNREADABLE}`` (or ``0..99 U {UNREADABLE}`` when ``roster``
    is empty) and renormalised. When ``skip_unreadable`` is on, crops where :data:`UNREADABLE` is the most probable
    class (including ties) are dropped before summing.

    Args:
        probs: Per-crop probability vectors ordered by ``classes``.
        roster: Allowed jersey numbers; empty keeps all ``0..99``.
        classes: Class names matching columns of ``probs``.
        skip_unreadable: Drop crops whose renormalised argmax is :data:`UNREADABLE`.
        eps: Probability floor before taking ``log``.

    Returns:
        Mapping from each candidate (plus :data:`UNREADABLE`) to its summed log-probability, or ``{}`` when no
        legible crop remains.

    Examples:
        >>> cls = (UNREADABLE, "3", "8")
        >>> sc = roster_logprob_scores([np.array([0.1, 0.8, 0.1]), np.array([0.1, 0.7, 0.2])], {"3", "8"}, cls)
        >>> sc["3"] > sc["8"]
        True
        >>> roster_logprob_scores([np.array([0.8, 0.1, 0.1])], {"3", "8"}, cls)
        {}

    """
    if not probs:
        return {}
    cls_idx = {c: i for i, c in enumerate(classes)}
    cands = sorted(c for c in roster if c in cls_idx) if roster else [c for c in classes if c != UNREADABLE]
    if not cands or UNREADABLE not in cls_idx:
        return {}
    idxs = [cls_idx[UNREADABLE], *(cls_idx[c] for c in cands)]
    arr = np.asarray(probs, dtype=np.float64)[:, idxs]
    sums = arr.sum(axis=1, keepdims=True)
    valid = sums[:, 0] > 0
    arr = arr[valid] / sums[valid]
    if skip_unreadable:
        legible = (arr.argmax(axis=1) > 0) & (arr[:, 0] < arr.max(axis=1))
        arr = arr[legible]
    if len(arr) == 0:
        return {}
    logp = np.log(np.clip(arr, eps, 1.0)).sum(axis=0)
    return {UNREADABLE: float(logp[0]), **{c: float(logp[j]) for j, c in enumerate(cands, 1)}}


def number_margins(scores: dict[str, float], against_unreadable: bool = False) -> dict[str, float]:
    """Convert summed log-probabilities into per-number margins over the best competing candidate.

    For each candidate number ``c``, its margin is ``scores[c] - max_{k != c} scores[k]``. Only the single top
    candidate has a strictly positive margin; every runner-up has margin ``<= 0``.

    Args:
        scores: Output of :func:`roster_logprob_scores`.
        against_unreadable: Also include :data:`UNREADABLE` among the competing candidates.

    Returns:
        Mapping from each candidate number to its margin, or ``{}`` when ``scores`` is empty.

    Examples:
        >>> ms = number_margins({UNREADABLE: -4.0, "3": -0.5, "8": -3.0})
        >>> round(ms["3"], 1), round(ms["8"], 1)
        (2.5, -2.5)

    """
    if not scores:
        return {}
    nums = {c: v for c, v in scores.items() if c != UNREADABLE}
    u_score = scores.get(UNREADABLE)
    out: dict[str, float] = {}
    for c, v in nums.items():
        others = [ov for oc, ov in nums.items() if oc != c]
        if against_unreadable and u_score is not None:
            others.append(u_score)
        out[c] = v - (max(others) if others else 0.0)
    return out


def vote_roster_logprob(
    probs: Seq[np.ndarray],
    roster: set[str],
    margin: float = 8.0,
    classes: tuple[str, ...] = NUMBER_CLASSES,
    skip_unreadable: bool = True,
    against_unreadable: bool = False,
    eps: float = 1e-6,
) -> tuple[str | None, dict[str, float]]:
    """Score one track's crop probabilities over ``roster`` and confirm when the top margin reaches ``margin``.

    Args:
        probs: Per-crop probability vectors ordered by ``classes``.
        roster: Allowed jersey numbers; empty keeps all ``0..99``.
        margin: Minimum log-likelihood margin over the runner-up to confirm.
        classes: Class names matching columns of ``probs``.
        skip_unreadable: Drop crops where :data:`UNREADABLE` wins before summing.
        against_unreadable: Include :data:`UNREADABLE` in the margin comparison.
        eps: Probability floor before taking ``log``.

    Returns:
        ``(confirmed_number_or_None, margins_by_number)``.

    Examples:
        >>> cls = (UNREADABLE, "3", "8")
        >>> p = [np.array([0.05, 0.90, 0.05])] * 3
        >>> num, _ = vote_roster_logprob(p, {"3", "8"}, margin=4.0, classes=cls)
        >>> num
        '3'

    """
    scores = roster_logprob_scores(probs, roster, classes, skip_unreadable=skip_unreadable, eps=eps)
    margins = number_margins(scores, against_unreadable=against_unreadable)
    if not margins:
        return None, margins
    best_c, best_m = max(margins.items(), key=lambda x: (x[1], x[0]))
    return (best_c if best_m >= margin else None), margins


def assign_owner_rule(tracks: Seq[TrackEvidence], confirmed: dict[int, str | None]) -> dict[int, str | None]:
    """Longest-track-wins uniqueness rule: at most one track per ``(team, number)`` keeps the number.

    Args:
        tracks: Tracks in one sequence.
        confirmed: Track id to confirmed number (or ``None``).

    Returns:
        Copy of ``confirmed`` with duplicate ``(team, number)`` entries cleared on shorter tracks.

    Examples:
        >>> t1 = TrackEvidence("s", 1, "home", boxes=10)
        >>> t2 = TrackEvidence("s", 2, "home", boxes=20)
        >>> assign_owner_rule([t1, t2], {1: "7", 2: "7"})
        {1: None, 2: '7'}

    """
    out = dict(confirmed)
    by_id = {t.tid: t for t in tracks}
    owner: dict[tuple[str, str], int] = {}
    for tid in sorted(out, key=lambda k: (-by_id[k].boxes, k)):
        team, num = by_id[tid].team, out[tid]
        if num and team in ("home", "away") and owner.setdefault((team, num), tid) != tid:
            out[tid] = None
    return out


def assign_rows_lsa(
    row_ids: Seq[int],
    margins_by_row: dict[int, dict[str, float]],
    candidates: Seq[str],
    threshold: float,
    team: str = "",
) -> tuple[dict[int, str | None], list[dict[str, Any]]]:
    """Assign unique jersey numbers across competing rows with ``linear_sum_assignment`` and an abstain column.

    Each row has one column per candidate number (cost ``-margin``) plus its own ``no number`` abstain column at
    cost ``-threshold``. Because only a row's top candidate has positive margin, a row that loses its preferred
    number to a stronger competitor abstains rather than being silently reassigned, and the conflict is flagged.

    Args:
        row_ids: Ordered ids of the competing tracks or clusters.
        margins_by_row: Row id to its candidate margins (from :func:`number_margins`).
        candidates: Sorted candidate jersey numbers.
        threshold: Minimum margin to beat the ``no number`` abstain column.
        team: Team label included in conflict records.

    Returns:
        ``(assigned_number_by_row_id, conflicts)``.

    Examples:
        >>> m = {1: {"10": 50.0, "20": -50.0}, 2: {"10": 90.0, "20": -90.0}}
        >>> assigned, conflicts = assign_rows_lsa([1, 2], m, ["10", "20"], threshold=8.0, team="home")
        >>> assigned, len(conflicts), conflicts[0]["winner"]
        ({1: None, 2: '10'}, 1, 2)

    """
    out: dict[int, str | None] = dict.fromkeys(row_ids)
    if not row_ids or not candidates:
        return out, []
    n_r, n_c = len(row_ids), len(candidates)
    cost = np.full((n_r, n_c + n_r), 1e6, dtype=np.float64)
    preferred: dict[str, list[int]] = defaultdict(list)
    for ri, rid in enumerate(row_ids):
        ms = margins_by_row.get(rid) or {}
        for ci, c in enumerate(candidates):
            if c in ms:
                cost[ri, ci] = -ms[c]
        cost[ri, n_c + ri] = -threshold
        if ms:
            top_c, top_m = max(ms.items(), key=lambda x: (x[1], x[0]))
            if top_m >= threshold:
                preferred[top_c].append(rid)
    r_ind, c_ind = linear_sum_assignment(cost)
    for ri, ci in zip(r_ind, c_ind, strict=True):
        if ci < n_c:
            out[row_ids[ri]] = candidates[ci]
    conflicts = [
        {
            "team": team,
            "number": num,
            "tracks": tids,
            "winner": next((tid for tid in tids if out[tid] == num), None),
        }
        for num, tids in sorted(preferred.items())
        if len(tids) > 1
    ]
    return out, conflicts


def _covisible_components(tracks: Seq[TrackEvidence]) -> list[list[TrackEvidence]]:
    """Partition tracks into connected components of frame co-visibility."""
    adj: dict[int, set[int]] = {t.tid: set() for t in tracks}
    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            if a.frames & b.frames:
                adj[a.tid].add(b.tid)
                adj[b.tid].add(a.tid)
    by_id = {t.tid: t for t in tracks}
    seen: set[int] = set()
    comps: list[list[TrackEvidence]] = []
    for t in tracks:
        if t.tid in seen:
            continue
        comp, stack = [], [t.tid]
        seen.add(t.tid)
        while stack:
            u = stack.pop()
            comp.append(by_id[u])
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp, key=lambda x: x.tid))
    return comps


def assign_covisible(
    tracks: Seq[TrackEvidence],
    margins_by_id: dict[int, dict[str, float]],
    rosters: dict[str, set[str]],
    threshold: float = 8.0,
) -> tuple[dict[int, str | None], list[dict[str, Any]]]:
    """Assign jersey numbers per ``(team, co-visible component)`` using :func:`assign_rows_lsa`.

    Args:
        tracks: All tracks in one sequence.
        margins_by_id: Track id to candidate margins.
        rosters: ``{"home": ..., "away": ...}`` roster sets.
        threshold: Confirmation margin threshold for the abstain column.

    Returns:
        ``(assigned_number_by_tid, conflicts)``.

    Examples:
        >>> t1 = TrackEvidence("s", 1, "home", boxes=10, frames={0, 5})
        >>> t2 = TrackEvidence("s", 2, "home", boxes=5, frames={0, 5})
        >>> m = {1: {"10": 15.0, "20": -15.0}, 2: {"10": 40.0, "20": -40.0}}
        >>> nums, confs = assign_covisible([t1, t2], m, {"home": {"10", "20"}, "away": set()}, threshold=8.0)
        >>> nums, len(confs)
        ({1: None, 2: '10'}, 1)

    """
    out: dict[int, str | None] = {t.tid: None for t in tracks}
    conflicts: list[dict[str, Any]] = []
    all_nums = [c for c in NUMBER_CLASSES if c != UNREADABLE]
    for side in ("home", "away"):
        side_tracks = [t for t in tracks if t.team == side]
        cands = sorted(rosters.get(side) or all_nums)
        for comp in _covisible_components(side_tracks):
            assigned, comp_conflicts = assign_rows_lsa(
                [t.tid for t in comp], margins_by_id, cands, threshold, team=side
            )
            out.update(assigned)
            conflicts.extend(comp_conflicts)
    return out, conflicts


def _pair_distance_stats(embeddings: Seq[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per track pair: cosine distance, whether both tracks have an embedding, and the complete-linkage seed.

    The seed is the distance where both have an embedding and ``-inf`` where either lacks one.

    """
    n = len(embeddings)
    have = np.array([e is not None for e in embeddings])
    dist = np.zeros((n, n))
    if have.any():
        e = np.stack([emb for emb in embeddings if emb is not None]).astype(np.float32)
        idx = np.flatnonzero(have)
        dist[np.ix_(idx, idx)] = 1.0 - (e @ e.T).astype(np.float64)
    count = (have[:, None] & have[None, :]).astype(np.int64)
    return dist, count, np.where(count > 0, dist, -np.inf)


def _covisible(tracks: Seq[TrackEvidence]) -> np.ndarray:
    """Whether two tracks share a frame of the same shot (never joined)."""
    n = len(tracks)
    out = np.zeros((n, n), dtype=bool)
    by_seq: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tracks):
        by_seq[t.seq].append(i)
    for members in by_seq.values():
        for x, i in enumerate(members):
            for j in members[x + 1 :]:
                if tracks[i].frames & tracks[j].frames:
                    out[i, j] = out[j, i] = True
    return out


def cluster_team_tracks(
    tracks: Seq[TrackEvidence],
    confirmed: Seq[str | None],
    max_dist: float = 0.18,
    linkage: str = "average",
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Agglomeratively cluster one team's tracks across a game by mean appearance embedding.

    Two hard constraints are enforced on every merge:
    1. Tracklets co-visible in any frame of the same shot are never joined.
    2. Tracklets with conflicting confirmed jersey numbers are never joined (kept split and logged).

    Each step merges the closest allowed pair (the first in cluster order on a tie) into the earlier cluster. The
    cluster distances are kept as matrices updated on each merge (sums and counts of the pairwise distances for
    average linkage, maxima for complete), so a full game's thousands of tracks cluster in seconds.

    Args:
        tracks: One team's tracks across all shots of one game.
        confirmed: Single-track confirmed number (or ``None``) aligned with ``tracks``.
        max_dist: Maximum cosine distance (``1 - cos``) allowed for a merge.
        linkage: ``"average"`` or ``"complete"``.

    Returns:
        ``(clusters, split_conflicts)`` where each cluster is a list of indices into ``tracks``.

    Examples:
        >>> e = np.array([1.0, 0.0], np.float32)
        >>> a = TrackEvidence("s1", 1, "home", boxes=10, frames={0}, embedding=e)
        >>> b = TrackEvidence("s2", 2, "home", boxes=10, frames={0}, embedding=e)
        >>> c = TrackEvidence("s2", 3, "home", boxes=10, frames={0}, embedding=e)
        >>> clusters, _ = cluster_team_tracks([a, b, c], ["7", None, None], max_dist=0.2)
        >>> clusters
        [[0, 1], [2]]

    """
    n = len(tracks)
    members = [[i] for i in range(n)]
    splits: list[dict[str, Any]] = []
    if n < 2:
        return members, splits
    dist, count, peak = _pair_distance_stats([t.embedding for t in tracks])
    total = dist * count
    blocked = _covisible(tracks)
    codes = {num: k for k, num in enumerate(sorted({c for c in confirmed if c is not None}))}
    number = np.array([codes[c] if c is not None else -1 for c in confirmed])
    alive = np.ones(n, dtype=bool)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    seen_splits: set[tuple[tuple[str, int], ...]] = set()
    while True:
        with np.errstate(invalid="ignore", divide="ignore"):
            d = peak if linkage == "complete" else total / count
        near = upper & alive[:, None] & alive[None, :] & (count > 0) & (d <= max_dist) & ~blocked
        conflict = (number[:, None] >= 0) & (number[None, :] >= 0) & (number[:, None] != number[None, :])
        for a, b in zip(*np.nonzero(near & conflict), strict=True):
            key = tuple(sorted((tracks[i].seq, tracks[i].tid) for i in members[a] + members[b]))
            if key not in seen_splits:
                seen_splits.add(key)
                nums = sorted({confirmed[i] for i in members[a] + members[b] if confirmed[i]})
                splits.append({
                    "kind": "cluster_split_conflict",
                    "dist": round(float(d[a, b]), 4),
                    "numbers": nums,
                    "tracks": [list(k) for k in key],
                })
        allowed = near & ~conflict
        if not allowed.any():
            break
        a, b = np.unravel_index(int(np.argmin(np.where(allowed, d, np.inf))), d.shape)
        members[a] = members[a] + members[b]
        alive[b] = False
        total[a, :] += total[b, :]
        total[:, a] = total[a, :]
        count[a, :] += count[b, :]
        count[:, a] = count[a, :]
        peak[a, :] = np.maximum(peak[a, :], peak[b, :])
        peak[:, a] = peak[a, :]
        blocked[a, :] |= blocked[b, :]
        blocked[:, a] = blocked[a, :]
        if number[a] < 0:
            number[a] = number[b]
    return [members[i] for i in np.flatnonzero(alive)], splits
