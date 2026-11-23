"""Shot-level identity backends: ``consecutive``, ``roster_logprob``, and ``covisible_assign``."""

from __future__ import annotations

from typing import Any, ClassVar

from bga.identity.base import (
    assign_covisible,
    assign_owner_rule,
    collect_track_evidence,
    sequence_rosters,
    vote_consecutive,
    vote_roster_logprob,
)
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("identity", "consecutive")
class ConsecutiveIdentity(BaseBackend):
    """Confirm each linked track's number by ``n_consecutive`` equal reads in a row, optionally filtered to roster.

    Examples:
        >>> ConsecutiveIdentity({}).p["n_consecutive"]
        3

    """

    defaults: ClassVar[dict[str, Any]] = {
        "n_consecutive": 3,
        "roster": True,
        "assign": "none",  # "none" or "owner_rule"
    }

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Vote on every linked track in ``seq`` with the consecutive-reads rule.

        Args:
            seq: Sequence to score.
            upstream: ``link``, ``track``, ``team`` and ``numbers`` outputs.
            read: Unused frame reader.

        Returns:
            ``{"numbers": {"<tid>": "23" | None}, "conflicts": []}``.

        """
        del read
        tracks, _ = collect_track_evidence(seq, upstream)
        rosters = sequence_rosters(seq, use_roster=self.p["roster"])
        confirmed = {
            t.tid: vote_consecutive(t.reads, rosters.get(t.team, set()), self.p["n_consecutive"]) for t in tracks
        }
        if self.p["assign"] == "owner_rule":
            confirmed = assign_owner_rule(tracks, confirmed)
        return {"numbers": {str(tid): val for tid, val in sorted(confirmed.items())}, "conflicts": []}


@register("identity", "roster_logprob")
class RosterLogprobIdentity(BaseBackend):
    """Confirm each linked track's number by summed log-probabilities over the team's roster.

    Examples:
        >>> RosterLogprobIdentity({}).p["margin"]
        8.0

    """

    defaults: ClassVar[dict[str, Any]] = {
        "roster": True,
        "margin": 8.0,
        "skip_unreadable": True,
        "against_unreadable": False,
        "eps": 1e-6,
        "assign": "none",  # "none" or "owner_rule"
    }

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Vote on every linked track in ``seq`` by summing renormalised crop log-probabilities.

        Args:
            seq: Sequence to score.
            upstream: ``link``, ``track``, ``team`` and ``numbers`` (with ``probs``) outputs.
            read: Unused frame reader.

        Returns:
            ``{"numbers": {"<tid>": "23" | None}, "conflicts": []}``.

        """
        del read
        tracks, classes = collect_track_evidence(seq, upstream)
        rosters = sequence_rosters(seq, use_roster=self.p["roster"])
        confirmed: dict[int, str | None] = {}
        for t in tracks:
            num, _ = vote_roster_logprob(
                t.probs,
                rosters.get(t.team, set()),
                margin=self.p["margin"],
                classes=classes,
                skip_unreadable=self.p["skip_unreadable"],
                against_unreadable=self.p["against_unreadable"],
                eps=self.p["eps"],
            )
            confirmed[t.tid] = num
        if self.p["assign"] == "owner_rule":
            confirmed = assign_owner_rule(tracks, confirmed)
        return {"numbers": {str(tid): val for tid, val in sorted(confirmed.items())}, "conflicts": []}


@register("identity", "covisible_assign")
class CovisibleAssignIdentity(BaseBackend):
    """Score tracks by roster log-probability and assign unique numbers to co-visible teammates via LSA.

    Examples:
        >>> CovisibleAssignIdentity({}).p["margin"]
        8.0

    """

    defaults: ClassVar[dict[str, Any]] = {
        "roster": True,
        "margin": 8.0,
        "skip_unreadable": True,
        "against_unreadable": False,
        "eps": 1e-6,
    }

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Score tracks with :func:`vote_roster_logprob` and assign co-visible teammates via :func:`assign_covisible`.

        Args:
            seq: Sequence to score.
            upstream: ``link``, ``track``, ``team`` and ``numbers`` (with ``probs``) outputs.
            read: Unused frame reader.

        Returns:
            ``{"numbers": {"<tid>": "23" | None}, "conflicts": [...]}``.

        """
        del read
        tracks, classes = collect_track_evidence(seq, upstream)
        rosters = sequence_rosters(seq, use_roster=self.p["roster"])
        margins_by_id: dict[int, dict[str, float]] = {}
        for t in tracks:
            _, ms = vote_roster_logprob(
                t.probs,
                rosters.get(t.team, set()),
                margin=self.p["margin"],
                classes=classes,
                skip_unreadable=self.p["skip_unreadable"],
                against_unreadable=self.p["against_unreadable"],
                eps=self.p["eps"],
            )
            margins_by_id[t.tid] = ms
        assigned, conflicts = assign_covisible(tracks, margins_by_id, rosters, threshold=self.p["margin"])
        return {"numbers": {str(tid): val for tid, val in sorted(assigned.items())}, "conflicts": conflicts}
