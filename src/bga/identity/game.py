"""Game-level identity backend (``identity/game``): cross-shot clustering with pooled roster log-prob voting.

Runs once per clip (``scope = "clip"``) over all selected shots of one game:
1. Clusters each team's linked tracklets by mean layer-3b appearance embedding, never joining tracklets co-visible
   in the same frame and never joining tracklets with conflicting confirmed single-track numbers.
2. Pools all crop probability distributions inside each cluster and scores them with ``roster_logprob``.
3. Assigns jersey numbers per ``(team, shot)`` over the clusters present in that shot via ``linear_sum_assignment``
   with a ``no number`` abstain column at ``margin``.

"""

from __future__ import annotations

from typing import Any, ClassVar

from bga.identity.base import (
    TrackEvidence,
    assign_rows_lsa,
    cluster_team_tracks,
    collect_track_evidence,
    sequence_rosters,
    vote_roster_logprob,
)
from bga.numbers.base import NUMBER_CLASSES, UNREADABLE
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("identity", "game")
class GameIdentity(BaseBackend):
    """Clip-scoped game-level identity: appearance clustering across shots + pooled roster log-prob + LSA.

    Examples:
        >>> GameIdentity({}).scope, GameIdentity({}).p["max_dist"]
        ('clip', 0.18)

    """

    scope: ClassVar[str] = "clip"
    defaults: ClassVar[dict[str, Any]] = {
        "roster": True,
        "margin": 8.0,
        "max_dist": 0.18,
        "linkage": "average",
        "skip_unreadable": True,
        "against_unreadable": False,
        "eps": 1e-6,
    }

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Fallback for single-sequence calls: delegates to :meth:`run_clip` on ``[seq]``.

        Args:
            seq: Single sequence.
            upstream: Upstream outputs for ``seq``.
            read: Unused frame reader.

        Returns:
            The sequence's identity output.

        """
        del read
        return self.run_clip([seq], {seq.name: upstream})[seq.name]

    def _vote(self, probs: list, roster: set[str], classes: tuple[str, ...]) -> tuple[str | None, dict[str, float]]:
        """Run :func:`vote_roster_logprob` with this backend's parameters."""
        return vote_roster_logprob(
            probs,
            roster,
            margin=self.p["margin"],
            classes=classes,
            skip_unreadable=self.p["skip_unreadable"],
            against_unreadable=self.p["against_unreadable"],
            eps=self.p["eps"],
        )

    def _assign_team(
        self,
        side: str,
        side_tracks: list[TrackEvidence],
        roster: set[str],
        classes: tuple[str, ...],
        seqs: list[Sequence],
        out: dict[str, dict[str, Any]],
    ) -> None:
        """Cluster one team's tracks across ``seqs``, pool probabilities, and assign numbers per shot."""
        if not side_tracks:
            return
        confirmed = [self._vote(t.probs, roster, classes)[0] for t in side_tracks]
        clusters, splits = cluster_team_tracks(
            side_tracks, confirmed, max_dist=self.p["max_dist"], linkage=self.p["linkage"]
        )
        cluster_margins: dict[int, dict[str, float]] = {}
        for cid, members in enumerate(clusters):
            pooled = [p for i in members for p in side_tracks[i].probs]
            _, cluster_margins[cid] = self._vote(pooled, roster, classes)
            label = f"{side}_{cid}"
            for i in members:
                out[side_tracks[i].seq]["clusters"][str(side_tracks[i].tid)] = label
        cands = sorted(roster) if roster else [c for c in classes if c != UNREADABLE]
        for seq in seqs:
            present = [cid for cid, ms in enumerate(clusters) if any(side_tracks[i].seq == seq.name for i in ms)]
            if not present:
                continue
            assigned, conflicts = assign_rows_lsa(present, cluster_margins, cands, self.p["margin"], team=side)
            out[seq.name]["conflicts"].extend(conflicts)
            for cid in present:
                num = assigned[cid]
                for i in clusters[cid]:
                    if side_tracks[i].seq == seq.name:
                        out[seq.name]["numbers"][str(side_tracks[i].tid)] = num
        for sp in splits:
            for seq_name in dict.fromkeys(name for name, _ in sp["tracks"]):
                if seq_name in out:
                    out[seq_name]["conflicts"].append(sp)

    def run_clip(self, seqs: list[Sequence], upstreams: dict[str, dict[str, dict]]) -> dict[str, dict]:
        """Cluster tracks across every shot of one clip and assign pooled jersey numbers per shot.

        Args:
            seqs: All selected sequences belonging to one clip/game.
            upstreams: Sequence name to its upstream outputs (``link``, ``track``, ``team``, ``numbers`` and
                optional ``appearance``).

        Returns:
            Sequence name to ``{"numbers", "clusters", "conflicts"}``.

        """
        all_tracks: list[TrackEvidence] = []
        classes: tuple[str, ...] = NUMBER_CLASSES
        out: dict[str, dict[str, Any]] = {}
        for seq in seqs:
            evs, classes = collect_track_evidence(seq, upstreams[seq.name])
            all_tracks.extend(evs)
            out[seq.name] = {
                "numbers": {str(t.tid): None for t in evs},
                "clusters": {},
                "conflicts": [],
            }
        rosters = sequence_rosters(seqs[0], use_roster=self.p["roster"]) if seqs else {"home": set(), "away": set()}
        for side in ("home", "away"):
            side_tracks = [t for t in all_tracks if t.team == side]
            self._assign_team(side, side_tracks, rosters.get(side, set()), classes, seqs, out)
        return out
