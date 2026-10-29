"""Tracker-health (churn) metrics, from the prior repo's ``export/churn.py``.

Ground-truth-free: computed from the per-frame sets of active track ids, so they can be reported
next to HOTA and IDF1 and on clips without gold labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean, median

DEFAULT_SHORT_LIVED_FRAMES = 25  # 0.5 s at 50 fps


def _ids(frame_ids: Iterable[int], ignore: frozenset[int]) -> set[int]:
    return {int(t) for t in frame_ids if int(t) not in ignore}


def id_set_change_events(per_frame_ids: Sequence[Iterable[int]], ignore: Iterable[int] = (-1,)) -> int:
    """Number of frames whose active id set differs from the previous frame's."""
    skip = frozenset(int(x) for x in ignore)
    events, prev = 0, None
    for frame_ids in per_frame_ids:
        cur = _ids(frame_ids, skip)
        if prev is not None and cur != prev:
            events += 1
        prev = cur
    return events


def track_lifetimes(per_frame_ids: Sequence[Iterable[int]], ignore: Iterable[int] = (-1,)) -> dict[int, int]:
    """Span (last - first + 1) per track id, in positions of ``per_frame_ids``."""
    skip = frozenset(int(x) for x in ignore)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, frame_ids in enumerate(per_frame_ids):
        for t in _ids(frame_ids, skip):
            first.setdefault(t, i)
            last[t] = i
    return {t: last[t] - first[t] + 1 for t in first}


def compute_churn(
    per_frame_ids: Sequence[Iterable[int]],
    short_lived_frames: int = DEFAULT_SHORT_LIVED_FRAMES,
    ignore: Iterable[int] = (-1,),
) -> dict[str, float | int]:
    per_frame_ids = list(per_frame_ids)
    t = max(1, int(short_lived_frames))
    n = len(per_frame_ids)
    lives = list(track_lifetimes(per_frame_ids, ignore).values())
    unique = len(lives)
    events = id_set_change_events(per_frame_ids, ignore)
    short = sum(1 for v in lives if v < t)
    return {
        "unique_track_ids": unique,
        "id_set_change_events": events,
        "id_set_delta_rate": events / max(n - 1, 1),
        "short_lived_id_count": short,
        "short_lived_id_fraction": short / unique if unique else 0.0,
        "mean_track_survival_frames": float(mean(lives)) if lives else 0.0,
        "median_track_survival_frames": float(median(lives)) if lives else 0.0,
        "short_lived_threshold_frames": t,
        "frame_count": n,
    }


def format_churn(c: dict) -> str:
    return (
        f"churn unique={c['unique_track_ids']} setΔ={c['id_set_change_events']} "
        f"setΔ_rate={c['id_set_delta_rate']:.4f} "
        f"short_lived={c['short_lived_id_count']}/{c['unique_track_ids']}"
        f"({c['short_lived_id_fraction']:.3f}) "
        f"mean_surv={c['mean_track_survival_frames']:.1f} med_surv={c['median_track_survival_frames']:.1f} "
        f"T={c['short_lived_threshold_frames']} frames={c['frame_count']}"
    )
