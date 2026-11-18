"""Gold-set export (plan section 8): one MOTChallenge sequence per gold shot, sampled every `stride` frames.

A sequence's frame k (1-based) is source frame ``start + (k - 1) * stride`` in the shot's own frame
grid (first labelled frame of the shot); ``frames.txt`` lists the source frame of every k, so a tracker
run on the full clip is scored on exactly these frames. Short gaps inside an identity are filled by
linear interpolation and written with visibility 0.5; longer gaps stay gaps.

"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

#: MOTChallenge class per labeller team value; anything else is 4 (other), as the labeller writes.
MOT_CLASS = {"home": 1, "away": 2, "ref": 3}


@dataclass
class Sequence:
    """One gold shot as a MOTChallenge sequence, ready for :func:`write_sequence`.

    Attributes:
        name: Sequence name, ``<youtube_id>_s<shot>``.
        clip: Clip the shot belongs to.
        shot: Shot index within the clip.
        held_out: Whether the clip's game is the held-out one.
        frames: Source frame of sequence frame k at index k - 1.
        rows: MOT rows ``(k, tid, x, y, w, h, 1, class, visibility)``, sorted.
        identities: Labeller identity record per track id used in ``rows``.

    """

    name: str
    clip: str
    shot: int
    held_out: bool
    frames: list[int]  # source frame of sequence frame k = index + 1
    rows: list[tuple]  # (k, tid, x, y, w, h, 1, class, visibility)
    identities: dict[str, dict]


def interpolate(obs: dict[int, dict], grid: list[int], max_gap: int) -> dict[int, tuple[dict, bool]]:
    """Fill gaps of up to ``max_gap`` missing grid frames between two observations of one identity.

    Args:
        obs: Labelled box (``x1``, ``y1``, ``x2``, ``y2``) per source frame.
        grid: The sequence's source frames, in order.
        max_gap: Longest run of missing grid frames that is filled; longer gaps stay gaps.

    Returns:
        ``(box, filled)`` per frame: the observations with ``filled`` False and the linear fills with True.

    Examples:
        >>> box = lambda x: {"x1": x, "y1": 0, "x2": x + 10, "y2": 20}
        >>> out = interpolate({0: box(0), 10: box(20)}, [0, 5, 10], max_gap=1)
        >>> out[5]
        ({'x1': 10.0, 'y1': 0.0, 'x2': 20.0, 'y2': 20.0}, True)
        >>> sorted(interpolate({0: box(0), 10: box(20)}, [0, 5, 10], max_gap=0))
        [0, 10]

    """
    out = {f: (b, False) for f, b in obs.items()}
    have = [i for i, f in enumerate(grid) if f in obs]
    for i0, i1 in itertools.pairwise(have):
        if 1 < i1 - i0 <= max_gap + 1:
            a, b = obs[grid[i0]], obs[grid[i1]]
            for i in range(i0 + 1, i1):
                t = (i - i0) / (i1 - i0)
                box = {c: a[c] + (b[c] - a[c]) * t for c in ("x1", "y1", "x2", "y2")}
                out[grid[i]] = (box, True)
    return out


def build_sequence(labels: dict, game: dict, shot: dict, stride: int, max_gap: int) -> Sequence:
    """Build the MOT sequence of one gold shot from the labeller's file.

    The grid starts at the shot's first labelled frame and steps by ``stride`` to the shot's end; occluded and
    interpolated boxes get visibility 0.5.

    Args:
        labels: The clip's labeller JSON (``frames``, ``identities``).
        game: The gold game entry (``youtube_id``, ``clip``, ``held_out``).
        shot: The shot (``shot``, ``start``, ``end`` inclusive).
        stride: Grid step in source frames.
        max_gap: Passed to :func:`interpolate`.

    Returns:
        The sequence with rows and the identities they use.

    Examples:
        >>> labels = {
        ...     "frames": {"100": {"boxes": [{"tid": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 20}]}},
        ...     "identities": {"1": {"team": "home"}},
        ... }
        >>> game = {"youtube_id": "abc", "clip": "c", "held_out": False}
        >>> seq = build_sequence(labels, game, {"shot": 3, "start": 98, "end": 106}, stride=5, max_gap=3)
        >>> seq.name, seq.frames, seq.rows
        ('abc_s3', [100, 105], [(1, 1, 0, 0, 10, 20, 1, 1, 1.0)])

    """
    lo, hi = shot["start"], shot["end"]
    labelled = sorted(int(k) for k in labels["frames"] if lo <= int(k) <= hi)
    first = labelled[0] if labelled else lo
    grid = list(range(first, hi + 1, stride))
    per_id: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in grid:
        for b in labels["frames"].get(str(f), {}).get("boxes", []):
            per_id[str(b["tid"])][f] = b
    ids = labels.get("identities", {})
    rows = []
    for tid, obs in per_id.items():
        team = ids.get(tid, {}).get("team", "")
        for f, (b, filled) in interpolate(obs, grid, max_gap).items():
            k = grid.index(f) + 1
            vis = 0.5 if filled or b.get("occluded") else 1.0
            x, y = round(b["x1"], 1), round(b["y1"], 1)
            w, h = round(b["x2"] - b["x1"], 1), round(b["y2"] - b["y1"], 1)
            rows.append((k, int(tid), x, y, w, h, 1, MOT_CLASS.get(team, 4), vis))
    rows.sort()
    name = f"{game['youtube_id']}_s{shot['shot']}"
    used = {str(r[1]) for r in rows}
    return Sequence(name, game["clip"], shot["shot"], game["held_out"], grid, rows, {t: ids.get(t, {}) for t in used})


def write_sequence(seq: Sequence, out: Path, meta: dict, fps: float, stride: int, wh: tuple[int, int]) -> None:
    """Write a sequence in MOTChallenge layout under ``out/<name>``.

    Files: ``gt/gt.txt``, ``frames.txt`` (source frame per sequence frame), ``seqinfo.ini`` and ``identities.csv``.

    Args:
        seq: The sequence to write.
        out: Gold version directory.
        meta: Clip meta; ``meta[side]["team"]`` names the home and away teams in ``identities.csv``.
        fps: Source frame rate of the clip.
        stride: Grid step, so the sequence frame rate is ``fps / stride``.
        wh: Frame width and height in pixels.

    Examples:
        >>> import tempfile
        >>> seq = Sequence("abc_s3", "c", 3, False, [100, 105], [(1, 1, 0, 0, 10, 20, 1, 1, 1.0)], {"1": {}})
        >>> with tempfile.TemporaryDirectory() as d:
        ...     write_sequence(seq, Path(d), {}, 50.0, 5, (1920, 1080))
        ...     (Path(d) / "abc_s3" / "gt" / "gt.txt").read_text().splitlines()
        ['1,1,0,0,10,20,1,1,1.0']

    """
    d = out / seq.name
    (d / "gt").mkdir(parents=True, exist_ok=True)
    (d / "gt" / "gt.txt").write_text("".join(",".join(str(v) for v in r) + "\n" for r in seq.rows))
    (d / "frames.txt").write_text("".join(f"{f}\n" for f in seq.frames))
    (d / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={seq.name}\nclip={seq.clip}\nshot={seq.shot}\nheld_out={str(seq.held_out).lower()}\n"
        f"frameRate={fps / stride:g}\nsourceFrameRate={fps:g}\nstride={stride}\n"
        f"seqLength={len(seq.frames)}\nimWidth={wh[0]}\nimHeight={wh[1]}\n"
    )
    with (d / "identities.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "team", "number", "name", "team_name", "note"])
        for tid, ident in sorted(seq.identities.items(), key=lambda kv: int(kv[0])):
            team = ident.get("team", "")
            team_name = (meta.get(team) or {}).get("team", "") if team in ("home", "away") else ""
            w.writerow([tid, team, ident.get("number", ""), ident.get("name", ""), team_name, ident.get("note", "")])


def load_manifest(path: Path) -> dict:
    """Read a gold version's ``manifest.json``.

    Args:
        path: Path of the manifest.

    Returns:
        The parsed manifest.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "manifest.json").write_text('{"stride": 5}')
        ...     load_manifest(Path(d) / "manifest.json")
        {'stride': 5}

    """
    return json.loads(path.read_text())
