"""Gold-set export (plan section 8): one MOTChallenge sequence per gold shot, sampled every `stride` frames.

A sequence's frame k (1-based) is source frame ``start + (k - 1) * stride`` in the shot's own frame
grid (first labelled frame of the shot); ``frames.txt`` lists the source frame of every k, so a tracker
run on the full clip is scored on exactly these frames. Short gaps inside an identity are filled by
linear interpolation and written with visibility 0.5; longer gaps stay gaps.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MOT_CLASS = {"home": 1, "away": 2, "ref": 3}  # anything else is 4 (other); as the labeller writes


@dataclass
class Sequence:
    name: str
    clip: str
    shot: int
    held_out: bool
    frames: list[int]  # source frame of sequence frame k = index + 1
    rows: list[tuple]  # (k, tid, x, y, w, h, 1, class, visibility)
    identities: dict[str, dict]


def interpolate(obs: dict[int, dict], grid: list[int], max_gap: int) -> dict[int, tuple[dict, bool]]:
    """Fill gaps of up to ``max_gap`` missing grid frames between two observations of one identity."""
    out = {f: (b, False) for f, b in obs.items()}
    have = [i for i, f in enumerate(grid) if f in obs]
    for i0, i1 in zip(have, have[1:], strict=False):
        if 1 < i1 - i0 <= max_gap + 1:
            a, b = obs[grid[i0]], obs[grid[i1]]
            for i in range(i0 + 1, i1):
                t = (i - i0) / (i1 - i0)
                box = {c: a[c] + (b[c] - a[c]) * t for c in ("x1", "y1", "x2", "y2")}
                out[grid[i]] = (box, True)
    return out


def build_sequence(labels: dict, game: dict, shot: dict, stride: int, max_gap: int) -> Sequence:
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
    return Sequence(
        name, game["clip"], shot["shot"], game["held_out"], grid, rows, {t: ids.get(t, {}) for t in used}
    )


def write_sequence(
    seq: Sequence, out: Path, meta: dict, fps: float, stride: int, wh: tuple[int, int]
) -> None:
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
            w.writerow(
                [tid, team, ident.get("number", ""), ident.get("name", ""), team_name, ident.get("note", "")]
            )


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())
