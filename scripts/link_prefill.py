#!/usr/bin/env python3
"""Propose merges of prefill tracklets that are one person cut by detector flicker (labelling aid).

    uv run scripts/link_prefill.py runs/<clip>/prefill/<model>/tracks.json > <id>.link.ops.txt

Within a shot, tracklet B continues tracklet A when B starts at most ``--max-gap`` sampled frames after
A ends, both are referees or both are not, B's first foot point lies near A's last foot point
extrapolated with A's recent velocity (distance in box heights), and the box heights agree. Candidate
links are taken one-to-one, cheapest first. The output is a review_ops.py file of ``merge`` lines,
which the implementer checks on filmstrip sheets before anything reaches the gold set. It never
merges tracklets that overlap in time.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ID_OFFSET = 1000  # as in import_prefill.py


def foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def tracklets_by_shot(tracks: dict) -> dict[int, dict[int, list]]:
    shots: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for fr in tracks["frames"]:
        for t in fr["tracks"]:
            shots[fr["shot"]][t["track_id"]].append((fr["frame_idx"], t["xyxy"], t.get("team") == "ref"))
    return shots


def link_shot(trs: dict[int, list], stride: int, max_gap: int, max_dist: float, max_size: float):
    info = {}
    for tid, obs in trs.items():
        obs.sort()
        (f0, b0, ref), (f1, b1, _) = obs[0], obs[-1]
        k = obs[-min(4, len(obs))]
        steps = max(1, (f1 - k[0]) // stride)
        v = ((foot(b1)[0] - foot(k[1])[0]) / steps, (foot(b1)[1] - foot(k[1])[1]) / steps)
        info[tid] = {"start": f0, "end": f1, "first": b0, "last": b1, "ref": ref, "v": v}
    cands = []
    for a, A in info.items():
        for b, B in info.items():
            gap = (B["start"] - A["end"]) // stride
            if a == b or gap < 1 or gap > max_gap or A["ref"] != B["ref"]:
                continue
            ha, hb = A["last"][3] - A["last"][1], B["first"][3] - B["first"][1]
            size = abs(math.log(max(hb, 1) / max(ha, 1)))
            px = foot(A["last"])[0] + A["v"][0] * gap
            py = foot(A["last"])[1] + A["v"][1] * gap
            dist = math.hypot(px - foot(B["first"])[0], py - foot(B["first"])[1]) / ((ha + hb) / 2)
            if dist <= max_dist * (1 + 0.15 * (gap - 1)) and size <= max_size:
                cands.append((dist + size + 0.05 * gap, a, b))
    succ, pred = {}, {}
    for _, a, b in sorted(cands):
        if a not in succ and b not in pred:
            succ[a], pred[b] = b, a
    root = {}
    for tid in sorted(info, key=lambda t: info[t]["start"]):
        root[tid] = root[pred[tid]] if tid in pred else tid
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-gap", type=int, default=10, help="sampled frames (10 = 1 s at 50 fps, stride 5)")
    ap.add_argument("--max-dist", type=float, default=0.5, help="foot-point distance in box heights")
    ap.add_argument("--max-size", type=float, default=0.35, help="|log height ratio|")
    a = ap.parse_args(argv)
    tracks = json.loads(a.tracks.read_text())
    print(f"clip {Path(tracks['clip_path']).stem}")
    print(f"# link_prefill.py {a.tracks} max_gap={a.max_gap} max_dist={a.max_dist} max_size={a.max_size}")
    for shot, trs in sorted(tracklets_by_shot(tracks).items()):
        root = link_shot(trs, a.stride, a.max_gap, a.max_dist, a.max_size)
        merges = [(t, r) for t, r in root.items() if t != r]
        print(f"# shot {shot}: {len(trs)} tracklets -> {len(set(root.values()))} after {len(merges)} merges")
        for t, r in merges:
            print(f"merge {ID_OFFSET + t} {ID_OFFSET + r}")


if __name__ == "__main__":
    main()
