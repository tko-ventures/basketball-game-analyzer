#!/usr/bin/env python3
"""Fuse SAM 2.1 identities with detector boxes into a labeller tracks.json (labelling aid).

    uv run scripts/sam_to_tracks.py runs/<clip>/prefill/sam21_hiera_large

Per sampled frame, SAM objects and the detector's person boxes (confidence >= --det-conf, no wider than
tall) are matched one-to-one by IoU, best first. A matched object takes the detector's box, so gold
geometry is the detector's, corrected by review; an object with a mask but no detector box keeps
its mask box and is marked ``src: sam``, but only inside a gap of at most --max-gap sampled frames
between matched boxes of the same track. Two SAM objects whose boxes overlap (IoU > --dup-iou) on
most frames where both exist are one person prompted twice and are merged. A track is a referee
when most of its matched detector boxes are referee boxes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PERSON = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def duplicates(frames: dict[int, dict[int, list]], dup_iou: float, min_share: float) -> dict[int, int]:
    """Map later object id -> earlier id for objects that track the same person."""
    both, close = Counter(), Counter()
    for objs in frames.values():
        ids = sorted(o for o, b in objs.items() if b)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                both[(a, b)] += 1
                if iou(objs[a], objs[b]) > dup_iou:
                    close[(a, b)] += 1
    parent: dict[int, int] = {}

    def find(x):
        while x in parent:
            x = parent[x]
        return x

    for (a, b), n in sorted(both.items()):
        if n >= 3 and close[(a, b)] / n >= min_share and find(a) != find(b):
            parent[find(b)] = find(a)
    return {o: find(o) for o in {o for objs in frames.values() for o in objs}}


def fuse(
    res: dict, dets: dict[int, list], det_conf: float, match_iou: float, dup_iou: float, max_gap: int = 3
) -> list[dict]:
    s = 1 / res["scale"]
    frames = {
        int(k): {int(o): ([v * s for v in b[:4]] if b else None) for o, b in objs.items()}
        for k, objs in res["frames"].items()
    }
    root = duplicates(frames, dup_iou, 0.8)
    out, ref_votes = [], defaultdict(Counter)
    for local, objs in sorted(frames.items()):
        g = res["start"] + local
        boxes = {}
        for o, b in objs.items():
            r = root[o]
            if b and (r not in boxes or (boxes[r][2] - boxes[r][0]) < (b[2] - b[0])):
                boxes[r] = b
        cands = [
            d
            for d in dets.get(g, [])
            if d["class"] in PERSON
            and d["score"] >= det_conf
            and d["xyxy"][2] - d["xyxy"][0] <= d["xyxy"][3] - d["xyxy"][1]
        ]
        pairs = sorted(
            ((iou(b, d["xyxy"]), o, j) for o, b in boxes.items() for j, d in enumerate(cands)), reverse=True
        )
        used_o, used_d, tracks = set(), set(), []
        for v, o, j in pairs:
            if v < match_iou or o in used_o or j in used_d:
                continue
            used_o.add(o)
            used_d.add(j)
            ref_votes[o][cands[j]["class"] == "referee"] += 1
            tracks.append({"track_id": o, "xyxy": [round(x, 1) for x in cands[j]["xyxy"]]})
        for o, b in boxes.items():
            if o not in used_o:
                tracks.append({"track_id": o, "xyxy": [round(x, 1) for x in b], "src": "sam"})
        out.append({"frame_idx": g, "shot": res["shot"], "tracks": tracks})
    # A SAM-only box survives only inside a short gap between detector-matched boxes of the same track;
    # longer runs are where a lost mask drifts onto legs, other players or the bench.
    matched = defaultdict(list)
    for i, fr in enumerate(out):
        for t in fr["tracks"]:
            if not t.get("src"):
                matched[t["track_id"]].append(i)
    for i, fr in enumerate(out):
        fr["tracks"] = [
            t
            for t in fr["tracks"]
            if not t.get("src")
            or (
                any(i - max_gap <= j < i for j in matched[t["track_id"]])
                and any(i < j <= i + max_gap for j in matched[t["track_id"]])
            )
        ]
    for fr in out:
        for t in fr["tracks"]:
            v = ref_votes.get(t["track_id"])
            if v and v[True] > v[False]:
                t["team"] = "ref"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sam_dir", type=Path)
    ap.add_argument("--prefill", default="basketball-player-detection-3-ycjdo_v4_c0.1_lb90_ar1")
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--dup-iou", type=float, default=0.7)
    ap.add_argument("--max-gap", type=int, default=3, help="sampled frames a SAM-only box may bridge")
    a = ap.parse_args(argv)
    clip_dir = a.sam_dir.parents[1]
    dets = {}
    for line in (clip_dir / "prefill" / a.prefill / "detections.jsonl").open():
        r = json.loads(line)
        dets[r["frame_idx"]] = r["detections"]
    frames, offset, meta = [], 0, []
    for p in sorted(a.sam_dir.glob("shot_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        res = json.loads(p.read_text())
        fused = fuse(res, dets, a.det_conf, a.match_iou, a.dup_iou, a.max_gap)
        ids = sorted({t["track_id"] for f in fused for t in f["tracks"]})
        remap = {o: offset + i + 1 for i, o in enumerate(ids)}
        for f in fused:
            for t in f["tracks"]:
                t["track_id"] = remap[t["track_id"]]
        offset += len(ids)
        frames += fused
        sam_boxes = sum(t.get("src") == "sam" for f in fused for t in f["tracks"])
        total = sum(len(f["tracks"]) for f in fused)
        meta.append(
            {
                "shot": res["shot"],
                "objects": res["objects"],
                "tracks": len(ids),
                "boxes": total,
                "sam_only_boxes": sam_boxes,
                "seconds": res["seconds"],
                "usd": res["usd"],
                "gpu": res["gpu"],
            }
        )
        print(json.dumps(meta[-1]))
    clip = json.loads((Path("data/labels") / f"{clip_dir.name}.json").read_text())
    doc = {"clip_path": clip["clip"], "model": "sam2.1-hiera-large + " + a.prefill, "frames": frames}
    (a.sam_dir / "tracks.json").write_text(json.dumps(doc))
    (a.sam_dir / "meta.json").write_text(
        json.dumps({"shots": meta, "args": vars(a) | {"sam_dir": str(a.sam_dir)}}, indent=1)
    )


if __name__ == "__main__":
    main()
