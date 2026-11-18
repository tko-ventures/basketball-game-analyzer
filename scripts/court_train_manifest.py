#!/usr/bin/env python3
"""Court-keypoint training manifest from Tomasz's training keyframes (M3; committed label file).

    uv run scripts/court_train_manifest.py --name plk-v1 [--window-s 1.5] [--min-ratio 0.85]

Every usable keyframe with purpose ``train`` in data/labels/court/queue.json is carried through its
shot every 5th frame (bga.court.propagate), up to ``window_s`` each way. A line gate keeps a frame
only while the projected court lines stay on the painted lines: walking out from the keyframe, a frame
passes while its line support (``line_support``: the share of projected line points within 2 px of a
painted line) is at least ``min_ratio`` of the keyframe's, and each side stops at its first failure.
Where the keyframe's own support is below ``min_support`` (faint or coloured lines), the measure cannot
vouch for anything, so only frames within ``faint_s`` of the keyframe are kept. Frames keep the 33
landmarks projected through the propagated homography: visibility 2 inside the frame, 0 outside.

Two gates were tried first and passed frames visibly 0.5-1 m off: a round-trip drift check (drift
that is the same both ways cancels) and the median distance to the nearest bright pixel (floor grain,
kits and adverts are always near).

Validation is held out by shot: the keyframes in VAL (one per hall that has two) go to ``valid``.
No frame of a gold clip may appear (data/labels/gold.json); the script refuses if one would.

Writes data/labels/court/train_manifest_<name>.json.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bga.court.propagate import Propagator, court_line_points, landmarks, line_support  # noqa: E402  # after the sys.path insert
from bga.train.leakage import assert_no_gold  # noqa: E402  # after the sys.path insert

GEO = json.loads((ROOT / "tools/labeler/static/fiba_court.json").read_text())
VERTS = np.array(GEO["vertices"], dtype=np.float64)
LINES = court_line_points(GEO)
# One keyframe per hall that has two usable ones, chosen before any training: (youtube_id, frame).
VAL = {("hUoD7ptXzCE", 6353), ("jXhvXy9iQeQ", None), ("Y4TmeMchTh0", None)}


class Reader:
    def __init__(self, path: str):
        self.cap, self.pos, self.cache = cv2.VideoCapture(path), -1, {}

    def __call__(self, i: int) -> np.ndarray:
        if i not in self.cache:
            if i != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = self.cap.read()
            if not ok:
                raise IndexError(i)
            self.pos, self.cache = i, {**dict(list(self.cache.items())[-40:]), i: img}
        return self.cache[i]


def val_keys(train: list[dict]) -> set[tuple[str, int]]:
    """Resolve VAL: an explicit frame, or the later keyframe of that clip's hall pair."""
    out = set()
    for yid, frame in VAL:
        frames = sorted(k["frame"] for k in train if k["youtube_id"] == yid)
        if frames:
            out.add((yid, frame if frame is not None else frames[-1]))
    return out


def line_gate(support: dict[int, float | None], key: int, min_ratio: float, min_support: float, faint: int) -> set[int]:
    """Frames kept by the line gate (see the module docstring); ``faint`` is in frames."""
    s0 = support.get(key)
    if s0 is None or s0 < min_support:
        return {f for f in support if abs(f - key) <= faint}
    keep = {key}
    for side in (
        sorted(f for f in support if f > key),
        sorted((f for f in support if f < key), reverse=True),
    ):
        for f in side:
            s = support[f]
            if s is None or s < min_ratio * s0:
                break
            keep.add(f)
    return keep


def _m(e: float | None) -> str:
    return "n/a" if e is None else f"{e:.2f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--window-s", type=float, default=1.5)
    ap.add_argument("--min-ratio", type=float, default=0.85)
    ap.add_argument("--min-support", type=float, default=0.4)
    ap.add_argument("--faint-s", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args(argv)

    queue = json.loads((ROOT / "data/labels/court/queue.json").read_text())["keyframes"]
    train = []
    for k in queue:
        rec = json.loads((ROOT / "data/labels/court" / f"{k['clip']}.json").read_text())["frames"].get(str(k["frame"]))
        if k["purpose"] == "train" and rec and rec.get("status") == "ok" and rec.get("H"):
            train.append({**k, "H": rec["H"]})
    assert_no_gold({k["clip"] for k in train})
    vals = val_keys(train)

    prop, frames, keyframes, sources = Propagator(), [], [], {}
    for k in sorted(train, key=lambda k: (k["clip"], k["frame"])):
        label_path = ROOT / "data/labels/court" / f"{k['clip']}.json"
        sources[label_path.name] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        labels = json.loads((ROOT / "data/labels" / f"{k['clip']}.json").read_text())
        shot = labels["shots"][k["shot"]]
        read = Reader(str(ROOT / labels["clip"]))
        fps = read.cap.get(cv2.CAP_PROP_FPS)
        wh = (int(read.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        key, H, lo, hi = k["frame"], np.array(k["H"]), shot["start"], shot["end"] - 1
        span = int(a.window_s * fps) // a.stride * a.stride
        want = [f for f in range(key - span, key + span + 1, a.stride) if lo <= f <= hi]
        steps = {f: st for f, st in prop.run(read, want, key, H).items() if st.H is not None}
        scores = {f: line_support(read(f), st.H, LINES) for f, st in steps.items()}
        keep = line_gate(scores, key, a.min_ratio, a.min_support, int(a.faint_s * fps))
        split = "valid" if (k["youtube_id"], key) in vals else "train"
        kept = []
        for f in sorted(keep):
            kp = landmarks(steps[f].H, VERTS, wh)
            if sum(1 for *_, v in kp if v) < 4:
                continue
            frames.append({
                "clip": k["clip"],
                "video": labels["clip"],
                "frame": f,
                "keyframe": key,
                "split": split,
                "size": list(wh),
                "H": np.round(steps[f].H, 10).tolist(),
                "line_support": None if scores[f] is None else round(scores[f], 3),
                "kp": kp,
            })
            kept.append(f)
        s0 = scores.get(key)
        keyframes.append({
            "clip": k["clip"],
            "hall": k["hall"],
            "frame": key,
            "split": split,
            "line_support": None if s0 is None else round(s0, 3),
            "from_s": round((min(kept) - key) / fps, 2) if kept else None,
            "to_s": round((max(kept) - key) / fps, 2) if kept else None,
            "frames": len(kept),
        })
        print(
            f"{k['hall'][:24]:24} f{key:<6} {split:5} key support {_m(s0)}, kept "
            f"{(min(kept) - key) / fps:+.2f}s..{(max(kept) - key) / fps:+.2f}s -> {len(kept)} frames",
            flush=True,
        )
        read.cap.release()

    assert_no_gold({f["clip"] for f in frames})
    out = ROOT / "data/labels/court" / f"train_manifest_{a.name}.json"
    manifest = {
        "name": a.name,
        "kind": "court_keypoints",
        "params": {
            "window_s": a.window_s,
            "stride": a.stride,
            "min_ratio": a.min_ratio,
            "min_support": a.min_support,
            "faint_s": a.faint_s,
        },
        "keypoint_names": GEO["labels"],
        "source_labels": sources,
        "keyframes": keyframes,
        "frames": frames,
    }
    out.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    n = {s: sum(1 for f in frames if f["split"] == s) for s in ("train", "valid")}
    summary = f"{len(frames)} frames ({n['train']} train, {n['valid']} valid)"
    print(f"{summary} from {len(keyframes)} keyframes -> {out}")


if __name__ == "__main__":
    main()
