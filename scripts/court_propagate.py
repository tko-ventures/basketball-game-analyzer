#!/usr/bin/env python3
"""Propagate court keyframes through their shots, or measure propagation drift.

    uv run scripts/court_propagate.py --check <clip-stem> [--shots 2,10]   # drift against the NBA model
    uv run scripts/court_propagate.py <clip-stem>                           # labels from corrected keyframes

--check (no human labels needed): in shots where the NBA court model is trustworthy (Warsaw, checked
on overlays), propagate from the first prefill frame and compare with the model's own mapping at
every later prefill frame. Reported: median and p90 distance in metres between where each mapping
puts the visible landmarks.

Without --check: each keyframe Tomasz saved in data/labels/court/<clip>.json is carried through its
shot every 5th frame. Output to runs/<clip>/court_labels/propagated.jsonl: frame, H, 33 landmarks
(x, y, visibility), keyframe, steps from it, inliers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.court.propagate import Propagator, landmarks  # noqa: E402

GEO = json.loads((Path(__file__).resolve().parents[1] / "tools/labeler/static/fiba_court.json").read_text())
VERTS = np.array(GEO["vertices"], dtype=np.float64)


class Reader:
    def __init__(self, path):
        self.cap, self.pos, self.cache = cv2.VideoCapture(path), -1, {}

    def __call__(self, i):
        if i not in self.cache:
            if i != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = self.cap.read()
            assert ok, i
            self.pos, self.cache = i, {**dict(list(self.cache.items())[-30:]), i: img}
        return self.cache[i]


def disagreement_m(Ha, Hb, wh) -> float | None:
    """Median distance (m) between court positions of the landmarks visible under Ha, mapped by Ha and Hb."""
    pts = [(x, y) for x, y, v in landmarks(Ha, VERTS, wh) if v]
    if len(pts) < 4:
        return None
    p = np.c_[np.array(pts), np.ones(len(pts))]
    a, b = p @ np.asarray(Ha).T, p @ np.asarray(Hb).T
    a, b = a[:, :2] / a[:, 2:3], b[:, :2] / b[:, 2:3]
    return float(np.median(np.linalg.norm(a - b, axis=1)) / 100.0)


def check(stem: str, shots: list[int]) -> None:
    labels = json.loads(Path(f"data/labels/{stem}.json").read_text())
    pre = [
        json.loads(line) for line in (Path("runs") / stem / "court_prefill").glob("*.jsonl").__next__().open()
    ]
    read, prop = Reader(labels["clip"]), Propagator()
    wh = (int(read.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    for si in shots:
        rows = [r for r in pre if r["shot"] == si and r["H"] and r["inliers"] >= 6]
        if len(rows) < 3:
            print(f"shot {si}: too few confident prefill frames ({len(rows)})")
            continue
        s = labels["shots"][si]
        key = rows[0]["frame"]
        frames = list(range(key, s["end"], 5))
        steps = prop.run(read, frames, key, np.array(rows[0]["H"]))
        errs = []
        for r in rows[1:]:
            st = steps.get(r["frame"])
            if st is not None and st.H is not None:
                e = disagreement_m(np.array(r["H"]), st.H, wh)
                if e is not None:
                    errs.append((r["frame"] - key, e))
        lost = next((f for f, st in sorted(steps.items()) if st.H is None), None)
        if errs:
            e = np.array([x for _, x in errs])
            print(
                f"shot {si}: {len(errs)} comparisons over {errs[-1][0] / 50:.1f} s, "
                f"median {np.median(e):.2f} m, p90 {np.percentile(e, 90):.2f} m, last {errs[-1][1]:.2f} m"
                + (f", lost at frame {lost}" if lost else "")
            )


def loop_check(stem: str, shots: list[int], spans_s=(1.0, 3.0)) -> None:
    """Round-trip drift: propagate forwards span seconds, then back to the start; no ground truth needed."""
    labels = json.loads(Path(f"data/labels/{stem}.json").read_text())
    pre = [json.loads(line) for line in next((Path("runs") / stem / "court_prefill").glob("*.jsonl")).open()]
    read, prop = Reader(labels["clip"]), Propagator()
    wh = (int(read.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    for si in shots:
        s = labels["shots"][si]
        start = next((r for r in pre if r["shot"] == si and r["H"] and r["inliers"] >= 6), None)
        if start is None:
            continue
        H0, key = np.array(start["H"]), start["frame"]
        for span in spans_s:
            end = min(s["end"] - 1, key + int(span * 50))
            fwd = prop.run(read, list(range(key, end + 1, 5)), key, H0)
            last = max(f for f, st in fwd.items() if st.H is not None)
            # Back with a 10-frame step from an odd offset: no frame pair repeats, so errors cannot cancel.
            back_frames = sorted({key} | set(range(last, key, -10)) | {last})
            back = prop.run(read, back_frames, last, fwd[last].H)
            if back.get(key) is None or back[key].H is None:
                print(f"shot {si} {span:.0f} s: lost (reached {(last - key) / 50:.1f} s)")
                continue
            e = disagreement_m(H0, back[key].H, wh)
            print(
                f"shot {si} {span:.0f} s: round trip over {(last - key) / 50:.1f} s each way, drift {e:.2f} m"
            )


def propagate(stem: str) -> None:
    labels = json.loads(Path(f"data/labels/{stem}.json").read_text())
    court = json.loads(Path(f"data/labels/court/{stem}.json").read_text())
    read, prop = Reader(labels["clip"]), Propagator()
    wh = (int(read.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = Path("runs") / stem / "court_labels" / "propagated.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fk, rec in sorted(court["frames"].items(), key=lambda kv: int(kv[0])):
        if rec.get("status") != "ok" or not rec.get("H"):
            continue
        fk = int(fk)
        s = next(s for s in labels["shots"] if s["start"] <= fk <= s["end"])
        frames = sorted(set(range(fk, s["start"] - 1, -5)) | set(range(fk, s["end"], 5)))
        for f, st in sorted(prop.run(read, frames, fk, np.array(rec["H"])).items()):
            if st.H is not None:
                rows.append(
                    {
                        "frame": f,
                        "H": np.round(st.H, 8).tolist(),
                        "kp": landmarks(st.H, VERTS, wh),
                        "keyframe": fk,
                        "steps": abs(f - fk) // 5,
                        "inliers": st.inliers,
                    }
                )
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{stem}: {len(rows)} labelled frames from {len(court['frames'])} keyframes -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--loop", action="store_true", help="round-trip drift instead of NBA comparison")
    ap.add_argument("--shots", default="")
    a = ap.parse_args(argv)
    if a.loop:
        loop_check(a.stem, [int(x) for x in a.shots.split(",") if x])
    elif a.check:
        check(a.stem, [int(x) for x in a.shots.split(",") if x])
    else:
        propagate(a.stem)


if __name__ == "__main__":
    main()
