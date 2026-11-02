#!/usr/bin/env python3
"""Court-landmark prefill for M3 (labelling aid): the NBA court keypoint model over every clip.

    uv run --extra prefill scripts/court_prefill.py [--every 25] [--clips <yid>,...]

Samples every ``--every`` frames inside each detected shot (not across cuts) and stores the model's
33 FIBA-ordered keypoints with confidences, plus a RANSAC homography and its inlier count, to
runs/<clip>/court_prefill/<model>.jsonl. A shot with many frames of >= 4 confident points is a
main-camera candidate for keyframes; the homography is only a starting point for the reviewer.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.detect.roboflow_inference import roboflow_key  # noqa: E402

MODEL = "basketball-court-detection-2/14"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--clips", default="")
    ap.add_argument("--kp-conf", type=float, default=0.5)
    a = ap.parse_args(argv)
    roboflow_key()
    from inference import get_model
    from sports.basketball import CourtConfiguration, League

    model = get_model(model_id=MODEL)
    vertices = np.array(CourtConfiguration(league=League.FIBA).vertices, dtype=np.float32)
    only = set(filter(None, a.clips.split(",")))
    for path in sorted(glob.glob("data/labels/2*.json")):
        labels = json.loads(Path(path).read_text())
        stem = Path(path).stem
        yid = stem[9:20]
        if only and yid not in only:
            continue
        out = Path("runs") / stem / "court_prefill" / f"{MODEL.replace('/', '_')}.jsonl"
        if out.exists():
            print(f"{yid}: cached")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(labels["clip"])
        t0, n, rows = time.time(), 0, []
        for si, s in enumerate(labels["shots"]):
            for f in range(s["start"] + a.every // 2, s["end"], a.every):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = cap.read()
                if not ok:
                    continue
                kp = model.infer(img, confidence=0.3)[0]
                pts = []
                if kp.predictions and len(kp.predictions[0].keypoints) == len(vertices):
                    pts = [
                        [round(p.x, 1), round(p.y, 1), round(p.confidence, 3)]
                        for p in kp.predictions[0].keypoints
                    ]
                H, inliers = None, 0
                if pts:
                    arr = np.array(pts, dtype=np.float32)
                    m = arr[:, 2] > a.kp_conf
                    if m.sum() >= 4:
                        Hm, mask = cv2.findHomography(arr[m, :2], vertices[m], cv2.RANSAC, 40.0)
                        if Hm is not None:
                            H, inliers = np.round(Hm, 8).tolist(), int(mask.sum())
                rows.append(
                    {
                        "frame": f,
                        "shot": si,
                        "kp": pts,
                        "H": H,
                        "inliers": inliers,
                        "n_conf": int(sum(p[2] > a.kp_conf for p in pts)),
                    }
                )
                n += 1
        cap.release()
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        good = sum(r["n_conf"] >= 4 for r in rows)
        print(f"{yid}: {n} frames, {good} with >= 4 confident points, {time.time() - t0:.0f} s", flush=True)
    os.environ.pop("ROBOFLOW_API_KEY", None)


if __name__ == "__main__":
    main()
