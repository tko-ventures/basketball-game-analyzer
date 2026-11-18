#!/usr/bin/env python3
"""M4 auto-label round 1 (plan section 7): training frames from the non-gold clips, prefilled for review.

    uv run --extra court scripts/autolabel_r1.py central     # court-landmark score per shot, all clips
    uv run --extra court scripts/autolabel_r1.py check       # the central rule against the shots tagged by hand

Step 1, central shots. The non-gold clips have no camera tags. A shot from the main camera shows the court, so the
PLK court model (court/rfdetr_keypoint/plk-v1) finds many confident landmarks there; replays, close-ups and graphics
give few. Each shot is scored by the median count of landmarks above confidence 0.5 on three frames (at 25, 50 and
75% of the shot). ``check`` measures the rule "median >= k" against the shots tagged central or other by hand (the
gold clips), for several k. Results go to runs/autolabel/r1/.

Checked 2026-11-18 on 186 hand-tagged shots (126 central): k = 12 gives precision 0.93 and recall 0.98 (9 false, all
other court views; 2 missed). It selects 534 of the 791 untagged shots of the non-gold clips.

"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path("runs/autolabel/r1")
COURT_WEIGHTS = "court/rfdetr_keypoint/plk-v1@c26b7841"
SKIP = {"gold.json", "box_gold.json", "box_gold_v1.json"}
MIN_SHOT_S = 1.0
CENTRAL_K = 12  # median confident landmarks for a central shot (see the module docstring)


def label_files():
    for p in sorted(Path("data/labels").glob("*.json")):
        if p.name in SKIP:
            continue
        d = json.loads(p.read_text())
        if "shots" in d and "clip" in d:
            yield p, d


def central(force: bool) -> None:
    from bga.court.rfdetr_keypoint import RFDETRKeypoint

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "central.json"
    res = json.loads(path.read_text()) if path.exists() and not force else {}
    court = RFDETRKeypoint({}, COURT_WEIGHTS)
    court.load()
    for p, d in label_files():
        if p.stem in res:
            continue
        cap = cv2.VideoCapture(d["clip"])
        shots = []
        for i, s in enumerate(d["shots"]):
            n = s["end"] - s["start"]
            rec = {"shot": i, "start": s["start"], "end": s["end"], "camera": s.get("camera", "unset"), "counts": []}
            if n / d["fps"] >= MIN_SHOT_S:
                for q in (0.25, 0.5, 0.75):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, s["start"] + int(q * n))
                    ok, img = cap.read()
                    if not ok:
                        continue
                    kp = court.model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=court.p["threshold"])
                    if len(kp) == 0:
                        rec["counts"].append(0)
                        continue
                    best = int(np.argmax(kp.detection_confidence)) if kp.detection_confidence is not None else 0
                    rec["counts"].append(int((kp.keypoint_confidence[best] > court.p["kp_conf"]).sum()))
            rec["median"] = statistics.median(rec["counts"]) if rec["counts"] else 0
            shots.append(rec)
        res[p.stem] = {"clip": d["clip"], "fps": d["fps"], "shots": shots}
        path.write_text(json.dumps(res))
        print(p.stem[:60], len(shots), "shots")
    court.unload()


def check() -> None:
    res = json.loads((OUT / "central.json").read_text())
    tagged = [s for c in res.values() for s in c["shots"] if s["camera"] in ("central", "other") and s["counts"]]
    print(f"{len(tagged)} tagged shots: {sum(s['camera'] == 'central' for s in tagged)} central")
    for k in (4, 6, 8, 10, 12, 14):
        tp = sum(s["median"] >= k and s["camera"] == "central" for s in tagged)
        fp = sum(s["median"] >= k and s["camera"] == "other" for s in tagged)
        fn = sum(s["median"] < k and s["camera"] == "central" for s in tagged)
        print(f"k={k:2d}: precision {tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f} (fp {fp}, fn {fn})")
    untagged = [s for c in res.values() for s in c["shots"] if s["camera"] == "unset" and s["counts"]]
    for k in (8, 10, 12):
        print(f"untagged shots with median >= {k}: {sum(s['median'] >= k for s in untagged)} of {len(untagged)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["central", "check"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"central": lambda: central(a.force), "check": check}[a.cmd]()


if __name__ == "__main__":
    main()
