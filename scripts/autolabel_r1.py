#!/usr/bin/env python3
"""M4 auto-label round 1 (plan section 7): training frames from the non-gold clips, prefilled for review.

    uv run --extra court scripts/autolabel_r1.py central     # court-landmark score per shot, all clips
    uv run --extra court scripts/autolabel_r1.py check       # the central rule against the shots tagged by hand
    uv run scripts/autolabel_r1.py sample                    # training frames from central shots, gold clips excluded
    uv run --extra prefill scripts/autolabel_r1.py prefill   # NBA detector v16 (all but ball) and v6 (ball), conf 0.1
    uv run --extra court scripts/autolabel_r1.py draft       # keep >= 0.3, NMS, drop people off the court
    uv run scripts/autolabel_r1.py ballqueue                 # data/labels/box_queue_r1balls.json for Tomasz

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


def sample(per_clip: int, seed: int) -> None:
    """Pick training frames from the central shots of the non-gold clips, at least 1 s apart within a shot."""
    import random

    gold = {g["youtube_id"] for g in json.loads(Path("data/labels/gold.json").read_text())["games"]}
    res = json.loads((OUT / "central.json").read_text())
    rng = random.Random(seed)
    frames = []
    for stem, c in sorted(res.items()):
        if any(g in stem for g in gold):
            continue
        step = int(c["fps"])
        pool = []
        for s in c["shots"]:
            if s["median"] >= CENTRAL_K:
                pool += [(s["shot"], f) for f in range(s["start"] + 5, s["end"] - 5, step)]
        for shot, f in sorted(rng.sample(pool, min(per_clip, len(pool)))):
            frames.append({"clip": stem, "video": c["clip"], "shot": shot, "frame": f})
    (OUT / "frames.json").write_text(json.dumps({"seed": seed, "per_clip": per_clip, "frames": frames}, indent=1))
    print(f"{len(frames)} frames from {len({f['clip'] for f in frames})} clips -> {OUT / 'frames.json'}")


#: Prefill models: version 16 for every class but the ball, version 6 for the ball (docs/decisions.md, 2026-11-18).
PREFILL_MODELS = {"basketball-player-detection-3-ycjdo/16": "not-ball", "basketball-player-detection-3-ycjdo/6": "ball"}


def prefill() -> None:
    """Run both prefill models at confidence 0.1 on the sampled frames; raw detections to prefill_raw.json."""
    from inference import get_model

    from bga.detect.roboflow_inference import roboflow_key

    roboflow_key()
    frames = json.loads((OUT / "frames.json").read_text())["frames"]
    raw = {}
    for model_id, keep in PREFILL_MODELS.items():
        model = get_model(model_id=model_id)
        caps = {}
        for fr in frames:
            cap = caps.setdefault(fr["video"], cv2.VideoCapture(fr["video"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame"])
            img = cap.read()[1]
            r = model.infer(img, confidence=0.1, iou_threshold=0.5)[0]
            dets = raw.setdefault(f"{fr['clip']}|{fr['frame']}", [])
            for q in r.predictions:
                if (q.class_name == "ball") == (keep == "ball"):
                    dets.append([
                        round(q.x - q.width / 2, 1),
                        round(q.y - q.height / 2, 1),
                        round(q.x + q.width / 2, 1),
                        round(q.y + q.height / 2, 1),
                        round(float(q.confidence), 4),
                        q.class_name,
                        model_id,
                    ])
        print(model_id, "done")
    (OUT / "prefill_raw.json").write_text(json.dumps(raw))
    print(f"{sum(len(v) for v in raw.values())} detections on {len(raw)} frames -> {OUT / 'prefill_raw.json'}")


PERSON = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee"}
KEEP, NMS_IOU, MARGIN_CM = 0.3, 0.5, 100.0


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def draft() -> None:
    """Boxes at score >= KEEP after per-class NMS, the rest as candidates; people off the court (foot point more than
    MARGIN_CM outside, under the PLK court model's mapping of the frame) and numbers outside every kept person
    dropped."""
    from bga.court.rfdetr_keypoint import RFDETRKeypoint

    frames = json.loads((OUT / "frames.json").read_text())["frames"]
    raw = json.loads((OUT / "prefill_raw.json").read_text())
    court = RFDETRKeypoint({}, COURT_WEIGHTS)
    court.load()
    caps, out, stats = {}, [], {"off_court": 0, "orphan_numbers": 0, "unmapped_frames": 0}
    for fr in frames:
        cap = caps.setdefault(fr["video"], cv2.VideoCapture(fr["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame"])
        img = cap.read()[1]
        h = court.homography(img)
        stats["unmapped_frames"] += h is None
        kept, cands = [], []
        for d in sorted(raw[f"{fr['clip']}|{fr['frame']}"], key=lambda d: -d[4]):
            if any(k[5] == d[5] and _iou(k, d) >= NMS_IOU for k in kept + cands):
                continue
            (kept if d[4] >= KEEP else cands).append(d)
        if h is not None:
            H = np.array(h)

            def on_court(d, H=H):
                w = H @ np.array([(d[0] + d[2]) / 2, d[3], 1.0])  # foot point: bottom centre of the box
                cx, cy = w[0] / w[2], w[1] / w[2]
                return -MARGIN_CM <= cx <= 2800 + MARGIN_CM and -MARGIN_CM <= cy <= 1500 + MARGIN_CM

            n0 = len(kept)
            kept = [d for d in kept if d[5] not in PERSON or on_court(d)]
            stats["off_court"] += n0 - len(kept)
        people = [d for d in kept if d[5] in PERSON]
        n0 = len(kept)
        kept = [
            d
            for d in kept
            if d[5] != "number"
            or any(d[0] >= p[0] - 5 and d[2] <= p[2] + 5 and d[1] >= p[1] - 5 and d[3] <= p[3] + 5 for p in people)
        ]
        stats["orphan_numbers"] += n0 - len(kept)
        out.append({**fr, "mapped": h is not None, "boxes": kept, "candidates": cands})
    court.unload()
    (OUT / "draft.json").write_text(json.dumps({"frames": out}, indent=1))
    from collections import Counter

    c = Counter(d[5] for f in out for d in f["boxes"])
    print(stats, dict(c), "frames with a ball box:", sum(any(d[5] == "ball" for d in f["boxes"]) for f in out))


BALL_QUEUE = Path("data/labels/box_queue_r1balls.json")
MIN_PEOPLE = 4  # frames with fewer person boxes are close-ups that passed the central rule
INSTRUCTIONS = (
    "<b>One question per frame: where is the game ball?</b><br>"
    "The loupe above shows the detector's best guess. <kbd>Space</kbd>: yes, that is the ball (frame done, next). "
    "<kbd>Tab</kbd>: show the next guess. Or <b>click</b> any guess on the frame (orange, or dashed: weaker).<br>"
    "No guess on the ball: <kbd>⇧</kbd>-drag a tight box around it, then <kbd>Enter</kbd>. "
    "A guess on the ball but loose: drag its edges in the loupe, then <kbd>Enter</kbd>.<br>"
    "<kbd>N</kbd> no ball visible · <kbd>S</kbd> can't tell, skip · <kbd>[</kbd> <kbd>]</kbd> previous / next frame "
    "· scroll to zoom, <kbd>F</kbd> fit · <kbd>U</kbd> undo.<br>"
    "Game ball only: in hands, in the air, on the floor. Nothing in the crowd, on boards or on shorts."
)


def ball_queue(force: bool) -> None:
    """Write the round-1 ball queue for the labeller (/boxes?set=r1balls): each frame's ball guesses, nothing else."""
    if BALL_QUEUE.exists() and not force:
        q = json.loads(BALL_QUEUE.read_text())
        raise SystemExit(f"{BALL_QUEUE} exists ({sum(f['status'] != 'todo' for f in q['frames'])} finished); --force")
    frames = []
    for f in json.loads((OUT / "draft.json").read_text())["frames"]:
        if sum(b[5] in PERSON for b in f["boxes"]) < MIN_PEOPLE:
            continue
        balls = [b for b in f["boxes"] if b[5] == "ball"]
        cands = sorted((b for b in f["candidates"] if b[5] == "ball"), key=lambda b: -b[4])[:3]
        clip_id = f["clip"]
        frames.append({
            "youtube_id": clip_id,
            "clip": clip_id,
            "video": f["video"],
            "shot": f["shot"],
            "frame": f["frame"],
            "status": "todo",
            "note": "",
            "boxes": [
                {
                    "id": i,
                    "xyxy": b[:4],
                    "cls": "ball",
                    "score": b[4],
                    "team": "",
                    "tid": None,
                    "src": "det",
                    "flags": [],
                    "hints": [],
                    "alt": {},
                }
                for i, b in enumerate(balls)
            ],
            "candidates": [{"xyxy": b[:4], "cls": "ball", "score": b[4]} for b in cands],
        })
    BALL_QUEUE.write_text(
        json.dumps(
            {
                "version": "r1balls",
                "mode": "ball",
                "title": "Ball queue, round 1",
                "instructions": INSTRUCTIONS,
                "classes": ["ball"],
                "frames": frames,
                "meta": {"review_seconds": {}},
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"{len(frames)} frames, {sum(len(f['boxes']) for f in frames)} guesses, "
        f"{sum(len(f['candidates']) for f in frames)} weaker guesses -> {BALL_QUEUE}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["central", "check", "sample", "prefill", "draft", "ballqueue"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--per-clip", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {
        "central": lambda: central(a.force),
        "check": check,
        "sample": lambda: sample(a.per_clip, a.seed),
        "prefill": prefill,
        "draft": draft,
        "ballqueue": lambda: ball_queue(a.force),
    }[a.cmd]()


if __name__ == "__main__":
    main()
