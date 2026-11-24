#!/usr/bin/env python3
"""M4 auto-labelling rounds (plan section 7): training frames from the non-gold clips, prefilled for review.

Every step takes ``--round N`` (default 1): round 1 prefills with Roboflow's NBA detectors, round 2 on with the
previous round's PLK detector, skips frames within 1 s of earlier rounds' frames, and accepts clear balls without
review.

    uv run --extra court scripts/autolabel.py central     # court-landmark score per shot, all clips
    uv run --extra court scripts/autolabel.py check       # the central rule against the shots tagged by hand
    uv run scripts/autolabel.py sample                    # training frames from central shots, gold clips excluded
    uv run scripts/autolabel.py mine --round 4            # round 4 instead of sample: hard frames (see mine())
    uv run --extra prefill scripts/autolabel.py prefill   # NBA detector v16 (all but ball) and v6 (ball), conf 0.1
    uv run --extra court scripts/autolabel.py draft       # keep >= 0.3, NMS, drop people off the court
    uv run scripts/autolabel.py ballqueue                 # data/labels/box_queue_r1balls.json for Tomasz
    uv run scripts/autolabel.py manifest                  # data/labels/detect/train_manifest_r1.json for bga train
    uv run scripts/autolabel.py possqueue --rounds 1 2    # data/labels/box_queue_possession.json for Tomasz
    uv run scripts/autolabel.py possmanifest --rounds 1 2 3   # train_manifest_r<N>_poss.json: possession applied

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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Shot scores of every clip, computed once (round 1) and shared by every round.
CENTRAL = Path("runs/autolabel/r1/central.json")
#: Round 2 on: the prefill is the previous round's PLK detector (docs/metrics.md, detection table).
PREFILL_WEIGHTS = {
    2: "detect/rfdetr_torch/plk-r1@7ab3f6fa",
    3: "detect/rfdetr_torch/plk-r2@b13bfb19",
    4: "detect/rfdetr_torch/plk-r3-poss@c62744fe",
}
#: Score at which a draft box is kept: 0.3 for the NBA detectors of round 1, 0.5 for the PLK detector.
KEEP_BY_ROUND = {1: 0.3}
#: A frame's ball is accepted without review when one ball scores at least AUTO_HI[round] and no other ball AUTO_LO,
#: calibrated on gold v1 dev for each round's prefill model: plk-r1 at 0.7 picked 22 of 76 frames, all right
#: (2026-11-19); plk-r2 at 0.7 picked 35 with 3 wrong, at 0.85 24 with 1 wrong (2026-11-20).
AUTO_HI_BY_ROUND, AUTO_LO = {2: 0.7, 3: 0.85}, 0.2


@dataclass(frozen=True)
class Round:
    """One auto-labelling round: every step reads and writes the files of its round."""

    n: int

    @property
    def out(self) -> Path:
        """The round's run folder (frames, prefill and draft)."""
        return Path(f"runs/autolabel/r{self.n}")

    @property
    def ball_queue(self) -> Path:
        """The round's ball queue for the labeller."""
        return Path(f"data/labels/box_queue_r{self.n}balls.json")

    @property
    def manifest(self) -> Path:
        """The round's training manifest for ``bga train``."""
        return Path(f"data/labels/detect/train_manifest_r{self.n}.json")

    @property
    def keep(self) -> float:
        """Score at which a draft box is kept."""
        return KEEP_BY_ROUND.get(self.n, 0.5)


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

    CENTRAL.parent.mkdir(parents=True, exist_ok=True)
    path = CENTRAL
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
    res = json.loads(CENTRAL.read_text())
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


def sample(r: Round, per_clip: int, seed: int) -> None:
    """Pick training frames from the central shots of the non-gold clips, at least 1 s apart within a shot."""
    import random

    gold = {g["youtube_id"] for g in json.loads(Path("data/labels/gold.json").read_text())["games"]}
    res = json.loads(CENTRAL.read_text())
    rng = random.Random(seed + 1000 * (r.n - 1))
    taken = {
        (f["clip"], f["frame"])
        for k in range(1, r.n)
        for f in json.loads(Path(f"runs/autolabel/r{k}/frames.json").read_text())["frames"]
    }
    frames = []
    for stem, c in sorted(res.items()):
        if any(g in stem for g in gold):
            continue
        step = int(c["fps"])
        pool = []
        for s in c["shots"]:
            if s["median"] >= CENTRAL_K:
                pool += [
                    (s["shot"], f)
                    for f in range(s["start"] + 5, s["end"] - 5, step)
                    if all(abs(f - g) >= step for c2, g in taken if c2 == stem)  # 1 s from earlier rounds' frames
                ]
        for shot, f in sorted(rng.sample(pool, min(per_clip, len(pool)))):
            frames.append({"clip": stem, "video": c["clip"], "shot": shot, "frame": f})
    (r.out / "frames.json").write_text(json.dumps({"seed": seed, "per_clip": per_clip, "frames": frames}, indent=1))
    print(f"{len(frames)} frames from {len({f['clip'] for f in frames})} clips -> {r.out / 'frames.json'}")


#: Cached plk-r3-poss detections (every 5th frame, confidence 0.05) of the crops-reid-r1 central shots: the pool that
#: round 4 mines for hard frames.
MINE_CACHE = "runs/*/detect/rfdetr_torch@eace3dfd/*_s*.json"
SHOT_CLASSES = {"player-jump-shot", "player-layup-dunk", "player-shot-block"}
#: Hard-frame buckets in priority order, with frames per clip (#40: crowded paint, shots, contact, unsure ball).
MINE_QUOTA = {"paint": 8, "shot": 5, "contact": 4, "ball": 3}
MINE_GAP_S, MINE_PER_SHOT, MINE_PREV_GAP_S = 0.5, 3, 0.5


def hard_bucket(dets: list) -> str | None:
    """The hard-frame bucket of one frame's detections, or None for an easy frame.

    paint: four or more players (score >= 0.5) under a rim, horizontally within 4 rim widths, with the ball near the
    rim, unsure or missing, or a shot class present. shot: a shot class at 0.3 or more. contact: a holder overlapping
    another player (IoU >= 0.3). ball: at least four players and the best ball below 0.5, or a second ball at 0.3.

    >>> rim = [900, 400, 960, 420, 0.9, "rim"]
    >>> ppl = [[900 + 30 * i, 450, 960 + 30 * i, 700, 0.9, "player"] for i in range(4)]
    >>> hard_bucket([rim, *ppl])
    'paint'
    >>> hard_bucket(ppl + [[100, 100, 110, 110, 0.9, "ball"]]) is None
    True

    """
    people = [d for d in dets if d[5] in PERSON and d[4] >= 0.5]
    players = [d for d in people if d[5] != "referee"]
    balls = sorted((d for d in dets if d[5] == "ball"), key=lambda d: -d[4])
    top = balls[0][4] if balls else 0.0
    shot = max([d[4] for d in dets if d[5] in SHOT_CLASSES] + [0.0])
    for rim in (d for d in dets if d[5] == "rim" and d[4] >= 0.5):
        w, cx = rim[2] - rim[0], (rim[0] + rim[2]) / 2
        crowd = sum(abs((q[0] + q[2]) / 2 - cx) < 4 * w and q[3] > rim[3] for q in players)
        ball_near = top >= 0.3 and abs((balls[0][0] + balls[0][2]) / 2 - cx) < 3 * w
        if crowd >= 4 and (ball_near or top < 0.5 or shot >= 0.3):
            return "paint"
    if shot >= 0.3:
        return "shot"
    holders = [d for d in players if d[5] == "player-in-possession" and d[4] >= 0.3]
    if any(_iou(h, q) >= 0.3 for h in holders for q in players if q[:4] != h[:4]):
        return "contact"
    if len(people) >= MIN_PEOPLE and (top < 0.5 or (len(balls) > 1 and balls[1][4] >= 0.3)):
        return "ball"
    return None


def mine(r: Round, seed: int) -> None:
    """Round 4: hard frames from the cached detections of non-gold central shots, instead of random ``sample``.

    Each cached frame gets a hard_bucket. Per clip, MINE_QUOTA frames are drawn from each bucket (unused quota goes to
    the next bucket), at least MINE_GAP_S apart within a shot, at most MINE_PER_SHOT per shot, and MINE_PREV_GAP_S from
    earlier rounds' frames.

    """
    import random
    from collections import Counter

    cen = json.loads(CENTRAL.read_text())
    rng = random.Random(seed + 1000 * (r.n - 1))
    pools = _mine_pools(r, cen)
    frames, counts = [], Counter()
    for stem, by_bucket in sorted(pools.items()):
        fps, picked, carry = cen[stem]["fps"], [], 0
        for bucket, quota in MINE_QUOTA.items():
            want, cands = quota + carry, by_bucket.get(bucket, [])
            rng.shuffle(cands)
            got = 0
            for shot, f in cands:
                if got == want:
                    break
                same = [g for s, g, _ in picked if s == shot]
                if len(same) < MINE_PER_SHOT and all(abs(f - g) >= MINE_GAP_S * fps for g in same):
                    picked.append((shot, f, bucket))
                    got += 1
            carry = want - got
            counts[bucket] += got
        for shot, f, bucket in sorted(picked):
            frames.append({"clip": stem, "video": cen[stem]["clip"], "shot": shot, "frame": f, "bucket": bucket})
    (r.out / "frames.json").write_text(
        json.dumps({"seed": seed, "mined": MINE_CACHE, "quota": MINE_QUOTA, "frames": frames}, indent=1)
    )
    print(f"{len(frames)} frames from {len(pools)} clips {dict(counts)} -> {r.out / 'frames.json'}")


def _mine_pools(r: Round, cen: dict) -> dict:
    """Hard frames of the cached detections per clip and bucket, gold clips and earlier rounds' frames left out."""
    import glob
    from collections import defaultdict

    gold = {g["youtube_id"] for g in json.loads(Path("data/labels/gold.json").read_text())["games"]}
    taken = defaultdict(list)
    for k in range(1, r.n):
        for f in json.loads(Path(f"runs/autolabel/r{k}/frames.json").read_text())["frames"]:
            taken[f["clip"]].append(f["frame"])
    pools = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(MINE_CACHE)):
        stem, shot = Path(path).parts[1], int(path.rsplit("_s", 1)[1][: -len(".json")])
        central_shot = next((s for s in cen[stem]["shots"] if s["shot"] == shot), None)
        if any(g in stem for g in gold) or central_shot is None or central_shot["median"] < CENTRAL_K:
            continue
        fps = cen[stem]["fps"]
        for k, dets in json.loads(Path(path).read_text())["frames"].items():
            f, b = int(k), hard_bucket(dets)
            if b and all(abs(f - g) >= MINE_PREV_GAP_S * fps for g in taken[stem]):
                pools[stem][b].append((shot, f))
    return pools


#: Prefill models: version 16 for every class but the ball, version 6 for the ball (docs/decisions.md, 2026-11-18).
PREFILL_MODELS = {"basketball-player-detection-3-ycjdo/16": "not-ball", "basketball-player-detection-3-ycjdo/6": "ball"}


def prefill(r: Round) -> None:
    """Prefill the sampled frames at confidence 0.1; raw detections to prefill_raw.json.

    Round 1 runs Roboflow's NBA detectors (PREFILL_MODELS); later rounds run the previous round's PLK detector.

    """
    if r.n >= 2:
        _prefill_plk(r)
        return
    from inference import get_model

    from bga.detect.roboflow_inference import roboflow_key

    roboflow_key()
    frames = json.loads((r.out / "frames.json").read_text())["frames"]
    raw = {}
    for model_id, keep in PREFILL_MODELS.items():
        model = get_model(model_id=model_id)
        caps = {}
        for fr in frames:
            cap = caps.setdefault(fr["video"], cv2.VideoCapture(fr["video"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame"])
            img = cap.read()[1]
            res = model.infer(img, confidence=0.1, iou_threshold=0.5)[0]
            dets = raw.setdefault(f"{fr['clip']}|{fr['frame']}", [])
            for q in res.predictions:
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
    (r.out / "prefill_raw.json").write_text(json.dumps(raw))
    print(f"{sum(len(v) for v in raw.values())} detections on {len(raw)} frames -> {r.out / 'prefill_raw.json'}")


def _prefill_plk(r: Round) -> None:
    from bga.detect.rfdetr_torch import RFDETRTorch

    det = RFDETRTorch({"confidence": 0.1}, PREFILL_WEIGHTS[r.n])
    det.load()
    frames = json.loads((r.out / "frames.json").read_text())["frames"]
    raw, caps = {}, {}
    for fr in frames:
        cap = caps.setdefault(fr["video"], cv2.VideoCapture(fr["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame"])
        rgb = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        raw[f"{fr['clip']}|{fr['frame']}"] = [[*d, PREFILL_WEIGHTS[r.n]] for d in det._predict(rgb)]
    det.unload()
    (r.out / "prefill_raw.json").write_text(json.dumps(raw))
    print(f"{sum(len(v) for v in raw.values())} detections on {len(raw)} frames -> {r.out / 'prefill_raw.json'}")


PERSON = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee"}
NMS_IOU, MARGIN_CM = 0.5, 100.0


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def proposals(f: dict, min_score: float = 0.15, max_iou: float = 0.3) -> list:
    """The draft's weaker people boxes that overlap no kept person (nor a stronger proposal) at IoU max_iou.

    >>> f = {"boxes": [[0, 0, 10, 20, 0.9, "player"]], "candidates": [[0, 0, 10, 20, 0.3, "player"],
    ...      [50, 0, 60, 20, 0.3, "player-jump-shot"], [51, 0, 61, 20, 0.2, "player"], [80, 0, 90, 9, 0.1, "player"]]}
    >>> [c[5] for c in proposals(f)]
    ['player-jump-shot']

    """
    kept, out = [b for b in f["boxes"] if b[5] in PERSON], []
    for c in sorted((c for c in f["candidates"] if c[5] in PERSON and c[4] >= min_score), key=lambda c: -c[4]):
        if all(_iou(c, k) < max_iou for k in kept + out):
            out.append(c)
    return out


def draft(r: Round) -> None:
    """Boxes at score >= the round's keep threshold after per-class NMS, the rest as candidates; people off the court
    (foot point more than MARGIN_CM outside, under the PLK court model's mapping of the frame) and numbers outside every
    kept person dropped."""
    from bga.court.rfdetr_keypoint import RFDETRKeypoint

    frames = json.loads((r.out / "frames.json").read_text())["frames"]
    raw = json.loads((r.out / "prefill_raw.json").read_text())
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
            (kept if d[4] >= r.keep else cands).append(d)
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
    (r.out / "draft.json").write_text(json.dumps({"frames": out}, indent=1))
    from collections import Counter

    c = Counter(d[5] for f in out for d in f["boxes"])
    print(stats, dict(c), "frames with a ball box:", sum(any(d[5] == "ball" for d in f["boxes"]) for f in out))


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


def ball_queue(r: Round, force: bool) -> None:
    """Write the round's ball queue for the labeller (/boxes?set=r<N>balls): each frame's ball guesses, nothing else.

    From round 2, a frame whose ball is clear by the AUTO_HI_BY_ROUND / AUTO_LO rule is written as done, with that
    ball and a
    note, so the reviewer only sees the rest (``]`` and ``[`` still reach it).

    """
    if r.ball_queue.exists() and not force:
        q = json.loads(r.ball_queue.read_text())
        raise SystemExit(f"{r.ball_queue} exists ({sum(f['status'] != 'todo' for f in q['frames'])} finished); --force")
    frames = []
    for f in json.loads((r.out / "draft.json").read_text())["frames"]:
        if sum(b[5] in PERSON for b in f["boxes"]) < MIN_PEOPLE:
            continue
        balls = [b for b in f["boxes"] if b[5] == "ball"]
        cands = sorted((b for b in f["candidates"] if b[5] == "ball"), key=lambda b: -b[4])[:3]
        every = sorted(balls + [b for b in f["candidates"] if b[5] == "ball"], key=lambda b: -b[4])
        hi = AUTO_HI_BY_ROUND.get(r.n)
        auto = hi is not None and every and every[0][4] >= hi and (len(every) < 2 or every[1][4] < AUTO_LO)
        if auto:
            balls, cands = every[:1], []
        clip_id = f["clip"]
        frames.append({
            "youtube_id": clip_id,
            "clip": clip_id,
            "video": f["video"],
            "shot": f["shot"],
            "frame": f["frame"],
            "status": "done" if auto else "todo",
            "note": f"accepted without review: one ball at {every[0][4]:.2f}" if auto else "",
            "boxes": [
                {
                    "id": i,
                    "xyxy": b[:4],
                    "cls": "ball",
                    "score": b[4],
                    "team": "",
                    "tid": None,
                    "src": "auto" if auto else "det",
                    "flags": [],
                    "hints": [],
                    "alt": {},
                }
                for i, b in enumerate(balls)
            ],
            "candidates": [{"xyxy": b[:4], "cls": "ball", "score": b[4]} for b in cands],
        })
    r.ball_queue.write_text(
        json.dumps(
            {
                "version": f"r{r.n}balls",
                "mode": "ball",
                "title": f"Ball queue, round {r.n}",
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
        f"{len(frames)} frames ({sum(f['status'] == 'done' for f in frames)} accepted without review), "
        f"{sum(len(f['boxes']) for f in frames)} guesses, "
        f"{sum(len(f['candidates']) for f in frames)} weaker guesses -> {r.ball_queue}"
    )


VALID_CLIPS = 3  # whole clips held out as the trainer's PLK validation frames


def manifest(r: Round) -> None:
    """The round's training manifest: Tomasz's ball decisions plus the draft's other boxes, finished frames only.

    A frame enters when its ball-queue status is ``done`` (a ball boxed, or none in view); skipped frames stay out.
    Its boxes are the draft's non-ball boxes (people on the court, numbers inside them, rims and the rest at score
    >= the round's keep threshold) and the ball box from the queue, if any. Every VALID_CLIPS-th clip, in name
    order, goes to validation.

    """
    import hashlib

    from bga.train.leakage import assert_no_gold

    q = json.loads(r.ball_queue.read_text())
    fp = r.out / "draft_fp.json"  # the implementer's first pass (scripts/firstpass.py), when the round has one
    draft = {
        (f["clip"], f["frame"]): f
        for f in json.loads((fp if fp.exists() else r.out / "draft.json").read_text())["frames"]
    }
    people = _people_fixes(r, draft) if fp.exists() else {}
    done = [f for f in q["frames"] if f["status"] == "done" and people.get((f["clip"], f["frame"]), True) is not None]
    clips = sorted({f["clip"] for f in done})
    valid = set(clips[:: max(1, len(clips) // VALID_CLIPS)][:VALID_CLIPS])
    if r.n >= 2:  # the same validation clips in every round, so no clip trains in one round and validates in another
        valid = set(json.loads(Path("data/labels/detect/train_manifest_r1.json").read_text())["params"]["valid_clips"])
    frames = []
    for f in done:
        d = draft[(f["clip"], f["frame"])]
        mine = people.get((f["clip"], f["frame"]))
        boxes = [[*b[:4], b[5]] for b in d["boxes"] if b[5] != "ball" and (mine is None or b[5] not in PERSON)]
        boxes += (mine or []) + [[*b["xyxy"], "ball"] for b in f["boxes"]]
        frames.append({
            "clip": f["clip"],
            "video": f["video"],
            "frame": f["frame"],
            "shot": f["shot"],
            "split": "valid" if f["clip"] in valid else "train",
            "size": [1920, 1080],
            "boxes": [[round(v, 1) for v in b[:4]] + [b[4]] for b in boxes],
        })
    assert_no_gold({f["clip"] for f in frames} | {f["video"] for f in frames})
    r.manifest.parent.mkdir(parents=True, exist_ok=True)
    r.manifest.write_text(
        json.dumps(
            {
                "name": f"r{r.n}",
                "kind": "detect",
                "params": {
                    "ball": "Tomasz's ball queue (done frames; from round 2, also balls accepted by rule)",
                    "other": f"draft boxes at score >= {r.keep}",
                    "prefill": PREFILL_WEIGHTS.get(r.n, PREFILL_MODELS),
                    "valid_clips": sorted(valid),
                },
                "source_labels": {str(r.ball_queue): hashlib.sha256(r.ball_queue.read_bytes()).hexdigest()},
                "frames": frames,
            },
            indent=1,
        )
        + "\n"
    )
    from collections import Counter

    c = Counter(b[4] for f in frames for b in f["boxes"])
    print(f"{len(frames)} frames ({sum(f['split'] == 'valid' for f in frames)} valid) -> {r.manifest}; {dict(c)}")


def _people_fixes(r: Round, draft: dict) -> dict:
    """People boxes from the round's people queue: (clip, frame) -> boxes when done, None when skipped.

    Raises SystemExit while a frame is still to do, so no manifest is built on half-fixed people.

    """
    path = Path(f"data/labels/box_queue_r{r.n}people.json")
    out = {}
    for f in json.loads(path.read_text())["frames"] if path.exists() else []:
        if f["status"] == "todo":
            raise SystemExit(f"{path}: {f['clip']} {f['frame']} is still to do")
        out[(f["clip"], f["frame"])] = (
            [[*map(float, b["xyxy"]), b["cls"]] for b in f["boxes"] if b["cls"] in PERSON]
            if f["status"] == "done"
            else None
        )
    return out


POSS_QUEUE = Path("data/labels/box_queue_possession.json")
POSS_INSTRUCTIONS = (
    "<b>One question per frame: who has the ball?</b> The ball is the orange box.<br>"
    "The loupe shows the most likely holder (the saved answer, else the model's guess, else the player nearest the "
    "ball). <kbd>Space</kbd>: yes, that player holds or dribbles the ball (frame done, next). "
    "<kbd>Tab</kbd>: the next player, nearest the ball first. Or <b>click</b> the player on the frame.<br>"
    "<kbd>N</kbd>: nobody has it (a pass or shot in the air, a loose ball) · <kbd>S</kbd>: can't tell, skip · "
    "<kbd>[</kbd> <kbd>]</kbd> previous / next frame, then answer again to change it · scroll to zoom, "
    "<kbd>F</kbd> fit · <kbd>U</kbd> undo on this frame.<br>"
    "The holder has no box: <kbd>⇧</kbd>-drag a box around him, then <kbd>Enter</kbd>. "
    "A shooter with the ball still in his hands counts as the holder.<br>"
    "<b>Dead ball</b> (a throw-in from behind a line, the referee has it, after a score until the throw-in): "
    "press <kbd>D</kbd> first, then answer as usual. Whoever holds the ball is still the holder."
)


def possession_queue(rounds: list[int]) -> None:
    """Add the finished frames with a ball of the given rounds to the possession queue (/boxes?set=possession).

    Each frame holds the round manifest's people boxes and its ball. Frames already in the queue keep their status
    and boxes, so later rounds can be added without touching decisions made. A round with a first pass
    (draft_fp.json) adds the frames whose holder it settled as done, with that holder (or nobody), for spot checks.

    """
    q = (
        json.loads(POSS_QUEUE.read_text())
        if POSS_QUEUE.exists()
        else {
            "version": "possession",
            "mode": "possession",
            "title": "Possession queue",
            "instructions": POSS_INSTRUCTIONS,
            "classes": [*sorted(PERSON), "ball"],
            "frames": [],
            "meta": {"review_seconds": {}},
        }
    )
    q["instructions"] = POSS_INSTRUCTIONS
    have = {(f["clip"], f["frame"]) for f in q["frames"]}
    added = 0
    for rnd in rounds:
        man = json.loads(Path(f"data/labels/detect/train_manifest_r{rnd}.json").read_text())
        fp_path = Round(rnd).out / "draft_fp.json"
        fp = (
            {(f["clip"], f["frame"]): f["fp"] for f in json.loads(fp_path.read_text())["frames"]}
            if fp_path.exists()
            else {}
        )
        for f in man["frames"]:
            balls = [b for b in f["boxes"] if b[4] == "ball"]
            if not balls or (f["clip"], f["frame"]) in have:
                continue
            boxes = [b for b in f["boxes"] if b[4] in PERSON] + balls[:1]
            settled = fp.get((f["clip"], f["frame"]), {})
            holder = next((i for i, b in enumerate(boxes) if b[:4] == settled.get("holder")), None)
            done = settled.get("holder_decided", False) and (settled.get("holder") is None or holder is not None)
            q["frames"].append({
                "youtube_id": f["clip"],
                "clip": f["clip"],
                "video": f["video"],
                "shot": f["shot"],
                "frame": f["frame"],
                "round": rnd,
                "status": "done" if done else "todo",
                "holder": holder if done else None,
                "dead_ball": False,
                "note": "first pass (implementer)" if done else settled.get("note", ""),
                "candidates": [],
                "boxes": [
                    {
                        "id": i,
                        "xyxy": b[:4],
                        "cls": b[4],
                        "team": "",
                        "tid": None,
                        "src": "ball-queue" if b[4] == "ball" else "draft",
                        "flags": [],
                        "hints": [],
                        "alt": {},
                    }
                    for i, b in enumerate(boxes)
                ],
            })
            added += 1
    POSS_QUEUE.write_text(json.dumps(q, indent=1) + "\n")
    todo = sum(f["status"] == "todo" for f in q["frames"])
    print(f"{added} frames added; {len(q['frames'])} in {POSS_QUEUE}, {todo} to do")


#: A people box overlapping the holder this much is the draft's second box on the same person (17 of 18 cases at 0.97+).
HOLDER_DUP_IOU = 0.8


def possession_manifest(rounds: list[int]) -> None:
    """Each round's manifest with the possession queue's decisions: ``train_manifest_r<N>_poss.json``.

    For a frame done in the possession queue, its people boxes (players, holders, shooters, referees, and holders
    drawn by hand) come from the queue, with any box the reviewer moved. The holder is ``player-in-possession`` or
    keeps a shot class, and every other player is ``player``; a second box on the holder (IoU >= HOLDER_DUP_IOU, the
    draft's other class for the same person) is dropped. Its other boxes (numbers, rims, balls) stay as in the
    round's manifest. Frames the queue skipped or never held (no ball in view) keep the manifest's boxes. The split
    is the round's.

    """
    import hashlib

    from bga.train.leakage import assert_no_gold

    q = json.loads(POSS_QUEUE.read_text())
    poss = {(f["clip"], f["frame"]): f for f in q["frames"] if f["status"] == "done"}
    for n in rounds:
        src = Path(f"data/labels/detect/train_manifest_r{n}.json")
        man = json.loads(src.read_text())
        changed = edited = dups = 0
        for f in man["frames"]:
            pf = poss.get((f["clip"], f["frame"]))
            if pf is None:
                continue
            ours = [b for b in f["boxes"] if b[4] in PERSON]
            theirs = [b for b in pf["boxes"] if b["cls"] in PERSON and b["src"] != "human"]
            same = len(ours) == len(theirs) and all(
                b.get("edited") or a[:4] == [float(v) for v in b["xyxy"]] for a, b in zip(ours, theirs, strict=True)
            )
            if not same:  # the queue was built from this manifest; only boxes the reviewer moved may differ
                raise SystemExit(f"{f['clip']} {f['frame']}: queue people differ from the round-{n} manifest")
            edited += sum(bool(b.get("edited")) for b in theirs)
            mine = [b for b in pf["boxes"] if b["cls"] in PERSON]
            if pf.get("holder") is not None:  # the draft's second box on the holder (another class) would say "not him"
                h = next(b["xyxy"] for b in mine if b["id"] == pf["holder"])
                keep = [b for b in mine if b["id"] == pf["holder"] or _iou(b["xyxy"], h) < HOLDER_DUP_IOU]
                dups += len(mine) - len(keep)
                mine = keep
            people = [[round(float(v), 1) for v in b["xyxy"]] + [b["cls"]] for b in mine]
            new = [b for b in f["boxes"] if b[4] not in PERSON] + people
            changed += sorted(map(str, new)) != sorted(map(str, f["boxes"]))
            f["boxes"] = new
        assert_no_gold({f["clip"] for f in man["frames"]} | {f["video"] for f in man["frames"]})
        man["name"] = f"r{n}_poss"
        man["params"]["people"] = "possession queue (done frames): the holder or none, Tomasz's review"
        man["source_labels"] = {
            **man["source_labels"],
            str(src): hashlib.sha256(src.read_bytes()).hexdigest(),
            str(POSS_QUEUE): hashlib.sha256(POSS_QUEUE.read_bytes()).hexdigest(),
        }
        out = src.with_name(f"train_manifest_r{n}_poss.json")
        out.write_text(json.dumps(man, indent=1) + "\n")
        from collections import Counter

        c = Counter(b[4] for f in man["frames"] for b in f["boxes"])
        print(
            f"{out}: {len(man['frames'])} frames, {changed} changed ({edited} boxes moved by the reviewer, "
            f"{dups} holder duplicates dropped); {dict(c)}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "central",
            "check",
            "sample",
            "mine",
            "prefill",
            "draft",
            "ballqueue",
            "manifest",
            "possqueue",
            "possmanifest",
        ],
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--per-clip", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--rounds", type=int, nargs="*", help="possqueue, possmanifest: rounds (default 1 2)")
    a = ap.parse_args(argv)
    r = Round(a.round)
    r.out.mkdir(parents=True, exist_ok=True)
    {
        "central": lambda: central(a.force),
        "check": check,
        "sample": lambda: sample(r, a.per_clip, a.seed),
        "mine": lambda: mine(r, a.seed),
        "prefill": lambda: prefill(r),
        "draft": lambda: draft(r),
        "ballqueue": lambda: ball_queue(r, a.force),
        "manifest": lambda: manifest(r),
        "possqueue": lambda: possession_queue(a.rounds or [1, 2]),
        "possmanifest": lambda: possession_manifest(a.rounds or [1, 2]),
    }[a.cmd]()


if __name__ == "__main__":
    main()
