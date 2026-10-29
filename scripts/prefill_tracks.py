#!/usr/bin/env python3
"""Prefill gold-shot tracks for the labeller with Skalski's NBA player model and ByteTrack.

    uv run --extra prefill scripts/prefill_tracks.py                  # every shot in data/labels/gold.json
    uv run --extra prefill scripts/prefill_tracks.py --clip-id 4XMD9FWmQws

Labelling aid only (plan section 7), never a pipeline layer. Per gold clip writes
runs/<clip>/prefill/<model-slug>/
  detections.jsonl  every detection on every sampled frame (all 10 classes, for ball and number crops)
  tracks.json       players and referees linked by ByteTrack within each shot, labeller prefill format
  meta.json         model id, thresholds, timings
Sampled frames are the multiples of the gold stride inside each shot, so they line up with the
labeller's keyframes. Tracks never cross a shot boundary (the tracker is reset per shot).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = "basketball-player-detection-3-ycjdo/4"  # as in Skalski's notebook
CONFIDENCE = 0.4
IOU_THRESHOLD = 0.9
PERSON_CLASSES = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
}


def main(argv=None):
    import supervision as sv
    from inference import get_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("data/labels/gold.json"))
    ap.add_argument("--clip-id", action="append", help="YouTube id(s) to run; default all gold clips")
    a = ap.parse_args(argv)
    gold = json.loads(a.gold.read_text())
    stride = gold["stride"]
    model = get_model(model_id=MODEL_ID)
    slug = MODEL_ID.replace("/", "_v")
    for g in gold["games"]:
        if a.clip_id and g["youtube_id"] not in a.clip_id:
            continue
        labels = json.loads((Path("data/labels") / f"{g['clip']}.json").read_text())
        out = Path("runs") / g["clip"] / "prefill" / slug
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(labels["clip"])
        frames_out, n_frames, t_infer = [], 0, 0.0
        next_tid = 1
        with (out / "detections.jsonl").open("w") as det_f:
            for shot in g["shots"]:
                tracker = sv.ByteTrack(frame_rate=round(labels["fps"] / stride))
                local_to_global: dict[int, int] = {}
                first = -(-shot["start"] // stride) * stride
                for idx in range(first, shot["end"] + 1, stride):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    t0 = time.time()
                    r = model.infer(frame, confidence=CONFIDENCE, iou_threshold=IOU_THRESHOLD)[0]
                    t_infer += time.time() - t0
                    n_frames += 1
                    dets = [
                        {
                            "xyxy": [
                                p.x - p.width / 2,
                                p.y - p.height / 2,
                                p.x + p.width / 2,
                                p.y + p.height / 2,
                            ],
                            "class": p.class_name,
                            "score": round(float(p.confidence), 4),
                        }
                        for p in r.predictions
                    ]
                    det_f.write(
                        json.dumps({"frame_idx": idx, "shot": shot["shot"], "detections": dets}) + "\n"
                    )
                    people = [d for d in dets if d["class"] in PERSON_CLASSES]
                    sv_dets = sv.Detections(
                        xyxy=np.array([d["xyxy"] for d in people], dtype=np.float32).reshape(-1, 4),
                        confidence=np.array([d["score"] for d in people], dtype=np.float32),
                        class_id=np.array([d["class"] == "referee" for d in people], dtype=int),
                    )
                    tracked = tracker.update_with_detections(sv_dets)
                    tracks = []
                    for xyxy, tid, cls in zip(
                        tracked.xyxy, tracked.tracker_id, tracked.class_id, strict=True
                    ):
                        if tid not in local_to_global:
                            local_to_global[tid] = next_tid
                            next_tid += 1
                        tracks.append(
                            {
                                "track_id": local_to_global[tid],
                                "xyxy": [round(float(v), 1) for v in xyxy],
                                "is_referee": bool(cls),
                            }
                        )
                    frames_out.append({"frame_idx": idx, "shot": shot["shot"], "tracks": tracks})
        cap.release()
        # one team label per track: referee if most of its boxes were referee boxes
        votes: dict[int, Counter] = {}
        for fr in frames_out:
            for t in fr["tracks"]:
                votes.setdefault(t["track_id"], Counter())[t.pop("is_referee")] += 1
        for fr in frames_out:
            for t in fr["tracks"]:
                if votes[t["track_id"]].most_common(1)[0][0]:
                    t["team"] = "ref"
        doc = {"clip_path": labels["clip"], "model": MODEL_ID, "frames": frames_out}
        (out / "tracks.json").write_text(json.dumps(doc))
        meta = {
            "model": MODEL_ID,
            "confidence": CONFIDENCE,
            "iou_threshold": IOU_THRESHOLD,
            "tracker": "supervision.ByteTrack per shot",
            "stride": stride,
            "frames": n_frames,
            "tracks": next_tid - 1,
            "ms_per_frame": round(1000 * t_infer / max(n_frames, 1), 1),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(g["youtube_id"], json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
