#!/usr/bin/env python3
"""Prefill gold-shot tracks for the labeller with Skalski's NBA player model and ByteTrack.

    uv run --extra prefill scripts/prefill_tracks.py                  # every shot in data/labels/gold.json
    uv run --extra prefill scripts/prefill_tracks.py --clip-id 4XMD9FWmQws
    uv run --extra prefill scripts/prefill_tracks.py --track-every 1   # detect and track every frame
    uv run --extra prefill scripts/prefill_tracks.py --det-conf 0.1 --lost-buffer 90  # low boxes

Labelling aid only (plan section 7), never a pipeline layer. Per gold clip writes
runs/<clip>/prefill/<model-slug>/
  detections.jsonl  every detection on every sampled frame (all 10 classes, for ball and number crops)
  tracks.json       players and referees linked by ByteTrack within each shot, labeller prefill format
  meta.json         model id, thresholds, timings
Sampled frames are the multiples of the gold stride inside each shot, so they line up with the
labeller's keyframes. With --track-every N the detector and tracker run on every Nth frame and only the
sampled frames are written. Tracks never cross a shot boundary (the tracker is reset per shot).
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
    ap.add_argument(
        "--track-every", type=int, default=0, help="frames between tracker updates (default: stride)"
    )
    ap.add_argument("--det-conf", type=float, default=CONFIDENCE, help="detector threshold fed to ByteTrack")
    ap.add_argument("--lost-buffer", type=int, default=30, help="ByteTrack lost_track_buffer (at 30 fps)")
    ap.add_argument(
        "--max-aspect", type=float, default=0.0, help="drop person boxes wider than this x height"
    )
    a = ap.parse_args(argv)
    gold = json.loads(a.gold.read_text())
    stride = gold["stride"]
    step = a.track_every or stride
    assert stride % step == 0, "--track-every must divide the gold stride"
    model = get_model(model_id=MODEL_ID)
    slug = MODEL_ID.replace("/", "_v") + ("" if step == stride else f"_track{step}")
    if a.det_conf != CONFIDENCE or a.lost_buffer != 30 or a.max_aspect:
        slug += f"_c{a.det_conf:g}_lb{a.lost_buffer}_ar{a.max_aspect:g}"
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
                # New tracks only from boxes at Skalski's threshold; lower boxes only extend tracks.
                tracker = sv.ByteTrack(
                    track_activation_threshold=CONFIDENCE,
                    lost_track_buffer=a.lost_buffer,
                    frame_rate=round(labels["fps"] / step),
                )
                local_to_global: dict[int, int] = {}
                first = -(-shot["start"] // step) * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                pos, ok, frame = first, False, None
                for idx in range(first, shot["end"] + 1, step):
                    while pos <= idx:  # sequential decode: cheaper than seeking for small steps
                        ok, frame = cap.read()
                        pos += 1
                    if not ok:
                        break
                    sampled = idx % stride == 0
                    t0 = time.time()
                    r = model.infer(frame, confidence=a.det_conf, iou_threshold=IOU_THRESHOLD)[0]
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
                    if sampled:
                        det_f.write(
                            json.dumps({"frame_idx": idx, "shot": shot["shot"], "detections": dets}) + "\n"
                        )
                    people = [
                        d
                        for d in dets
                        if d["class"] in PERSON_CLASSES
                        and not (
                            a.max_aspect
                            and d["xyxy"][2] - d["xyxy"][0] > a.max_aspect * (d["xyxy"][3] - d["xyxy"][1])
                        )
                    ]
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
                    if sampled:
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
            "confidence": a.det_conf,
            "track_activation": CONFIDENCE,
            "lost_track_buffer": a.lost_buffer,
            "max_aspect": a.max_aspect,
            "iou_threshold": IOU_THRESHOLD,
            "tracker": "supervision.ByteTrack per shot",
            "track_every": step,
            "stride": stride,
            "frames": n_frames,
            "tracks": next_tid - 1,
            "ms_per_frame": round(1000 * t_infer / max(n_frames, 1), 1),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(g["youtube_id"], json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
