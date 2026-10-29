#!/usr/bin/env python3
"""Write every frame of each gold shot as a JPEG folder for SAM 2 video propagation, plus prompts.

    uv run scripts/extract_shot_frames.py --prefill <prefill-variant-dir-name>

Per gold shot: runs/<clip>/sam_frames/shot_<n>/00000.jpg ... (long side 1024, SAM 2's input size) and
prompts.json with the detector's person boxes (confidence >= 0.4, no wider than tall) on the sampled
frames, scaled to the JPEG size, keyed by local frame index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

PERSON = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("data/labels/gold.json"))
    ap.add_argument("--prefill", default="basketball-player-detection-3-ycjdo_v4_c0.1_lb90_ar1")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--prompt-conf", type=float, default=0.4)
    a = ap.parse_args(argv)
    gold = json.loads(a.gold.read_text())
    for g in gold["games"]:
        clip = json.loads((Path("data/labels") / f"{g['clip']}.json").read_text())
        dets = {}
        for line in (Path("runs") / g["clip"] / "prefill" / a.prefill / "detections.jsonl").open():
            r = json.loads(line)
            dets[r["frame_idx"]] = r["detections"]
        cap = cv2.VideoCapture(clip["clip"])
        scale = a.long_side / max(clip["width"], clip["height"])
        size = (round(clip["width"] * scale), round(clip["height"] * scale))
        for shot in g["shots"]:
            out = Path("runs") / g["clip"] / "sam_frames" / f"shot_{shot['shot']}"
            out.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot["start"])
            prompts = {}
            for i, f in enumerate(range(shot["start"], shot["end"] + 1)):
                ok, img = cap.read()
                if not ok:
                    break
                cv2.imwrite(
                    str(out / f"{i:05d}.jpg"),
                    cv2.resize(img, size, cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                boxes = [
                    [round(v * scale, 1) for v in d["xyxy"]]
                    for d in dets.get(f, [])
                    if d["class"] in PERSON
                    and d["score"] >= a.prompt_conf
                    and d["xyxy"][2] - d["xyxy"][0] <= d["xyxy"][3] - d["xyxy"][1]
                ]
                if f in dets:
                    prompts[i] = boxes
            meta = {
                "clip": g["clip"],
                "shot": shot["shot"],
                "start": shot["start"],
                "end": shot["end"],
                "scale": scale,
                "size": size,
                "prompts": prompts,
            }
            (out / "prompts.json").write_text(json.dumps(meta))
            print(out, i + 1, "frames", sum(map(len, prompts.values())), "prompt boxes", flush=True)
        cap.release()


if __name__ == "__main__":
    main()
