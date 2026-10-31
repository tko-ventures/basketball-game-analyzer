#!/usr/bin/env python3
"""Client demo video: detection, tracking, teams, jersey numbers and a 2D FIBA court map, end to end.

    uv run --extra prefill --extra demo scripts/demo_video.py --out runs/demo/demo.mp4

A showcase, not a measured pipeline (no metrics row, no gold set): it follows Skalski's notebook with
the models already used for the gold-set prefill, and every output is automatic and uncorrected.
  tracks   SAM 2.1 identities fused with NBA-model boxes (runs/<clip>/prefill/sam21_hiera_large)
  teams    sports.TeamClassifier (SigLIP + UMAP + KMeans) on the central 40% of player boxes, per clip;
           the brighter cluster is the team whose meta kit is "light"
  numbers  roboflow-jvuqo basketball-jersey-numbers-ocr/3 (SmolVLM) on `number` boxes lying inside a
           player box (IoS >= 0.9), padded 10 px, "Read the number."; a track's number needs two
           agreeing reads and is shown with the roster name when the roster has it
  court    basketball-court-detection-2/14 keypoints (confidence > 0.5, >= 4) -> homography per frame
           onto League.FIBA; foot point = box bottom-centre, smoothed per track
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SHOTS = [("4XMD9FWmQws", 4), ("OSp445e0mXU", 10), ("1jDaJBkUGmc", 3)]
PREFILL = "basketball-player-detection-3-ycjdo_v4_c0.1_lb90_ar1"
COURT_MODEL = "basketball-court-detection-2/14"
OCR_MODEL = "basketball-jersey-numbers-ocr/3"
REF_BGR = (40, 40, 40)


def clip_files(yid):
    labels = Path(glob.glob(f"data/labels/*_{yid}_*.json")[0])
    clip = labels.stem
    return (
        labels,
        Path("runs") / clip / "prefill" / "sam21_hiera_large" / "tracks.json",
        (Path("runs") / clip / "prefill" / PREFILL / "detections.jsonl"),
    )


def interpolate(tracks_json, shot, step, max_gap=15):
    """Per track: box for every rendered frame, linear between sampled boxes up to max_gap frames apart."""
    obs = defaultdict(list)
    ref = Counter()
    for fr in tracks_json["frames"]:
        if fr["shot"] != shot:
            continue
        for t in fr["tracks"]:
            obs[t["track_id"]].append((fr["frame_idx"], np.array(t["xyxy"], dtype=float)))
            ref[t["track_id"]] += t.get("team") == "ref"
    per_frame = defaultdict(dict)
    for tid, seq in obs.items():
        seq.sort(key=lambda x: x[0])
        for (f0, b0), (f1, b1) in zip(seq, seq[1:] + [seq[-1]], strict=True):
            if f1 - f0 > max_gap:
                f1, b1 = f0, b0
            for f in range(f0 - f0 % step, max(f1, f0) + 1, step):
                if f < f0:
                    continue
                w = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
                per_frame[f][tid] = b0 + (b1 - b0) * w
    is_ref = {tid: ref[tid] > len(obs[tid]) / 2 for tid in obs}
    return per_frame, obs, is_ref


def crop(frame, xyxy, factor=1.0, pad=0):
    x1, y1, x2, y2 = xyxy
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    x1, x2, y1, y2 = cx - w / 2 - pad, cx + w / 2 + pad, cy - h / 2 - pad, cy + h / 2 + pad
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
    return frame[y1:y2, x1:x2]


def ios(inner, outer):
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def roster_names(labels, side):
    key = labels["meta"][side].get("roster")
    p = Path("data/rosters") / f"{key}.json"
    if not key or not p.exists():
        return {}
    season = json.loads(p.read_text())["seasons"].get(labels["meta"]["season"], {})
    return {pl["number"]: pl["name"] for pl in season.get("players", []) if pl.get("number")}


def main(argv=None):
    import supervision as sv
    from inference import get_model
    from sports import TeamClassifier, ViewTransformer
    from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/demo/demo.mp4"))
    ap.add_argument("--step", type=int, default=2, help="render every Nth frame (2: 25 fps from 50 fps)")
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    court_model, ocr_model = get_model(model_id=COURT_MODEL), get_model(model_id=OCR_MODEL)
    cfg = CourtConfiguration(league=League.FIBA)
    vertices = np.array(cfg.vertices, dtype=np.float32)
    scale, padding = 0.25, 20
    base_court = draw_court(cfg, scale=scale, padding=padding, line_thickness=2)
    writer, summary = None, []

    for yid, shot in SHOTS:
        labels_path, tracks_path, dets_path = clip_files(yid)
        labels = json.loads(labels_path.read_text())
        tracks_json = json.loads(tracks_path.read_text())
        dets = {r["frame_idx"]: r["detections"] for r in map(json.loads, dets_path.open())}
        per_frame, obs, is_ref = interpolate(tracks_json, shot, a.step)
        cap = cv2.VideoCapture(labels["clip"])

        def read(i, cap=cap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            return cap.read()[1]

        # Teams: SigLIP + UMAP + KMeans on central crops, majority per track.
        crops, owners = [], []
        for tid, seq in obs.items():
            if is_ref[tid]:
                continue
            for f, b in seq[:: max(1, len(seq) // 8)][:8]:
                c = crop(read(f), b, factor=0.4)
                if c.size:
                    crops.append(c)
                    owners.append(tid)
        tc = TeamClassifier(device="mps")
        tc.fit(crops)
        pred = tc.predict(crops)
        votes = defaultdict(Counter)
        for tid, p in zip(owners, pred, strict=True):
            votes[tid][int(p)] += 1
        cluster = {tid: v.most_common(1)[0][0] for tid, v in votes.items()}
        bright = [
            np.mean(
                [
                    cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2].mean()
                    for c, p in zip(crops, pred, strict=True)
                    if p == k
                ]
                or [0]
            )
            for k in (0, 1)
        ]
        meta = labels["meta"]
        light_side = "home" if meta["home"].get("kit") == "light" else "away"
        dark_side = "away" if light_side == "home" else "home"
        side_of_cluster = {int(np.argmax(bright)): light_side, int(np.argmin(bright)): dark_side}
        team_bgr = {}
        for k, side in side_of_cluster.items():
            cs = [c for c, p in zip(crops, pred, strict=True) if p == k]
            team_bgr[side] = tuple(int(v) for v in np.median([c.reshape(-1, 3).mean(0) for c in cs], axis=0))

        # Numbers: OCR on number boxes inside player boxes, every sampled frame of the shot.
        reads = defaultdict(Counter)
        for fr in tracks_json["frames"]:
            if fr["shot"] != shot:
                continue
            nums = [d for d in dets.get(fr["frame_idx"], []) if d["class"] == "number" and d["score"] >= 0.3]
            if not nums:
                continue
            frame = read(fr["frame_idx"])
            for n in nums:
                owners_n = [(ios(n["xyxy"], t["xyxy"]), t["track_id"]) for t in fr["tracks"]]
                best = max(owners_n, default=(0, None))
                if best[0] < 0.9 or is_ref.get(best[1]):
                    continue
                c = crop(frame, n["xyxy"], pad=10)
                if c.size == 0:
                    continue
                # inference >= 1.8: infer(image, prompt=...) (the notebook's predict(image, prompt) is gone)
                text = ocr_model.infer(cv2.resize(c, (224, 224)), prompt="Read the number.")[0].response
                digits = re.sub(r"\D", "", str(text))
                if digits and len(digits) <= 2:
                    reads[best[1]][digits] += 1
        number = {tid: v.most_common(1)[0][0] for tid, v in reads.items() if v.most_common(1)[0][1] >= 2}
        names = {side: roster_names(labels, side) for side in ("home", "away")}

        def label(
            tid, is_ref=is_ref, side_of_cluster=side_of_cluster, cluster=cluster, number=number, names=names
        ):
            if is_ref[tid]:
                return "REF"
            side = side_of_cluster.get(cluster.get(tid))
            num = number.get(tid)
            if not num:
                return ""
            name = names.get(side, {}).get(num, "")
            return f"#{num} {name.split()[-1] if name else ''}".strip()

        # Render.
        title = f"{meta['home']['team']} – {meta['away']['team']}  ·  {meta['season']}"
        smooth: dict[int, np.ndarray] = {}
        H_last = None
        frames = sorted(per_frame)
        if writer is None:
            fps = labels["fps"] / a.step
            writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
        n_h = 0
        for f in frames:
            frame = read(f)
            boxes = per_frame[f]
            tids = list(boxes)
            xyxy = np.array([boxes[t] for t in tids]).reshape(-1, 4)
            kp = court_model.infer(frame, confidence=0.3)[0]
            if kp.predictions:
                pts = kp.predictions[0].keypoints
                conf = np.array([p.confidence for p in pts])
                mask = conf > 0.5
                if mask.sum() >= 4 and len(pts) == len(vertices):
                    src = np.array([[p.x, p.y] for p in pts], dtype=np.float32)[mask]
                    try:
                        H_last = ViewTransformer(source=src, target=vertices[mask])
                        n_h += 1
                    except ValueError:
                        pass
            court = base_court.copy()
            if H_last is not None and len(tids):
                feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
                xy = H_last.transform_points(feet)
                for t, p in zip(tids, xy, strict=True):
                    smooth[t] = p if t not in smooth else 0.6 * smooth[t] + 0.4 * p
                for t in tids:
                    p = smooth[t]
                    if not (-200 < p[0] < 3000 and -200 < p[1] < 1700):
                        continue
                    side = side_of_cluster.get(cluster.get(t))
                    bgr = REF_BGR if is_ref[t] else team_bgr.get(side, (128, 128, 128))
                    num = "" if is_ref[t] else number.get(t, "")
                    court = draw_points_on_court(
                        cfg,
                        xy=p[None],
                        labels=[num],
                        fill_color=sv.Color(bgr[2], bgr[1], bgr[0]),
                        edge_color=sv.Color.WHITE,
                        size=11,
                        scale=scale,
                        padding=padding,
                        line_thickness=2,
                        court=court,
                    )
            for t, b in zip(tids, xyxy, strict=True):
                side = side_of_cluster.get(cluster.get(t))
                bgr = REF_BGR if is_ref[t] else team_bgr.get(side, (128, 128, 128))
                x1, y1, x2, y2 = b.astype(int)
                cx, w = (x1 + x2) // 2, max(12, (x2 - x1) // 2)
                cv2.ellipse(
                    frame, (cx, y2), (w, max(5, w // 3)), 0, -45, 235, (255, 255, 255), 5, cv2.LINE_AA
                )
                cv2.ellipse(frame, (cx, y2), (w, max(5, w // 3)), 0, -45, 235, bgr, 3, cv2.LINE_AA)
                txt = label(t)
                if txt:
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(
                        frame, (cx - tw // 2 - 4, y1 - th - 12), (cx + tw // 2 + 4, y1 - 4), bgr, -1
                    )
                    cv2.putText(
                        frame,
                        txt,
                        (cx - tw // 2, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            # Court inset top-right, title and legend top-left.
            ch, cw = court.shape[:2]
            x0, y0 = 1920 - cw - 24, 24
            roi = frame[y0 : y0 + ch, x0 : x0 + cw]
            frame[y0 : y0 + ch, x0 : x0 + cw] = cv2.addWeighted(roi, 0.15, court, 0.85, 0)
            cv2.rectangle(frame, (20, 20), (760, 112), (20, 20, 20), -1)
            cv2.putText(
                frame, title, (34, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
            lx = 34
            for side in ("home", "away"):
                cv2.circle(frame, (lx + 8, 88), 9, team_bgr.get(side, (128, 128, 128)), -1)
                cv2.putText(
                    frame,
                    meta[side]["team"],
                    (lx + 24, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                lx += 30 + cv2.getTextSize(meta[side]["team"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] + 30
            cv2.putText(
                frame,
                "automatic, uncorrected",
                (lx + 10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
        cap.release()
        summary.append(
            {
                "clip": yid,
                "shot": shot,
                "frames": len(frames),
                "tracks": len(obs),
                "numbers": {str(t): n for t, n in number.items()},
                "homography_frames": n_h,
                "team_crops": len(crops),
            }
        )
        print(json.dumps(summary[-1]), flush=True)
    writer.release()
    (a.out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
