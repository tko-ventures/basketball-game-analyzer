#!/usr/bin/env python3
"""Client demo video: detection, tracking, teams, jersey numbers and a 2D FIBA court map, end to end.

    uv run --extra prefill --extra demo scripts/demo_video.py --out runs/demo/demo.mp4
    uv run --extra prefill --extra demo scripts/demo_video.py --recompute   # ignore the per-shot cache

A showcase, not a measured pipeline (no metrics row, no gold set): it follows Skalski's notebook with
the models already used for the gold-set prefill, and every output is automatic and uncorrected.
  tracks   SAM 2.1 identities fused with NBA-model boxes (runs/<clip>/prefill/sam21_hiera_large)
  teams    sports.TeamClassifier (SigLIP + UMAP + KMeans) on the central 40% of player boxes, per shot;
           the brighter cluster is the team whose meta kit is "light"
  numbers  roboflow-jvuqo basketball-jersey-numbers-ocr/3 (SmolVLM) on the Mac GPU, max 4 new tokens,
           on `number` boxes inside a player box (IoS >= 0.9), padded 10 px, "Read the number.";
           a track stops being read once two reads agree (or after --max-reads), and its number needs
           two agreeing reads; names come from the roster
  court    basketball-court-detection-2/14 keypoints (confidence > 0.5, >= 4) -> homography per frame
           onto League.FIBA; foot point = box bottom-centre, smoothed per track
Teams, numbers and homographies are cached per shot in <out dir>/cache, so a re-render only redraws.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SHOTS = [("4XMD9FWmQws", 4), ("OSp445e0mXU", 10), ("1jDaJBkUGmc", 3)]
PREFILL = "basketball-player-detection-3-ycjdo_v4_c0.1_lb90_ar1"
COURT_MODEL = "basketball-court-detection-2/14"
OCR_MODEL = "basketball-jersey-numbers-ocr/3"
REF_BGR = (40, 40, 40)
# Display colours (BGR) per team and kit, chosen by eye for the demo; the measured kit colour (median of
# crop means) is muddy because the crops include floor and crowd. Unknown teams fall back to it.
DISPLAY_BGR = {
    ("PGE Start", "light"): (60, 60, 220),  # white kit with red trim: shown red
    ("Dziki Warszawa", "dark"): (120, 50, 20),  # navy
    ("Dziki Warszawa", "light"): (235, 235, 235),  # cream/white
    ("Legia", "dark"): (60, 140, 30),  # green
    ("Energa Trefl", "light"): (0, 200, 245),  # yellow
    ("Zastal", "dark"): (60, 140, 30),  # green
}
GREY = (128, 128, 128)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def clip_files(yid):
    labels = Path(glob.glob(f"data/labels/*_{yid}_*.json")[0])
    run = Path("runs") / labels.stem / "prefill"
    return labels, run / "sam21_hiera_large" / "tracks.json", run / PREFILL / "detections.jsonl"


class FrameReader:
    """Decode forward with grab() when the wanted frame is close ahead; seek only on long jumps."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.next = 0

    def __call__(self, i):
        if i < self.next or i - self.next > 60:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.next = i
        while self.next < i:
            self.cap.grab()
            self.next += 1
        self.next += 1
        return self.cap.read()[1]

    def release(self):
        self.cap.release()


def interpolate(tracks_json, shot, step, max_gap=15):
    """Per track: box for every rendered frame, linear between sampled boxes up to max_gap frames apart."""
    obs, ref = defaultdict(list), Counter()
    for fr in tracks_json["frames"]:
        if fr["shot"] != shot:
            continue
        for t in fr["tracks"]:
            obs[t["track_id"]].append((fr["frame_idx"], np.array(t["xyxy"], dtype=float)))
            ref[t["track_id"]] += t.get("team") == "ref"
    per_frame = defaultdict(dict)
    for tid, seq in obs.items():
        seq.sort(key=lambda x: x[0])
        for (f0, b0), (f1, b1) in zip(seq, [*seq[1:], seq[-1]], strict=True):
            if f1 - f0 > max_gap:
                f1, b1 = f0, b0
            for f in range(f0 + (-f0) % step, max(f1, f0) + 1, step):
                w = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
                per_frame[f][tid] = b0 + (b1 - b0) * w
    is_ref = {tid: ref[tid] > len(obs[tid]) / 2 for tid in obs}
    return per_frame, obs, is_ref


def crop(frame, xyxy, factor=1.0, pad=0):
    x1, y1, x2, y2 = xyxy
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(cx - w / 2 - pad)), max(0, int(cy - h / 2 - pad))
    x2, y2 = min(W, int(cx + w / 2 + pad)), min(H, int(cy + h / 2 + pad))
    return frame[y1:y2, x1:x2]


def ios(inner, outer):
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def teams(obs, is_ref, read, meta):
    """SigLIP + UMAP + KMeans (Skalski) on central crops; majority per track; brighter cluster = light kit."""
    from sports import TeamClassifier

    wanted = defaultdict(list)  # frame -> [(tid, box)], read in frame order
    for tid, seq in obs.items():
        if not is_ref[tid]:
            for f, b in seq[:: max(1, len(seq) // 8)][:8]:
                wanted[f].append((tid, b))
    crops, owners = [], []
    for f in sorted(wanted):
        frame = read(f)
        for tid, b in wanted[f]:
            c = crop(frame, b, factor=0.4)
            if c.size:
                crops.append(c)
                owners.append(tid)
    tc = TeamClassifier(device="mps")
    tc.fit(crops)
    pred = [int(p) for p in tc.predict(crops)]
    votes = defaultdict(Counter)
    for tid, p in zip(owners, pred, strict=True):
        votes[tid][p] += 1
    cluster = {tid: v.most_common(1)[0][0] for tid, v in votes.items()}
    of = {k: [c for c, p in zip(crops, pred, strict=True) if p == k] for k in (0, 1)}
    bright = [np.mean([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2].mean() for c in of[k]] or [0]) for k in (0, 1)]
    light = "home" if meta["home"].get("kit") == "light" else "away"
    dark = "away" if light == "home" else "home"
    side_of_cluster = {int(np.argmax(bright)): light, int(np.argmin(bright)): dark}
    team_bgr = {
        side: [int(v) for v in np.median([c.reshape(-1, 3).mean(0) for c in of[k]], axis=0)]
        for k, side in side_of_cluster.items()
        if of[k]
    }
    return cluster, side_of_cluster, team_bgr, len(crops)


def load_ocr():
    """Skalski's number reader on the Mac GPU (MPS); `inference` alone defaults to CPU and 400 new tokens."""
    import torch
    from inference_models import AutoModel

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return AutoModel.from_pretrained(OCR_MODEL, api_key=os.environ.get("ROBOFLOW_API_KEY"), device=device)


def numbers(tracks_json, shot, dets, is_ref, read, model, max_reads):
    reads, n = defaultdict(Counter), 0
    for fr in tracks_json["frames"]:
        if fr["shot"] != shot:
            continue
        frame = None
        for d in dets.get(fr["frame_idx"], []):
            if d["class"] != "number" or d["score"] < 0.3:
                continue
            score, tid = max(((ios(d["xyxy"], t["xyxy"]), t["track_id"]) for t in fr["tracks"]), default=(0, None))
            if score < 0.9 or is_ref.get(tid):
                continue
            r = reads[tid]
            if (r and r.most_common(1)[0][1] >= 2) or sum(r.values()) >= max_reads:
                continue  # settled, or not readable enough
            frame = read(fr["frame_idx"]) if frame is None else frame
            c = crop(frame, d["xyxy"], pad=10)
            if c.size == 0:
                continue
            text = model.prompt(images=cv2.resize(c, (224, 224)), prompt="Read the number.", max_new_tokens=4)[0]
            digits = re.sub(r"\D", "", str(text))
            n += 1
            if 0 < len(digits) <= 2:
                r[digits] += 1
    return {tid: v.most_common(1)[0][0] for tid, v in reads.items() if v.most_common(1)[0][1] >= 2}, n


def homographies(frames, read, court_model, vertices):
    """Per rendered frame: 3x3 image-to-court matrix, or None when fewer than 4 confident landmarks."""
    out = {}
    for f in frames:
        kp = court_model.infer(read(f), confidence=0.3)[0]
        out[f] = None
        if kp.predictions and len(kp.predictions[0].keypoints) == len(vertices):
            pts = kp.predictions[0].keypoints
            mask = np.array([p.confidence for p in pts]) > 0.5
            if mask.sum() >= 4:
                src = np.array([[p.x, p.y] for p in pts], dtype=np.float32)[mask]
                m, _ = cv2.findHomography(src, vertices[mask])
                out[f] = m.tolist() if m is not None else None
    return out


def roster_names(labels, side):
    key = labels["meta"][side].get("roster")
    p = Path("data/rosters") / f"{key}.json"
    if not key or not p.exists():
        return {}
    season = json.loads(p.read_text())["seasons"].get(labels["meta"]["season"], {})
    return {pl["number"]: pl["name"] for pl in season.get("players", []) if pl.get("number")}


def text(frame, s, org, size=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(frame, s, org, FONT, size, color, thick, cv2.LINE_AA)


@dataclass(frozen=True)
class _Look:
    """How one shot's tracks are drawn: referee flags, team colours, numbers and roster names."""

    is_ref: dict
    cluster: dict
    side_of_cluster: dict
    team_bgr: dict
    number: dict
    names: dict

    def colour(self, t):
        return REF_BGR if self.is_ref[t] else self.team_bgr.get(self.side_of_cluster.get(self.cluster.get(t)), GREY)

    def label(self, t):
        if self.is_ref[t]:
            return "REF"
        num = self.number.get(t)
        if not num:
            return ""
        name = self.names.get(self.side_of_cluster.get(self.cluster.get(t)), {}).get(num, "")
        return f"#{num} {name.split()[-1] if name else ''}".strip()


def ink(bgr):  # readable text on a team colour
    return (20, 20, 20) if sum(bgr) > 500 else (255, 255, 255)


@dataclass(frozen=True)
class _Court:
    """The FIBA court drawing: configuration, its vertices, the empty court and its scale, and the drawing API."""

    sv: object  # the supervision module
    draw_points: object  # sports.basketball.draw_points_on_court
    cfg: object
    vertices: np.ndarray
    base: np.ndarray
    scale: float
    padding: int


def _shot_cache(cache, a, models, shot_in):
    """Teams, numbers and homographies for one shot: read from the cache, or computed and cached, plus timings."""
    timing = {}
    if cache.exists() and not a.recompute:
        return json.loads(cache.read_text()), timing
    obs, is_ref, read, meta, tracks_json, shot, dets, frames, vertices = shot_in
    c = {}
    t0 = time.time()
    cluster, side_of_cluster, team_bgr, c["team_crops"] = teams(obs, is_ref, read, meta)
    c.update(cluster=cluster, side_of_cluster=side_of_cluster, team_bgr=team_bgr)
    timing["teams_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    models["ocr"] = models["ocr"] or load_ocr()
    c["number"], c["ocr_reads"] = numbers(tracks_json, shot, dets, is_ref, read, models["ocr"], a.max_reads)
    timing["numbers_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    models["court"] = models["court"] or models["get_model"](model_id=COURT_MODEL)
    c["H"] = homographies(frames, read, models["court"], vertices)
    timing["court_s"] = round(time.time() - t0, 1)
    cache.write_text(json.dumps(c))
    return json.loads(cache.read_text()), timing  # one code path: JSON keys are strings from here on


def _draw_points(court, smooth, tids, xyxy, H_last, look, ct):
    """Project the tracks' foot points through the homography, smooth them, and draw them on the court."""
    sv = ct.sv
    feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1).astype(np.float32)
    xy = cv2.perspectiveTransform(feet[None], H_last.astype(np.float32))[0]
    for t, p in zip(tids, xy, strict=True):
        smooth[t] = p if t not in smooth else 0.6 * smooth[t] + 0.4 * p
        if -200 < smooth[t][0] < 3000 and -200 < smooth[t][1] < 1700:
            bgr = look.colour(t)
            court = ct.draw_points(
                ct.cfg,
                xy=smooth[t][None],
                labels=["" if look.is_ref[t] else look.number.get(t, "")],
                fill_color=sv.Color(bgr[2], bgr[1], bgr[0]),
                text_color=sv.Color(*ink(bgr)[::-1]),
                edge_color=sv.Color.BLACK if sum(bgr) > 500 else sv.Color.WHITE,
                size=14,
                scale=ct.scale,
                padding=ct.padding,
                line_thickness=2,
                court=court,
            )
    return court


def _draw_boxes(frame, tids, xyxy, look):
    """Team-coloured ellipse at each player's feet and a number-and-name tag above the box."""
    for t, b in zip(tids, xyxy, strict=True):
        bgr = look.colour(t)
        x1, y1, x2, y2 = b.astype(int)
        cx, w = (x1 + x2) // 2, max(12, (x2 - x1) // 2)
        cv2.ellipse(frame, (cx, y2), (w, max(5, w // 3)), 0, -45, 235, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, y2), (w, max(5, w // 3)), 0, -45, 235, bgr, 3, cv2.LINE_AA)
        s = look.label(t)
        if s:
            (tw, th), _ = cv2.getTextSize(s, FONT, 0.6, 2)
            cv2.rectangle(frame, (cx - tw // 2 - 4, y1 - th - 12), (cx + tw // 2 + 4, y1 - 4), bgr, -1)
            text(frame, s, (cx - tw // 2, y1 - 8), color=ink(bgr))


def _draw_overlay(frame, court, title, meta, team_bgr):
    """Court map top right, and the title bar with the team legend top left."""
    ch, cw = court.shape[:2]
    x0, y0 = 1920 - cw - 24, 24
    frame[y0 : y0 + ch, x0 : x0 + cw] = cv2.addWeighted(frame[y0 : y0 + ch, x0 : x0 + cw], 0.15, court, 0.85, 0)
    cv2.rectangle(frame, (20, 20), (760, 112), (20, 20, 20), -1)
    text(frame, title, (34, 54), 0.8)
    lx = 34
    for side in ("home", "away"):
        cv2.circle(frame, (lx + 8, 88), 9, team_bgr.get(side, GREY), -1)
        text(frame, meta[side]["team"], (lx + 24, 95))
        lx += 54 + cv2.getTextSize(meta[side]["team"], FONT, 0.6, 2)[0][0]
    text(frame, "automatic, uncorrected", (lx + 10, 95), 0.5, (180, 180, 180), 1)


def _render(writer, frames, read, per_frame, H, look, ct, title, meta):
    """Draw and write every rendered frame of one shot."""
    smooth, H_last = {}, None
    for f in frames:
        frame = read(f)
        tids = list(per_frame[f])
        xyxy = np.array([per_frame[f][t] for t in tids]).reshape(-1, 4)
        H_last = H.get(f) if H.get(f) is not None else H_last
        court = ct.base.copy()
        if H_last is not None and tids:
            court = _draw_points(court, smooth, tids, xyxy, H_last, look, ct)
        _draw_boxes(frame, tids, xyxy, look)
        _draw_overlay(frame, court, title, meta, look.team_bgr)
        writer.write(frame)


def main(argv=None):
    import supervision as sv
    from inference import get_model
    from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/demo/demo.mp4"))
    ap.add_argument("--step", type=int, default=2, help="render every Nth frame (2: 25 fps from 50 fps)")
    ap.add_argument("--recompute", action="store_true", help="ignore cached teams, numbers and homographies")
    ap.add_argument("--max-reads", type=int, default=8, help="number reads per track at most")
    a = ap.parse_args(argv)
    cache_dir = a.out.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = CourtConfiguration(league=League.FIBA)
    scale, padding = 0.25, 20
    ct = _Court(
        sv=sv,
        draw_points=draw_points_on_court,
        cfg=cfg,
        vertices=np.array(cfg.vertices, dtype=np.float32),
        base=draw_court(cfg, scale=scale, padding=padding, line_thickness=2),
        scale=scale,
        padding=padding,
    )
    models = {"court": None, "ocr": None, "get_model": get_model}
    writer = None
    summary = []

    for yid, shot in SHOTS:
        labels_path, tracks_path, dets_path = clip_files(yid)
        labels = json.loads(labels_path.read_text())
        meta = labels["meta"]
        tracks_json = json.loads(tracks_path.read_text())
        dets = {r["frame_idx"]: r["detections"] for r in map(json.loads, dets_path.open())}
        per_frame, obs, is_ref = interpolate(tracks_json, shot, a.step)
        frames = sorted(per_frame)
        read = FrameReader(labels["clip"])
        cache = cache_dir / f"{yid}_s{shot}_step{a.step}.json"
        shot_in = (obs, is_ref, read, meta, tracks_json, shot, dets, frames, ct.vertices)
        c, timing = _shot_cache(cache, a, models, shot_in)
        number = {int(k): v for k, v in c["number"].items()}
        look = _Look(
            is_ref=is_ref,
            cluster={int(k): v for k, v in c["cluster"].items()},
            side_of_cluster={int(k): v for k, v in c["side_of_cluster"].items()},
            team_bgr={
                side: DISPLAY_BGR.get((meta[side]["team"], meta[side].get("kit")), tuple(v))
                for side, v in c["team_bgr"].items()
            },
            number=number,
            names={side: roster_names(labels, side) for side in ("home", "away")},
        )
        H = {int(k): (np.array(v) if v else None) for k, v in c["H"].items()}

        t0 = time.time()
        title = f"{meta['home']['team']} – {meta['away']['team']}  ·  {meta['season']}"
        if writer is None:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            fps = labels["fps"] / a.step
            writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
        _render(writer, frames, read, per_frame, H, look, ct, title, meta)
        read.release()
        timing["render_s"] = round(time.time() - t0, 1)
        summary.append({
            "clip": yid,
            "shot": shot,
            "frames": len(frames),
            "tracks": len(obs),
            "numbers": {str(t): v for t, v in number.items()},
            "ocr_reads": c["ocr_reads"],
            "homography_frames": sum(v is not None for v in H.values()),
            "team_crops": c["team_crops"],
            **timing,
        })
        print(json.dumps(summary[-1]), flush=True)
    writer.release()
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
