#!/usr/bin/env python3
"""Client demo as two web pages over the same broadcast videos (one video per shot, labels burnt in):
  /          broadcast and a 2D FIBA court video side by side, one set of controls, full screen per panel
  /analysis/ broadcast and a live 2D court drawn in the browser from per-frame positions: coverage
             (nearest defender, most open attacker), team shape, player selection with speed/distance

    uv run --extra prefill --extra demo scripts/demo_web.py                # -> runs/demo/web/
    uv run --extra prefill --extra demo scripts/demo_web.py --recompute    # redo number reads and court
    open runs/demo/web/index.html

Showcase, not a measured pipeline (no metrics row): automatic and uncorrected. Differences from the
first demo (scripts/demo_video.py), after Tomasz's review on 2026-10-31:
  court    homography per frame with RANSAC (>= 5 confident landmarks, 40 cm threshold); frames whose
           mapping jumps against their neighbours are dropped; the camera mapping is smoothed in time
           (a grid of image points is projected, smoothed with Savitzky-Golay, and the homography refit);
           a player's 2D path is one key position per --key-s seconds (median of the projected foot
           points in that window) joined by a monotone cubic, after Tomasz's suggestion
  2D view  players only; referees appear in the broadcast view but not on the court map
  numbers  more reads per track (up to --max-reads, stop once a value leads by 3); a track's number can
           change once in time when the reads before and after a point disagree (SAM track switched
           players); at any frame one (team, number) has one owner, the track with more agreeing reads
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_video import (  # noqa: E402
    COURT_MODEL,
    DISPLAY_BGR,
    FONT,
    GREY,
    REF_BGR,
    SHOTS,
    FrameReader,
    clip_files,
    crop,
    interpolate,
    ios,
    load_ocr,
    roster_names,
    teams,
    text,
)

INF = 10**9
# Shot sets. The NBA-trained court keypoint model maps the Warsaw hall (blue keys) well and misplaces
# landmarks in Lublin (red keys) and Sopot (orange floor), checked with scripts/qa_court_overlay.py on
# 2026-10-31; PLK halls need the M3 court fine-tune. The default demo therefore uses Warsaw shots.
DEMO_SETS = {
    "warsaw": [("OSp445e0mXU", 2), ("OSp445e0mXU", 10), ("OSp445e0mXU", 19)],
    "three-halls": SHOTS,
}
# Court mapping shown only inside these frame windows (visual QA); elsewhere the 2D panel says so.
COURT_WINDOW = {("OSp445e0mXU", 10): (3130, 10**9)}
COURT_SCALE, COURT_PAD = 0.6, 40  # court video: px per cm, padding px


# --------------------------------------------------------------------------- numbers
def read_numbers(tracks_json, shot, dets, is_ref, read, model, max_reads):
    """Raw reads [(frame, track, digits)]; a track stops being read once one value leads by 3."""
    reads, out = defaultdict(Counter), []
    for fr in tracks_json["frames"]:
        if fr["shot"] != shot:
            continue
        frame = None
        for d in dets.get(fr["frame_idx"], []):
            if d["class"] != "number" or d["score"] < 0.3:
                continue
            score, tid = max(
                ((ios(d["xyxy"], t["xyxy"]), t["track_id"]) for t in fr["tracks"]), default=(0, None)
            )
            if score < 0.9 or is_ref.get(tid):
                continue
            r = reads[tid]
            top = r.most_common(2) + [("", 0), ("", 0)]
            if top[0][1] - top[1][1] >= 3 or sum(r.values()) >= max_reads:
                continue
            frame = read(fr["frame_idx"]) if frame is None else frame
            c = crop(frame, d["xyxy"], pad=10)
            if c.size == 0:
                continue
            raw = model.prompt(images=cv2.resize(c, (224, 224)), prompt="Read the number.", max_new_tokens=4)[
                0
            ]
            digits = re.sub(r"\D", "", str(raw))
            if 0 < len(digits) <= 2:
                r[digits] += 1
                out.append((fr["frame_idx"], tid, digits))
    return out


def number_segments(reads):
    """[(start, end, number, support)] for one track from its time-ordered reads [(frame, digits)].
    One change point is allowed when both sides agree internally (>= 2 reads, >= 75% purity)."""
    vals = [d for _, d in reads]
    best = None
    for k in range(2, len(vals) - 1):
        (lv, lc), (rv, rc) = Counter(vals[:k]).most_common(1)[0], Counter(vals[k:]).most_common(1)[0]
        if lv != rv and lc >= 2 and rc >= 2 and lc / k >= 0.75 and rc / (len(vals) - k) >= 0.75:
            if best is None or lc + rc > best[0]:
                best = (lc + rc, k, lv, lc, rv, rc)
    if best:
        _, k, lv, lc, rv, rc = best
        cut = (reads[k - 1][0] + reads[k][0]) // 2
        return [(-INF, cut, lv, lc), (cut, INF, rv, rc)]
    if vals:
        v, c = Counter(vals).most_common(1)[0]
        if c >= 2 and c / len(vals) >= 0.6:
            return [(-INF, INF, v, c)]
    return []


def numbers_at(frames, tracks_at, segments, side_of):
    """frame -> {track: number}, with one owner per (side, number) per frame (most agreeing reads wins)."""
    out = {}
    for f in frames:
        claims = defaultdict(list)
        for t in tracks_at(f):
            for s, e, num, sup in segments.get(t, []):
                if s <= f < e:
                    claims[(side_of(t), num)].append((sup, t))
        out[f] = {max(c)[1]: key[1] for key, c in claims.items()}
    return out


# --------------------------------------------------------------------------- court mapping
def court_mapping(frames, read, court_model, vertices, wh, jump_cm=150.0, window=11):
    """frame -> smoothed 3x3 image-to-court (cm) homography, or None before the first usable frame."""
    from scipy.signal import savgol_filter

    w, h = wh
    gx, gy = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 5), np.linspace(0.45 * h, 0.95 * h, 3))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    G = np.full((len(frames), len(grid), 2), np.nan)
    for i, f in enumerate(frames):
        kp = court_model.infer(read(f), confidence=0.3)[0]
        if not kp.predictions or len(kp.predictions[0].keypoints) != len(vertices):
            continue
        pts = kp.predictions[0].keypoints
        mask = np.array([p.confidence for p in pts]) > 0.5
        if mask.sum() < 5:
            continue
        src = np.array([[p.x, p.y] for p in pts], dtype=np.float32)[mask]
        H, inl = cv2.findHomography(src, vertices[mask], cv2.RANSAC, 40.0)
        if H is not None and inl is not None and inl.sum() >= 5:
            G[i] = cv2.perspectiveTransform(grid[None], H)[0]
    valid = ~np.isnan(G[:, 0, 0])
    # Drop frames whose mapped grid jumps away from the median of their neighbours.
    for i in np.flatnonzero(valid):
        nb = [j for j in range(max(0, i - 3), min(len(frames), i + 4)) if j != i and valid[j]]
        if len(nb) >= 2 and np.median(np.linalg.norm(G[i] - np.median(G[nb], 0), axis=1)) > jump_cm:
            G[i] = np.nan
    valid = ~np.isnan(G[:, 0, 0])
    if valid.sum() < 2:
        return {f: None for f in frames}, int(valid.sum())
    idx = np.arange(len(frames))
    for p in range(len(grid)):
        for c in range(2):
            G[:, p, c] = np.interp(idx, idx[valid], G[valid, p, c])
    win = min(window, (len(frames) // 2) * 2 - 1)
    if win >= 5:
        G = savgol_filter(G, win, 2, axis=0)
    first = int(np.flatnonzero(valid)[0])
    out = {}
    for i, f in enumerate(frames):
        out[f] = None if i < first else cv2.findHomography(grid, G[i].astype(np.float32))[0]
    return out, int(valid.sum())


def court_paths(frames, per_frame, H, key_frames, is_ref, img_h, edge_px=3):
    """Track -> {frame: (x_m, y_m)} for the frames where the track is visible.

    Foot points (box bottom-centre) are projected every frame; each visible run of a track is cut into
    windows of key_frames frames, each window becomes one key position (the median, so a single bad
    frame cannot move it), and a monotone cubic (PCHIP) through the key positions gives the path.
    A box that reaches the bottom edge of the frame has no visible feet: such frames do not make key
    positions unless a run has nothing else. Players are kept on the 28 x 15 m court; referees are not
    clamped (they work outside the lines).
    """
    from scipy.interpolate import PchipInterpolator

    tids = sorted({t for f in frames for t in per_frame[f]})
    raw = {t: {} for t in tids}
    for i, f in enumerate(frames):
        if H[f] is None or not per_frame[f]:
            continue
        ts = list(per_frame[f])
        b = np.array([per_frame[f][t] for t in ts])
        feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1).astype(np.float32)
        for t, p, y2 in zip(ts, cv2.perspectiveTransform(feet[None], H[f])[0] / 100.0, b[:, 3], strict=True):
            raw[t][i] = (p, y2 < img_h - edge_px)
    out = {}
    for t, pos in raw.items():
        out[t] = {}
        idx = sorted(pos)
        runs, cur = [], [idx[0]] if idx else []
        for a, b in zip(idx, idx[1:], strict=False):
            if b - a > 5:  # out of view or lost for > 0.2 s: separate run
                runs.append(cur)
                cur = []
            cur.append(b)
        if cur:
            runs.append(cur)
        for run in runs:
            usable = [i for i in run if pos[i][1]] or run  # frames with visible feet, else all
            keys_t, keys_xy = [], []
            for k in range(0, len(usable), key_frames):
                win = usable[k : k + key_frames]
                keys_t.append(float(np.mean(win)))
                keys_xy.append(np.median([pos[i][0] for i in win], axis=0))
            keys_xy = np.array(keys_xy)
            curve = (
                PchipInterpolator(keys_t, keys_xy, axis=0, extrapolate=False) if len(keys_t) >= 2 else None
            )
            for i in run:
                v = curve(min(max(i, keys_t[0]), keys_t[-1])) if curve else keys_xy[0]
                x, y = float(v[0]), float(v[1])
                if not is_ref.get(t):
                    x, y = min(max(x, 0.0), 28.0), min(max(y, 0.0), 15.0)
                out[t][frames[i]] = (x, y)
    return out


# --------------------------------------------------------------------------- analysis for the page
def kinematics(paths, src_fps, k=3, max_kmh=34.0):
    """Per track and frame: speed (km/h, central difference over +-k frames) and distance so far (m).
    Steps across a gap or a jump > 0.6 m do not count."""
    speed, dist = {}, {}
    for t, path in paths.items():
        keys = sorted(path)
        speed[t], dist[t], total = {}, {}, 0.0
        for j, f in enumerate(keys):
            if j and f - keys[j - 1] <= 4:
                step = math.dist(path[f], path[keys[j - 1]])
                total += step if step <= 0.6 else 0.0
            dist[t][f] = round(total, 1)
            lo = hi = j
            while lo > 0 and j - lo < k and keys[lo] - keys[lo - 1] <= 4:
                lo -= 1
            while hi < len(keys) - 1 and hi - j < k and keys[hi + 1] - keys[hi] <= 4:
                hi += 1
            v = (
                math.dist(path[keys[hi]], path[keys[lo]]) / ((keys[hi] - keys[lo]) / src_fps) * 3.6
                if hi > lo
                else 0.0
            )
            speed[t][f] = round(min(v, max_kmh), 1)
    return speed, dist


def attack_frames(rows, window=12):
    """Per frame [attacking side, basket 'L'/'R'] or [None, None] in transition. In a half-court set the
    defence stands closer to the basket it protects, so the attacking side is the one farther from it."""
    raw = []
    for frame_rows in rows:
        sided = [r for r in frame_rows if r[0] in ("home", "away")]
        sides = {r[0] for r in sided}
        xs = sorted(r[1] for r in sided)
        if len(sided) < 6 or len(sides) < 2 or abs(xs[len(xs) // 2] - 14.0) < 3.0:
            raw.append(None)
            continue
        basket = "L" if xs[len(xs) // 2] < 14.0 else "R"
        bx = 1.575 if basket == "L" else 26.425
        mean = {sd: np.mean([math.hypot(r[1] - bx, r[2] - 7.5) for r in sided if r[0] == sd]) for sd in sides}
        raw.append((max(mean, key=mean.get), basket))
    out = []
    for i in range(len(raw)):
        top = Counter(raw[max(0, i - window) : i + window + 1]).most_common(1)[0][0]
        out.append(list(top) if top else [None, None])
    return out


def hexcol(bgr):
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in bgr[::-1]])


# --------------------------------------------------------------------------- drawing
def to_px(x_m, y_m):
    return int(round(x_m * 100 * COURT_SCALE + COURT_PAD)), int(round(y_m * 100 * COURT_SCALE + COURT_PAD))


def ink(bgr):
    return (20, 20, 20) if sum(bgr) > 500 else (255, 255, 255)


def legend(img, x, y, meta, team_bgr, size=0.6):
    for side in ("home", "away"):
        cv2.circle(img, (x + 9, y - 6), 10, team_bgr.get(side, GREY), -1)
        cv2.circle(img, (x + 9, y - 6), 10, (255, 255, 255), 1)
        text(img, meta[side]["team"], (x + 26, y), size)
        x += 60 + cv2.getTextSize(meta[side]["team"], FONT, size, 2)[0][0]
    return x


def draw_broadcast(frame, f, per_frame, colour, label, title, meta, team_bgr):
    for t, b in per_frame[f].items():
        bgr = colour(t)
        x1, y1, x2, y2 = b.astype(int)
        cx, w = (x1 + x2) // 2, max(12, (x2 - x1) // 2)
        cv2.ellipse(frame, (cx, y2), (w, max(5, w // 3)), 0, -45, 235, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, y2), (w, max(5, w // 3)), 0, -45, 235, bgr, 3, cv2.LINE_AA)
        s = label(t, f)
        if s:
            (tw, th), _ = cv2.getTextSize(s, FONT, 0.6, 2)
            cv2.rectangle(frame, (cx - tw // 2 - 4, y1 - th - 12), (cx + tw // 2 + 4, y1 - 4), bgr, -1)
            text(frame, s, (cx - tw // 2, y1 - 8), color=ink(bgr))
    cv2.rectangle(frame, (20, 20), (820, 112), (20, 20, 20), -1)
    text(frame, title, (34, 54), 0.8)
    x = legend(frame, 34, 95, meta, team_bgr)
    text(frame, "automatycznie, bez korekty", (x + 10, 95), 0.5, (180, 180, 180), 1)
    return frame


def draw_court_frame(base, f, paths, colour, number_of, title, meta, team_bgr, mapped, trail=25):
    canvas = np.full((1080, 1920, 3), 24, np.uint8)
    court = base.copy()
    if not mapped:
        court = cv2.addWeighted(court, 0.35, np.zeros_like(court), 0.65, 0)
        msg = "brak wiarygodnej kalibracji kamery w tym fragmencie"
        (tw, _), _ = cv2.getTextSize(msg, FONT, 1.0, 2)
        text(court, msg, ((court.shape[1] - tw) // 2, court.shape[0] // 2), 1.0)
    trails = court.copy()
    for t, path in paths.items():
        if f not in path:
            continue
        keys = list(path)
        k = keys.index(f)
        recent = keys[max(0, k - trail) : k + 1]
        # Draw only the continuous tail: stop at a gap (player out of view or lost) or a jump > 0.6 m.
        start = len(recent) - 1
        while start > 0:
            g0, g1 = recent[start - 1], recent[start]
            (x0, y0), (x1, y1) = path[g0], path[g1]
            if g1 - g0 > 4 or np.hypot(x1 - x0, y1 - y0) > 0.6:
                break
            start -= 1
        pts = [to_px(*path[g]) for g in recent[start:]]
        if len(pts) > 1:
            cv2.polylines(trails, [np.array(pts, np.int32)], False, colour(t), 4, cv2.LINE_AA)
    court = cv2.addWeighted(trails, 0.55, court, 0.45, 0)
    for t, path in paths.items():
        if f not in path:
            continue
        x, y = path[f]
        if not (-1.0 < x < 29.0 and -1.0 < y < 16.0):
            continue
        cx, cy = to_px(x, y)
        bgr = colour(t)
        cv2.circle(court, (cx, cy), 22, bgr, -1, cv2.LINE_AA)
        cv2.circle(court, (cx, cy), 22, (0, 0, 0) if sum(bgr) > 500 else (255, 255, 255), 2, cv2.LINE_AA)
        num = number_of(t, f)
        if num:
            (tw, th), _ = cv2.getTextSize(num, FONT, 0.75, 2)
            text(court, num, (cx - tw // 2, cy + th // 2), 0.75, ink(bgr))
    ch, cw = court.shape[:2]
    y0, x0 = 1080 - ch - 10, (1920 - cw) // 2
    canvas[y0 : y0 + ch, x0 : x0 + cw] = court
    text(canvas, title, (x0, 52), 0.9)
    x = legend(canvas, x0 + 760, 52, meta, team_bgr, 0.7)
    text(canvas, "pozycje na boisku FIBA 28 x 15 m, automatycznie", (x + 10, 52), 0.55, (170, 170, 170), 1)
    return canvas


# --------------------------------------------------------------------------- page
PAGE_VIDEO = Path(__file__).with_name("demo_page_video.html").read_text(encoding="utf-8")
PAGE_ANALYSIS = Path(__file__).with_name("demo_page_analysis.html").read_text(encoding="utf-8")


def encode(src: Path, dst: Path, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(int(fps)),
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )
    src.unlink()


# --------------------------------------------------------------------------- main
def main(argv=None):
    from inference import get_model
    from sports.basketball import CourtConfiguration, League, draw_court

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/demo/web"))
    ap.add_argument("--step", type=int, default=2, help="render every Nth frame (2: 25 fps from 50 fps)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--max-reads", type=int, default=16)
    ap.add_argument("--key-s", type=float, default=0.5, help="seconds per key position on the 2D court")
    ap.add_argument("--set", choices=list(DEMO_SETS), default="warsaw")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cache_dir = a.out.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = CourtConfiguration(league=League.FIBA)
    vertices = np.array(cfg.vertices, dtype=np.float32)
    base = draw_court(cfg, scale=COURT_SCALE, padding=COURT_PAD, line_thickness=3)
    court_model = ocr_model = None
    clips, summary, fps = [], [], 25.0
    for old in [*a.out.glob("*.mp4"), *a.out.glob("clip_*.json")]:  # drop files of earlier builds
        old.unlink()
    cv2.imwrite(str(a.out / "court_bg.png"), base)

    shots = DEMO_SETS[a.set]
    same_game = len({y for y, _ in shots}) < len(shots)
    for yid, shot in shots:
        labels_path, tracks_path, dets_path = clip_files(yid)
        labels = json.loads(labels_path.read_text())
        meta = labels["meta"]
        tracks_json = json.loads(tracks_path.read_text())
        dets = {r["frame_idx"]: r["detections"] for r in map(json.loads, dets_path.open())}
        per_frame, obs, is_ref = interpolate(tracks_json, shot, a.step)
        frames = sorted(per_frame)
        read = FrameReader(labels["clip"])
        fps = labels["fps"] / a.step
        timing = {}

        team_cache = cache_dir / f"{yid}_s{shot}_step{a.step}.json"  # shared with demo_video.py
        if team_cache.exists():
            tc = json.loads(team_cache.read_text())
        else:
            cl, sc, tb, n = teams(obs, is_ref, read, meta)
            tc = json.loads(
                json.dumps({"cluster": cl, "side_of_cluster": sc, "team_bgr": tb, "team_crops": n})
            )
        cluster = {int(k): v for k, v in tc["cluster"].items()}
        side_of_cluster = {int(k): v for k, v in tc["side_of_cluster"].items()}
        team_bgr = {
            s: DISPLAY_BGR.get((meta[s]["team"], meta[s].get("kit")), tuple(v))
            for s, v in tc["team_bgr"].items()
        }

        def side_of(t, cluster=cluster, side_of_cluster=side_of_cluster):
            return side_of_cluster.get(cluster.get(t))

        def colour(t, is_ref=is_ref, team_bgr=team_bgr, side_of=side_of):
            return REF_BGR if is_ref[t] else team_bgr.get(side_of(t), GREY)

        cache = cache_dir / f"{yid}_s{shot}_step{a.step}_web.json"
        c = {} if a.recompute or not cache.exists() else json.loads(cache.read_text())
        if "reads" not in c:
            t0 = time.time()
            ocr_model = ocr_model or load_ocr()
            c["reads"] = read_numbers(tracks_json, shot, dets, is_ref, read, ocr_model, a.max_reads)
            timing["numbers_s"] = round(time.time() - t0, 1)
        if "H" not in c:
            t0 = time.time()
            court_model = court_model or get_model(model_id=COURT_MODEL)
            H, n_valid = court_mapping(
                frames, read, court_model, vertices, (labels["width"], labels["height"])
            )
            c["H"] = {str(f): (h.tolist() if h is not None else None) for f, h in H.items()}
            c["court_valid_frames"] = n_valid
            timing["court_s"] = round(time.time() - t0, 1)
        cache.write_text(json.dumps(c))
        H = {int(f): (np.array(h, dtype=np.float32) if h else None) for f, h in c["H"].items()}
        lo, hi = COURT_WINDOW.get((yid, shot), (-INF, INF))
        H = {f: (h if lo <= f <= hi else None) for f, h in H.items()}

        by_track = defaultdict(list)
        for f, t, d in c["reads"]:
            by_track[t].append((f, d))
        segments = {t: number_segments(sorted(r)) for t, r in by_track.items()}
        owner = numbers_at(frames, lambda f, pf=per_frame: pf[f].keys(), segments, side_of)
        names = {s: roster_names(labels, s) for s in ("home", "away")}

        def number_of(t, f, owner=owner):
            return owner.get(f, {}).get(t, "")

        def label(t, f, is_ref=is_ref, names=names, side_of=side_of, number_of=number_of):
            if is_ref[t]:
                return "SĘDZIA"
            num = number_of(t, f)
            name = names.get(side_of(t), {}).get(num, "") if num else ""
            return f"#{num} {name.split()[-1]}" if name else (f"#{num}" if num else "")

        paths = court_paths(frames, per_frame, H, max(1, round(a.key_s * fps)), is_ref, labels["height"])
        paths = {t: v for t, v in paths.items() if not is_ref[t]}  # 2D view shows players only (Tomasz)
        title = f"{meta['home']['team']} – {meta['away']['team']}  ·  {meta['season']}"
        n = len(clips) + 1
        raw_a, raw_b = a.out / f"broadcast_{n}_raw.mp4", a.out / f"court_{n}_raw.mp4"
        wa = cv2.VideoWriter(str(raw_a), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
        wb = cv2.VideoWriter(str(raw_b), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
        t0 = time.time()
        for f in frames:
            wa.write(draw_broadcast(read(f), f, per_frame, colour, label, title, meta, team_bgr))
            wb.write(
                draw_court_frame(base, f, paths, colour, number_of, title, meta, team_bgr, H[f] is not None)
            )
        read.release()
        wa.release()
        wb.release()
        encode(raw_a, a.out / f"broadcast_{n}.mp4", fps)
        encode(raw_b, a.out / f"court_{n}.mp4", fps)
        timing["render_s"] = round(time.time() - t0, 1)
        # Data for the live 2D view in the page: one row per player per rendered frame.
        speed, dist = kinematics(paths, labels["fps"])
        rows = [
            [
                [t, round(x, 2), round(y, 2), number_of(t, f), speed[t][f], dist[t][f]]
                for t, path in paths.items()
                if f in path
                for x, y in [path[f]]
            ]
            for f in frames
        ]
        sided = [[(side_of(r[0]), r[1], r[2]) for r in fr] for fr in rows]
        data = {
            "court": {"scale": COURT_SCALE, "pad": COURT_PAD},
            "teams": {
                sd: {"name": meta[sd]["team"], "color": hexcol(team_bgr.get(sd, GREY))}
                for sd in ("home", "away")
            },
            "names": names,
            "players": {str(t): {"side": side_of(t)} for t in paths},
            "f": rows,
            "boxes": [
                [[t, *[int(v) for v in b]] for t, b in per_frame[f].items() if not is_ref[t]] for f in frames
            ],
            "mapped": [H[f] is not None for f in frames],
            "atk": attack_frames(sided),
        }
        (a.out / f"clip_{n}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        clips.append(
            {
                "a": f"broadcast_{n}.mp4",
                "b": f"court_{n}.mp4",
                "data": f"clip_{n}.json",
                "title": title,
                "short": f"{meta['home']['team'].split()[0]} – {meta['away']['team'].split()[0]}"
                + (f" · {sum(y == yid for y, _ in shots[:n])}" if same_game else ""),
            }
        )
        numbered = {t for t, s in segments.items() if s}
        summary.append(
            {
                "clip": yid,
                "shot": shot,
                "frames": len(frames),
                "tracks": len(obs),
                "ocr_reads": len(c["reads"]),
                "tracks_with_number": len(numbered),
                "tracks_with_two_numbers": sum(len(s) == 2 for s in segments.values()),
                "court_valid_frames": c["court_valid_frames"],
                **timing,
            }
        )
        print(json.dumps(summary[-1]), flush=True)

    # Two independent pages over the same videos: / (two synced videos) and /analysis/ (live 2D analysis).
    def page(template, cl):
        return (
            template.replace("__CLIPS__", json.dumps(cl, ensure_ascii=False))
            .replace("__FPS__", f"{fps:g}")
            .replace("__N__", str(len(cl)))
        )

    (a.out / "index.html").write_text(page(PAGE_VIDEO, clips), encoding="utf-8")
    sub = [{**c, "a": "../" + c["a"], "data": "../" + c["data"]} for c in clips]
    shutil.rmtree(a.out / "analiza", ignore_errors=True)  # path before 2026-11-01; redirected on Vercel
    (a.out / "analysis").mkdir(exist_ok=True)
    (a.out / "analysis" / "index.html").write_text(page(PAGE_ANALYSIS, sub), encoding="utf-8")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
