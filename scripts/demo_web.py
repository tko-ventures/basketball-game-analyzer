#!/usr/bin/env python3
"""Client demo: one web page over the broadcast videos (one video per shot, labels burnt in):
  /          broadcast and a live 2D court drawn in the browser from per-frame positions: coverage
             (nearest defender, most open attacker), team shape, player selection with speed/distance

    uv run --extra prefill --extra demo --extra court scripts/demo_web.py              # -> runs/demo/web/
    uv run --extra prefill --extra demo --extra court scripts/demo_web.py --recompute  # redo reads and court
    uv run scripts/serve_demo.py                                           # http://127.0.0.1:8770
    uv run scripts/demo_audit.py                                           # what the pages show vs gold

The site is English only (text in scripts/demo_strings.py); the Polish edition was dropped on 2026-11-18.
On 2026-11-18 the page with two synchronised videos (broadcast and a rendered 2D court video) was dropped;
/analysis/ moved to / and the old path redirects there on Vercel.

Showcase, not a measured pipeline (no metrics row): automatic and uncorrected. Differences from the
first demo (scripts/demo_video.py), after Tomasz's review on 2026-10-31:
  court    the court layer of a bga run config (--court-config, default configs/court-plk-v1-smooth4.yaml:
           the PLK fine-tuned RF-DETR Keypoint model with median smoothing within the shot, scored in
           docs/metrics.md), run on the rendered frames; until 2026-11-18 the NBA Roboflow court model
           with a Savitzky-Golay smoothing of the demo's own
  paths    a player's 2D path is one key position per --key-s seconds (median of the projected foot
           points in that window) joined by a monotone cubic, after Tomasz's suggestion
  2D view  players only; referees appear in the broadcast view but not on the court map
  numbers  more reads per track (up to --max-reads, stop once a value leads by 3); a track's number can
           change once in time when the reads before and after a point disagree (SAM track switched
           players); at any frame one (team, number) has one owner, the track with more agreeing reads;
           a number that is not on the team's roster for that season is not shown (2026-11-11, after
           scripts/demo_audit.py found a Dziki player shown as #32)
  names    label = number and the name after the first given name ("#0 Vander Plas", not "#0 Plas")
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_strings import JS, PAGE, VIDEO, surname
from scripts.demo_video import (
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
# Shot sets. The default is three Warsaw shots; three-halls adds Lublin and Sopot, which the NBA court
# model mapped badly (scripts/qa_court_overlay.py, 2026-10-31) and the PLK court model was trained for.
DEMO_SETS = {
    "warsaw": [("OSp445e0mXU", 2), ("OSp445e0mXU", 10), ("OSp445e0mXU", 19)],
    "three-halls": SHOTS,
}
# Court mapping shown only inside these frame windows (visual QA); elsewhere the 2D panel says so. Rechecked
# with the PLK court model on 2026-11-18 (scripts/qa_court_overlay.py): shot 10 is still off before 3130, a
# mid-court view with few landmarks; shots 2 and 19 sit on the painted lines throughout.
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
            score, tid = max(((ios(d["xyxy"], t["xyxy"]), t["track_id"]) for t in fr["tracks"]), default=(0, None))
            if score < 0.9 or is_ref.get(tid):
                continue
            r = reads[tid]
            top = [*r.most_common(2), ("", 0), ("", 0)]
            if top[0][1] - top[1][1] >= 3 or sum(r.values()) >= max_reads:
                continue
            frame = read(fr["frame_idx"]) if frame is None else frame
            c = crop(frame, d["xyxy"], pad=10)
            if c.size == 0:
                continue
            raw = model.prompt(images=cv2.resize(c, (224, 224)), prompt="Read the number.", max_new_tokens=4)[0]
            digits = re.sub(r"\D", "", str(raw))
            if 0 < len(digits) <= 2:
                r[digits] += 1
                out.append((fr["frame_idx"], tid, digits))
    return out


def number_segments(reads):
    """[(start, end, number, support)] for one track from its time-ordered reads [(frame, digits)].

    One change point is allowed when both sides agree internally (>= 2 reads, >= 75% purity).

    """
    vals = [d for _, d in reads]
    best = None
    for k in range(2, len(vals) - 1):
        (lv, lc), (rv, rc) = Counter(vals[:k]).most_common(1)[0], Counter(vals[k:]).most_common(1)[0]
        agree = lv != rv and lc >= 2 and rc >= 2 and lc / k >= 0.75 and rc / (len(vals) - k) >= 0.75
        if agree and (best is None or lc + rc > best[0]):
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
    """Frame -> {track: number}, with one owner per (side, number) per frame (most agreeing reads wins)."""
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
def court_mapping(frames, read, court_cfg, seq_name, wh):
    """Frame -> 3x3 image-to-court (cm) homography or None, from the court layer of a bga run config.

    The backend runs on the rendered frames of one shot as one sequence, so its smoothing (if any) stays
    inside the shot. Returns the mapping and the number of mapped frames.

    """
    from bga import registry
    from bga.stage import Sequence

    entry = court_cfg.layers["court"]
    backend = registry.get("court", entry.backend)(entry.params, entry.weights)
    backend.load()
    try:
        seq = Sequence(seq_name, seq_name, "", 0, list(frames), 0.0)
        out = backend.run(seq, {}, read)["frames"]
    finally:
        backend.unload()
    H = {f: (None if out.get(str(f)) is None else np.array(out[str(f)], dtype=np.float64)) for f in frames}
    return H, sum(h is not None for h in H.values())


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
        for a, b in itertools.pairwise(idx):
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
            curve = PchipInterpolator(keys_t, keys_xy, axis=0, extrapolate=False) if len(keys_t) >= 2 else None
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

    Steps across a gap or a jump > 0.6 m do not count.

    """
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
            v = math.dist(path[keys[hi]], path[keys[lo]]) / ((keys[hi] - keys[lo]) / src_fps) * 3.6 if hi > lo else 0.0
            speed[t][f] = round(min(v, max_kmh), 1)
    return speed, dist


def attack_frames(rows, window=12):
    """Per frame [attacking side, basket 'L'/'R'] or [None, None] in transition.

    In a half-court set the
    defence stands closer to the basket it protects, so the attacking side is the one farther from it.

    """
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
def ink(bgr):
    return (20, 20, 20) if sum(bgr) > 500 else (255, 255, 255)


def legend(img, x, y, meta, team_bgr, size=0.6):
    for side in ("home", "away"):
        cv2.circle(img, (x + 9, y - 6), 10, team_bgr.get(side, GREY), -1)
        cv2.circle(img, (x + 9, y - 6), 10, (255, 255, 255), 1)
        text(img, meta[side]["team"], (x + 26, y), size)
        x += 60 + cv2.getTextSize(meta[side]["team"], FONT, size, 2)[0][0]
    return x


def draw_broadcast(frame, f, per_frame, colour, label, title, meta, team_bgr, vs):
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
    text(frame, vs["automatic"], (x + 10, 95), 0.5, (180, 180, 180), 1)
    return frame


# --------------------------------------------------------------------------- page
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/demo/web"))
    ap.add_argument("--step", type=int, default=2, help="render every Nth frame (2: 25 fps from 50 fps)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--max-reads", type=int, default=16)
    ap.add_argument("--key-s", type=float, default=0.5, help="seconds per key position on the 2D court")
    ap.add_argument("--set", choices=list(DEMO_SETS), default="warsaw")
    ap.add_argument("--cache", type=Path, default=Path("runs/demo/cache"))
    ap.add_argument("--court-config", type=Path, default=Path("configs/court-plk-v1-smooth4.yaml"))
    a = ap.parse_args(argv)
    build(a, a.out)


def build(a, out):
    """The site: broadcast videos with their text burnt in, the analysis page, clip data."""
    from sports.basketball import CourtConfiguration, League, draw_court

    from bga.config import load_config

    vs = VIDEO
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = a.cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = CourtConfiguration(league=League.FIBA)
    base = draw_court(cfg, scale=COURT_SCALE, padding=COURT_PAD, line_thickness=3)
    court_cfg = load_config(a.court_config)
    models = {"ocr": None, "court_cfg": court_cfg, "court_id": f"{court_cfg.name}@{court_cfg.layer_hash('court')}"}
    clips, summary, fps = [], [], 25.0
    for old in [*out.glob("*.mp4"), *out.glob("clip_*.json")]:  # drop files of earlier builds (and court videos)
        old.unlink()
    cv2.imwrite(str(out / "court_bg.png"), base)

    shots = DEMO_SETS[a.set]
    same_game = len({y for y, _ in shots}) < len(shots)
    for yid, shot in shots:
        n = len(clips) + 1
        clip, row, fps = _build_shot(a, vs, out, models, yid, shot, n)
        clip["short"] += f" · {sum(y == yid for y, _ in shots[:n])}" if same_game else ""
        clips.append(clip)
        summary.append(row)
        print(json.dumps(summary[-1]), flush=True)

    # One page at /: the broadcast with a live 2D analysis. Older paths are redirected on Vercel.
    (out / "index.html").write_text(_page(PAGE_ANALYSIS, clips, fps), encoding="utf-8")
    shutil.rmtree(out / "analiza", ignore_errors=True)  # path before 2026-11-01
    shutil.rmtree(out / "en", ignore_errors=True)  # English edition before 2026-11-18
    shutil.rmtree(out / "analysis", ignore_errors=True)  # analysis page before it moved to / on 2026-11-18
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def _page(template, cl, fps):
    """Fill one page template with the strings, the clips and the frame rate."""
    html = template
    for k, v in PAGE.items():
        html = html.replace("{{" + k + "}}", v)
    return (
        html
        .replace("__CLIPS__", json.dumps(cl, ensure_ascii=False))
        .replace("__T__", json.dumps(JS, ensure_ascii=False))
        .replace("__FPS__", f"{fps:g}")
        .replace("__N__", str(len(cl)))
    )


def _team_cache(team_cache, obs, is_ref, read, meta):
    """Team clusters for one shot, from the cache shared with demo_video.py or computed and cached."""
    if team_cache.exists():
        return json.loads(team_cache.read_text())
    cl, sc, tb, n = teams(obs, is_ref, read, meta)
    tc = json.loads(json.dumps({"cluster": cl, "side_of_cluster": sc, "team_bgr": tb, "team_crops": n}))
    team_cache.write_text(json.dumps(tc))
    return tc


def _reads_and_mapping(a, cache, models, shot_in, timing):
    """Number reads and the per-frame court mapping of one shot, reused from the cache unless --recompute."""
    tracks_json, shot, dets, is_ref, read, frames, wh = shot_in
    c = {} if a.recompute or not cache.exists() else json.loads(cache.read_text())
    if "reads" not in c:
        t0 = time.time()
        models["ocr"] = models["ocr"] or load_ocr()
        c["reads"] = read_numbers(tracks_json, shot, dets, is_ref, read, models["ocr"], a.max_reads)
        timing["numbers_s"] = round(time.time() - t0, 1)
    if c.get("court") != models["court_id"]:  # mappings of another court config (or the NBA model) are redone
        t0 = time.time()
        H, n_valid = court_mapping(frames, read, models["court_cfg"], cache.stem, wh)
        c["H"] = {str(f): (h.tolist() if h is not None else None) for f, h in H.items()}
        c["court"], c["court_valid_frames"] = models["court_id"], n_valid
        timing["court_s"] = round(time.time() - t0, 1)
    cache.write_text(json.dumps(c))
    return c


@dataclass(frozen=True)
class _Tracks:
    """How one shot's tracks are shown: referee flags, sides, team colours, numbers over time and names."""

    is_ref: dict
    cluster: dict
    side_of_cluster: dict
    team_bgr: dict
    owner: dict
    names: dict
    referee: str

    def side_of(self, t):
        return self.side_of_cluster.get(self.cluster.get(t))

    def colour(self, t):
        return REF_BGR if self.is_ref[t] else self.team_bgr.get(self.side_of(t), GREY)

    def number_of(self, t, f):
        return self.owner.get(f, {}).get(t, "")

    def label(self, t, f):
        if self.is_ref[t]:
            return self.referee
        num = self.number_of(t, f)
        name = self.names.get(self.side_of(t), {}).get(num, "") if num else ""
        return f"#{num} {surname(name)}" if name else (f"#{num}" if num else "")


def _write_video(out, n, fps, src, tk, title, meta, vs):
    """Render the broadcast video of one shot with its labels and encode it; returns render seconds."""
    frames, read, per_frame = src
    raw = out / f"broadcast_{n}_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    t0 = time.time()
    for f in frames:
        writer.write(draw_broadcast(read(f), f, per_frame, tk.colour, tk.label, title, meta, tk.team_bgr, vs))
    read.release()
    writer.release()
    encode(raw, out / f"broadcast_{n}.mp4", fps)
    return round(time.time() - t0, 1)


def _clip_data(frames, per_frame, paths, H, tk, meta, src_fps):
    """Data for the live 2D view in the page: one row per player per rendered frame."""
    speed, dist = kinematics(paths, src_fps)
    rows = [
        [
            [t, round(x, 2), round(y, 2), tk.number_of(t, f), speed[t][f], dist[t][f]]
            for t, path in paths.items()
            if f in path
            for x, y in [path[f]]
        ]
        for f in frames
    ]
    sided = [[(tk.side_of(r[0]), r[1], r[2]) for r in fr] for fr in rows]
    is_ref = tk.is_ref
    return {
        "court": {"scale": COURT_SCALE, "pad": COURT_PAD},
        "teams": {
            sd: {"name": meta[sd]["team"], "color": hexcol(tk.team_bgr.get(sd, GREY))} for sd in ("home", "away")
        },
        "names": tk.names,
        "players": {str(t): {"side": tk.side_of(t)} for t in paths},
        "f": rows,
        "boxes": [[[t, *[int(v) for v in b]] for t, b in per_frame[f].items() if not is_ref[t]] for f in frames],
        "mapped": [H[f] is not None for f in frames],
        "atk": attack_frames(sided),
    }


def _build_shot(a, vs, out, models, yid, shot, n):
    """Videos and page data for one shot (clip number n); returns the clip entry, its summary row and fps."""
    cache_dir = a.cache
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

    tc = _team_cache(cache_dir / f"{yid}_s{shot}_step{a.step}.json", obs, is_ref, read, meta)
    cluster = {int(k): v for k, v in tc["cluster"].items()}
    side_of_cluster = {int(k): v for k, v in tc["side_of_cluster"].items()}
    team_bgr = {s: DISPLAY_BGR.get((meta[s]["team"], meta[s].get("kit")), tuple(v)) for s, v in tc["team_bgr"].items()}

    def side_of(t):
        return side_of_cluster.get(cluster.get(t))

    cache = cache_dir / f"{yid}_s{shot}_step{a.step}_web.json"
    shot_in = (tracks_json, shot, dets, is_ref, read, frames, (labels["width"], labels["height"]))
    c = _reads_and_mapping(a, cache, models, shot_in, timing)
    H = {int(f): (np.array(h, dtype=np.float32) if h else None) for f, h in c["H"].items()}
    lo, hi = COURT_WINDOW.get((yid, shot), (-INF, INF))
    H = {f: (h if lo <= f <= hi else None) for f, h in H.items()}

    by_track = defaultdict(list)
    for f, t, d in c["reads"]:
        by_track[t].append((f, d))
    names = {s: roster_names(labels, s) for s in ("home", "away")}
    segments = {
        t: [seg for seg in number_segments(sorted(r)) if seg[2] in names.get(side_of(t), {})]
        for t, r in by_track.items()
    }
    owner = numbers_at(frames, lambda f, pf=per_frame: pf[f].keys(), segments, side_of)
    tk = _Tracks(
        is_ref=is_ref,
        cluster=cluster,
        side_of_cluster=side_of_cluster,
        team_bgr=team_bgr,
        owner=owner,
        names=names,
        referee=vs["referee"],
    )

    # What the pages show per frame (team, number, box), for scripts/demo_audit.py against gold.
    shown = {
        str(f): [
            [t, "ref" if is_ref[t] else tk.side_of(t), tk.number_of(t, f), *[round(float(v)) for v in b]]
            for t, b in per_frame[f].items()
        ]
        for f in frames
    }
    (cache_dir / f"{yid}_s{shot}_step{a.step}_shown.json").write_text(json.dumps(shown))

    paths = court_paths(frames, per_frame, H, max(1, round(a.key_s * fps)), is_ref, labels["height"])
    paths = {t: v for t, v in paths.items() if not is_ref[t]}  # 2D view shows players only (Tomasz)
    title = f"{meta['home']['team']} – {meta['away']['team']}  ·  {meta['season']}"
    timing["render_s"] = _write_video(out, n, fps, (frames, read, per_frame), tk, title, meta, vs)
    data = _clip_data(frames, per_frame, paths, H, tk, meta, labels["fps"])
    (out / f"clip_{n}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    clip = {
        "a": f"broadcast_{n}.mp4",
        "data": f"clip_{n}.json",
        "title": title,
        "short": f"{meta['home']['team'].split()[0]} – {meta['away']['team'].split()[0]}",
    }
    numbered = {t for t, s in segments.items() if s}
    row = {
        "clip": yid,
        "shot": shot,
        "frames": len(frames),
        "tracks": len(obs),
        "ocr_reads": len(c["reads"]),
        "tracks_with_number": len(numbered),
        "tracks_with_two_numbers": sum(len(s) == 2 for s in segments.values()),
        "court": c["court"],
        "court_valid_frames": c["court_valid_frames"],
        **timing,
    }
    return clip, row, fps


if __name__ == "__main__":
    main()
