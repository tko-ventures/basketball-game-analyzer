#!/usr/bin/env python3
"""Review sheets for the implementer's full track pass (gold v0), larger than track_sheets.py.

    uv run scripts/review_sheets.py data/labels/<clip>.json --shot 10

Writes to runs/<clip>/review/s<shot>/:
  overview_NN.jpg  4 full frames per sheet (every --every-s seconds), every box drawn with its tid,
                   so crossings, bench and staff are seen in context.
  tracks_NN.jpg    one row per track, --per-track crops at 1.5x the box with the frame number; rows
                   are captioned with tid, team, number, span, and the frames where the track's
                   torso colour or position jumps (candidate identity switches, marked "!").
  zoom_<tid>.jpg   (--zoom tid:frame,...) dense crops around a frame to place a split exactly.

A labelling aid, not a pipeline layer: it only decides what the reviewer looks at.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bga.review.sheets import contact_sheet  # noqa: E402

TEAM_BGR = {"home": (240, 240, 240), "away": (60, 200, 60), "ref": (0, 200, 255), "other": (200, 0, 200)}


class Frames:
    def __init__(self, path: str):
        self.cap, self.cache = cv2.VideoCapture(path), {}

    def __call__(self, i: int) -> np.ndarray:
        if i not in self.cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = self.cap.read()
            assert ok, i
            self.cache[i] = img
        return self.cache[i]


def torso_hist(img: np.ndarray, b: dict) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(b[c]) for c in ("x1", "y1", "x2", "y2"))
    h, w = y2 - y1, x2 - x1
    roi = img[max(0, y1 + h // 5) : max(0, y1 + h // 2), max(0, x1 + w // 4) : max(0, x2 - w // 4)]
    if roi.size < 30:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def jumps(obs: list[tuple[int, dict]], frames: Frames, top: int = 3) -> list[int]:
    """Frames where the torso colour (vs the median of the previous 4) or the box centre jumps."""
    hists = [torso_hist(frames(fi), b) for fi, b in obs]
    scores = []
    for i in range(1, len(obs)):
        (f0, b0), (f1, b1) = obs[i - 1], obs[i]
        h = max(1.0, b1["y2"] - b1["y1"])
        move = np.hypot((b1["x1"] + b1["x2"] - b0["x1"] - b0["x2"]) / 2, (b1["y2"] - b0["y2"])) / h
        move /= max(1, (f1 - f0) / 5)
        prev = [x for x in hists[max(0, i - 4) : i] if x is not None]
        col = 0.0
        if prev and hists[i] is not None:
            ref = np.median(np.stack(prev), axis=0).astype(np.float32)
            col = cv2.compareHist(ref, hists[i], cv2.HISTCMP_BHATTACHARYYA)
        scores.append((max(col / 0.5, move / 0.6), f1))
    return sorted(f for s, f in sorted(scores, reverse=True)[:top] if s >= 1.0)


def crop(img: np.ndarray, b: dict, pad: float = 0.25) -> np.ndarray:
    x1, y1, x2, y2 = (int(b[c]) for c in ("x1", "y1", "x2", "y2"))
    w, h = x2 - x1, y2 - y1
    img = img.copy()  # the box itself is drawn, so the padding never hides whose crop it is
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(1, h // 120))
    return img[
        max(0, int(y1 - h * pad / 2)) : int(y2 + h * pad / 2), max(0, int(x1 - w * pad)) : int(x2 + w * pad)
    ]


def kit_votes(d: dict, tr: dict, frames: Frames, cache: Path) -> dict[tuple[str, int], str]:
    """Light/dark kit per box: Skalski's TeamClassifier (SigLIP + UMAP + KMeans) fitted on this shot's
    central box crops; the brighter cluster is "L". Cached per shot; a review aid, not the team layer."""
    if cache.is_file():
        return {(t, int(f)): v for t, f, v in json.loads(cache.read_text())}
    from sports import TeamClassifier

    keys, crops = [], []
    for tid, obs in tr.items():
        for fi, b in obs:
            x1, y1, x2, y2 = (b[c] for c in ("x1", "y1", "x2", "y2"))
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.4, (y2 - y1) * 0.4
            c = frames(fi)[int(cy - h / 2) : int(cy + h / 2), max(0, int(cx - w / 2)) : int(cx + w / 2)]
            if c.size:
                keys.append((tid, fi))
                crops.append(c)
    # Referees would take one of the two clusters: fit on player tracks only, predict on every box.
    is_ref = [d["identities"].get(t, {}).get("team") == "ref" for t, _ in keys]
    tc = TeamClassifier(device="mps")
    tc.fit([c for c, r in zip(crops, is_ref, strict=True) if not r])
    pred = [int(p) for p in tc.predict(crops)]
    bright = [
        np.mean(
            [
                cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2].mean()
                for c, p, r in zip(crops, pred, is_ref, strict=True)
                if p == k and not r
            ]
            or [0]
        )
        for k in (0, 1)
    ]
    light = int(np.argmax(bright))
    out = {k: ("R" if r else "L" if p == light else "D") for k, p, r in zip(keys, pred, is_ref, strict=True)}
    cache.write_text(json.dumps([[t, f, v] for (t, f), v in out.items()]))
    return out


def runs_of(seq: list[tuple[int, str]]) -> list[tuple[str, int, int, int]]:
    """Run-length encode [(frame, kit)] into [(kit, first frame, count, last frame)]."""
    out: list[list] = []
    for f, v in seq:
        if out and out[-1][0] == v:
            out[-1][2] += 1
            out[-1][3] = f
        else:
            out.append([v, f, 1, f])
    return [tuple(r) for r in out]


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def propose(d: dict, tr: dict, kits: dict, next_id: int, min_run: int = 8) -> list[str]:
    """Ops proposed from kit votes, for the reviewer to check on the sheets before applying:
    team by majority kit; an opposite-kit run of >= min_run boxes becomes its own identity (split if
    it runs to the end, move otherwise); a shorter run is cut when it has 3 or more boxes or overlaps
    another track's box (contact: the box sits on the wrong player), else kept (classifier noise)."""
    side = {
        d["meta"]["home"].get("kit", "light")[0].upper(): "home",
        d["meta"]["away"].get("kit", "dark")[0].upper(): "away",
    }
    ops = []
    for tid, obs in sorted(tr.items(), key=lambda kv: kv[1][0][0]):
        if d["identities"].get(tid, {}).get("reviewed"):
            continue
        for fi, b in obs:  # SAM-only boxes spread over two people or a whole group
            if b["x2"] - b["x1"] > 1.2 * (b["y2"] - b["y1"]):
                ops.append(f"cut {tid} {fi} {fi}  # box wider than tall")
        if d["identities"].get(tid, {}).get("team") == "ref":
            continue
        seq = [(fi, kits.get((tid, fi), "?")) for fi, _ in obs]
        votes = [v for _, v in seq if v in "LD"]
        if not votes:
            continue
        major = max("LD", key=votes.count)
        ops.append(f"team {tid} {side.get(major, 'other')}  # kit {major} {votes.count(major)}/{len(votes)}")
        r = runs_of(seq)
        for i, (v, f0, n, f1) in enumerate(r):
            if v == major or v == "?":
                continue
            if n >= min_run:
                if i + 1 == len(r):
                    ops.append(f"split {tid} {f0} {next_id}  # kit {v} x{n} at the end")
                    ops.append(f"team {next_id} {side.get(v, 'other')}")
                else:
                    ops.append(f"move {tid} {f0} {f1} {next_id}  # kit {v} x{n} inside")
                    ops.append(f"team {next_id} {side.get(v, 'other')}")
                next_id += 1
            else:
                boxes = {fi: b for fi, b in obs}
                contact = any(
                    iou(boxes[fi], o) > 0.2
                    for fi in boxes
                    if f0 <= fi <= f1
                    for o in d["frames"][str(fi)]["boxes"]
                    if str(o["tid"]) != tid
                )
                if contact or n >= 3:
                    ops.append(f"cut {tid} {f0} {f1}  # kit {v} x{n}{' in contact' if contact else ''}")
    return ops


def label(d: dict, tid: str) -> str:
    ident = d["identities"].get(tid, {})
    num = f"#{ident['number']}" if ident.get("number") else ""
    rev = " OK" if ident.get("reviewed") else ""
    return f"{tid} {ident.get('team') or '?'}{num}{rev}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--per-track", type=int, default=10)
    ap.add_argument("--tracks-per-sheet", type=int, default=6)
    ap.add_argument("--zoom", default="", help="tid:frame[,tid:frame...] dense crops around a frame")
    ap.add_argument("--propose", action="store_true", help="print kit-vote ops for the reviewer")
    ap.add_argument("--only", default="", help="overview,tracks or zoom (default: overview,tracks)")
    a = ap.parse_args(argv)
    d = json.loads(a.labels.read_text())
    s = d["shots"][a.shot]
    fps = d.get("fps") or 50.0
    frames = Frames(d["clip"])
    keys = sorted(int(k) for k in d["frames"] if s["start"] <= int(k) <= s["end"] and d["frames"][k]["boxes"])
    tr = defaultdict(list)
    for k in keys:
        for b in d["frames"][str(k)]["boxes"]:
            tr[str(b["tid"])].append((k, b))
    out = Path("runs") / a.labels.stem / "review" / f"s{a.shot}"
    out.mkdir(parents=True, exist_ok=True)
    kits = kit_votes(d, tr, frames, out / "kits.json")
    for tid, obs in sorted(tr.items(), key=lambda kv: kv[1][0][0]):
        r = runs_of([(fi, kits.get((tid, fi), "?")) for fi, _ in obs])
        long_runs = " ".join(f"{v}{n}@{f}" for v, f, n, _ in r if n >= 3)
        print(f"{label(d, tid):<16} {obs[0][0]}-{obs[-1][0]} n{len(obs):<3} {long_runs}")
    if a.propose:
        top = max([int(t) for t in d["identities"] if t.isdigit()] + [1000])
        print(f"# shot {a.shot} ({s['start']}-{s['end']}): kit-vote proposals")
        print("\n".join(propose(d, tr, kits, (top // 100 + 1) * 100 + 1)))
    only = set((a.only or ("zoom" if a.zoom else "overview,tracks")).split(","))

    if "overview" in only:
        step = max(1, round(a.every_s * fps / 5)) * 5
        picks = [k for k in keys if (k - keys[0]) % step == 0]
        items = []
        for k in picks:
            img = frames(k).copy()
            for b in d["frames"][str(k)]["boxes"]:
                tid = str(b["tid"])
                col = TEAM_BGR.get(d["identities"].get(tid, {}).get("team"), (0, 0, 255))
                p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
                cv2.rectangle(img, p1, p2, col, 3)
                cv2.putText(img, tid, (p1[0], p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
                cv2.putText(img, tid, (p1[0], p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            items.append((img, f"f{k}"))
        for n in range(0, len(items), 4):
            p = out / f"overview_{n // 4:02d}.jpg"
            cv2.imwrite(
                str(p),
                contact_sheet(items[n : n + 4], cols=2, tile_wh=(960, 540)),
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            print(p)

    if "tracks" in only:
        rows = sorted(tr.items(), key=lambda kv: (kv[1][0][0], kv[0]))
        for n in range(0, len(rows), a.tracks_per_sheet):
            items = []
            for tid, obs in rows[n : n + a.tracks_per_sheet]:
                jf = jumps(obs, frames)
                k = a.per_track
                idx = sorted(
                    {round(i * (len(obs) - 1) / (k - 1)) for i in range(k)}
                    if len(obs) >= k
                    else range(len(obs))
                )
                # Make sure the frames just before and at each jump are shown.
                fidx = {fi: i for i, (fi, _) in enumerate(obs)}
                for f in jf:
                    idx = sorted(set(idx) | {fidx[f], max(0, fidx[f] - 1)})
                idx = idx[: k + 2 * len(jf)]
                for j, i in enumerate(idx[: k + 4]):
                    fi, b = obs[i]
                    mark = "!" if fi in jf else ""
                    kit = kits.get((tid, fi), "?")
                    cap = f"{label(d, tid)} {obs[0][0]}-{obs[-1][0]}" if j == 0 else f"{mark}{fi}{kit}"
                    items.append((crop(frames(fi), b), cap))
                for _ in range(len(idx[: k + 4]), k + 4):
                    items.append((np.zeros((4, 4, 3), np.uint8), ""))
            p = out / f"tracks_{n // a.tracks_per_sheet:02d}.jpg"
            cv2.imwrite(
                str(p),
                contact_sheet(items, cols=a.per_track + 4, tile_wh=(130, 210), caption_h=18),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            print(p, [t for t, _ in rows[n : n + a.tracks_per_sheet]])

    if "zoom" in only and a.zoom:
        for spec in a.zoom.split(","):
            tid, f = spec.split(":")
            obs = tr[tid]
            i0 = min(range(len(obs)), key=lambda i: abs(obs[i][0] - int(f)))
            sel = obs[max(0, i0 - 6) : i0 + 6]
            items = [(crop(frames(fi), b, 0.6), f"{tid} f{fi} {kits.get((tid, fi), '?')}") for fi, b in sel]
            p = out / f"zoom_{tid}_{f}.jpg"
            cv2.imwrite(
                str(p), contact_sheet(items, cols=6, tile_wh=(240, 320)), [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            print(p)


if __name__ == "__main__":
    main()
