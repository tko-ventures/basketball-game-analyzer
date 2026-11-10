#!/usr/bin/env python3
"""Probe: how well do off-the-shelf re-identification encoders tell PLK players apart?

Scores the pretrained encoders of Roboflow's ``reid`` package (https://reid.roboflow.com) on the gold
track boxes of the dev shots. It is a probe behind the 2026-11-10 scope decision, not a pipeline layer:
its numbers go to the M6 issue and ``docs/decisions.md``, never to ``docs/metrics.md``. The held-out
game is not read.

Three questions, each answered with rank-1 (is the nearest candidate the same player?) and MRR:

- ``reassoc``: one crop of a player against the crops of everyone on the floor 1, 3 and 5 s later in
  the same shot. ``all`` ranks every person, ``team`` only the player's teammates (same kit, the hard
  case once the team layer has done its job). ``position`` is the reference a tracker has without
  appearance: nearest box centre in pixels, no camera compensation.
- ``halves``: the mean embedding of a player's boxes in the first half of a shot against the means of
  his teammates in the second half, 1 s apart. This is the offline linking decision.
- ``cross_shot``: the mean embedding of a numbered player in one shot against the numbered teammates
  in another shot of the same game. This is identity carried over a cut without reading the number.

Runs outside the project environment, so the lock file stays as it is:

    uv run --no-project --python 3.12 --with reid==0.1.0 scripts/probe_reid.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

GAPS_S = (1, 3, 5)
QUERY_STRIDE = 5  # sequence frames between query frames (0.5 s), so neighbouring queries differ
MIN_BOXES = 5  # boxes needed for a mean embedding
PLAYER_CLASSES = (1, 2)  # gold classes: 1 home, 2 away, 3 referee, 4 other


def load_sequences(gold: Path, raw: Path) -> list[dict]:
    """Dev sequences only: the held-out game is scored once per milestone by ``bga eval``."""
    out = []
    for s in json.loads((gold / "manifest.json").read_text())["sequences"]:
        if s["held_out"]:
            continue
        d = gold / s["name"]
        number = {}
        for r in csv.DictReader((d / "identities.csv").open()):
            if r["number"]:
                number[int(r["tid"])] = r["number"]
        rows = np.loadtxt(d / "gt" / "gt.txt", delimiter=",", ndmin=2)
        out.append(
            {
                "name": s["name"],
                "game": s["clip"],
                "video": raw / f"{s['clip']}.mp4",
                "source_frames": [int(x) for x in (d / "frames.txt").read_text().split()],
                # visibility 0.5 marks interpolated or occluded boxes; they are neither query nor candidate
                "rows": rows[rows[:, 8] >= 1.0],
                "number": number,
            }
        )
    return out


def read_frames(video: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    keep, out, pos = set(wanted), {}, wanted[0]
    while pos <= wanted[-1]:
        if pos in keep:
            ok, img = cap.read()
            if not ok:
                raise IndexError(f"{video.name}: frame {pos} unreadable")
            out[pos] = img
        elif not cap.grab():
            raise IndexError(f"{video.name}: frame {pos} unreadable")
        pos += 1
    cap.release()
    return out


def embed(seqs: list[dict], names: list[str]) -> tuple[dict, dict[str, np.ndarray], dict[str, float]]:
    """Embed every clean gold box with every model. Returns box metadata, embeddings and ms per crop."""
    import supervision as sv
    from reid import ReIDModel

    models = {}
    for n in names:
        try:
            models[n] = ReIDModel.from_pretrained(n)
        except Exception as e:  # a weights host can be down; the others still answer the question
            print(f"skipped {n}: {type(e).__name__}: {e}")
    meta = defaultdict(list)
    parts, seconds = {n: [] for n in models}, dict.fromkeys(models, 0.0)
    for si, s in enumerate(seqs):
        rows = s["rows"]
        seq_frames = sorted({int(f) for f in rows[:, 0]})
        frames = read_frames(s["video"], [s["source_frames"][f - 1] for f in seq_frames])
        for f in seq_frames:
            r = rows[rows[:, 0] == f]
            x, y, w, h = r[:, 2], r[:, 3], r[:, 4], r[:, 5]
            xyxy = np.column_stack([x, y, x + w, y + h]).astype(np.float32)
            img = frames[s["source_frames"][f - 1]]
            for n, m in models.items():
                t0 = time.perf_counter()
                parts[n].append(m.extract_embeddings(sv.Detections(xyxy=xyxy), img))
                seconds[n] += time.perf_counter() - t0
            meta["seq"] += [si] * len(r)
            meta["frame"] += [f] * len(r)
            meta["tid"] += [int(t) for t in r[:, 1]]
            meta["cls"] += [int(c) for c in r[:, 7]]
            meta["cx"] += list((xyxy[:, 0] + xyxy[:, 2]) / 2)
            meta["cy"] += list((xyxy[:, 1] + xyxy[:, 3]) / 2)
        print(f"embedded {s['name']}: {len(rows)} boxes on {len(seq_frames)} frames")
    meta = {k: np.array(v) for k, v in meta.items()}
    embs = {}
    for n in models:
        e = np.concatenate(parts[n])
        embs[n] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return meta, embs, {n: 1000 * seconds[n] / len(meta["tid"]) for n in models}


class Score:
    def __init__(self) -> None:
        self.ranks: list[int] = []
        self.candidates: list[int] = []

    def add(self, distances: np.ndarray, correct: int) -> None:
        self.ranks.append(1 + int((distances < distances[correct]).sum()))
        self.candidates.append(len(distances))

    def summary(self) -> dict:
        r, c = np.array(self.ranks), np.array(self.candidates)
        if not len(r):
            return {"n": 0}
        return {
            "n": len(r),
            "candidates": round(float(c.mean()), 1),
            "chance": round(float((1 / c).mean()), 3),
            "rank1": round(float((r == 1).mean()), 3),
            "mrr": round(float((1 / r).mean()), 3),
        }


def reassoc(meta: dict, emb: np.ndarray | None, fps: int) -> dict:
    """``emb`` None scores the position reference instead of an encoder."""
    xy = np.column_stack([meta["cx"], meta["cy"]])
    scores = {(g, pool): Score() for g in GAPS_S for pool in ("all", "team")}
    for si in np.unique(meta["seq"]):
        in_seq = meta["seq"] == si
        seq_frames = np.unique(meta["frame"][in_seq])
        by_frame = {f: np.flatnonzero(in_seq & (meta["frame"] == f)) for f in seq_frames}
        for f, q_idx in by_frame.items():
            if (f - 1) % QUERY_STRIDE:
                continue
            for g in GAPS_S:
                later = by_frame.get(f + g * fps)
                if later is None:
                    continue
                for q in q_idx:
                    if meta["cls"][q] not in PLAYER_CLASSES or meta["tid"][q] not in meta["tid"][later]:
                        continue
                    for pool in ("all", "team"):
                        cand = later if pool == "all" else later[meta["cls"][later] == meta["cls"][q]]
                        if len(cand) < 2:
                            continue
                        if emb is None:
                            d = np.linalg.norm(xy[cand] - xy[q], axis=1)
                        else:
                            d = 1 - emb[cand] @ emb[q]
                        scores[(g, pool)].add(d, int(np.flatnonzero(meta["tid"][cand] == meta["tid"][q])[0]))
    return {f"{g}s_{pool}": s.summary() for (g, pool), s in scores.items()}


def mean_embedding(emb: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    if len(idx) < MIN_BOXES:
        return None
    m = emb[idx].mean(axis=0)
    return m / np.linalg.norm(m)


def rank_groups(score: Score, queries: dict, gallery: dict) -> None:
    """``queries`` and ``gallery`` map (team, identity) to a mean embedding; teammates are the candidates."""
    for (team, ident), q in queries.items():
        keys = [k for k in gallery if k[0] == team]
        if (team, ident) not in gallery or len(keys) < 2:
            continue
        score.add(1 - np.stack([gallery[k] for k in keys]) @ q, keys.index((team, ident)))


def halves(meta: dict, emb: np.ndarray, fps: int) -> dict:
    score = Score()
    for si in np.unique(meta["seq"]):
        in_seq = (meta["seq"] == si) & np.isin(meta["cls"], PLAYER_CLASSES)
        mid = (meta["frame"][in_seq].min() + meta["frame"][in_seq].max()) / 2
        first, second = {}, {}
        for tid in np.unique(meta["tid"][in_seq]):
            mine = in_seq & (meta["tid"] == tid)
            key = (int(meta["cls"][mine][0]), int(tid))
            a = mean_embedding(emb, np.flatnonzero(mine & (meta["frame"] <= mid - fps / 2)))
            b = mean_embedding(emb, np.flatnonzero(mine & (meta["frame"] > mid + fps / 2)))
            if a is not None:
                first[key] = a
            if b is not None:
                second[key] = b
        rank_groups(score, first, second)
    return score.summary()


def cross_shot(meta: dict, emb: np.ndarray, seqs: list[dict]) -> dict:
    per_seq = []
    for si, s in enumerate(seqs):
        groups = {}
        for tid, number in s["number"].items():
            mine = np.flatnonzero((meta["seq"] == si) & (meta["tid"] == tid))
            if len(mine) and meta["cls"][mine[0]] in PLAYER_CLASSES:
                groups.setdefault((int(meta["cls"][mine[0]]), number), []).extend(mine)
        means = {k: mean_embedding(emb, np.array(v)) for k, v in groups.items()}
        per_seq.append({k: m for k, m in means.items() if m is not None})
    total, by_game = Score(), defaultdict(Score)
    for a, sa in enumerate(seqs):
        for b, sb in enumerate(seqs):
            if a != b and sa["game"] == sb["game"]:
                for score in (total, by_game[sa["game"].split("_")[1]]):
                    rank_groups(score, per_seq[a], per_seq[b])
    return {"all": total.summary(), **{g: s.summary() for g, s in sorted(by_game.items())}}


def main(argv=None) -> None:
    from reid import list_aliases

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", type=Path, default=Path("data/gold/v0"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--models", nargs="*", default=list_aliases())
    ap.add_argument("--out", type=Path, default=Path("runs/_scratch/probe_reid/result.json"))
    a = ap.parse_args(argv)

    seqs = load_sequences(a.gold, a.raw)
    fps = 10  # gold sequences keep every 5th frame of 50 fps footage
    meta, embs, ms = embed(seqs, a.models)
    result = {
        "gold": a.gold.name,
        "sequences": [s["name"] for s in seqs],
        "boxes": len(meta["tid"]),
        "position": reassoc(meta, None, fps),
        "models": {
            n: {
                "ms_per_crop": round(ms[n], 2),
                "reassoc": reassoc(meta, e, fps),
                "halves": halves(meta, e, fps),
                "cross_shot": cross_shot(meta, e, seqs),
            }
            for n, e in embs.items()
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))

    def cell(s: dict) -> str:
        return f"{s['rank1']:.3f} / {s['mrr']:.3f}" if s["n"] else "n/a"

    cols = [f"{g}s_{pool}" for pool in ("all", "team") for g in GAPS_S]
    first = next(iter(result["models"].values()))
    print(f"\n{result['boxes']} clean boxes, {len(seqs)} dev shots, gold {result['gold']}.")
    print("Cells: rank-1 / MRR.\n")
    print("| encoder | " + " | ".join(cols) + " | halves | cross-shot | ms/crop |")
    print("|---|" + "---|" * (len(cols) + 3))
    counts = [f"{first['reassoc'][c]['n']} ({first['reassoc'][c]['candidates']})" for c in cols]
    counts += [f"{first[k]['n']} ({first[k]['candidates']})" for k in ("halves",)]
    counts += [f"{first['cross_shot']['all']['n']} ({first['cross_shot']['all']['candidates']})"]
    print("| queries (mean candidates) | " + " | ".join(counts) + " | |")
    chance = [f"{first['reassoc'][c]['chance']:.3f}" for c in cols]
    chance += [f"{first['halves']['chance']:.3f}", f"{first['cross_shot']['all']['chance']:.3f}"]
    print("| chance, rank-1 | " + " | ".join(chance) + " | |")
    print("| position only | " + " | ".join(cell(result["position"][c]) for c in cols) + " | n/a | n/a | |")
    for n, m in result["models"].items():
        cells = [cell(m["reassoc"][c]) for c in cols] + [cell(m["halves"]), cell(m["cross_shot"]["all"])]
        print(f"| `{n}` | " + " | ".join(cells) + f" | {m['ms_per_crop']:.1f} |")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
