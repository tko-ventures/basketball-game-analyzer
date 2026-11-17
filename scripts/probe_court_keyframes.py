#!/usr/bin/env python3
"""Probe: how good are the NBA court mapping and keyframe propagation, judged against Tomasz's keyframes.

    uv run scripts/probe_court_keyframes.py [--spans 1,3]

Every queued keyframe sits in its own shot, so propagation cannot be scored against a second human
frame. Per usable keyframe (held-out games skipped) this reports:

- nba_m: median distance in metres, over the landmarks visible under Tomasz's mapping, between where
  his mapping and the NBA model's prefill put them on the court.
- reach: the share of the shot's every-5th frames that propagation from his keyframe labels.
- loop_<span>: round-trip drift in metres; propagate span seconds away from the keyframe, then back
  with a different step, and compare with his mapping.

A probe for M3's labelling aid, not a metrics row. Output: runs/_probe/court_keyframes.tsv.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from court_propagate import Reader, disagreement_m  # noqa: E402

from bga.court.propagate import Propagator  # noqa: E402


def round_trip(prop, read, key, H0, lo, hi, span_f, wh) -> float | None:
    """Drift after propagating span_f frames away from key (forwards if room, else backwards) and back."""
    far = key + span_f if key + span_f <= hi else key - span_f
    if not lo <= far <= hi:
        return None
    sgn = 1 if far > key else -1
    out = prop.run(read, sorted(range(key, far + sgn, 5 * sgn)), key, H0)
    if any(st.H is None for st in out.values()):
        return float("nan")  # lost before the span ended
    last = max(out) if sgn > 0 else min(out)
    back = sorted({key, last} | set(range(last, key, -10 * sgn)))
    res = prop.run(read, back, last, out[last].H)
    if res.get(key) is None or res[key].H is None:
        return float("nan")
    return disagreement_m(H0, res[key].H, wh)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="1,3", help="round-trip spans in seconds")
    a = ap.parse_args(argv)
    spans = [float(s) for s in a.spans.split(",")]

    gold = json.loads((ROOT / "data/labels/gold.json").read_text())["games"]
    held = {g["clip"] for g in gold if g["held_out"]}
    queue = json.loads((ROOT / "data/labels/court/queue.json").read_text())["keyframes"]
    prop, rows = Propagator(), []
    for k in queue:
        if k["clip"] in held:
            continue
        court = json.loads((ROOT / "data/labels/court" / f"{k['clip']}.json").read_text())
        rec = court["frames"].get(str(k["frame"]))
        if not rec or rec.get("status") != "ok" or not rec.get("H"):
            continue
        labels = json.loads((ROOT / "data/labels" / f"{k['clip']}.json").read_text())
        shot = labels["shots"][k["shot"]]
        read = Reader(labels["clip"])
        fps = read.cap.get(cv2.CAP_PROP_FPS)
        wh = (int(read.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        H, key, lo, hi = np.array(rec["H"]), k["frame"], shot["start"], shot["end"] - 1

        nba = disagreement_m(H, np.array(k["prefill_H"]), wh) if k.get("prefill_H") else None
        frames = sorted(set(range(key, lo - 1, -5)) | set(range(key, hi + 1, 5)))
        reach = sum(st.H is not None for st in prop.run(read, frames, key, H).values()) / len(frames)
        loops = [round_trip(prop, read, key, H, lo, hi, int(s * fps), wh) for s in spans]
        rows.append(
            {
                "hall": k["hall"],
                "purpose": k["purpose"],
                "clip": k["youtube_id"],
                "shot": k["shot"],
                "frame": key,
                "points": len(rec.get("fixed", {})),
                "nba_m": nba,
                "reach": reach,
                **{f"loop_{s:g}s": v for s, v in zip(spans, loops, strict=True)},
            }
        )
        r = rows[-1]
        print(
            f"{r['hall'][:24]:24} {r['purpose']:5} s{r['shot']:<3} nba "
            + (f"{nba:6.2f} m" if nba is not None else "     -  ")
            + f"  reach {reach:4.0%}  "
            + "  ".join(
                f"loop {s:g}s " + (f"{v:.2f} m" if v is not None else "-")
                for s, v in zip(spans, loops, strict=True)
            ),
            flush=True,
        )

    out = ROOT / "runs/_probe/court_keyframes.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    out.write_text("\t".join(cols) + "\n" + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r in rows))

    def med(xs):
        xs = [x for x in xs if x is not None and x == x]
        return f"{statistics.median(xs):.2f}" if xs else "-"

    print(f"\n{len(rows)} usable keyframes outside the held-out game -> {out}")
    print(f"NBA prefill vs Tomasz: median {med([r['nba_m'] for r in rows])} m")
    print(f"propagation reach: median {statistics.median(r['reach'] for r in rows):.0%} of the shot")
    for s in spans:
        vals = [r[f"loop_{s:g}s"] for r in rows]
        lost = sum(1 for v in vals if v is not None and v != v)
        tried = sum(v is not None for v in vals)
        print(f"round trip {s:g} s: median drift {med(vals)} m, lost {lost} of {tried}")


if __name__ == "__main__":
    main()
