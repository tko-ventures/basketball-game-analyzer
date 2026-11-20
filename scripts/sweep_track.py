#!/usr/bin/env python3
"""Grid sweep of tracker params on gold dev, in one process: run the track layer per setting and score it.

    uv run scripts/sweep_track.py configs/track-bytetrack.yaml \
        --grid track.params.min_score=0.05,0.1,0.3 --grid track.params.lost_track_buffer_s=1,2,3

Upstream layers must already be cached (run ``bga eval`` on the base config first); only the track layer runs per
setting. Every setting's scores go to runs/_sweeps/<config>_<timestamp>.csv, and the best by HOTA are printed. The
sweep only picks settings to commit as configs: metrics rows come from ``bga eval`` on those configs, never from here.
Never sweep on the held-out game.

"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.config import load_config
from bga.eval.scores import score
from bga.pipeline import run

COLS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "team_acc", "pred_ids_per_seq"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--grid", action="append", required=True, help="key=v1,v2,...")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    keys, values = [], []
    for g in a.grid:
        k, vs = g.split("=", 1)
        keys.append(k)
        values.append(vs.split(","))
    out = Path("runs/_sweeps") / f"{a.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*keys, *COLS])
        for combo in itertools.product(*values):
            cfg = load_config(a.config, [f"{k}={v}" for k, v in zip(keys, combo, strict=True)])
            if cfg.layers["shots"].params.get("held_out"):
                sys.exit("refusing to sweep on the held-out game")
            seqs = run(cfg, layers=["shots", "track"])
            s = score(cfg, seqs)
            row = [*combo, *(round(float(s[c]), 4) for c in COLS)]
            w.writerow(row)
            fh.flush()
            rows.append(row)
    n = len(keys)
    rows.sort(key=lambda r: -r[n])
    print(" ".join(k.split(".")[-1] for k in keys), *COLS)
    for r in rows[: a.top]:
        print(*r)
    print(out, len(rows), "settings")


if __name__ == "__main__":
    main()
