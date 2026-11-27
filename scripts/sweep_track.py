#!/usr/bin/env python3
"""Grid sweep of tracker params on gold dev, in one process: run the track layer per setting and score it.

    uv run scripts/sweep_track.py configs/track-bytetrack.yaml \
        --grid track.params.min_score=0.05,0.1,0.3 --grid track.params.lost_track_buffer_s=1,2,3
    uv run scripts/sweep_track.py configs/track-trackers-botsort-reid.yaml --remote-gpu L4 \
        --grid track.params.reid_config.appearance_threshold=0.1,0.25

With ``--remote-gpu`` the track layer of every setting and sequence missing from ``runs/`` runs on Modal in one
parallel batch first (cost rows go to docs/costs.md); scoring then runs locally on the stored outputs.

Upstream layers must already be cached (run ``bga eval`` on the base config first); only the track layer runs per
setting. Every setting's scores go to runs/_sweeps/<config>_<timestamp>.csv, and the best by HOTA are printed. The
sweep only picks settings to commit as configs: metrics rows come from ``bga eval`` on those configs, never from here.
Never sweep on the held-out game.

"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bga.config import load_config
from bga.eval.scores import score
from bga.pipeline import run

COLS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "team_acc", "pred_ids_per_seq"]


def run_remote(config, keys, values, gpu):
    """Run the track layer of every setting on Modal in one parallel batch, writing outputs where bga reads them.

    Like ``bga.pipeline.run``, a setting reruns outputs made by other track code and refuses to run on detect or court
    outputs made by other code. Finished outputs are recorded with their code key even when the sweep stops part way.

    """
    from bga.config import LAYERS
    from bga.pipeline import (
        _write_meta,
        check_upstreams,
        code_keys,
        load_output,
        output_path,
        sequences,
        stale_sequences,
    )
    from bga.remote.modal_runner import run_layer

    runs = Path("runs")
    items = []
    for combo in itertools.product(*values):
        cfg = load_config(config, [f"{k}={v}" for k, v in zip(keys, combo, strict=True)])
        cfg_keys = code_keys(cfg)
        seqs = sequences(cfg)
        stale = {s.name for s in stale_sequences(cfg, runs, seqs, "track", cfg_keys["track"])}
        todo = [s for s in seqs if s.name in stale or not output_path(cfg, runs, s, "track").exists()]
        check_upstreams(cfg, runs, todo, "track", cfg_keys)
        for seq in todo:
            ups = {u: load_output(cfg, runs, seq, u) for u in LAYERS["track"] if u in cfg.layers}
            items.append((cfg, seq, ups, cfg_keys["track"]))
    if not items:
        return
    t0 = time.perf_counter()
    entries = [cfg.layers["track"].model_dump() for cfg, _, _, _ in items]
    finished = []

    def keep(i, out):  # written as each call finishes, so a stopped sweep keeps what finished
        cfg, seq, _, _ = items[i]
        path = output_path(cfg, runs, seq, "track")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, separators=(",", ":")))
        finished.append(i)

    try:
        run_layer(
            "track",
            entries,
            [s for _, s, _, _ in items],
            [u for _, _, u, _ in items],
            None if gpu == "cpu" else gpu,
            keep,
        )
    finally:
        done = {}
        for i in finished:
            cfg, seq, _, key = items[i]
            done.setdefault(cfg.name, (cfg, key, []))[2].append(seq)
        for cfg, key, seqs in done.values():
            _write_meta(cfg, runs, "track", seqs, t0, key)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--grid", action="append", required=True, help="key=v1,v2,...")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--remote-gpu", default=None, help="run every setting's track layer on Modal in one batch")
    a = ap.parse_args(argv)
    keys, values = [], []
    for g in a.grid:
        k, vs = g.split("=", 1)
        keys.append(k)
        values.append(vs.split(","))
    out = Path("runs/_sweeps") / f"{a.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.remote_gpu:
        run_remote(a.config, keys, values, a.remote_gpu)
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
