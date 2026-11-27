"""Break down Layer 7 errors on dev gold: holder error classes, outcome swaps, shooter swaps, missed shots.

Diagnostic aid for #46 (not a pipeline layer). Reuses cached pipeline outputs exactly as ``bga eval-ball`` does.

    uv run python scripts/ball_errors.py --config configs/ball-plk-fused-v2.yaml [--set ...] [--frames]

"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bga import pipeline
from bga.ball.base import bbox_iou, point_to_box_dist
from bga.config import load_config
from bga.eval import ball as eb
from bga.eval.scores import read_gt

RUNS = Path("runs")


def collect(cfg):
    track_layer = "link" if "link" in cfg.layers else "track"
    cfgs = {}
    for s in pipeline.run(cfg, runs=RUNS):
        cfgs[s.name] = (cfg, s)
    sg = eb._shot_gold_config(cfg, eb.GOLD_V1_SHOT_GOLD)
    for s in pipeline.run(sg, runs=RUNS):
        cfgs.setdefault(s.name, (sg, s))
    out = {}
    for name, (c, s) in cfgs.items():
        out[name] = {
            "ball": pipeline.load_output(c, RUNS, s, "ball"),
            "tracks": pipeline.load_output(c, RUNS, s, track_layer)["frames"],
            "detect": pipeline.load_output(c, RUNS, s, "detect")["frames"],
        }
    return out


def gold():
    frames = json.loads(eb.GOLD_V1_BOXES.read_text())["frames"]
    holders = eb.load_holders() or {}
    events_doc = json.loads(eb.GOLD_V1_EVENTS.read_text())
    events = list(events_doc["events"])
    verified = set(events_doc["verified_sequences"])
    sf, sh, se, sv = eb.load_shot_gold()
    return [*frames, *sf], {**holders, **sh}, [*events, *se], verified | sv


def holder_class(gt_h, row, trows):
    pred = row.get("holder_track_id")
    if not gt_h:
        return "ok" if pred is None else "fp_held_when_nobody"
    if pred is None:
        return "fn_missed_holder"
    pbox = eb._box_for_tid(trows, pred)
    if pbox is not None and bbox_iou(gt_h[0]["xyxy"], pbox) >= 0.5:
        return "ok"
    team = gt_h[0].get("team")
    return "wrong_team_player" if row.get("holder_team") not in (None, team) else "wrong_player_box"


def frame_features(fr, row, trows, dets):
    gt_ball = next((b["xyxy"] for b in fr["boxes"] if b["cls"] == "ball"), None)
    bpx = row.get("ball_px")
    players = [[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in trows]
    d_pred = min((point_to_box_dist(bpx[0], bpx[1], b) for b in players), default=None) if bpx else None
    gc = ((gt_ball[0] + gt_ball[2]) / 2, (gt_ball[1] + gt_ball[3]) / 2) if gt_ball else None
    d_gt = min((point_to_box_dist(gc[0], gc[1], b) for b in players), default=None) if gc else None
    pip = max((float(b[4]) for b in dets if b[5] == "player-in-possession"), default=0.0)
    return {
        "gt_ball": gt_ball is not None,
        "pred_ball": bpx is not None,
        "ball_conf": round(float(row.get("ball_conf") or 0.0), 2),
        "d_pred_px": None if d_pred is None else round(d_pred),
        "d_gt_px": None if d_gt is None else round(d_gt),
        "pip_max": round(pip, 2),
    }


def report_holders(data, frames, holders, show_frames):
    cls_count = Counter()
    for fr in frames:
        name = f"{fr['youtube_id']}_s{fr['shot']}"
        if name not in data:
            continue
        d = data[name]
        row = d["ball"]["frames"].get(str(fr["frame"]), {})
        trows = d["tracks"].get(str(fr["frame"]), [])
        c = holder_class(eb._gold_holder(fr, holders), row, trows)
        cls_count[c] += 1
        if show_frames and c != "ok":
            feats = frame_features(fr, row, trows, d["detect"].get(str(fr["frame"]), []))
            print(f"{c:22s} {name:22s} f={fr['frame']:6d} {feats}")
    print("holders:", dict(cls_count))


def report_sequence_shots(name, d, events):
    gt = [e for e in events if e["sequence"] == name]
    pred = d["ball"].get("shots", [])
    m = eb._match_shots(gt, pred, eb.DEFAULT_RIM_TOL_FRAMES)
    mg = {g for g, _ in m}
    mp = {p for _, p in m}
    gt_by_f = {}
    if (eb.GOLD_V0_ROOT / name / "gt" / "gt.txt").exists():
        _, gt_by_f, _ = read_gt(eb.GOLD_V0_ROOT, name)
    for gi, g in enumerate(gt):
        if gi not in mg:
            print(f"MISSED   {name:22s} rim={g['rim_frame']} {g.get('outcome')} {g.get('points', '')}")
    for pi, p in enumerate(pred):
        if pi not in mp:
            print(f"FP       {name:22s} rim={p['rim_frame']} {p['outcome']}")
    for gi, pi in m:
        g, p = gt[gi], pred[pi]
        if p["outcome"] != g.get("outcome"):
            print(f"OUTCOME  {name:22s} rim={g['rim_frame']} gt={g.get('outcome')} pred={p['outcome']}")
        has_shooter = any(g.get(k) is not None for k in ("shooter_gt_tid", "shooter_xyxy", "shooter_team"))
        if has_shooter and not eb._shooter_matches(g, p, d["tracks"], gt_by_f):
            print(
                f"SHOOTER  {name:22s} rim={g['rim_frame']} gt_team={g.get('shooter_team')} "
                f"pred_team={p.get('shooter_team')} start={p['start_frame']} gt_rel={g.get('release_frame')}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--frames", action="store_true", help="print each holder error frame with features")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    cfg.layers["shots"].params["held_out"] = "exclude"
    data = collect(cfg)
    frames, holders, events, verified = gold()
    report_holders(data, frames, holders, a.frames)
    for name, d in sorted(data.items()):
        if name in verified:
            report_sequence_shots(name, d, events)


if __name__ == "__main__":
    main()
