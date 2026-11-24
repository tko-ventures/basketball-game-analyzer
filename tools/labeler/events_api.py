"""Backend helpers for the ``/events`` gold shot-event verification UI.

Serves gold ``v0`` sequence metadata, player boxes, and roster identities for all 12 gold shots (9 dev,
3 held-out), pre-seeds dev shots from cached ``ball-plk-fused`` outputs when ``data/gold/v1/events.json`` does
not yet exist (never touching pipeline predictions for held-out sequences), and atomically writes verified
events to ``data/gold/v1/events.json``.

"""

from __future__ import annotations

import configparser
import csv
import json
from pathlib import Path
from typing import Any

from bga.ball.base import bbox_iou
from bga.config import load_config
from bga.eval.scores import read_gt
from bga.pipeline import load_output
from bga.stage import Sequence

#: Canonical path of the gold v1 shot-events file.
EVENTS_PATH = Path("data/gold/v1/events.json")
#: Root of exported gold v0 sequences.
GOLD_V0 = Path("data/gold/v0")
#: Labels folder holding per-clip metadata (video path, kits, teams).
LABELS_DIR = Path("data/labels")
#: Default dev config used solely to pre-seed draft dev shot events when ``events.json`` is missing.
PREFILL_CONFIG = Path("configs/ball-plk-fused.yaml")


def _read_identities(seq_dir: Path) -> list[dict[str, Any]]:
    """Read ``identities.csv`` for one gold sequence as a list of player dicts."""
    path = seq_dir / "identities.csv"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            team = (row.get("team") or "").strip()
            if team not in ("home", "away"):
                continue
            out.append({
                "gt_tid": int(row["tid"]),
                "team": team,
                "number": (row.get("number") or "").strip(),
                "name": (row.get("name") or "").strip(),
                "team_name": (row.get("team_name") or "").strip(),
            })
    out.sort(
        key=lambda r: (
            0 if r["team"] == "home" else 1,
            int(r["number"]) if r["number"].isdigit() else 999,
            r["gt_tid"],
        )
    )
    return out


def _read_gt_boxes(gold_v0: Path, seq_name: str) -> dict[str, list[list[Any]]]:
    """Return ``{frame_str: [[gt_tid, x1, y1, x2, y2, team], ...]}`` from ``gt/gt.txt``."""
    _, per_f, ids = read_gt(gold_v0, seq_name)
    out: dict[str, list[list[Any]]] = {}
    for f, rows in per_f.items():
        out[str(f)] = [
            [
                int(r[0]),
                round(float(r[1]), 1),
                round(float(r[2]), 1),
                round(float(r[3]), 1),
                round(float(r[4]), 1),
                (ids.get(str(r[0])) or {}).get("team", ""),
            ]
            for r in rows
        ]
    return out


def _match_gt_tid(
    pbox: list[float] | None,
    sf: int,
    gt_boxes: dict[str, list[list[Any]]],
) -> int | None:
    """Match a predicted shooter box at ``sf`` to the closest gold ``v0`` track id."""
    if pbox is None or not gt_boxes:
        return None
    near_f = min((int(k) for k in gt_boxes), key=lambda f: abs(f - sf))
    best_tid, best_iou = None, 0.25
    for g in gt_boxes[str(near_f)]:
        ov = bbox_iou(pbox, g[1:5])
        if ov > best_iou:
            best_iou, best_tid = ov, int(g[0])
    return best_tid


def _dev_prefill(
    seq_rows: list[dict[str, Any]],
    gt_by_seq: dict[str, dict[str, list[list[Any]]]],
    runs: Path = Path("runs"),
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float] | None]]]:
    """Load cached dev ``ball-plk-fused`` predictions to pre-seed draft events and ball overlay dots.

    Never touches held-out sequences.

    """
    if not PREFILL_CONFIG.exists():
        return [], {}
    cfg = load_config(PREFILL_CONFIG)
    draft_events: list[dict[str, Any]] = []
    ball_px_by_seq: dict[str, dict[str, list[float] | None]] = {}
    for meta in seq_rows:
        if meta["held_out"]:
            continue
        seq = Sequence(
            name=meta["sequence"],
            clip=meta["clip"],
            video=meta["video"],
            shot=meta["shot"],
            frames=meta["frames"],
            fps=meta["fps"],
            held_out=False,
        )
        ball_file = cfg.run_dir(runs, seq.clip, "ball") / f"{seq.name}.json"
        if not ball_file.exists():
            continue
        ball_out = load_output(cfg, runs, seq, "ball")
        track_layer = "link" if "link" in cfg.layers else "track"
        tracks = load_output(cfg, runs, seq, track_layer)["frames"]
        ball_px_by_seq[seq.name] = {
            f: row.get("ball_px") for f, row in ball_out.get("frames", {}).items() if row.get("ball_px") is not None
        }
        gt_boxes = gt_by_seq.get(seq.name, {})
        for sh in ball_out.get("shots", []):
            sf = int(sh.get("start_frame", sh["rim_frame"]))
            tid = sh.get("shooter_track_id")
            pbox = (
                next((r[1:5] for r in tracks.get(str(sf), []) if int(r[0]) == tid), None) if tid is not None else None
            )
            gt_tid = _match_gt_tid(pbox, sf, gt_boxes)
            draft_events.append({
                "sequence": seq.name,
                "youtube_id": meta["youtube_id"],
                "shot": meta["shot"],
                "rim_frame": int(sh["rim_frame"]),
                "release_frame": sf,
                "outcome": str(sh.get("outcome", "missed")),
                "points": int(sh.get("points") or 2),
                "shot_type": str(sh.get("shot_type") or "jump"),
                "shooter_team": sh.get("shooter_team") or "home",
                "shooter_gt_tid": gt_tid,
                "note": "",
            })
    return draft_events, ball_px_by_seq


def load_events_state(
    gold_v0: Path = GOLD_V0,
    labels_dir: Path = LABELS_DIR,
    events_path: Path = EVENTS_PATH,
    runs: Path = Path("runs"),
) -> dict[str, Any]:
    """Build the full JSON payload for ``/api/events/state``.

    Args:
        gold_v0: Root directory of exported gold ``v0`` sequences.
        labels_dir: Directory containing per-clip label JSONs.
        events_path: Path to ``data/gold/v1/events.json``.
        runs: Root of cached pipeline runs (read only for dev sequences).

    Returns:
        Dict with ``sequences``, ``events``, ``verified_sequences``, ``run_held_out``, ``saved``, and ``ball_px``.

    Examples:
        >>> st = load_events_state()
        >>> len(st["sequences"]) >= 9
        True

    """
    manifest = json.loads((gold_v0 / "manifest.json").read_text(encoding="utf-8"))
    seq_rows: list[dict[str, Any]] = []
    gt_by_seq: dict[str, dict[str, list[list[Any]]]] = {}
    for s in manifest["sequences"]:
        sdir = gold_v0 / s["name"]
        ini = configparser.ConfigParser()
        ini.read(sdir / "seqinfo.ini")
        grid = [int(x) for x in (sdir / "frames.txt").read_text(encoding="utf-8").split()]
        labels = json.loads((labels_dir / f"{s['clip']}.json").read_text(encoding="utf-8"))
        meta = labels.get("meta", {})
        gt_boxes = _read_gt_boxes(gold_v0, s["name"])
        gt_by_seq[s["name"]] = gt_boxes
        seq_rows.append({
            "sequence": s["name"],
            "youtube_id": s["name"].rsplit("_s", 1)[0],
            "shot": int(s["shot"]),
            "clip": s["clip"],
            "video": labels["clip"],
            "held_out": bool(s["held_out"]),
            "fps": float(ini["Sequence"]["sourceFrameRate"]),
            "frames": list(range(grid[0], grid[-1] + 1, 5)),
            "home_team": (meta.get("home") or {}).get("team", "Home"),
            "away_team": (meta.get("away") or {}).get("team", "Away"),
            "home_kit": (meta.get("home") or {}).get("kit", ""),
            "away_kit": (meta.get("away") or {}).get("kit", ""),
            "identities": _read_identities(sdir),
            "gt_boxes": gt_boxes,
        })

    draft_events, ball_px_by_seq = _dev_prefill(seq_rows, gt_by_seq, runs=runs)
    if events_path.exists():
        saved = json.loads(events_path.read_text(encoding="utf-8"))
        events = saved.get("events", [])
        verified = saved.get("verified_sequences", [])
        run_held_out = saved.get("run_held_out")
        is_saved = True
    else:
        events = draft_events
        verified = []
        run_held_out = None
        is_saved = False

    return {
        "sequences": seq_rows,
        "events": events,
        "verified_sequences": verified,
        "run_held_out": run_held_out,
        "saved": is_saved,
        "ball_px": ball_px_by_seq,
    }


def save_events_state(body: dict[str, Any], events_path: Path = EVENTS_PATH) -> dict[str, Any]:
    """Validate and atomically save ``data/gold/v1/events.json``.

    Args:
        body: JSON payload from ``POST /api/events/save``.
        events_path: Destination JSON file path.

    Returns:
        Summary dict ``{"count": int, "verified": int, "path": str}``.

    Raises:
        ValueError: When ``events`` is missing or malformed.

    Examples:
        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "events.json"
        >>> res = save_events_state({"events": [], "verified_sequences": ["a_s1"]}, events_path=p)
        >>> (res["count"], res["verified"], p.exists())
        (0, 1, True)

    """
    raw_events = body.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("events must be a list")
    cleaned: list[dict[str, Any]] = []
    for ev in raw_events:
        outcome = str(ev.get("outcome", "missed"))
        if outcome not in ("made", "missed"):
            raise ValueError(f"invalid outcome: {outcome!r}")
        pts = int(ev.get("points", 2))
        if pts not in (2, 3):
            raise ValueError(f"invalid points: {pts!r}")
        gt_tid = ev.get("shooter_gt_tid")
        cleaned.append({
            "sequence": str(ev["sequence"]),
            "youtube_id": str(ev.get("youtube_id") or str(ev["sequence"]).rsplit("_s", 1)[0]),
            "shot": int(ev["shot"]),
            "rim_frame": int(ev["rim_frame"]),
            "release_frame": int(ev["release_frame"]) if ev.get("release_frame") is not None else None,
            "outcome": outcome,
            "points": pts,
            "shot_type": str(ev.get("shot_type") or "jump"),
            "shooter_team": str(ev.get("shooter_team") or "home"),
            "shooter_gt_tid": int(gt_tid) if gt_tid not in (None, "") else None,
            "note": str(ev.get("note") or ""),
        })
    cleaned.sort(key=lambda e: (e["sequence"], e["rim_frame"]))
    verified = sorted({str(x) for x in body.get("verified_sequences", [])})
    payload: dict[str, Any] = {
        "version": "v1",
        "verified_sequences": verified,
        "events": cleaned,
    }
    if body.get("run_held_out") is not None:
        payload["run_held_out"] = str(body["run_held_out"])
    events_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = events_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(events_path)
    return {"count": len(cleaned), "verified": len(verified), "path": str(events_path)}
