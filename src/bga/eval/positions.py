"""2D positions evaluation on gold shots (plan section 8, M8).

Measures every layer-6 backend on the selected gold shots:
- ``kf_med_m`` / ``kf_p90_m``: median and p90 position error in metres on the gold court keyframes inside the
  shots, comparing the pipeline's ``(x_m, y_m)`` against the foot point projected through Tomasz's labelled
  ground-truth homography ``H_gt`` (one gold keyframe falls inside every gold shot).
- ``on_court``: fraction of non-referee player positions inside ``[-margin, 28 + margin] x [-margin, 15 + margin]``.
- ``speed_p95`` and ``jump_pct``: 95th-percentile step speed (m/s) and fraction of steps above ``10.0 m/s``
  (unphysical projection or airborne spikes).
- ``cov``: fraction of tracked person-frames that have a finite ``(x_m, y_m)``.
- ``pose_pct`` and ``airborne``: ankle-keypoint match share and held-airborne frame count when reported by the
  backend.
- ``ms_per_frame``: positions-stage latency per frame when reported (``0.0`` for pure geometric backends).

"""

from __future__ import annotations

import itertools
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from bga import pipeline
from bga.config import RunConfig
from bga.eval.court import gold_keyframes
from bga.export.radar import track_teams
from bga.positions.base import COURT_M, project
from bga.stage import Sequence

#: Sideline tolerance in metres when measuring player on-court rate (allows inbounders/corner shooters).
COURT_MARGIN_M = 0.5
#: Physical ceiling on frame-to-frame player speed (m/s); steps above this flag projection/airborne spikes.
MAX_SPRINT_MPS = 10.0


def _index_points(mapping: dict[str, list[list[float]]]) -> dict[tuple[int, int], tuple[float, float]]:
    """Index ``{tid_str: [[frame, x, y], ...]}`` into ``{(frame, tid): (x, y)}``."""
    return {(int(row[0]), int(tid)): (float(row[1]), float(row[2])) for tid, pts in mapping.items() for row in pts}


def _track_speeds(pos_tracks: dict[str, list[list[float]]], fps: float) -> list[float]:
    """Compute consecutive frame-to-frame player speeds in m/s across all tracks."""
    speeds: list[float] = []
    for pts in pos_tracks.values():
        for prev, curr in itertools.pairwise(pts):
            dt_s = (float(curr[0]) - float(prev[0])) / float(fps)
            if dt_s > 0.0:
                speeds.append(float(np.hypot(float(curr[1]) - float(prev[1]), float(curr[2]) - float(prev[2]))) / dt_s)
    return speeds


def _keyframe_errors(
    seq: Sequence,
    tracks: dict[str, list],
    pos_idx: dict[tuple[int, int], tuple[float, float]],
    foot_idx: dict[tuple[int, int], tuple[float, float]],
    kfs: list[dict[str, Any]],
) -> list[float]:
    """Compute position errors (m) on gold court keyframes inside ``seq`` against ``H_gt(foot_px)``."""
    errs: list[float] = []
    for kf in kfs:
        if not (seq.frames[0] <= int(kf["frame"]) <= seq.frames[-1]):
            continue
        f_near = min(seq.frames, key=lambda fr: abs(fr - int(kf["frame"])))
        h_gt = np.asarray(kf["H"], dtype=np.float64)
        for r in tracks.get(str(f_near), []):
            tid = int(r[0])
            xy = pos_idx.get((f_near, tid))
            if xy is None:
                continue
            foot = foot_idx.get((f_near, tid), ((float(r[1]) + float(r[3])) / 2.0, float(r[4])))
            gt_xy = project(h_gt, np.array([foot], dtype=np.float64))[0] * 0.01
            if np.all(np.isfinite(gt_xy)):
                errs.append(float(np.hypot(xy[0] - gt_xy[0], xy[1] - gt_xy[1])))
    return errs


def _is_on_court(xy: tuple[float, float]) -> bool:
    """Return True if ``xy`` lies inside the 28x15 m FIBA court with ``COURT_MARGIN_M`` sideline margin."""
    return (
        -COURT_MARGIN_M <= xy[0] <= COURT_M[0] + COURT_MARGIN_M
        and -COURT_MARGIN_M <= xy[1] <= COURT_M[1] + COURT_MARGIN_M
    )


def _sequence_metrics(
    cfg: RunConfig,
    seq: Sequence,
    runs: Path,
    kfs_by_clip: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute per-sequence 2D position diagnostics from cached run outputs."""
    track_layer = "link" if "link" in cfg.layers else "track"
    tracks = pipeline.load_output(cfg, runs, seq, track_layer)["frames"]
    teams = pipeline.load_output(cfg, runs, seq, "team")["frames"] if "team" in cfg.layers else {}
    team_of = track_teams(tracks, teams)
    pos_out = pipeline.load_output(cfg, runs, seq, "positions")
    pos_idx = _index_points(pos_out.get("tracks", {}))
    foot_idx = _index_points(pos_out.get("foot_px", {}))

    total_obs = mapped_obs = player_obs = on_court_obs = 0
    for f_str, rows in tracks.items():
        f = int(f_str)
        for r in rows:
            tid = int(r[0])
            total_obs += 1
            xy = pos_idx.get((f, tid))
            if xy is None or not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                continue
            mapped_obs += 1
            if team_of.get(tid) != "ref":
                player_obs += 1
                on_court_obs += int(_is_on_court(xy))

    stats = pos_out.get("stats") or {}
    return {
        "total_obs": total_obs,
        "mapped_obs": mapped_obs,
        "player_obs": player_obs,
        "on_court_obs": on_court_obs,
        "speeds": _track_speeds(pos_out.get("tracks", {}), seq.fps),
        "kf_errs": _keyframe_errors(seq, tracks, pos_idx, foot_idx, kfs_by_clip.get(seq.clip, [])),
        "pose_hits": int(stats.get("both_ankles", 0)) + int(stats.get("one_ankle", 0)),
        "pose_total": int(stats.get("both_ankles", 0))
        + int(stats.get("one_ankle", 0))
        + int(stats.get("bbox_bottom", 0)),
        "airborne_frames": int(stats.get("airborne_frames", 0)),
        "ms_per_frame": float(pos_out.get("ms_per_frame", 0.0)),
    }


def score(cfg: RunConfig, runs: Path = Path("runs"), held_out: bool = False) -> dict[str, Any]:
    """Run ``cfg`` (reusing cached outputs) and score its ``positions`` layer across the selected gold split.

    Args:
        cfg: Run config including ``positions``.
        runs: Root of pipeline run outputs.
        held_out: Score the held-out gold game instead of dev.

    Returns:
        Summary metrics dict across all selected sequences.

    """
    seqs = pipeline.run(cfg, runs=runs)
    kfs_by_clip: dict[str, list[dict[str, Any]]] = {}
    for kf in gold_keyframes(held_out=held_out):
        kfs_by_clip.setdefault(kf["clip"], []).append(kf)

    total_obs = mapped_obs = player_obs = on_court_obs = pose_hits = pose_total = airborne = 0
    speeds: list[float] = []
    kf_errs: list[float] = []
    ms_vals: list[float] = []
    for seq in seqs:
        m = _sequence_metrics(cfg, seq, runs, kfs_by_clip)
        total_obs += m["total_obs"]
        mapped_obs += m["mapped_obs"]
        player_obs += m["player_obs"]
        on_court_obs += m["on_court_obs"]
        pose_hits += m["pose_hits"]
        pose_total += m["pose_total"]
        airborne += m["airborne_frames"]
        speeds.extend(m["speeds"])
        kf_errs.extend(m["kf_errs"])
        ms_vals.append(m["ms_per_frame"])

    return {
        "sequences": len(seqs),
        "kf_points": len(kf_errs),
        "kf_med_m": statistics.median(kf_errs) if kf_errs else float("nan"),
        "kf_p90_m": float(np.percentile(kf_errs, 90)) if kf_errs else float("nan"),
        "on_court": on_court_obs / player_obs if player_obs else float("nan"),
        "cov": mapped_obs / total_obs if total_obs else float("nan"),
        "speed_p50": float(np.percentile(speeds, 50)) if speeds else float("nan"),
        "speed_p95": float(np.percentile(speeds, 95)) if speeds else float("nan"),
        "jump_pct": sum(v > MAX_SPRINT_MPS for v in speeds) / len(speeds) if speeds else float("nan"),
        "pose_pct": pose_hits / pose_total if pose_total else float("nan"),
        "airborne_frames": airborne,
        "ms_per_frame": statistics.mean(ms_vals) if ms_vals else 0.0,
    }
