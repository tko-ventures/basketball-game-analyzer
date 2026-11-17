"""Radar video per sequence (plan section 9, M2): broadcast frame above, FIBA court with players below.

Team colour per track is the majority of its boxes' team labels; referees are drawn grey. Written to
``runs/<clip>/radar/<config>@<hash8>/<sequence>.mp4`` at the sequence's frame rate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from bga.config import RunConfig
from bga.pipeline import load_output
from bga.stage import FrameReader, Sequence

TEAM_BGR = {"home": (235, 235, 235), "away": (40, 40, 200), "ref": (120, 120, 120), "": (200, 0, 200)}


def track_teams(tracks: dict, teams: dict) -> dict[int, str]:
    votes = defaultdict(Counter)
    for f, rows in tracks.items():
        for r in rows:
            if r[7] >= 0 and f in teams:
                votes[r[0]][teams[f][r[7]]] += 1
    return {t: v.most_common(1)[0][0] for t, v in votes.items()}


def render(cfg: RunConfig, seq: Sequence, runs: Path = Path("runs"), width: int = 960) -> Path:
    from sports.basketball import CourtConfiguration, League, draw_court

    court_cfg = CourtConfiguration(league=League.FIBA)
    scale = width / (court_cfg.court_length + 200)  # px per cm, 1 m margin each side
    pad = int(100 * scale)
    court = draw_court(court_cfg, scale=scale, padding=pad, line_thickness=2)
    tracks = load_output(cfg, runs, seq, "track")["frames"]
    teams = load_output(cfg, runs, seq, "team")["frames"] if "team" in cfg.layers else {}
    team_of = track_teams(tracks, teams)
    pos = defaultdict(dict)
    for tid, pts in load_output(cfg, runs, seq, "positions")["tracks"].items():
        for f, x, y in pts:
            pos[f][int(tid)] = (x, y)
    out = runs / seq.clip / "radar" / f"{cfg.name.split('[')[0]}@{cfg.config_hash()}" / f"{seq.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    stride = seq.frames[1] - seq.frames[0] if len(seq.frames) > 1 else 1
    read = FrameReader(seq.video)
    h_top = round(width * 9 / 16)
    writer = None
    for f in seq.frames:
        top = cv2.resize(read(f), (width, h_top))
        rad = court.copy()
        for tid, (x, y) in pos.get(f, {}).items():
            px = (int(x * 100 * scale) + pad, int(y * 100 * scale) + pad)
            team = team_of.get(tid, "")
            cv2.circle(rad, px, 9, TEAM_BGR.get(team, TEAM_BGR[""]), -1)
            cv2.circle(rad, px, 9, (0, 0, 0), 1)
            cv2.putText(rad, str(tid), (px[0] + 10, px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        frame = np.vstack([top, cv2.resize(rad, (width, round(rad.shape[0] * width / rad.shape[1])))])
        cv2.putText(frame, f"{seq.name} f{f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*"avc1"), seq.fps / stride, frame.shape[1::-1]
            )
        writer.write(frame)
    read.close()
    if writer is not None:
        writer.release()
    return out
