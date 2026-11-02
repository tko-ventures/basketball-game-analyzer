#!/usr/bin/env python3
"""Import a prefill tracks.json into a clip's labels, the way the labeller's Prefill button does.

    uv run scripts/import_prefill.py runs/<clip>/prefill/<model>/tracks.json [--force]

Boxes land on their sampled frames as unreviewed prefill (identity = 1000 + track id, team set when
the track is a referee). Frames that already have boxes are left alone unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.labeler import server  # noqa: E402

ID_OFFSET = 1000


def import_tracks(labels: dict, tracks: dict, force: bool = False) -> int:
    n = 0
    for fr in tracks["frames"]:
        key = str(fr["frame_idx"])
        if labels["frames"].get(key, {}).get("boxes") and not force:
            continue
        boxes = []
        for t in fr["tracks"]:
            tid = ID_OFFSET + int(t["track_id"])
            labels["identities"].setdefault(
                str(tid), {"team": t.get("team", ""), "number": "", "name": "", "source": "prefill"}
            )
            x1, y1, x2, y2 = t["xyxy"]
            boxes.append(
                {"tid": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "occluded": False, "num_visible": False}
            )
        labels["frames"][key] = {"done": False, "interp": False, "prefill": True, "boxes": boxes}
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    tracks = json.loads(a.tracks.read_text())
    stem = Path(tracks["clip_path"]).stem
    path = a.labels_dir / f"{stem}.json"
    labels = server.migrate_labels(json.loads(path.read_text()))
    n = import_tracks(labels, tracks, a.force)
    server.save_labels(a.labels_dir, labels)
    print(f"{stem}: {n} frames prefilled, {len(labels['identities'])} identities")


if __name__ == "__main__":
    main()
