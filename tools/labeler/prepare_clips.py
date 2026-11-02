#!/usr/bin/env python3
"""Pre-process downloaded videos for the labeller (run from the repo root).

For every video in data/raw (or given paths) without a label file yet:
  * detect camera cuts with the layer-0 ``hist_diff`` detector,
  * fill meta.season from the upload date (Sep to Jun season) and meta.home / meta.away from the title,
  * write data/labels/<stem>.json so the labeller opens with shots ready to tag.

    uv run tools/labeler/prepare_clips.py                 # all of data/raw
    uv run tools/labeler/prepare_clips.py --force         # redo shots (keeps boxes and camera tags)
    uv run tools/labeler/prepare_clips.py --labels-dir runs/_shots --jobs 6
    uv run tools/labeler/prepare_clips.py --meta-only     # refresh team/roster meta, keep shots
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bga.ingest.hist_diff import DEFAULT_MIN_LEN_S, DEFAULT_THRESHOLD, detect_shots  # noqa: E402
from tools.labeler import server  # noqa: E402


def season_for(upload_date: str | None) -> str:
    if not upload_date or len(upload_date) < 6:
        return ""
    y, m = int(upload_date[:4]), int(upload_date[4:6])
    start = y if m >= 8 else y - 1
    return f"{start}/{str(start + 1)[2:]}"


def prepare(video: Path, labels_dir: Path, force: bool, meta_only: bool = False) -> str:
    rel = str(video.resolve().relative_to(server.REPO))
    out = labels_dir / f"{video.stem}.json"
    if out.exists() and not (force or meta_only):
        return f"skip {video.name}"
    data = server.load_labels(labels_dir, rel, server.Clip(video.resolve()))
    info = video.with_suffix(".info.json")
    if info.exists():
        meta = json.loads(info.read_text(encoding="utf-8"))
        m = data["meta"]
        m["season"] = m.get("season") or season_for(meta.get("upload_date"))
        m["source"] = meta.get("webpage_url", "")
        m["title"] = meta.get("title", "")
        home, away = server.teams_from_title(m["title"])
        for side, team in (("home", home), ("away", away)):
            if not m[side].get("team"):
                m[side] = server.side_meta(team, m[side].get("kit", ""))
            elif not m[side].get("roster"):
                m[side]["roster"] = server.roster_key_for(m[side]["team"])
    if meta_only:
        server.save_labels(labels_dir, data)
        return (
            f"{video.name}: {data['meta']['home']['roster'] or '?'} - {data['meta']['away']['roster'] or '?'}"
        )
    old = data.get("shots", [])
    data["shots"] = [
        {
            **s,
            "camera": next(
                (o["camera"] for o in old if o["start"] <= s["start"] <= o["end"] and len(old) > 1), "unset"
            ),
        }
        for s in detect_shots(video, DEFAULT_THRESHOLD, DEFAULT_MIN_LEN_S)
    ]
    server.save_labels(labels_dir, data)
    m = data["meta"]
    return (
        f"{video.name}: {len(data['shots'])} shots, season {m['season']}, "
        f"{m['home']['team'] or '?'} - {m['away']['team'] or '?'}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="clips processed in parallel")
    ap.add_argument("--meta-only", action="store_true", help="refresh season/team/roster meta only")
    a = ap.parse_args(argv)
    labels_dir = server.REPO / a.labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    vids = [Path(v) for v in a.videos] or sorted((server.REPO / "data/raw").glob("*.mp4"))
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        n = len(vids)
        for line in pool.map(prepare, vids, [labels_dir] * n, [a.force] * n, [a.meta_only] * n):
            print(line, flush=True)


if __name__ == "__main__":
    main()
