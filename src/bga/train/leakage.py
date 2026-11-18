"""Gold-leakage guard: no frame of a gold clip may appear in any training or validation set."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def gold_clips(gold_json: Path = ROOT / "data/labels/gold.json") -> set[str]:
    """Label stems of every gold game, held out or not."""
    return {g["clip"] for g in json.loads(gold_json.read_text())["games"]}


def gold_ids(gold_json: Path = ROOT / "data/labels/gold.json") -> set[str]:
    return {
        g.get("youtube_id") or g["clip"].split("_")[1] for g in json.loads(gold_json.read_text())["games"]
    }


def assert_no_gold(clips: Iterable[str]) -> None:
    """Raise if any clip (label stem, or a path or name containing one) belongs to a gold game."""
    gold, ids = gold_clips(), gold_ids()
    bad = sorted({c for c in clips if Path(c).stem in gold or any(i in c for i in ids)})
    if bad:
        raise ValueError(f"gold clips in a training manifest: {bad}")


def manifest_clips(manifest: dict) -> set[str]:
    clips = {f["clip"] for f in manifest.get("frames", [])}
    clips |= {f.get("video", "") for f in manifest.get("frames", [])}
    clips |= {k["clip"] for k in manifest.get("keyframes", [])}
    return {c for c in clips if c}
