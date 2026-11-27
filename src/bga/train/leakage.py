"""Gold-leakage guard: no frame of a gold clip may appear in any training or validation set."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

#: Repository root; the gold list is ``data/labels/gold.json`` under it.
ROOT = Path(__file__).resolve().parents[3]


def gold_clips(gold_json: Path = ROOT / "data/labels/gold.json") -> set[str]:
    """List the label stems of every gold game, held out or not.

    Args:
        gold_json: the gold-set index.

    Returns:
        Label stems (``<date>_<youtube id>_<title>``).

    Examples:
        >>> "20250213_4XMD9FWmQws_PGE_Start_Dziki_Skrot_meczu" in gold_clips()
        True

    """
    return {g["clip"] for g in json.loads(gold_json.read_text())["games"]}


def gold_ids(gold_json: Path = ROOT / "data/labels/gold.json") -> set[str]:
    """List the YouTube ids of every gold game and of every other upload of the same game.

    A game's own id is ``youtube_id``, else the second field of its label stem. ``same_game_ids`` names other
    uploads of that game, such as the highlights of a full-game recording, which show the same broadcast frames.

    Args:
        gold_json: the gold-set index.

    Returns:
        YouTube video ids.

    Examples:
        >>> "4XMD9FWmQws" in gold_ids()
        True
        >>> "Fulq9md9DZc" in gold_ids()  # the highlights of the full-game gold recording 1A79NXz9CFM
        True

    """
    ids = set()
    for g in json.loads(gold_json.read_text())["games"]:
        ids.add(g.get("youtube_id") or g["clip"].split("_")[1])
        ids.update(g.get("same_game_ids", []))
    return ids


def assert_no_gold(clips: Iterable[str]) -> None:
    """Raise if any clip (label stem, or a path or name containing one) belongs to a gold game.

    A clip matches on its file stem being a gold label stem, or on containing a gold YouTube id anywhere.

    Args:
        clips: label stems, video paths or file names.

    Raises:
        ValueError: listing the gold clips found, sorted.

    Examples:
        >>> assert_no_gold(["20251113_kW4HG-5MLNM_SKROT_Dziki_-_King_8._kolejka"])
        >>> assert_no_gold(["data/clips/20250213_4XMD9FWmQws_PGE_Start_Dziki_Skrot_meczu.mp4"])
        Traceback (most recent call last):
        ...
        ValueError: gold clips in a training manifest: ['data/clips/20250213_4XMD9FWmQws_...']

    """
    gold, ids = gold_clips(), gold_ids()
    bad = sorted({c for c in clips if Path(c).stem in gold or any(i in c for i in ids)})
    if bad:
        raise ValueError(f"gold clips in a training manifest: {bad}")


def manifest_clips(manifest: dict) -> set[str]:
    """Collect every clip a training manifest draws on, so gold clips can be refused.

    Frame clips and videos, keyframe clips, and the clips and videos of appearance identities, number-crop tracklets
    and their strays.

    Args:
        manifest: a parsed training manifest (``frames``, ``keyframes``, ``identities`` or ``tracklets`` lists, and
            ``strays`` keyed by shot).

    Returns:
        The non-empty clip names and video paths.

    Examples:
        >>> frames = [{"clip": "a", "video": "data/clips/a.mp4"}, {"clip": "b"}]
        >>> manifest = {"frames": frames, "keyframes": [{"clip": "c"}]}
        >>> sorted(manifest_clips(manifest))
        ['a', 'b', 'c', 'data/clips/a.mp4']
        >>> sorted(manifest_clips({"tracklets": [{"clip": "d", "video": "v.mp4"}], "strays": {"s": {"clip": "e"}}}))
        ['d', 'e', 'v.mp4']

    """
    clips = {f["clip"] for f in manifest.get("frames", [])}
    clips |= {f.get("video", "") for f in manifest.get("frames", [])}
    clips |= {k["clip"] for k in manifest.get("keyframes", [])}
    for rec in [*manifest.get("identities", []), *manifest.get("tracklets", []), *manifest.get("strays", {}).values()]:
        clips |= {rec.get("clip", ""), rec.get("video", "")}
    return {c for c in clips if c}
