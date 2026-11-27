"""``shots/video``: Layer 0 for raw game video, with no hand-labelled shots (M10, #32).

A configured cut detector (:mod:`bga.ingest.cuts`) splits each video into shots. A configured camera classifier
(:mod:`bga.ingest.camera`) then scores each shot from a few sampled frames. Every shot scored ``central`` and lasting
at least ``min_s`` becomes a sequence named ``<youtube id>_v<shot index>``. The ``v`` keeps these apart from the
``_s<i>`` sequences of the label files, whose cuts may differ.

From a game's label file only ``clip`` (the video), ``fps`` and ``meta`` (teams, rosters, kits) are read, never its
``shots``. Cuts and camera scores are cached per video under ``<cache>/<clip>/ingest/``. Each cache key covers the
backend's config entry and its source hash (:func:`bga.codehash.source_hash`); a camera key also covers the cuts it
scored.

"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from bga import registry
from bga.codehash import source_hash
from bga.ingest import camera as _camera  # noqa: F401  # registers the camera backends and puts them in the code key
from bga.ingest import cuts as _cuts  # noqa: F401  # registers the cut detectors and puts them in the code key
from bga.ingest.camera import read_shot_frames
from bga.ingest.clip_shots import shot_frames
from bga.paths import write_atomic
from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence

log = logging.getLogger("bga")

T = TypeVar("T")


def entry_key(kind: str, entry: dict, extra: str = "") -> str:
    """Cache key of a cuts or camera entry: its backend, params, the backend's source hash and ``extra``.

    Args:
        kind: ``"cuts"`` or ``"camera"``.
        entry: ``{"backend": name, "params": {...}}``.
        extra: Anything else the result depends on, such as the cuts key a camera score was made on.

    Returns:
        The first 8 hex digits of a SHA-256.

    Examples:
        >>> len(entry_key("cuts", {"backend": "hist_diff", "params": {}}))
        8

    """
    module = registry.get(kind, entry["backend"]).__module__
    blob = json.dumps({"entry": entry, "source": source_hash(module), "extra": extra}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def make_component(kind: str, entry: dict) -> BaseBackend:
    """Instantiate the cuts or camera backend an entry names.

    Args:
        kind: ``"cuts"`` or ``"camera"``.
        entry: ``{"backend": name, "params": {...}}``.

    Returns:
        The backend, not yet loaded.

    Examples:
        >>> make_component("cuts", {"backend": "hist_diff", "params": {"min_len_s": 1.0}}).p["min_len_s"]
        1.0

    """
    return registry.get(kind, entry["backend"])(entry.get("params") or {})


@register("shots", "video")
class VideoShots(BaseBackend):
    """Shots backend over raw game videos: cut, tag the camera, keep the central shots.

    Params (``defaults``): ``clips`` (label stems giving video, fps and meta), ``cuts`` and ``camera`` (backend
    entries), ``samples`` (frames per shot for the camera), ``min_s`` (shortest shot kept), ``stride`` (frame grid),
    ``labels`` (label folder) and ``cache`` (root of the cached cuts and scores).

    Examples:
        >>> VideoShots({}).p["cuts"]
        {'backend': 'hist_diff', 'params': {}}

    """

    defaults: ClassVar[dict[str, Any]] = {
        "clips": [],
        "cuts": {"backend": "hist_diff", "params": {}},
        "camera": {"backend": "siglip_zeroshot", "params": {}},
        "samples": 3,
        "min_s": 3.0,
        "stride": 5,
        "labels": "data/labels",
        "cache": "runs",
    }

    def _cached(self, path: Path, make: Callable[[], T]) -> T:
        """Read ``path`` if present, else compute ``make()`` and write it."""
        if path.exists():
            return json.loads(path.read_text())
        value = make()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, json.dumps(value))
        return value

    def game_cuts(self, clip: str, video: str) -> list[dict]:
        """The cut detector's shots of one video, from the cache or computed once.

        Args:
            clip: Label stem, the ``runs/<clip>`` directory.
            video: Path of the video.

        Returns:
            The shots ``{"start", "end"}``.

        """
        key = entry_key("cuts", self.p["cuts"])

        def cut() -> list[dict]:
            log.info("cuts/%s on %s", self.p["cuts"]["backend"], clip)
            return make_component("cuts", self.p["cuts"]).shots(video)

        return self._cached(
            Path(self.p["cache"]) / clip / "ingest" / f"cuts-{self.p['cuts']['backend']}@{key}.json", cut
        )

    def game_shots(self, clip: str, video: str) -> tuple[list[dict], list[float]]:
        """Cuts and camera scores of one video, from the cache or computed once.

        Args:
            clip: Label stem, the ``runs/<clip>`` directory.
            video: Path of the video.

        Returns:
            The shots ``{"start", "end"}`` and one camera score per shot.

        """
        shots = self.game_cuts(clip, video)
        cuts_key = entry_key("cuts", self.p["cuts"])
        cam_key = entry_key("camera", self.p["camera"], f"{cuts_key}/{self.p['samples']}")
        root = Path(self.p["cache"]) / clip / "ingest"

        def score() -> list[float]:
            log.info("camera/%s on %d shots of %s", self.p["camera"]["backend"], len(shots), clip)
            cam = make_component("camera", self.p["camera"])
            cam.load()
            try:
                return cam.score_shots(read_shot_frames(video, shots, self.p["samples"]))
            finally:
                cam.unload()

        scores = self._cached(root / f"camera-{self.p['camera']['backend']}@{cam_key}.json", score)
        return shots, scores

    def sequences(self) -> list[Sequence]:
        """List the central shots lasting at least ``min_s`` of every configured clip.

        Returns:
            One sequence per kept shot, framed on the multiples of ``stride``.

        """
        out = []
        for clip in self.p["clips"]:
            labels = json.loads((Path(self.p["labels"]) / f"{clip}.json").read_text())
            fps, ytid = float(labels["fps"]), clip.split("_")[1]
            shots, scores = self.game_shots(clip, labels["clip"])
            for i, (s, score) in enumerate(zip(shots, scores, strict=True)):
                if score < 0.5 or (s["end"] - s["start"] + 1) / fps < self.p["min_s"]:
                    continue
                frames = shot_frames(s["start"], s["end"], self.p["stride"])
                out.append(Sequence(f"{ytid}_v{i}", clip, labels["clip"], i, frames, fps, labels.get("meta", {})))
        return out

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Return the sequence's fields as the layer's output.

        Args:
            seq: The sequence.
            upstream: Unused.
            read: Unused.

        Returns:
            ``dataclasses.asdict(seq)``.

        """
        return asdict(seq)
