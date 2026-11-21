"""Appearance training data: identity folders of player crops cut from the clips by a crop manifest.

The layout is what ``reid.data.identity_crops.load_identity_folders`` reads: ``train/<identity>/<frame>.jpg`` and
``valid/<identity>/<frame>.jpg``. Sources are crop manifests (``data/labels/appearance/crops_*.json``, written by
``scripts/reid_crops.py``), checked against their sha256 and the gold games; identities marked ``dropped`` (mixed,
by eye) are left out. An identity goes to ``valid`` when its clip is in ``recipe.val_clips``, else to ``train``, so
validation never shares a game with training.

"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from bga.stage import FrameReader
from bga.train.config import ROOT, ManifestSource, TrainConfig, file_sha256
from bga.train.leakage import assert_no_gold

#: Splits written, with the identity folders under each.
SPLITS = ("train", "valid")


def split_of(clip: str, val_clips: list[str]) -> str:
    """Split an identity goes to: ``valid`` for the validation games, else ``train``.

    Args:
        clip: The identity's label stem.
        val_clips: Label stems of the validation games.

    Returns:
        ``train`` or ``valid``.

    Examples:
        >>> split_of("20251113_kW4HG-5MLNM_x", ["20251113_kW4HG-5MLNM_x"]), split_of("a", [])
        ('valid', 'train')

    """
    return "valid" if clip in val_clips else "train"


def kept_identities(src: ManifestSource) -> list[dict]:
    """Identities of a crop manifest that are not dropped, after the sha256 and gold checks.

    Args:
        src: The manifest source.

    Returns:
        The manifest's identity records without a ``dropped`` mark.

    Raises:
        ValueError: When the sha256 differs from the config, or an identity comes from a gold game.

    Examples:
        >>> kept_identities(ManifestSource(path="data/labels/appearance/crops_r1.json", sha256="0"))  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: data/labels/appearance/crops_r1.json: sha256 differs from the config; ...

    """
    path = ROOT / src.path
    if file_sha256(path) != src.sha256:
        raise ValueError(f"{src.path}: sha256 differs from the config; rebuild the config or the manifest")
    ids = json.loads(path.read_text())["identities"]
    assert_no_gold({i["clip"] for i in ids})
    return [i for i in ids if not i.get("dropped")]


def build(cfg: TrainConfig, out: Path) -> dict:
    """Cut every kept identity's crops into identity folders under ``out``, replacing what is there.

    Args:
        cfg: The training config: crop manifest sources and ``recipe.val_clips``.
        out: Dataset directory.

    Returns:
        ``{"counts": {split: {manifest: crops}}, "identities": {split: n}, "sha256": hex}``; the sha256 covers the
        identity names and crop boxes written, in order.

    Raises:
        ValueError: When a manifest's sha256 differs from the config, or names a gold game.

    Examples:
        >>> from bga.train.config import load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/appearance-reid-plk-r1.yaml")  # doctest: +SKIP
        >>> build(cfg, ROOT / "runs/train/example/dataset")["identities"]  # doctest: +SKIP
        {'train': 1012, 'valid': 212}

    """
    if out.exists():
        shutil.rmtree(out)
    val = cfg.recipe.val_clips or []
    jobs: dict[str, list[tuple[int, list[float], Path]]] = defaultdict(list)  # video -> (frame, box, file)
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    identities = dict.fromkeys(SPLITS, 0)
    digest = hashlib.sha256()
    for src in cfg.sources:
        for ident in kept_identities(src):
            split = split_of(ident["clip"], val)
            folder = out / split / ident["id"]
            folder.mkdir(parents=True)
            identities[split] += 1
            for f, *box in ident["crops"]:
                jobs[ident["video"]].append((f, box, folder / f"{f:06d}.jpg"))
                counts[split][src.path] = counts[split].get(src.path, 0) + 1
                digest.update(json.dumps([ident["id"], f, box]).encode())
    for video, crops in sorted(jobs.items()):
        read = FrameReader(str(ROOT / video))
        try:
            for f, (x1, y1, x2, y2), path in sorted(crops, key=lambda c: c[0]):
                img = read(f)
                cv2.imwrite(str(path), img[int(y1) : int(y2), int(x1) : int(x2)], [cv2.IMWRITE_JPEG_QUALITY, 95])
        finally:
            read.close()
    return {"counts": counts, "identities": identities, "sha256": digest.hexdigest()}
