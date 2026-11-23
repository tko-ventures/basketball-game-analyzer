"""Number-reader training data: class folders of ``number`` crops cut from the clips by a number-crop manifest.

Sources are manifests from ``scripts/number_crops.py`` (``data/labels/numbers/train_*.json``), checked against their
sha256 and the gold games. Every crop of a tracklet read as a number goes to that number's folder. Crops of
tracklets read as ``not_number`` and the manifest's ``strays`` (number boxes inside no person) go to
``unreadable``, and so do the crops of ``illegible`` tracklets when ``recipe.illegible_as_unreadable`` is set.
``mixed`` and unread tracklets are left out. Crops are cut with ``recipe.pad`` pixels around the box, as the numbers
backends cut them (``bga.numbers.base.number_crop``). A crop goes to ``valid`` when its clip is in
``recipe.val_clips``, else to ``train``.

Layout: ``train/<class>/<id>.jpg`` and ``valid/<class>/<id>.jpg``, one folder per class of
``bga.numbers.base.NUMBER_CLASSES`` that has crops.

"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from bga.numbers.base import NUMBER_CLASSES, UNREADABLE, number_crop
from bga.stage import FrameReader
from bga.train.appearance_data import split_of
from bga.train.config import ROOT, ManifestSource, TrainConfig, file_sha256
from bga.train.leakage import assert_no_gold

#: Splits written, with the class folders under each.
SPLITS = ("train", "valid")
#: Padding around a number box when ``recipe.pad`` is unset: the numbers backends' default.
DEFAULT_PAD = 10


def class_of(label: str | None, illegible_as_unreadable: bool) -> str | None:
    """The class folder a tracklet's crops go to, or ``None`` when they are left out.

    Args:
        label: The tracklet's label: digits, ``illegible``, ``mixed``, ``not_number`` or ``None`` (not read).
        illegible_as_unreadable: Whether ``illegible`` tracklets train the ``unreadable`` class.

    Returns:
        A class of ``NUMBER_CLASSES``, or ``None``.

    Examples:
        >>> [class_of(v, False) for v in ("23", "not_number", "illegible", "mixed", None)]
        ['23', 'unreadable', None, None, None]
        >>> class_of("illegible", True)
        'unreadable'

    """
    if label == "not_number" or (label == "illegible" and illegible_as_unreadable):
        return UNREADABLE
    return label if label in NUMBER_CLASSES else None


def labelled_crops(src: ManifestSource, illegible_as_unreadable: bool) -> list[tuple[str, str, str, list]]:
    """The crops of a number-crop manifest with their class, after the sha256 and gold checks.

    Args:
        src: The manifest source.
        illegible_as_unreadable: Whether ``illegible`` tracklets train the ``unreadable`` class.

    Returns:
        ``(clip, video, class, [frame, x1, y1, x2, y2, ...])`` per crop: tracklet crops, then strays.

    Raises:
        ValueError: When the sha256 differs from the config, or a crop comes from a gold game.

    Examples:
        >>> src = ManifestSource(path="data/labels/numbers/train_r1.json", sha256="0")
        >>> labelled_crops(src, False)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: data/labels/numbers/train_r1.json: sha256 differs from the config; ...

    """
    path = ROOT / src.path
    if file_sha256(path) != src.sha256:
        raise ValueError(f"{src.path}: sha256 differs from the config; rebuild the config or the manifest")
    m = json.loads(path.read_text())
    assert_no_gold({t["clip"] for t in m["tracklets"]} | {s["clip"] for s in m.get("strays", {}).values()})
    out = []
    for t in m["tracklets"]:
        cls = class_of(t.get("label"), illegible_as_unreadable)
        if cls is not None:
            out += [(t["clip"], t["video"], cls, c) for c in t["crops"]]
    for s in m.get("strays", {}).values():
        out += [(s["clip"], s["video"], UNREADABLE, b) for b in s["boxes"]]
    return out


def build(cfg: TrainConfig, out: Path) -> dict:
    """Cut every labelled crop into class folders under ``out``, replacing what is there.

    Args:
        cfg: The training config: number-crop manifest sources and ``recipe.val_clips``, ``pad`` and
            ``illegible_as_unreadable``.
        out: Dataset directory.

    Returns:
        ``{"counts": {split: {manifest: crops}}, "classes": {split: {class: crops}}, "sha256": hex}``; the sha256
        covers the classes and crop boxes written, in order.

    Raises:
        ValueError: When a manifest's sha256 differs from the config, or names a gold game.

    Examples:
        >>> from bga.train.config import load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/numbers-crop_cls-resnet18-r1.yaml")  # doctest: +SKIP
        >>> build(cfg, ROOT / "runs/train/example/dataset")["classes"]["train"]["unreadable"]  # doctest: +SKIP
        700

    """
    if out.exists():
        shutil.rmtree(out)
    r = cfg.recipe
    val, pad = r.val_clips or [], DEFAULT_PAD if r.pad is None else r.pad
    jobs: dict[str, list[tuple[int, list[float], Path]]] = defaultdict(list)  # video -> (frame, box, file)
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    classes: dict[str, Counter] = {s: Counter() for s in SPLITS}
    digest = hashlib.sha256()
    for src in cfg.sources:
        for clip, video, cls, (f, *rest) in labelled_crops(src, bool(r.illegible_as_unreadable)):
            box = rest[:4]
            split = split_of(clip, val)
            folder = out / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            n = sum(classes[s][cls] for s in SPLITS)
            jobs[video].append((f, box, folder / f"{n:06d}.jpg"))
            counts[split][src.path] = counts[split].get(src.path, 0) + 1
            classes[split][cls] += 1
            digest.update(json.dumps([cls, clip, f, box]).encode())
    for video, crops in sorted(jobs.items()):
        read = FrameReader(str(ROOT / video))
        try:
            for f, box, path in sorted(crops, key=lambda c: c[0]):
                cv2.imwrite(str(path), number_crop(read(f), box, pad), [cv2.IMWRITE_JPEG_QUALITY, 95])
        finally:
            read.close()
    return {
        "counts": counts,
        "classes": {s: dict(sorted(c.items())) for s, c in classes.items()},
        "sha256": digest.hexdigest(),
    }
