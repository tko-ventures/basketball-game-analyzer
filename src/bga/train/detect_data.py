"""Detection training data: Roboflow COCO exports in the 10-class schema, optionally cut into square tiles.

The layout is what ``rfdetr`` reads (``dataset_file="roboflow"``): ``train/``, ``valid/`` and ``test/``, each with its
images and ``_annotations.coco.json``. Categories are id 0, the supercategory placeholder ``basketball``, then the
10 classes of ``bga.types.DETECTION_CLASSES`` as ids 1 to 10, matched by name in every source. With ``recipe.tiles``
set, each image becomes that many overlapping square crops (``bga.detect.tiles``), keeping boxes at least half inside.

Manifest sources (PLK auto-labels, ``data/labels/detect/train_manifest_*.json``) are checked against their sha256 and
the gold clips, and their frames are decoded from the local videos and resized to the source's ``image_width`` (no
resize when the video already has that width): ``boxes`` as ``[x1, y1, x2, y2, class]`` per frame, scaled with the
frame, and the split each frame names.

"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from bga.detect.tiles import crop_box, square_tiles
from bga.train.config import ROOT, ManifestSource, RoboflowSource, TrainConfig, file_sha256
from bga.train.court_data import SPLITS, fetch_roboflow
from bga.train.leakage import assert_no_gold, manifest_clips
from bga.types import DETECTION_CLASSES

#: COCO categories written to every split: the placeholder, then the schema.
CATEGORIES = [
    {"id": 0, "name": "basketball", "supercategory": "none"},
    *({"id": i + 1, "name": n, "supercategory": "basketball"} for i, n in enumerate(DETECTION_CLASSES)),
]


def category_map(categories: list[dict]) -> dict[int, int]:
    """Map a source's category ids to the schema's ids by class name; the placeholder and unknown names are dropped.

    Args:
        categories: The source's COCO categories.

    Returns:
        Source id to schema id.

    Examples:
        >>> category_map([{"id": 0, "name": "basketball"}, {"id": 7, "name": "Ball"}, {"id": 8, "name": "net"}])
        {7: 1}

    """
    ids = {c["name"]: c["id"] for c in CATEGORIES[1:]}
    return {c["id"]: ids[c["name"].lower()] for c in categories if c["name"].lower() in ids}


def _write_image(
    src: Path | np.ndarray, dst_dir: Path, name: str, size: tuple[int, int], anns: list[dict], tiles: int | None
) -> list[tuple]:
    """Write one image (a file to copy, or a decoded frame) or its tiles; ``(file name, width, height, boxes)`` each.

    ``anns`` carry COCO ``bbox`` (x, y, w, h) and the schema's ``category_id``.

    """
    w, h = size
    if not tiles or tiles <= 1:
        if isinstance(src, Path):
            shutil.copyfile(src, dst_dir / name)
        else:
            cv2.imwrite(str(dst_dir / name), src, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return [(name, w, h, [(a["bbox"], a["category_id"]) for a in anns])]
    img = cv2.imread(str(src)) if isinstance(src, Path) else src
    out = []
    for k, tile in enumerate(square_tiles(w, h, tiles)):
        boxes = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            c = crop_box((x, y, x + bw, y + bh), tile)
            if c is not None:
                boxes.append(([c[0], c[1], c[2] - c[0], c[3] - c[1]], a["category_id"]))
        fn = f"{Path(name).stem}_t{k}.jpg"
        cv2.imwrite(str(dst_dir / fn), img[tile[1] : tile[3], tile[0] : tile[2]], [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append((fn, tile[2] - tile[0], tile[3] - tile[1], boxes))
    return out


class _Coco:
    """One split's COCO detection annotations, with box counts per class."""

    def __init__(self) -> None:
        self.coco: dict = {"images": [], "annotations": [], "categories": CATEGORIES}
        self.boxes: Counter = Counter()

    def add(self, written: list[tuple], source: str) -> None:
        for fn, w, h, bs in written:
            iid = len(self.coco["images"])
            self.coco["images"].append({"id": iid, "file_name": fn, "width": w, "height": h, "source": source})
            for (x, y, bw, bh), cid in bs:
                self.coco["annotations"].append({
                    "id": len(self.coco["annotations"]),
                    "image_id": iid,
                    "category_id": cid,
                    "bbox": [round(v, 2) for v in (x, y, bw, bh)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                })
                self.boxes[DETECTION_CLASSES[cid - 1]] += 1


def _add_roboflow(src: RoboflowSource, split: str, out: Path, coco: _Coco, tiles: int | None) -> int:
    """Add a Roboflow export's images of one split; return how many images were written."""
    d = fetch_roboflow(src)
    sc = json.loads((d / split / "_annotations.coco.json").read_text())
    cmap = category_map(sc["categories"])
    by_img: dict[int, list[dict]] = {}
    for a in sc["annotations"]:
        if a["category_id"] in cmap:
            by_img.setdefault(a["image_id"], []).append({**a, "category_id": cmap[a["category_id"]]})
    n = 0
    for im in sc["images"]:
        written = _write_image(
            d / split / im["file_name"],
            out,
            f"rf_{im['file_name']}",
            (im["width"], im["height"]),
            by_img.get(im["id"], []),
            tiles,
        )
        coco.add(written, src.ref.replace("/", "_"))
        n += len(written)
    return n


def _add_manifest(src: ManifestSource, split: str, out: Path, coco: _Coco, tiles: int | None) -> int:
    """Decode a detection manifest's frames of one split from the local videos and add them at ``image_width``."""
    path = ROOT / src.path
    if file_sha256(path) != src.sha256:
        raise ValueError(f"{src.path}: sha256 differs from the config; rebuild the config or the manifest")
    man = json.loads(path.read_text())
    assert_no_gold(manifest_clips(man))
    ids = {c["name"]: c["id"] for c in CATEGORIES[1:]}
    caps: dict[str, cv2.VideoCapture] = {}
    n = 0
    for f in (f for f in man["frames"] if f["split"] == split):
        cap = caps.setdefault(f["video"], cv2.VideoCapture(str(ROOT / f["video"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        ok, img = cap.read()
        if not ok:
            raise IndexError(f"{f['video']} frame {f['frame']}")
        k = src.image_width / img.shape[1]
        if k != 1:
            img = cv2.resize(img, (src.image_width, round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        anns = [
            {"bbox": [x1 * k, y1 * k, (x2 - x1) * k, (y2 - y1) * k], "category_id": ids[c]}
            for x1, y1, x2, y2, c in f["boxes"]
        ]
        name = f"plk_{f['clip'].split('_')[1]}_{f['frame']:06d}.jpg"
        coco.add(_write_image(img, out, name, (img.shape[1], img.shape[0]), anns, tiles), src.path)
        n += 1
    for cap in caps.values():
        cap.release()
    return n


def build(cfg: TrainConfig, out: Path) -> dict:
    """Build the COCO detection dataset of a training config under ``out``, replacing what is there.

    Args:
        cfg: The training config; Roboflow exports keep their own splits, manifest frames go to the split they name.
        out: Dataset directory.

    Returns:
        ``{"counts": {split: {source: images}}, "boxes": {split: {class: n}}, "tiles": n or None, "sha256": hex}``.

    Raises:
        ValueError: When a manifest's sha256 differs from the config, or it names a gold clip.
        IndexError: When a manifest frame cannot be decoded.

    Examples:
        >>> from bga.train.config import ROOT, load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/detect-rfdetr_torch-nba-704.yaml")
        >>> build(cfg, ROOT / "runs/train/example/dataset")["boxes"]["train"]["ball"]  # doctest: +SKIP
        432

    """
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    boxes: dict[str, dict[str, int]] = {}
    digest = hashlib.sha256()
    for s in SPLITS:
        coco = _Coco()
        (out / s).mkdir(parents=True)
        for src in cfg.sources:
            tag = src.ref.replace("/", "_") if isinstance(src, RoboflowSource) else src.path
            add = _add_roboflow if isinstance(src, RoboflowSource) else _add_manifest
            counts[s][tag] = add(src, s, out / s, coco, cfg.recipe.tiles)
        p = out / s / "_annotations.coco.json"
        p.write_text(json.dumps(coco.coco, separators=(",", ":")))
        digest.update(p.read_bytes())
        boxes[s] = dict(coco.boxes)
    return {
        "counts": counts,
        "boxes": boxes,
        "tiles": cfg.recipe.tiles,
        "sha256": digest.hexdigest(),
    }
