"""Detection training data: Roboflow COCO exports in the 10-class schema, optionally cut into square tiles.

The layout is what ``rfdetr`` reads (``dataset_file="roboflow"``): ``train/``, ``valid/`` and ``test/``, each with its
images and ``_annotations.coco.json``. Categories are id 0, the supercategory placeholder ``basketball``, then the
10 classes of ``bga.types.DETECTION_CLASSES`` as ids 1 to 10, matched by name in every source. With ``recipe.tiles``
set, each image becomes that many overlapping square crops (``bga.detect.tiles``), keeping boxes at least half inside.

Manifest sources (PLK auto-labels) are not supported yet; the builder refuses them rather than skip them silently.

"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

from bga.detect.tiles import crop_box, square_tiles
from bga.train.config import ManifestSource, RoboflowSource, TrainConfig
from bga.train.court_data import SPLITS, fetch_roboflow
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


def _write_image(src: Path, dst_dir: Path, im: dict, anns: list[dict], tiles: int | None) -> list[tuple]:
    """Copy one image (or write its tiles) and return ``(file name, width, height, boxes)`` per written image."""
    name, w, h = f"rf_{im['file_name']}", im["width"], im["height"]
    if not tiles or tiles <= 1:
        shutil.copyfile(src, dst_dir / name)
        return [(name, w, h, [(a["bbox"], a["category_id"]) for a in anns])]
    img = cv2.imread(str(src))
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


def build(cfg: TrainConfig, out: Path) -> dict:
    """Build the COCO detection dataset of a training config under ``out``, replacing what is there.

    Args:
        cfg: The training config; its Roboflow sources are fetched and merged, keeping their own splits.
        out: Dataset directory.

    Returns:
        ``{"counts": {split: {source: images}}, "boxes": {split: {class: n}}, "tiles": n or None, "sha256": hex}``.

    Raises:
        NotImplementedError: For a manifest source.

    Examples:
        >>> from bga.train.config import ROOT, load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/detect-rfdetr_torch-nba-704.yaml")
        >>> build(cfg, ROOT / "runs/train/example/dataset")["boxes"]["train"]["ball"]  # doctest: +SKIP
        432

    """
    if any(isinstance(s, ManifestSource) for s in cfg.sources):
        raise NotImplementedError("detection manifests (PLK auto-labels) are not supported yet")
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    boxes: dict[str, Counter] = {s: Counter() for s in SPLITS}
    digest = hashlib.sha256()
    for s in SPLITS:
        coco = {"images": [], "annotations": [], "categories": CATEGORIES}
        (out / s).mkdir(parents=True)
        for src in (x for x in cfg.sources if isinstance(x, RoboflowSource)):
            d = fetch_roboflow(src)
            tag = src.ref.replace("/", "_")
            sc = json.loads((d / s / "_annotations.coco.json").read_text())
            cmap = category_map(sc["categories"])
            by_img: dict[int, list[dict]] = {}
            for a in sc["annotations"]:
                if a["category_id"] in cmap:
                    by_img.setdefault(a["image_id"], []).append({**a, "category_id": cmap[a["category_id"]]})
            for im in sc["images"]:
                written = _write_image(d / s / im["file_name"], out / s, im, by_img.get(im["id"], []), cfg.recipe.tiles)
                for fn, w, h, bs in written:
                    iid = len(coco["images"])
                    coco["images"].append({"id": iid, "file_name": fn, "width": w, "height": h, "source": tag})
                    for (x, y, bw, bh), cid in bs:
                        coco["annotations"].append({
                            "id": len(coco["annotations"]), "image_id": iid, "category_id": cid,
                            "bbox": [round(v, 2) for v in (x, y, bw, bh)], "area": round(bw * bh, 2), "iscrowd": 0,
                        })
                        boxes[s][DETECTION_CLASSES[cid - 1]] += 1
                counts[s][tag] = counts[s].get(tag, 0) + len(written)
        p = out / s / "_annotations.coco.json"
        p.write_text(json.dumps(coco, separators=(",", ":")))
        digest.update(p.read_bytes())
    return {"counts": counts, "boxes": {s: dict(b) for s, b in boxes.items()}, "tiles": cfg.recipe.tiles,
            "sha256": digest.hexdigest()}
