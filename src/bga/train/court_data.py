"""Court-keypoint training data: Roboflow COCO exports and committed manifests merged into one COCO dataset.

The layout is what ``rfdetr`` reads (``dataset_file="roboflow"``): ``train/``, ``valid/`` and ``test/``,
each with its images and ``_annotations.coco.json``. Categories follow the Roboflow export (id 0, the
supercategory placeholder; id 1, ``court`` with the 33 landmark names), so every source must share the
keypoint names. Manifest frames are decoded from the local videos and resized to ``image_width``.

"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from bga.court.propagate import court_mask
from bga.train.config import ROOT, ManifestSource, RoboflowSource, TrainConfig, file_sha256
from bga.train.leakage import assert_no_gold, manifest_clips

#: Dataset splits, in the order they are written and hashed.
SPLITS = ("train", "valid", "test")


def fetch_roboflow(src: RoboflowSource, cache: Path = ROOT / "runs/train/_sources") -> Path:
    """Download a Roboflow dataset version as COCO into its own directory (once).

    Needs ``ROBOFLOW_API_KEY`` (``bga.detect.roboflow_inference.roboflow_key``) unless the export is already cached.

    Args:
        src: the dataset version.
        cache: parent directory; the export lands in ``<cache>/<project>-v<version>``.

    Returns:
        The export directory, with ``train/``, ``valid/`` and ``test/``.

    Examples:
        >>> src = RoboflowSource(ref="roboflow-jvuqo/basketball-court-detection-2/20", licence="CC-BY-4.0")
        >>> fetch_roboflow(src).name  # doctest: +SKIP
        'basketball-court-detection-2-v20'

    """
    import urllib.request

    from bga.detect.roboflow_inference import roboflow_key

    ws, proj, ver = src.ref.split("/")
    out = cache / f"{proj}-v{ver}"
    if (out / "train" / "_annotations.coco.json").exists():
        return out
    import os

    roboflow_key()
    key = os.environ["ROBOFLOW_API_KEY"]
    with urllib.request.urlopen(f"https://api.roboflow.com/{ws}/{proj}/{ver}/coco?api_key={key}") as r:
        link = json.load(r)["export"]["link"]
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "export.zip"
    urllib.request.urlretrieve(link, zpath)  # noqa: S310  # link is the export URL Roboflow's https API returns
    shutil.unpack_archive(zpath, out)
    zpath.unlink()
    return out


def _court_bbox(hom: np.ndarray, wh: tuple[int, int]) -> list[float] | None:
    """COCO ``[x, y, w, h]`` of the court (no margin) in a ``wh`` frame under ``hom``; None when it is out of view."""
    m = court_mask(np.asarray(hom), (wh[1], wh[0]), margin_cm=0.0)
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


class _Split:
    """One split's image directory and COCO annotations, built up image by image."""

    def __init__(self, root: Path, categories: list[dict]) -> None:
        self.dir = root
        self.dir.mkdir(parents=True, exist_ok=True)
        self.coco = {"images": [], "annotations": [], "categories": categories}

    def add(self, file_name: str, w: int, h: int, bbox: list[float], kp: list[float], source: str) -> None:
        iid = len(self.coco["images"])
        self.coco["images"].append({"id": iid, "file_name": file_name, "width": w, "height": h, "source": source})
        n = sum(1 for v in kp[2::3] if v > 0)
        self.coco["annotations"].append({
            "id": len(self.coco["annotations"]),
            "image_id": iid,
            "category_id": 1,
            "bbox": [round(b, 2) for b in bbox],
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
            "segmentation": [],
            "keypoints": [round(float(v), 2) for v in kp],
            "num_keypoints": n,
        })

    def write(self) -> Path:
        p = self.dir / "_annotations.coco.json"
        p.write_text(json.dumps(self.coco, separators=(",", ":")))
        return p


def _add_roboflow(
    src: RoboflowSource, names: list[str], splits: dict[str, _Split], counts: dict[str, dict[str, int]]
) -> None:
    """Copy a Roboflow export's annotated court images into ``splits``, keeping its train/valid/test split."""
    d = fetch_roboflow(src)
    tag = src.ref.replace("/", "_")
    for s in SPLITS:
        coco = json.loads((d / s / "_annotations.coco.json").read_text())
        if coco["categories"][1]["keypoints"] != names:
            raise ValueError(f"{src.ref}: keypoint names differ from the first source")
        anns = {a["image_id"]: a for a in coco["annotations"] if a["category_id"] == 1}
        for im in coco["images"]:
            a = anns.get(im["id"])
            if a is None:
                continue
            fn = f"rf_{im['file_name']}"
            shutil.copyfile(d / s / im["file_name"], splits[s].dir / fn)
            splits[s].add(fn, im["width"], im["height"], a["bbox"], a["keypoints"], tag)
            counts[s][tag] = counts[s].get(tag, 0) + 1


def _add_manifest(
    src: ManifestSource, names: list[str], splits: dict[str, _Split], counts: dict[str, dict[str, int]]
) -> None:
    """Decode a manifest's frames from the local videos, resize them and add them to the split each names."""
    path = ROOT / src.path
    if file_sha256(path) != src.sha256:
        raise ValueError(f"{src.path}: sha256 differs from the config; rebuild the config or the manifest")
    man = json.loads(path.read_text())
    assert_no_gold(manifest_clips(man))
    if man["keypoint_names"] != names:
        raise ValueError(f"{src.path}: keypoint names differ from the Roboflow source")
    caps: dict[str, cv2.VideoCapture] = {}
    for f in man["frames"]:
        cap = caps.setdefault(f["video"], cv2.VideoCapture(str(ROOT / f["video"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        ok, img = cap.read()
        if not ok:
            raise IndexError(f"{f['video']} frame {f['frame']}")
        w0, h0 = f["size"]
        k = src.image_width / w0
        w, h = src.image_width, round(h0 * k)
        bbox = _court_bbox(np.array(f["H"]), (w0, h0))
        if bbox is None:
            continue
        fn = f"plk_{Path(f['clip']).stem.split('_')[1]}_{f['frame']:06d}.jpg"
        cv2.imwrite(
            str(splits[f["split"]].dir / fn),
            cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        kp = []
        for x, y, v in f["kp"]:
            kp += [x * k, y * k, v] if v else [0.0, 0.0, 0]
        splits[f["split"]].add(fn, w, h, [b * k for b in bbox], kp, src.path)
        counts[f["split"]][src.path] = counts[f["split"]].get(src.path, 0) + 1
    for cap in caps.values():
        cap.release()


def build(cfg: TrainConfig, out: Path) -> dict:
    """Build the merged COCO dataset of a court training config under ``out``, replacing what is there.

    Roboflow sources are copied first, then manifest frames; every manifest is checked against its sha256 and the gold
    clips (``bga.train.leakage``).

    Args:
        cfg: the training config; its first Roboflow source sets the categories and keypoint names.
        out: the dataset directory.

    Returns:
        ``{"counts": {split: {source: images}}, "sha256": hex digest of the three annotation files}``.

    Raises:
        ValueError: with no Roboflow source, on differing keypoint names, a manifest sha256 mismatch or a gold clip.
        IndexError: when a manifest frame cannot be decoded.

    Examples:
        >>> from bga.train.config import load_train_config
        >>> cfg = load_train_config(ROOT / "configs/train/court-rfdetr_keypoint-plk-v1.yaml")
        >>> build(cfg, ROOT / "runs/train/example/dataset")["counts"]["train"]  # doctest: +SKIP
        {'roboflow-jvuqo_basketball-court-detection-2_20': ..., 'data/labels/court/train_manifest_plk-v1.json': ...}

    """
    if out.exists():
        shutil.rmtree(out)
    roboflow = [s for s in cfg.sources if isinstance(s, RoboflowSource)]
    manifests = [s for s in cfg.sources if isinstance(s, ManifestSource)]
    first = fetch_roboflow(roboflow[0]) if roboflow else None
    if first is None:
        raise ValueError("the court dataset takes its categories from a Roboflow source")
    categories = json.loads((first / "train" / "_annotations.coco.json").read_text())["categories"]
    names = categories[1]["keypoints"]
    splits = {s: _Split(out / s, categories) for s in SPLITS}
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}

    for src in roboflow:
        _add_roboflow(src, names, splits, counts)

    for src in manifests:
        _add_manifest(src, names, splits, counts)

    digest = hashlib.sha256()
    for s in SPLITS:
        digest.update(splits[s].write().read_bytes())
    return {"counts": counts, "sha256": digest.hexdigest()}
