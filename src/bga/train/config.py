"""Training configs (plan section 5): one committed YAML per training setup, hashed into the weight id.

A weight id is ``<layer>/<backend>/<name>@<hash8>``; the hash covers the whole config, so any change to the
base weights, data or recipe gives new weights a new id. Weights land in ``weights/<id>/`` with a
``train.json`` beside them.

"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

#: Repository root; configs, weights and runs are addressed from here.
ROOT = Path(__file__).resolve().parents[3]

#: The committed config of the first PLK court model, used by the examples below.
_EXAMPLE = "configs/train/court-rfdetr_keypoint-plk-v1.yaml"


class BaseWeights(BaseModel):
    """The pretrained checkpoint a fine-tune starts from: file name, download URL and licence."""

    model_config = ConfigDict(extra="forbid")
    name: str
    url: str
    licence: str


class RoboflowSource(BaseModel):
    """A Roboflow Universe dataset version, exported as COCO; train/valid/test kept as published."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["roboflow"] = "roboflow"
    ref: str  # workspace/project/version
    licence: str


class ManifestSource(BaseModel):
    """A committed label manifest (data/labels/...), pinned by sha256."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["manifest"] = "manifest"
    path: str
    sha256: str
    image_width: int = 1280  # frames are resized to this width for upload; keypoints scale with them


class Recipe(BaseModel):
    """Trainer hyperparameters, and the validation metric the best checkpoint is selected on."""

    model_config = ConfigDict(extra="forbid")
    resolution: int
    epochs: int
    batch_size: int
    grad_accum_steps: int = 1
    lr: float
    seed: int = 0
    select_on: str = "val/keypoint_map_50_95"
    variant: str | None = None  # rfdetr model class for detection fine-tunes, e.g. RFDETRLarge
    tiles: int | None = None  # detection: 2 trains and infers on two overlapping square crops per frame


class Platform(BaseModel):
    """Where the fine-tune runs: Modal, on which GPU, for at most ``timeout_h`` hours."""

    model_config = ConfigDict(extra="forbid")
    name: Literal["modal"]
    gpu: str
    timeout_h: float = 6.0


class TrainConfig(BaseModel):
    """One training setup (``configs/train/*.yaml``): what is trained, from which weights, on which data, how and where.

    Examples:
        >>> cfg = load_train_config(ROOT / _EXAMPLE)
        >>> cfg.layer, cfg.backend, cfg.recipe.epochs, cfg.platform.gpu
        ('court', 'rfdetr_keypoint', 50, 'L40S')

    """

    model_config = ConfigDict(extra="forbid")
    name: str
    layer: str
    backend: str
    description: str = ""
    base_weights: BaseWeights
    sources: list[Annotated[RoboflowSource | ManifestSource, Field(discriminator="kind")]]
    recipe: Recipe
    platform: Platform
    package: str  # the trainer package pinned, e.g. rfdetr[train]==1.12.1

    def config_hash(self) -> str:
        """Hash the whole config: the first 8 hex digits of the sha256 of its sorted-key JSON dump.

        Examples:
            >>> load_train_config(ROOT / _EXAMPLE).config_hash()
            'c26b7841'

        """
        blob = json.dumps(self.model_dump(exclude_none=True), sort_keys=True)  # new optional fields keep old ids
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def weight_id(self) -> str:
        """Address the weights this config trains: ``<layer>/<backend>/<name>@<hash8>``.

        Examples:
            >>> load_train_config(ROOT / _EXAMPLE).weight_id()
            'court/rfdetr_keypoint/plk-v1@c26b7841'

        """
        return f"{self.layer}/{self.backend}/{self.name}@{self.config_hash()}"

    def weights_dir(self, root: Path = ROOT) -> Path:
        """Locate the directory the weights and their ``train.json`` land in: ``<root>/weights/<weight id>``.

        Args:
            root: repository root.

        Returns:
            The weights directory (not created here).

        Examples:
            >>> load_train_config(ROOT / _EXAMPLE).weights_dir(Path("/repo")).as_posix()
            '/repo/weights/court/rfdetr_keypoint/plk-v1@c26b7841'

        """
        return root / "weights" / self.weight_id()


def load_train_config(path: str | Path) -> TrainConfig:
    """Read and validate a training config YAML.

    Args:
        path: the YAML file.

    Returns:
        The validated config.

    Raises:
        pydantic.ValidationError: on a missing, unknown or mistyped key.

    Examples:
        >>> load_train_config(ROOT / _EXAMPLE).name
        'plk-v1'

    """
    return TrainConfig(**yaml.safe_load(Path(path).read_text()))


def file_sha256(path: Path) -> str:
    """Hash a file's bytes with sha256, as manifests are pinned in training configs.

    Args:
        path: the file.

    Returns:
        The hex digest.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "x.txt"
        ...     _ = path.write_bytes(b"abc")
        ...     file_sha256(path)
        'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
