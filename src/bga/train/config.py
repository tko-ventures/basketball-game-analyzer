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

ROOT = Path(__file__).resolve().parents[3]


class BaseWeights(BaseModel):
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
    model_config = ConfigDict(extra="forbid")
    resolution: int
    epochs: int
    batch_size: int
    grad_accum_steps: int = 1
    lr: float
    seed: int = 0
    select_on: str = "val/keypoint_map_50_95"


class Platform(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["modal"]
    gpu: str
    timeout_h: float = 6.0


class TrainConfig(BaseModel):
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
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def weight_id(self) -> str:
        return f"{self.layer}/{self.backend}/{self.name}@{self.config_hash()}"

    def weights_dir(self, root: Path = ROOT) -> Path:
        return root / "weights" / self.weight_id()


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig(**yaml.safe_load(Path(path).read_text()))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
