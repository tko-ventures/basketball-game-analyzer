"""No gold frame in any training setup (plan section 5): every committed training config and manifest."""

import json
from pathlib import Path

import pytest

from bga.train.config import ROOT, ManifestSource, file_sha256, load_train_config
from bga.train.leakage import assert_no_gold, gold_clips, manifest_clips

CONFIGS = sorted((ROOT / "configs/train").glob("*.yaml"))
MANIFESTS = sorted((ROOT / "data/labels").rglob("train_manifest_*.json"))


def test_there_is_something_to_check():
    assert CONFIGS and MANIFESTS


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_has_no_gold_clip(path: Path):
    assert_no_gold(manifest_clips(json.loads(path.read_text())))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_manifests_pinned_and_clean(path: Path):
    cfg = load_train_config(path)
    assert path.name == f"{cfg.layer}-{cfg.backend}-{cfg.name}.yaml"
    for src in cfg.sources:
        if isinstance(src, ManifestSource):
            p = ROOT / src.path
            assert file_sha256(p) == src.sha256, f"{src.path} changed since {path.name} pinned it"
            assert_no_gold(manifest_clips(json.loads(p.read_text())))


def test_guard_refuses_a_gold_clip():
    clip = sorted(gold_clips())[0]
    with pytest.raises(ValueError):
        assert_no_gold({clip})
    with pytest.raises(ValueError):
        assert_no_gold({f"data/raw/{clip}.mp4"})
    assert_no_gold({"20251113_kW4HG-5MLNM_SKROT_Dziki_-_King_8._kolejka"})


def test_weight_id_tracks_the_config():
    cfg = load_train_config(CONFIGS[0])
    other = cfg.model_copy(update={"recipe": cfg.recipe.model_copy(update={"epochs": cfg.recipe.epochs + 1})})
    assert cfg.weight_id().startswith(f"{cfg.layer}/{cfg.backend}/{cfg.name}@")
    assert other.config_hash() != cfg.config_hash()
