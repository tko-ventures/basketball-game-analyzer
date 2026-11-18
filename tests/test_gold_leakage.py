"""No gold frame in any training setup (plan section 5): every committed training config and manifest."""

import json
from pathlib import Path

import pytest

from bga.train.config import ROOT, ManifestSource, file_sha256, load_train_config
from bga.train.leakage import assert_no_gold, gold_clips, manifest_clips

CONFIGS = sorted((ROOT / "configs/train").glob("*.yaml"))
MANIFESTS = sorted((ROOT / "data/labels").rglob("train_manifest_*.json"))


MANIFEST_SOURCES = [
    pytest.param(path, src, id=f"{path.name}:{Path(src.path).name}")
    for path in CONFIGS
    for src in load_train_config(path).sources
    if isinstance(src, ManifestSource)
]


@pytest.fixture
def gold_clip() -> str:
    return sorted(gold_clips())[0]


@pytest.mark.parametrize("found", [pytest.param(CONFIGS, id="configs"), pytest.param(MANIFESTS, id="manifests")])
def test_there_is_something_to_check(found):
    assert found


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_has_no_gold_clip(path: Path):
    assert_no_gold(manifest_clips(json.loads(path.read_text())))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_file_named_after_its_weight_id(path: Path):
    cfg = load_train_config(path)
    assert path.name == f"{cfg.layer}-{cfg.backend}-{cfg.name}.yaml"


@pytest.mark.parametrize(("path", "src"), MANIFEST_SOURCES)
class TestConfigManifest:
    def test_pinned(self, path: Path, src: ManifestSource):
        assert file_sha256(ROOT / src.path) == src.sha256, f"{src.path} changed since {path.name} pinned it"

    def test_has_no_gold_clip(self, path: Path, src: ManifestSource):
        assert_no_gold(manifest_clips(json.loads((ROOT / src.path).read_text())))


class TestGuard:
    @pytest.mark.parametrize(
        "pattern", [pytest.param("{}", id="clip-stem"), pytest.param("data/raw/{}.mp4", id="raw-path")]
    )
    def test_refuses_a_gold_clip(self, gold_clip, pattern):
        with pytest.raises(ValueError, match="gold clips in a training manifest"):
            assert_no_gold({pattern.format(gold_clip)})

    def test_accepts_a_non_gold_clip(self):
        assert_no_gold({"20251113_kW4HG-5MLNM_SKROT_Dziki_-_King_8._kolejka"})


def test_weight_id_tracks_the_config():
    cfg = load_train_config(CONFIGS[0])
    other = cfg.model_copy(update={"recipe": cfg.recipe.model_copy(update={"epochs": cfg.recipe.epochs + 1})})
    assert cfg.weight_id().startswith(f"{cfg.layer}/{cfg.backend}/{cfg.name}@")
    assert other.config_hash() != cfg.config_hash()
