"""Collected weights, platform resolution and remote volume sync for ``bga train``."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from modal.exception import NotFoundError

from bga.remote import modal_runner, modal_train
from bga.train import run as train_run
from bga.train.config import BaseWeights, ManifestSource, Platform, Recipe, TrainConfig

WEIGHT_ID = "numbers/crop_cls/demo@00000000"


class Entry:
    """A volume directory entry, as ``modal.Volume.listdir`` returns them."""

    def __init__(self, path: str) -> None:
        self.path = path


class FakeVolume:
    """Stands in for the ``bga-data`` Modal volume: a set of paths that can be listed, uploaded to and removed."""

    def __init__(self, paths=()) -> None:
        self.paths, self.removed, self.uploaded = set(paths), [], []

    def listdir(self, path: str) -> list[Entry]:
        found = [Entry(p) for p in sorted(self.paths) if p.rsplit("/", 1)[0] == path]
        if not found:
            raise NotFoundError(path)
        return found

    def remove_file(self, path: str, recursive: bool = False) -> None:
        self.removed.append((path, recursive))

    @contextlib.contextmanager
    def batch_upload(self):
        yield self

    def put_file(self, local: str, remote: str) -> None:
        self.uploaded.append(remote)


def _train_cfg(platform: str = "modal", gpu: str = "L4") -> TrainConfig:
    """Minimal valid ``TrainConfig`` for platform-resolution and collect tests."""
    return TrainConfig(
        name="demo",
        layer="numbers",
        backend="crop_cls",
        base_weights=BaseWeights(name="resnet18", url="https://example.invalid/w", licence="Apache-2.0"),
        sources=[ManifestSource(path="data/labels/numbers/demo.json", sha256="0" * 64)],
        recipe=Recipe(resolution=64, epochs=2, batch_size=8, lr=1e-3, select_on="val/top1"),
        platform=Platform(name=platform, gpu=gpu),
        package="timm==1.0.30",
    )


@pytest.fixture
def local_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local weights directory holding a classifier's collected files, its trainer logs among them."""
    monkeypatch.setenv("BGA_WEIGHTS", str(tmp_path))
    monkeypatch.setitem(
        train_run.PULLS, ("numbers", "crop_cls"), ("weights.safetensors", "cls_config.json", "metrics.csv")
    )
    folder = tmp_path / WEIGHT_ID
    folder.mkdir(parents=True)
    for name in ("train.json", "weights.safetensors", "cls_config.json", "metrics.csv", "result.json"):
        (folder / name).write_text("x")
    return folder


class TestWeightFiles:
    def test_every_collected_file_but_the_logs_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(train_run.PULLS, ("demo", "demo"), ("demo.bin", "demo_config.json", "metrics.csv"))
        assert set(train_run.weight_files()) >= {"train.json", "demo.bin", "demo_config.json"}

    @pytest.mark.parametrize(
        "log", [pytest.param("metrics.csv", id="metrics"), pytest.param("result.json", id="result")]
    )
    def test_trainer_logs_are_left_out(self, log: str) -> None:
        assert log not in train_run.weight_files()


class TestUploadWeights:
    def test_uploads_every_file_the_backend_loads(self, local_weights: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vol = FakeVolume()
        monkeypatch.setattr(modal_runner, "vol", vol)
        modal_runner.upload_weights(WEIGHT_ID)
        assert sorted(vol.uploaded) == [
            f"weights/{WEIGHT_ID}/{n}" for n in ("cls_config.json", "train.json", "weights.safetensors")
        ]

    def test_files_already_on_the_volume_are_skipped(
        self, local_weights: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vol = FakeVolume({f"weights/{WEIGHT_ID}/train.json", f"weights/{WEIGHT_ID}/weights.safetensors"})
        monkeypatch.setattr(modal_runner, "vol", vol)
        modal_runner.upload_weights(WEIGHT_ID)
        assert vol.uploaded == [f"weights/{WEIGHT_ID}/cls_config.json"]

    def test_pretrained_ids_upload_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vol = FakeVolume()
        monkeypatch.setattr(modal_runner, "vol", vol)
        modal_runner.upload_weights("pretrained:roboflow/x/6")
        assert vol.uploaded == []


class TestDropRemoteWeights:
    def test_removes_the_volumes_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vol = FakeVolume({f"weights/{WEIGHT_ID}", f"weights/{WEIGHT_ID}/train.json"})
        monkeypatch.setattr(modal_train, "vol", vol)
        assert (train_run.drop_remote_weights(WEIGHT_ID), vol.removed) == (True, [(f"weights/{WEIGHT_ID}", True)])

    def test_does_nothing_when_the_volume_has_no_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vol = FakeVolume({"weights/numbers/crop_cls/other@11111111"})
        monkeypatch.setattr(modal_train, "vol", vol)
        assert (train_run.drop_remote_weights(WEIGHT_ID), vol.removed) == (False, [])


class TestResolvePlatform:
    def test_uses_config_platform_and_gpu_when_no_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BGA_TRAIN_PLATFORM", raising=False)
        monkeypatch.delenv("BGA_TRAIN_GPU", raising=False)
        assert train_run.resolve_platform(_train_cfg("modal", "L40S")) == ("modal", "L40S")

    def test_env_var_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BGA_TRAIN_PLATFORM", "colab")
        monkeypatch.setenv("BGA_TRAIN_GPU", "a100")
        assert train_run.resolve_platform(_train_cfg("modal", "L40S")) == ("colab", "A100")

    def test_explicit_args_override_env_and_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BGA_TRAIN_PLATFORM", "colab")
        monkeypatch.setenv("BGA_TRAIN_GPU", "T4")
        assert train_run.resolve_platform(_train_cfg("colab", "L4"), platform="modal", gpu="L40S") == ("modal", "L40S")

    def test_rejects_unknown_platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BGA_TRAIN_PLATFORM", raising=False)
        with pytest.raises(ValueError, match="unsupported training platform"):
            train_run.resolve_platform(_train_cfg(), platform="runpod")

    def test_colab_override_rejects_modal_only_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BGA_TRAIN_GPU", raising=False)
        with pytest.raises(ValueError, match="Colab does not offer GPU"):
            train_run.resolve_platform(_train_cfg("modal", "L40S"), platform="colab")


class TestColabCollect:
    def test_collect_reads_local_colab_output_and_logs_ccu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _train_cfg("modal", "L4")
        monkeypatch.setattr(train_run, "ROOT", tmp_path)
        train_md = tmp_path / "docs/train.md"
        costs_md = tmp_path / "docs/costs.md"
        train_md.parent.mkdir(parents=True)
        train_md.write_text(
            "| Date | Commit | Config | Weights | Images | Best | Spent | Note |\n|---|---|---|---|---|---|---|---|\n"
        )
        costs_md.write_text("| Date | Platform | GPU | Job | GPU seconds | USD |\n|---|---|---|---|---|---|\n")
        monkeypatch.setattr(train_run, "TRAIN_MD", train_md)
        monkeypatch.setattr(train_run, "COSTS_MD", costs_md)
        monkeypatch.setattr(
            train_run, "drop_remote_weights", lambda _wid: (_ for _ in ()).throw(RuntimeError("no auth"))
        )

        work = tmp_path / "runs/train" / cfg.weight_id().replace("/", "__")
        out = work / "output"
        out.mkdir(parents=True)
        (work / "call.json").write_text(json.dumps({"platform": "colab", "gpu": "L4"}))
        (work / "dataset_info.json").write_text(json.dumps({"counts": {"train": {"a": 10}, "valid": {"b": 4}}}))
        (out / "weights.safetensors").write_bytes(b"weights")
        (out / "cls_config.json").write_text('{"classes": ["0"]}')
        (out / "metrics.csv").write_text("epoch,val/top1\n0,0.75\n1,0.92\n")
        (out / "result.json").write_text(
            json.dumps({
                "platform": "colab",
                "gpu": "L4",
                "seconds": 42.5,
                "ccu": 0.056,
                "commit": "abc1234",
                "package": "timm==1.0.31",
                "torch": "2.14.1+cu128",
            })
        )

        wdir = train_run.collect(cfg)
        meta = json.loads((wdir / "train.json").read_text())

        assert meta["platform"] == "colab"
        assert meta["ccu"] == 0.056
        assert meta["gpu"] == "L4"
        assert "colab L4, 42.5 s, 0.056 CCU" in train_md.read_text()
        assert f"| colab | L4 | bga train {cfg.weight_id()} | 42.5 | 0.056 CCU |" in costs_md.read_text()
