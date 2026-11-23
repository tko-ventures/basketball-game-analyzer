"""Collected weights and the volume: which files a remote run uploads, and dropping a stale copy on collect."""

import contextlib

import pytest
from modal.exception import NotFoundError

from bga.remote import modal_runner, modal_train
from bga.train import run as train_run

WEIGHT_ID = "numbers/crop_cls/demo@00000000"


class Entry:
    """A volume directory entry, as ``modal.Volume.listdir`` returns them."""

    def __init__(self, path):
        self.path = path


class FakeVolume:
    """Stands in for the ``bga-data`` Modal volume: a set of paths that can be listed, uploaded to and removed."""

    def __init__(self, paths=()):
        self.paths, self.removed, self.uploaded = set(paths), [], []

    def listdir(self, path):
        found = [Entry(p) for p in sorted(self.paths) if p.rsplit("/", 1)[0] == path]
        if not found:
            raise NotFoundError(path)
        return found

    def remove_file(self, path, recursive=False):
        self.removed.append((path, recursive))

    @contextlib.contextmanager
    def batch_upload(self):
        yield self

    def put_file(self, local, remote):
        self.uploaded.append(remote)


@pytest.fixture
def local_weights(tmp_path, monkeypatch):
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
    def test_every_collected_file_but_the_logs_is_named(self, monkeypatch):
        monkeypatch.setitem(train_run.PULLS, ("demo", "demo"), ("demo.bin", "demo_config.json", "metrics.csv"))
        assert set(train_run.weight_files()) >= {"train.json", "demo.bin", "demo_config.json"}

    @pytest.mark.parametrize(
        "log", [pytest.param("metrics.csv", id="metrics"), pytest.param("result.json", id="result")]
    )
    def test_trainer_logs_are_left_out(self, log):
        assert log not in train_run.weight_files()


class TestUploadWeights:
    def test_uploads_every_file_the_backend_loads(self, local_weights, monkeypatch):
        vol = FakeVolume()
        monkeypatch.setattr(modal_runner, "vol", vol)
        modal_runner.upload_weights(WEIGHT_ID)
        assert sorted(vol.uploaded) == [
            f"weights/{WEIGHT_ID}/{n}" for n in ("cls_config.json", "train.json", "weights.safetensors")
        ]

    def test_files_already_on_the_volume_are_skipped(self, local_weights, monkeypatch):
        vol = FakeVolume({f"weights/{WEIGHT_ID}/train.json", f"weights/{WEIGHT_ID}/weights.safetensors"})
        monkeypatch.setattr(modal_runner, "vol", vol)
        modal_runner.upload_weights(WEIGHT_ID)
        assert vol.uploaded == [f"weights/{WEIGHT_ID}/cls_config.json"]

    def test_pretrained_ids_upload_nothing(self, monkeypatch):
        vol = FakeVolume()
        monkeypatch.setattr(modal_runner, "vol", vol)
        modal_runner.upload_weights("pretrained:roboflow/x/6")
        assert vol.uploaded == []


class TestDropRemoteWeights:
    def test_removes_the_volumes_copy(self, monkeypatch):
        vol = FakeVolume({f"weights/{WEIGHT_ID}", f"weights/{WEIGHT_ID}/train.json"})
        monkeypatch.setattr(modal_train, "vol", vol)
        assert (train_run.drop_remote_weights(WEIGHT_ID), vol.removed) == (True, [(f"weights/{WEIGHT_ID}", True)])

    def test_does_nothing_when_the_volume_has_no_copy(self, monkeypatch):
        vol = FakeVolume({"weights/numbers/crop_cls/other@11111111"})
        monkeypatch.setattr(modal_train, "vol", vol)
        assert (train_run.drop_remote_weights(WEIGHT_ID), vol.removed) == (False, [])
