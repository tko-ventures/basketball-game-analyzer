"""Appearance training: identity splits, P x K batches, the triplet loss and the dataset's identity folders."""

from collections import Counter

import numpy as np
import pytest
import torch

from bga.train.appearance_data import split_of
from bga.train.reid_trainer import batch_hard_triplet, identity_folders, pk_batches, shot_batches


class TestPkBatches:
    def test_each_batch_holds_k_crops_of_each_identity(self):
        labels = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5
        batch = next(pk_batches(labels, 4, 8, np.random.default_rng(0)))
        assert sorted(Counter(labels[i] for i in batch).values()) == [4, 4]

    def test_short_identity_is_repeated_to_k(self):
        batch = next(pk_batches([0, 1, 1, 1], 3, 6, np.random.default_rng(0)))
        assert len(batch) == 6

    def test_epoch_covers_the_crops_once_on_average(self):
        assert len(list(pk_batches(list(range(4)) * 8, 4, 8, np.random.default_rng(0)))) == 4


class TestShotBatches:
    def test_batch_stays_inside_one_shot(self):
        labels, shots = [0, 0, 1, 1, 2, 2, 3, 3], ["a", "a", "a", "a", "b", "b", "b", "b"]
        batch = next(shot_batches(labels, shots, 2, 8, np.random.default_rng(0)))
        assert len({shots[i] for i in batch}) == 1

    def test_shot_with_one_identity_is_skipped(self):
        assert list(shot_batches([0, 0], ["a", "a"], 2, 4, np.random.default_rng(0))) == []


class TestBatchHardTriplet:
    def test_margin_is_paid_when_negatives_are_as_close_as_positives(self):
        e = torch.ones(4, 2)
        assert float(batch_hard_triplet(e, torch.tensor([0, 0, 1, 1]), 0.3)) == pytest.approx(0.3)


class TestIdentityFolders:
    def test_labels_follow_sorted_folders(self, tmp_path):
        for name in ("b", "a"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "000005.jpg").write_bytes(b"")
        _, labels = identity_folders(tmp_path)
        assert labels == [0, 1]


class TestSplitOf:
    def test_validation_game_goes_to_valid(self):
        assert split_of("game_a", ["game_a"]) == "valid"
