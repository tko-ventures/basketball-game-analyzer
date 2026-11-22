"""Remote layer runs: results handed over as each call finishes, in any order, at their sequence's index."""

import contextlib

import pytest

from bga.remote import modal_runner
from bga.stage import Sequence


class FakeFunction:
    """Stands in for the Modal function: answers the calls in reverse order, as a parallel batch may."""

    def __init__(self, fail_job=None):
        self.fail_job = fail_job

    def starmap(self, args, order_outputs=True, return_exceptions=False):
        for _layer, _entry, seq, _upstream, gpu, job in reversed(args):
            if job == self.fail_job:
                yield RuntimeError("remote failure")
                continue
            yield {"name": seq["name"], "remote": {"gpu": gpu, "seconds": 1.0, "usd": 0.0, "job": job}}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Modal and the volume replaced at the process boundary; no cost log in the working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modal_runner, "run_sequence", FakeFunction())
    monkeypatch.setattr(modal_runner.app, "run", contextlib.nullcontext)
    monkeypatch.setattr(modal_runner.modal, "enable_output", contextlib.nullcontext)
    monkeypatch.setattr(modal_runner, "upload_clips", lambda videos: None)
    monkeypatch.setattr(modal_runner, "upload_weights", lambda weight_id: None)


SEQS = [Sequence(name=f"s{i}", clip="c", video="v.mp4", shot=i, frames=[0], fps=50.0) for i in range(3)]


class TestRunLayer:
    def test_outputs_come_back_in_sequence_order(self, offline):
        outs = modal_runner.run_layer("track", {"backend": "trackers"}, SEQS, [{}] * 3, None)
        assert [o["name"] for o in outs] == ["s0", "s1", "s2"]

    def test_each_result_is_handed_over_with_its_index(self, offline):
        seen = []
        modal_runner.run_layer("track", {"backend": "trackers"}, SEQS, [{}] * 3, None, lambda i, o: seen.append(i))
        assert seen == [2, 1, 0]

    def test_failed_call_does_not_stop_the_others(self, offline, monkeypatch):
        monkeypatch.setattr(modal_runner, "run_sequence", FakeFunction(fail_job=1))
        seen = []
        with pytest.raises(modal_runner.RemoteCallsFailedError):
            modal_runner.run_layer("track", {"backend": "trackers"}, SEQS, [{}] * 3, None, lambda i, o: seen.append(i))
        assert seen == [2, 0]

    def test_job_index_is_not_left_in_the_output(self, offline):
        outs = modal_runner.run_layer("track", {"backend": "trackers"}, SEQS, [{}] * 3, None)
        assert "job" not in outs[0]["remote"]
