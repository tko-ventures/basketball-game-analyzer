"""Layer outputs on disk: written whole, and ``meta.json`` keeps a record per sequence across runs."""

import json
import time

import numpy as np
import pytest

from bga import pipeline
from bga.config import RunConfig
from bga.stage import Sequence

CFG = RunConfig(name="demo", layers={"shots": {"backend": "gold"}, "detect": {"backend": "roboflow_inference"}})
S1, S2 = (Sequence(name=f"clip_s{i}", clip="clip", video="clip.mp4", shot=i, frames=[0], fps=50.0) for i in (1, 2))
REMOTE = {"platform": "modal", "gpu": "L4", "seconds": 9.0, "usd": 0.002}


@pytest.fixture
def commits(monkeypatch):
    """The checked-out commit, read through git, replaced by a value the test sets."""
    current = {"commit": "aaaaaaa"}
    monkeypatch.setattr(pipeline, "git_commit", lambda: current["commit"])
    return current


def write(runs, seq, out):
    """Store one sequence's detect output and record it in the clip's meta, as a run over that sequence does."""
    pipeline.write_output(pipeline.output_path(CFG, runs, seq, "detect"), out)
    pipeline._write_meta(CFG, runs, "detect", [seq], time.perf_counter())


def meta(runs) -> dict:
    """The detect layer's ``meta.json`` of the clip."""
    return json.loads((CFG.run_dir(runs, "clip", "detect") / "meta.json").read_text())


class TestWriteOutput:
    def test_leaves_only_the_finished_files(self, tmp_path):
        path = pipeline.output_path(CFG, tmp_path, S1, "detect")
        pipeline.write_output(path, {"frames": {}, "arrays": {"e": np.ones((2, 3), np.float16)}})
        assert sorted(p.name for p in path.parent.iterdir()) == ["clip_s1.e.npy", "clip_s1.json"]

    def test_failed_write_leaves_no_output(self, tmp_path):
        path = pipeline.output_path(CFG, tmp_path, S1, "detect")
        with pytest.raises(TypeError):
            pipeline.write_output(path, {"frames": {1, 2}})
        assert not path.exists()

    def test_failed_rewrite_keeps_the_earlier_output(self, tmp_path):
        path = pipeline.output_path(CFG, tmp_path, S1, "detect")
        pipeline.write_output(path, {"frames": {"0": []}})
        with pytest.raises(TypeError):
            pipeline.write_output(path, {"frames": {1, 2}})
        assert json.loads(path.read_text()) == {"frames": {"0": []}}


class TestMeta:
    def test_run_over_one_sequence_keeps_the_record_of_another(self, tmp_path, commits):
        write(tmp_path, S1, {"wall_s": 1.5})
        commits["commit"] = "bbbbbbb"
        write(tmp_path, S2, {"wall_s": 2.5})
        got = {name: (rec["commit"], rec["wall_s"]) for name, rec in meta(tmp_path)["sequences"].items()}
        assert got == {"clip_s1": ("aaaaaaa", 1.5), "clip_s2": ("bbbbbbb", 2.5)}

    def test_rerun_replaces_the_sequences_record(self, tmp_path, commits):
        write(tmp_path, S1, {"wall_s": 1.5})
        commits["commit"] = "bbbbbbb"
        write(tmp_path, S1, {"wall_s": 0.5})
        assert meta(tmp_path)["sequences"]["clip_s1"]["commit"] == "bbbbbbb"

    def test_top_level_commit_is_the_last_runs(self, tmp_path, commits):
        write(tmp_path, S1, {"wall_s": 1.5})
        commits["commit"] = "bbbbbbb"
        write(tmp_path, S2, {"wall_s": 2.5})
        assert meta(tmp_path)["commit"] == "bbbbbbb"

    def test_remote_total_covers_sequences_of_earlier_runs(self, tmp_path, commits):
        write(tmp_path, S1, {"remote": REMOTE})
        write(tmp_path, S2, {"remote": REMOTE})
        assert meta(tmp_path)["remote"] == {"platform": "modal", "gpu": "L4", "seconds": 18.0, "usd": 0.004}

    def test_platform_is_recorded_per_sequence(self, tmp_path, commits):
        write(tmp_path, S1, {"wall_s": 1.5})
        write(tmp_path, S2, {"remote": REMOTE})
        got = {name: rec["platform"] for name, rec in meta(tmp_path)["sequences"].items()}
        assert got == {"clip_s1": "local", "clip_s2": "modal"}

    def test_meta_written_before_sequence_records_is_extended(self, tmp_path, commits):
        path = CFG.run_dir(tmp_path, "clip", "detect") / "meta.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"layer": "detect", "commit": "0000000"}))
        write(tmp_path, S1, {"wall_s": 1.5})
        assert list(meta(tmp_path)["sequences"]) == ["clip_s1"]
