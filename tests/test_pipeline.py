"""Layer outputs on disk: written whole, ``meta.json`` keeps a record per sequence, and stale code reruns."""

import importlib
import json
import sys
import time

import numpy as np
import pytest

from bga import pipeline, registry
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
    pipeline._write_meta(CFG, runs, "detect", [seq], time.perf_counter(), "c0de0000")


def legacy_dir(runs) -> None:
    """A detect directory as runs before 2026-11-22 left it: an output of S2 and a ``meta.json`` with no records."""
    pipeline.write_output(pipeline.output_path(CFG, runs, S2, "detect"), {"wall_s": 1.0})
    (CFG.run_dir(runs, "clip", "detect") / "meta.json").write_text(json.dumps({"layer": "detect", "commit": "0000000"}))


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

    def test_untouched_legacy_output_stays_reusable(self, tmp_path, commits):
        legacy_dir(tmp_path)
        write(tmp_path, S1, {"wall_s": 1.5})
        assert pipeline.stale_sequences(CFG, tmp_path, [S2], "detect", "c0de1111") == []

    def test_legacy_record_is_not_counted_as_remote(self, tmp_path, commits):
        legacy_dir(tmp_path)
        write(tmp_path, S1, {"wall_s": 1.5})
        assert "remote" not in meta(tmp_path)

    def test_meta_written_before_sequence_records_is_extended(self, tmp_path, commits):
        path = CFG.run_dir(tmp_path, "clip", "detect") / "meta.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"layer": "detect", "commit": "0000000"}))
        write(tmp_path, S1, {"wall_s": 1.5})
        assert list(meta(tmp_path)["sequences"]) == ["clip_s1"]


FAKE_LAYERS = {
    "shots": {"backend": "_fake_shots"},
    "detect": {"backend": "_fake_detect"},
    "team": {"backend": "_fake_team"},
}
FAKE_CFG = RunConfig(name="fake", layers=FAKE_LAYERS)
#: A detect ``value`` that makes the fake detect backend fail on the second sequence.
FAIL_ON_SECOND = 9

#: Source of a throwaway backend module; ``{value}`` is what its detect output records, ``{doc}`` its docstring.
FAKE_SOURCE = '''"""{doc}"""

from bga.stage import BaseBackend, Sequence


class Shots(BaseBackend):
    def sequences(self):
        return [
            Sequence(name=f"clip_s{{i}}", clip="clip", video="missing.mp4", shot=i, frames=[0], fps=50.0)
            for i in (1, 2)
        ]


class Detect(BaseBackend):
    def run(self, seq, upstream, read):
        if {value} == 9 and seq.name == "clip_s2":
            raise RuntimeError("detector failed")
        return {{"frames": {{}}, "value": {value}}}


class Team(BaseBackend):
    def run(self, seq, upstream, read):
        return {{"value": upstream["detect"]["value"]}}
'''


@pytest.fixture
def install(tmp_path, monkeypatch):
    """Write, import and register a throwaway backend module, as editing a backend's code and rerunning does."""
    pkg = tmp_path / "pkgs" / "fakebackends"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(pkg.parent))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # a stale .pyc must not mask the rewrite

    def _install(value: int, doc: str = "Fake backends.") -> None:
        (pkg / "backend.py").write_text(FAKE_SOURCE.format(value=value, doc=doc))
        monkeypatch.delitem(sys.modules, "fakebackends.backend", raising=False)
        importlib.invalidate_caches()
        mod = importlib.import_module("fakebackends.backend")
        for layer, cls in (("shots", mod.Shots), ("detect", mod.Detect), ("team", mod.Team)):
            cls.layer, cls.name = layer, f"_fake_{layer}"
            monkeypatch.setitem(registry._REGISTRY, (layer, cls.name), cls)

    yield _install
    sys.modules.pop("fakebackends.backend", None)
    sys.modules.pop("fakebackends", None)


def fake_output(runs, layer: str = "detect") -> dict:
    """The fake output of the first sequence."""
    seq = pipeline.sequences(FAKE_CFG)[0]
    return pipeline.load_output(FAKE_CFG, runs, seq, layer)


def fake_meta(runs) -> dict:
    """The fake detect layer's ``meta.json``."""
    return json.loads((FAKE_CFG.run_dir(runs, "clip", "detect") / "meta.json").read_text())


class TestStaleCode:
    def test_output_of_changed_code_is_recomputed(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        install(2)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        assert fake_output(tmp_path)["value"] == 2

    def test_output_is_kept_when_only_a_docstring_changed(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        path = pipeline.output_path(FAKE_CFG, tmp_path, pipeline.sequences(FAKE_CFG)[0], "detect")
        before = path.stat().st_mtime_ns
        install(1, doc="Edited docstring.")
        pipeline.run(FAKE_CFG, runs=tmp_path)
        assert path.stat().st_mtime_ns == before

    def test_output_without_a_code_record_is_kept(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        meta_path = FAKE_CFG.run_dir(tmp_path, "clip", "detect") / "meta.json"
        legacy = json.loads(meta_path.read_text())
        del legacy["sequences"]["clip_s1"]["code"]
        meta_path.write_text(json.dumps(legacy))
        install(2)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        assert fake_output(tmp_path)["value"] == 1

    def test_record_carries_the_code_key(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        meta_path = FAKE_CFG.run_dir(tmp_path, "clip", "detect") / "meta.json"
        record = json.loads(meta_path.read_text())["sequences"]["clip_s1"]
        assert record["code"] == pipeline.code_keys(FAKE_CFG)["detect"]

    def test_output_of_a_stopped_run_keeps_its_code_key(self, tmp_path, install):
        install(FAIL_ON_SECOND)
        with pytest.raises(RuntimeError, match="detector failed"):
            pipeline.run(FAKE_CFG, runs=tmp_path)
        assert fake_meta(tmp_path)["sequences"]["clip_s1"]["code"] == pipeline.code_keys(FAKE_CFG)["detect"]

    def test_unrecorded_output_in_a_keyed_directory_is_recomputed(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        meta = fake_meta(tmp_path)
        del meta["sequences"]["clip_s1"]
        (FAKE_CFG.run_dir(tmp_path, "clip", "detect") / "meta.json").write_text(json.dumps(meta))
        install(2)
        pipeline.run(FAKE_CFG, runs=tmp_path, layers=["detect"])
        assert fake_output(tmp_path)["value"] == 2

    def test_layer_refuses_stale_upstream_outputs_it_did_not_remake(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        install(2)
        with pytest.raises(RuntimeError, match="team would read 2 detect outputs made by other code"):
            pipeline.run(FAKE_CFG, runs=tmp_path, layers=["team"])

    def test_downstream_reruns_on_fresh_upstream(self, tmp_path, install):
        install(1)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        install(2)
        pipeline.run(FAKE_CFG, runs=tmp_path)
        assert fake_output(tmp_path, "team")["value"] == 2
