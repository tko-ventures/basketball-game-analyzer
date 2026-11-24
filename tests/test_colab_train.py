"""Tests for Google Colab training orchestration (``bga.remote.colab_train``)."""

from __future__ import annotations

import base64
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from bga.remote import colab_train
from bga.train.config import BaseWeights, ManifestSource, Platform, Recipe, TrainConfig


def _sample_config(*, gpu: str = "L4", devices: int | None = None, timeout_h: float = 4.0) -> TrainConfig:
    """Build a minimal valid ``TrainConfig`` for unit tests."""
    return TrainConfig(
        name="demo",
        layer="numbers",
        backend="crop_cls",
        base_weights=BaseWeights(name="resnet18", url="https://example.invalid/w", licence="Apache-2.0"),
        sources=[ManifestSource(path="data/labels/numbers/demo.json", sha256="0" * 64)],
        recipe=Recipe(resolution=64, epochs=1, batch_size=8, lr=1e-3, select_on="val/top1", devices=devices),
        platform=Platform(name="colab", gpu=gpu, timeout_h=timeout_h),
        package="timm==1.0.30",
    )


def _poll_line(
    state: str,
    *,
    offset: int = 0,
    chunk: bytes = b"",
    more: bool = False,
    exit_code: int | None = None,
) -> str:
    """Format a synthetic ``_BGA_POLL_JSON_:`` line matching ``_remote_poll_script``."""
    payload = {
        "state": state,
        "offset": offset,
        "data_b64": base64.b64encode(chunk).decode("ascii"),
        "more": more,
        "exit_code": exit_code,
    }
    return f"{colab_train._POLL_MARKER}{json.dumps(payload)}\n"


class FakeColabCli:
    """In-memory stub for the ``colab`` CLI process boundary."""

    def __init__(
        self,
        *,
        usage_outputs: list[str] | None = None,
        remote_files: dict[str, bytes] | None = None,
        fail_on: str | None = None,
        poll_responses: list[str | Exception] | None = None,
        sessions_stdout: str | None = None,
        sessions_returncode: int = 0,
    ) -> None:
        self.calls: list[list[str]] = []
        self.uploaded: dict[str, bytes] = {}
        self.remote_files: dict[str, bytes] = dict(remote_files or {})
        self.usage_outputs = list(usage_outputs or [])
        self.fail_on = fail_on
        self.poll_responses: list[str | Exception] = list(poll_responses or [])
        self.sessions_stdout = sessions_stdout
        self.sessions_returncode = sessions_returncode
        self.last_session: str = ""

    def __call__(
        self,
        cmd: list[str],
        *,
        input: str | None = None,  # noqa: A002  # matches subprocess.run keyword
        capture_output: bool = True,
        text: bool = True,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        args = cmd[1:]
        self.calls.append(args)
        subcmd = args[0]
        if subcmd == "new" and "-s" in args:
            self.last_session = args[args.index("-s") + 1]
        if self.fail_on == subcmd:
            raise subprocess.CalledProcessError(1, cmd)
        if subcmd == "usage":
            out = self.usage_outputs.pop(0) if self.usage_outputs else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        if subcmd == "sessions":
            out = (
                f"[{self.last_session}] gpu-l4-abc | Hardware: L4 | Shape: Standard | Variant: GPU\n"
                if self.sessions_stdout is None
                else self.sessions_stdout
            )
            return subprocess.CompletedProcess(cmd, self.sessions_returncode, stdout=out, stderr="")
        if subcmd == "upload":
            local_path, remote_path = Path(args[3]), args[4]
            self.uploaded[remote_path] = local_path.read_bytes()
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if subcmd == "download":
            remote_path, local_path = args[3], Path(args[4])
            local_path.write_bytes(self.remote_files[remote_path])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if subcmd == "exec" and input and colab_train._POLL_MARKER in input:
            return self._handle_poll(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _handle_poll(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if not self.poll_responses:
            return subprocess.CompletedProcess(cmd, 0, stdout=_poll_line("done", exit_code=0), stderr="")
        nxt = self.poll_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return subprocess.CompletedProcess(cmd, 0, stdout=nxt, stderr="")


@pytest.fixture
def patch_colab(monkeypatch: pytest.MonkeyPatch):
    """Patch ``shutil.which``, ``subprocess.run``, and ``time.sleep`` in ``colab_train``."""

    def _install(fake: FakeColabCli) -> FakeColabCli:
        monkeypatch.setattr(
            colab_train.shutil, "which", lambda name: "/usr/local/bin/colab" if name == "colab" else None
        )
        monkeypatch.setattr(colab_train.subprocess, "run", fake)
        monkeypatch.setattr(colab_train.time, "sleep", lambda _s: None)
        return fake

    return _install


class TestValidateColabGpu:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("t4", "T4", id="t4-lower"),
            pytest.param("L4", "L4", id="l4-upper"),
            pytest.param(" a100 ", "A100", id="a100-padded"),
            pytest.param("g4", "G4", id="g4-lower"),
            pytest.param("H100", "H100", id="h100-upper"),
        ],
    )
    def test_normalises_supported_gpus(self, raw: str, expected: str) -> None:
        assert colab_train.validate_colab_gpu(raw) == expected

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param("L40S", id="modal-only-l40s"),
            pytest.param("H200", id="unsupported-h200"),
            pytest.param("", id="empty"),
        ],
    )
    def test_rejects_unsupported_gpu(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Colab does not offer GPU"):
            colab_train.validate_colab_gpu(bad)


class TestUsageAndCcu:
    def test_parses_colab_usage_stdout(self) -> None:
        raw = "Current balance: 198.45 compute units\nUsage rate: 4.82/hr\nActive assignments: 1\n"
        assert colab_train.parse_usage(raw) == colab_train.UsageInfo(198.45, 4.82, 1)

    def test_returns_none_for_unrecognised_usage_output(self) -> None:
        assert colab_train.parse_usage("not authenticated") is None

    def test_prefers_positive_balance_delta_over_rate_estimate(self) -> None:
        before = colab_train.UsageInfo(200.0, 0.0, 0)
        during = colab_train.UsageInfo(200.0, 4.82, 1)
        after = colab_train.UsageInfo(199.10, 0.0, 0)
        assert colab_train.compute_ccu(before, during, after, 600.0) == (0.9, 4.82)

    def test_falls_back_to_rate_times_seconds_when_balance_unchanged(self) -> None:
        before = colab_train.UsageInfo(200.0, 0.0, 0)
        during = colab_train.UsageInfo(200.0, 4.82, 1)
        after = colab_train.UsageInfo(200.0, 0.0, 0)
        assert colab_train.compute_ccu(before, during, after, 360.0) == (0.482, 4.82)

    def test_returns_none_when_neither_delta_nor_rate_available(self) -> None:
        assert colab_train.compute_ccu(None, None, None, 120.0) == (None, None)


class TestRemoteScriptsSyntax:
    @pytest.mark.parametrize(
        ("label", "source"),
        [
            pytest.param("train", colab_train._remote_train_script(), id="train-script"),
            pytest.param("start", colab_train._remote_start_script(), id="start-script"),
            pytest.param("poll", colab_train._remote_poll_script(0), id="poll-script"),
            pytest.param(
                "prepare-downloads",
                colab_train._remote_prepare_downloads_script(("result.json",), 1024),
                id="prepare-downloads-script",
            ),
        ],
    )
    def test_generated_remote_scripts_compile(self, label: str, source: str) -> None:
        code = compile(source, f"<remote-{label}>", "exec")
        assert code is not None


class TestPackAndChunkTransfer:
    def test_pack_directory_excludes_pycache_and_pyc(self, tmp_path: Path) -> None:
        src = tmp_path / "pkg"
        (src / "__pycache__").mkdir(parents=True)
        (src / "mod.py").write_text("x = 1\n")
        (src / "mod.pyc").write_bytes(b"pyc")
        (src / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"pyc")

        archive = colab_train.pack_directory(src, tmp_path / "pkg.tar.gz", compress=True, arcroot="pkg")
        with tarfile.open(archive) as tf:
            names = sorted(tf.getnames())

        assert names == ["pkg/mod.py"]

    def test_upload_file_sends_single_request_below_chunk_limit(self, tmp_path: Path, patch_colab) -> None:
        fake = patch_colab(FakeColabCli())
        payload = tmp_path / "small.bin"
        payload.write_bytes(b"hello")

        parts = colab_train.upload_file("sess", payload, "content/bga_run/upload/small.bin", chunk_bytes=16)

        assert parts == ["content/bga_run/upload/small.bin"]
        assert fake.uploaded == {"content/bga_run/upload/small.bin": b"hello"}

    def test_upload_file_splits_large_file_into_ordered_parts(self, tmp_path: Path, patch_colab) -> None:
        fake = patch_colab(FakeColabCli())
        payload = tmp_path / "big.bin"
        payload.write_bytes(b"abcdefghij")

        parts = colab_train.upload_file("sess", payload, "content/bga_run/upload/big.bin", chunk_bytes=4)

        assert parts == [
            "content/bga_run/upload/big.bin.part000",
            "content/bga_run/upload/big.bin.part001",
            "content/bga_run/upload/big.bin.part002",
        ]
        assert b"".join(fake.uploaded[p] for p in parts) == b"abcdefghij"

    def test_download_outputs_reassembles_chunked_file(self, tmp_path: Path, patch_colab) -> None:
        manifest = {
            "files": {
                "weights.safetensors": [
                    f"{colab_train.REMOTE_ROOT}/download/weights.safetensors.part000",
                    f"{colab_train.REMOTE_ROOT}/download/weights.safetensors.part001",
                ],
                "result.json": [f"{colab_train.REMOTE_ROOT}/output/result.json"],
            },
            "missing": [],
            "error": None,
        }
        fake = patch_colab(
            FakeColabCli(
                remote_files={
                    f"{colab_train.REMOTE_ROOT}/download/_manifest.json": json.dumps(manifest).encode(),
                    f"{colab_train.REMOTE_ROOT}/download/weights.safetensors.part000": b"part1-",
                    f"{colab_train.REMOTE_ROOT}/download/weights.safetensors.part001": b"part2",
                    f"{colab_train.REMOTE_ROOT}/output/result.json": b'{"seconds": 12.0}',
                }
            )
        )

        out_dir = tmp_path / "out"
        pulled = colab_train.download_outputs("sess", ("weights.safetensors", "result.json"), out_dir, chunk_bytes=6)

        assert pulled["missing"] == []
        assert (out_dir / "weights.safetensors").read_bytes() == b"part1-part2"
        assert json.loads((out_dir / "result.json").read_text()) == {"seconds": 12.0}
        assert fake.calls[0][0] == "exec"


class TestWaitForRemoteTraining:
    def test_streams_multibyte_utf8_chunks_until_done(self, patch_colab, capsys: pytest.CaptureFixture[str]) -> None:
        raw = "┏━┓ ok\n".encode()
        split = 4  # splits the 3-byte box-drawing character across polls
        patch_colab(
            FakeColabCli(
                poll_responses=[
                    _poll_line("running", offset=split, chunk=raw[:split], more=True),
                    _poll_line("done", offset=len(raw), chunk=raw[split:], more=False, exit_code=0),
                ]
            )
        )

        rc = colab_train._wait_for_remote_training("sess", timeout_s=60.0, poll_interval_s=0.0)

        assert rc == 0
        assert capsys.readouterr().out == "┏━┓ ok\n"

    def test_recovers_from_transient_poll_errors_when_session_stays_alive(
        self, patch_colab, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeColabCli(
            poll_responses=[
                subprocess.CalledProcessError(1, ["colab", "exec"]),
                subprocess.TimeoutExpired(["colab", "exec"], 120.0),
                _poll_line("done", offset=5, chunk=b"done\n", exit_code=0),
            ],
            sessions_stdout="[sess] gpu-l4-abc | Hardware: L4 | Shape: Standard | Variant: GPU\n",
        )
        patch_colab(fake)

        rc = colab_train._wait_for_remote_training("sess", timeout_s=60.0, poll_interval_s=0.0, max_poll_errors=5)

        assert rc == 0
        assert capsys.readouterr().out == "done\n"
        assert sum(call[0] == "sessions" for call in fake.calls) == 2

    def test_fails_fast_when_session_disappears_from_server(self, patch_colab) -> None:
        fake = FakeColabCli(
            poll_responses=[subprocess.CalledProcessError(1, ["colab", "exec"])],
            sessions_stdout="[colab] No active sessions found on server.\n",
        )
        patch_colab(fake)

        with pytest.raises(RuntimeError, match="no longer active on server"):
            colab_train._wait_for_remote_training("sess", timeout_s=60.0, poll_interval_s=0.0, max_poll_errors=10)

        assert sum(call[0] == "exec" for call in fake.calls) == 1

    def test_aborts_after_consecutive_network_errors(self, patch_colab) -> None:
        fake = FakeColabCli(
            poll_responses=[
                subprocess.CalledProcessError(1, ["colab", "exec"]),
                subprocess.CalledProcessError(1, ["colab", "exec"]),
                subprocess.CalledProcessError(1, ["colab", "exec"]),
            ],
            sessions_returncode=1,
        )
        patch_colab(fake)

        with pytest.raises(RuntimeError, match="3 consecutive poll failures"):
            colab_train._wait_for_remote_training("sess", timeout_s=60.0, poll_interval_s=0.0, max_poll_errors=3)

    def test_raises_when_remote_process_crashes_without_exit_code(self, patch_colab) -> None:
        patch_colab(FakeColabCli(poll_responses=[_poll_line("crashed", offset=0, chunk=b"")]))

        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            colab_train._wait_for_remote_training("sess", timeout_s=60.0, poll_interval_s=0.0)

    def test_raises_timeout_when_deadline_exceeded(self, patch_colab, monkeypatch: pytest.MonkeyPatch) -> None:
        ticks = iter([1000.0, 1005.0, 2000.0])
        monkeypatch.setattr(colab_train.time, "time", lambda: next(ticks))
        patch_colab(FakeColabCli(poll_responses=[_poll_line("running"), _poll_line("running")]))

        with pytest.raises(TimeoutError, match="exceeded"):
            colab_train._wait_for_remote_training("sess", timeout_s=10.0, poll_interval_s=0.0)


class TestLaunchColab:
    def test_stops_session_and_returns_result_with_ccu(self, tmp_path: Path, patch_colab) -> None:
        ds = tmp_path / "dataset"
        ds.mkdir()
        (ds / "sample.txt").write_text("ok")
        out = tmp_path / "output"
        manifest = {
            "files": {
                "weights.safetensors": [f"{colab_train.REMOTE_ROOT}/output/weights.safetensors"],
                "metrics.csv": [f"{colab_train.REMOTE_ROOT}/output/metrics.csv"],
                "result.json": [f"{colab_train.REMOTE_ROOT}/output/result.json"],
            },
            "missing": [],
            "error": None,
        }
        fake = patch_colab(
            FakeColabCli(
                usage_outputs=[
                    "Current balance: 200.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n",
                    "Current balance: 200.00 compute units\nUsage rate: 4.82/hr\nActive assignments: 1\n",
                    "Current balance: 199.50 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n",
                ],
                remote_files={
                    f"{colab_train.REMOTE_ROOT}/download/_manifest.json": json.dumps(manifest).encode(),
                    f"{colab_train.REMOTE_ROOT}/output/weights.safetensors": b"weights",
                    f"{colab_train.REMOTE_ROOT}/output/metrics.csv": b"epoch,val/top1\n0,0.9\n",
                    f"{colab_train.REMOTE_ROOT}/output/result.json": json.dumps({
                        "gpu": "L4",
                        "seconds": 360.0,
                        "commit": "abc1234",
                    }).encode(),
                },
            )
        )
        spec = colab_train.ColabJobSpec(
            cfg=_sample_config(gpu="L4"),
            trainer_name="train_numbers",
            dataset_dir=ds,
            output_dir=out,
            pull_files=("weights.safetensors", "metrics.csv", "result.json"),
            gpu="L4",
            commit="abc1234",
        )

        result = colab_train.launch_colab(spec)

        assert result["platform"] == "colab"
        assert result["ccu"] == 0.5
        assert result["ccu_rate"] == 4.82
        assert result["ccu_balance_after"] == 199.5
        assert f"{colab_train.REMOTE_ROOT}/upload/run_train.py" in fake.uploaded
        assert any(call[0] == "stop" for call in fake.calls)
        assert "--high-mem" not in next(call for call in fake.calls if call[0] == "new")

    def test_requests_high_mem_for_a100(self, tmp_path: Path, patch_colab) -> None:
        ds = tmp_path / "dataset"
        ds.mkdir()
        (ds / "sample.txt").write_text("ok")
        out = tmp_path / "output"
        manifest = {
            "files": {"result.json": [f"{colab_train.REMOTE_ROOT}/output/result.json"]},
            "missing": [],
            "error": None,
        }
        fake = patch_colab(
            FakeColabCli(
                remote_files={
                    f"{colab_train.REMOTE_ROOT}/download/_manifest.json": json.dumps(manifest).encode(),
                    f"{colab_train.REMOTE_ROOT}/output/result.json": b'{"gpu": "A100", "seconds": 60.0}',
                }
            )
        )
        spec = colab_train.ColabJobSpec(
            cfg=_sample_config(gpu="A100"),
            trainer_name="train_numbers",
            dataset_dir=ds,
            output_dir=out,
            pull_files=("result.json",),
            gpu="a100",
            commit="abc1234",
        )

        colab_train.launch_colab(spec)

        new_call = next(call for call in fake.calls if call[0] == "new")
        assert "--high-mem" in new_call
        assert "A100" in new_call

    def test_stops_session_even_when_exec_raises(self, tmp_path: Path, patch_colab) -> None:
        ds = tmp_path / "dataset"
        ds.mkdir()
        fake = patch_colab(FakeColabCli(fail_on="exec"))
        spec = colab_train.ColabJobSpec(
            cfg=_sample_config(gpu="L4"),
            trainer_name="train_numbers",
            dataset_dir=ds,
            output_dir=tmp_path / "output",
            pull_files=("result.json",),
            gpu="L4",
            commit="abc1234",
        )

        with pytest.raises(subprocess.CalledProcessError):
            colab_train.launch_colab(spec)

        assert any(call[0] == "stop" for call in fake.calls)

    def test_raises_when_remote_writes_error_file_despite_zero_exit(self, tmp_path: Path, patch_colab) -> None:
        ds = tmp_path / "dataset"
        ds.mkdir()
        manifest = {"files": {}, "missing": ["result.json"], "error": "RuntimeError: boom"}
        fake = patch_colab(
            FakeColabCli(
                remote_files={f"{colab_train.REMOTE_ROOT}/download/_manifest.json": json.dumps(manifest).encode()}
            )
        )
        spec = colab_train.ColabJobSpec(
            cfg=_sample_config(gpu="L4"),
            trainer_name="train_numbers",
            dataset_dir=ds,
            output_dir=tmp_path / "output",
            pull_files=("result.json",),
            gpu="L4",
            commit="abc1234",
        )

        with pytest.raises(RuntimeError, match="RuntimeError: boom"):
            colab_train.launch_colab(spec)

        assert any(call[0] == "stop" for call in fake.calls)

    def test_rejects_multi_gpu_recipe_before_creating_session(self, tmp_path: Path, patch_colab) -> None:
        fake = patch_colab(FakeColabCli())
        spec = colab_train.ColabJobSpec(
            cfg=_sample_config(gpu="L4", devices=2),
            trainer_name="train_detect",
            dataset_dir=tmp_path / "dataset",
            output_dir=tmp_path / "output",
            pull_files=("result.json",),
            gpu="L4",
            commit="abc1234",
        )

        with pytest.raises(ValueError, match="single GPU"):
            colab_train.launch_colab(spec)

        assert fake.calls == []
