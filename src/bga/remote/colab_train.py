"""``bga train`` on Google Colab GPUs via ``google-colab-cli`` (``colab``).

Provisions an ephemeral Colab GPU runtime, uploads the local ``bga`` package and built dataset (chunked so
large archives stay below Jupyter's Contents API body limit), starts :func:`bga.train.dispatch.run_trainer`
detached on the VM so laptop sleep, transient network drops, or proxy-token expiry do not kill training,
polls status and streams remote logs over short reconnecting calls, downloads the collected checkpoints and
logs into ``runs/train/<id>/output/``, and always terminates the session in ``finally`` so idle runtimes
never burn Compute Units.

"""

from __future__ import annotations

import base64
import codecs
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bga.train.config import ROOT, TrainConfig
from bga.train.dispatch import PARSEQ_PACKAGE, REID_PACKAGE, RFDETR_PACKAGE, RFDETR_PLUS_PACKAGE

log = logging.getLogger("bga")

#: GPU accelerators accepted by ``colab new --gpu`` (uppercase canonical names).
COLAB_GPUS: tuple[str, ...] = ("T4", "L4", "G4", "A100", "H100")
#: Accelerators where ``--high-mem`` is requested (``L4`` has a single high-RAM shape already).
HIGH_MEM_GPUS: frozenset[str] = frozenset({"A100"})
#: Raw bytes per chunk when transferring large archives/checkpoints over Jupyter's Contents API (~20 MiB base64).
CHUNK_BYTES: int = 15 * 1024 * 1024
#: Working directory on the Colab VM (relative to Jupyter's root ``/``).
REMOTE_ROOT: str = "content/bga_run"
#: Seconds between status/log polls while detached training runs on Colab.
POLL_INTERVAL_S: float = 15.0
#: Maximum raw bytes read from ``train.log`` per poll call.
POLL_LOG_CHUNK_BYTES: int = 262_144
#: Consecutive transient poll errors tolerated before aborting (covers laptop sleep / Wi-Fi reconnect).
MAX_POLL_ERRORS: int = 12
#: Maximum attempts for post-training download steps so a transient blip cannot lose finished weights.
DOWNLOAD_RETRIES: int = 4
#: Base delay (seconds) before retrying a failed post-training download call.
DOWNLOAD_RETRY_DELAY_S: float = 5.0
#: Extra packages to install per trainer on top of Colab's preinstalled CUDA PyTorch environment.
SETUP_PACKAGES: dict[str, tuple[str, ...]] = {
    "train_keypoint": (RFDETR_PACKAGE, RFDETR_PLUS_PACKAGE),
    "train_detect": (RFDETR_PACKAGE, RFDETR_PLUS_PACKAGE),
    "train_appearance": (REID_PACKAGE,),
    "train_numbers": (),
    "train_parseq": ("pytorch-lightning==2.6.6", "nltk==3.10.4", PARSEQ_PACKAGE),
}

_BALANCE_RE = re.compile(r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)\s+compute units")
_RATE_RE = re.compile(r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)/hr")
_ASSIGNMENTS_RE = re.compile(r"Active assignments:\s*([0-9]+)")
_POLL_MARKER = "_BGA_POLL_JSON_:"


@dataclass(frozen=True)
class UsageInfo:
    """Compute-unit balance, burn rate and active assignments parsed from ``colab usage``."""

    balance: float
    rate_per_hour: float
    active_assignments: int


@dataclass(frozen=True)
class ColabJobSpec:
    """Inputs for a single Colab training execution."""

    cfg: TrainConfig
    trainer_name: str
    dataset_dir: Path
    output_dir: Path
    pull_files: tuple[str, ...]
    gpu: str
    commit: str
    init_file: Path | None = None


@dataclass(frozen=True)
class _PollResult:
    """Status and log delta returned by a single remote poll."""

    state: Literal["running", "done", "crashed"]
    offset: int
    chunk: bytes
    more: bool
    exit_code: int | None


def validate_colab_gpu(gpu: str) -> str:
    """Normalise and validate a GPU name for ``colab new --gpu``.

    Args:
        gpu: Requested GPU name (case-insensitive, e.g. ``l4``, ``A100``).

    Returns:
        Canonical uppercase Colab GPU name.

    Raises:
        ValueError: When the GPU is not offered by Colab (e.g. Modal's ``L40S``).

    Examples:
        >>> validate_colab_gpu("l4")
        'L4'
        >>> validate_colab_gpu("L40S")
        Traceback (most recent call last):
            ...
        ValueError: Colab does not offer GPU 'L40S' (supported: T4, L4, G4, A100, H100); pass --gpu, e.g. --gpu L4

    """
    upper = gpu.strip().upper()
    if upper not in COLAB_GPUS:
        supported = ", ".join(COLAB_GPUS)
        raise ValueError(f"Colab does not offer GPU {gpu!r} (supported: {supported}); pass --gpu, e.g. --gpu L4")
    return upper


def parse_usage(text: str) -> UsageInfo | None:
    r"""Parse the stdout of ``colab usage`` into a :class:`UsageInfo`.

    Args:
        text: Output text from ``colab usage``.

    Returns:
        Parsed usage info, or ``None`` when the expected lines are absent.

    Examples:
        >>> parse_usage("Current balance: 200.00 compute units\nUsage rate: 4.82/hr\nActive assignments: 1\n")
        UsageInfo(balance=200.0, rate_per_hour=4.82, active_assignments=1)
        >>> parse_usage("unexpected output") is None
        True

    """
    bal = _BALANCE_RE.search(text)
    rate = _RATE_RE.search(text)
    active = _ASSIGNMENTS_RE.search(text)
    if not (bal and rate and active):
        return None
    return UsageInfo(
        balance=float(bal.group(1)),
        rate_per_hour=float(rate.group(1)),
        active_assignments=int(active.group(1)),
    )


def compute_ccu(
    before: UsageInfo | None,
    during: UsageInfo | None,
    after: UsageInfo | None,
    seconds: float,
) -> tuple[float | None, float | None]:
    """Compute Compute Units spent and the observed hourly rate from ``colab usage`` snapshots.

    Prefers the actual balance delta reported by Colab when positive; falls back to Colab's reported hourly rate
    times trainer GPU seconds when the balance tick has not settled yet.

    Args:
        before: Usage snapshot before creating the session.
        during: Usage snapshot while the GPU session is active.
        after: Usage snapshot after stopping the session.
        seconds: Trainer GPU seconds recorded inside the remote run.

    Returns:
        ``(ccu_spent, rate_per_hour)``, each ``None`` when unavailable.

    Examples:
        >>> b = UsageInfo(200.0, 0.0, 0)
        >>> d = UsageInfo(200.0, 4.82, 1)
        >>> a = UsageInfo(199.15, 0.0, 0)
        >>> compute_ccu(b, d, a, 600.0)
        (0.85, 4.82)
        >>> compute_ccu(b, d, UsageInfo(200.0, 0.0, 0), 360.0)
        (0.482, 4.82)

    """
    rate = during.rate_per_hour if during is not None and during.rate_per_hour > 0 else None
    if before is not None and after is not None:
        delta = round(before.balance - after.balance, 4)
        if delta > 0:
            return delta, rate
    if rate is not None and seconds > 0:
        return round(seconds / 3600.0 * rate, 4), rate
    return None, rate


def _colab_bin() -> str:
    """Resolve the ``colab`` CLI binary on ``PATH``."""
    exe = shutil.which("colab")
    if exe is None:
        raise RuntimeError(
            "google-colab-cli (`colab`) is not on PATH; install with `uv tool install google-colab-cli` "
            "and authenticate with `colab auth`"
        )
    return exe


def _run_colab(
    args: Sequence[str],
    *,
    capture: bool = True,
    check: bool = True,
    input_text: str | None = None,
    cmd_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the ``colab`` CLI with fixed argument lists."""
    cmd = [_colab_bin(), *args]
    kwargs: dict[str, Any] = {
        "input": input_text,
        "capture_output": capture,
        "text": True,
        "check": check,
    }
    if cmd_timeout is not None:
        kwargs["timeout"] = cmd_timeout
    return subprocess.run(cmd, **kwargs)  # noqa: S603  # resolved colab CLI executable with controlled arguments


def _staging_tempdir(preferred_dir: Path | None = None) -> tempfile.TemporaryDirectory[str]:
    """Create a temporary staging directory, resilient to a deleted inherited ``TMPDIR``."""
    if preferred_dir is not None:
        preferred_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=".bga-tmp-", dir=preferred_dir)
    default_tmp = Path(tempfile.gettempdir())
    if default_tmp.is_dir():
        return tempfile.TemporaryDirectory(prefix="bga-colab-")
    fallback = Path.home() / ".cache" / "bga" / "tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="bga-colab-", dir=fallback)


def get_usage() -> UsageInfo | None:
    """Query Colab Compute Unit balance and rate via ``colab usage``.

    Returns:
        Parsed :class:`UsageInfo`, or ``None`` if the command fails.

    Examples:
        >>> get_usage()  # doctest: +SKIP
        UsageInfo(balance=200.0, rate_per_hour=0.0, active_assignments=0)

    """
    try:
        proc = _run_colab(["usage"], capture=True, check=False)
    except RuntimeError:
        return None
    if proc.returncode != 0:
        return None
    return parse_usage(proc.stdout)


def pack_directory(src_dir: Path, archive_path: Path, *, compress: bool = False, arcroot: str = "") -> Path:
    r"""Pack a directory into a tar archive, skipping ``__pycache__`` and ``.pyc`` files.

    Args:
        src_dir: Directory to pack.
        archive_path: Destination ``.tar`` or ``.tar.gz`` path.
        compress: Write gzip-compressed tar when True.
        arcroot: Optional top-level folder name inside the archive.

    Returns:
        The written archive path.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     src = Path(tmp) / "pkg"
        ...     src.mkdir()
        ...     _ = (src / "a.py").write_text("x = 1\n")
        ...     out = pack_directory(src, Path(tmp) / "pkg.tar.gz", compress=True, arcroot="pkg")
        ...     with tarfile.open(out) as tf:
        ...         sorted(tf.getnames())
        ['pkg/a.py']

    """
    mode = "w:gz" if compress else "w"
    with tarfile.open(archive_path, mode) as tf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src_dir)
            if "__pycache__" in rel.parts or path.suffix == ".pyc":
                continue
            arcname = f"{arcroot}/{rel.as_posix()}" if arcroot else rel.as_posix()
            tf.add(path, arcname=arcname)
    return archive_path


def upload_file(
    session: str,
    local_path: Path,
    remote_path: str,
    *,
    chunk_bytes: int = CHUNK_BYTES,
) -> list[str]:
    """Upload a local file to Colab, splitting into chunks when larger than ``chunk_bytes``.

    Args:
        session: Active Colab session name.
        local_path: Local file to upload.
        remote_path: Target path on the VM (relative to Jupyter root, e.g. ``content/bga_run/upload/a.tar``).
        chunk_bytes: Maximum raw bytes per single ``colab upload`` call.

    Returns:
        Remote paths uploaded: ``[remote_path]`` for a single upload, or the ordered part paths when chunked.

    Examples:
        >>> upload_file("bga-demo", Path("a.tar"), "content/bga_run/upload/a.tar")  # doctest: +SKIP
        ['content/bga_run/upload/a.tar']

    """
    size = local_path.stat().st_size
    if size <= chunk_bytes:
        _run_colab(["upload", "-s", session, str(local_path), remote_path])
        return [remote_path]
    parts: list[str] = []
    with _staging_tempdir() as tmp, local_path.open("rb") as src:
        idx = 0
        while True:
            data = src.read(chunk_bytes)
            if not data:
                break
            part_local = Path(tmp) / f"{local_path.name}.part{idx:03d}"
            part_local.write_bytes(data)
            part_remote = f"{remote_path}.part{idx:03d}"
            _run_colab(["upload", "-s", session, str(part_local), part_remote])
            parts.append(part_remote)
            part_local.unlink()
            idx += 1
    return parts


def _exec_code(
    session: str,
    code: str,
    *,
    timeout: float = 0.0,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a Python snippet on a Colab session via stdin."""
    return _run_colab(
        ["exec", "-s", session, "--timeout", str(timeout)],
        capture=not stream,
        check=True,
        input_text=code,
        cmd_timeout=timeout + 60.0 if timeout > 0 else None,
    )


def _remote_train_script() -> str:
    """Return the self-contained Python trainer script run detached on the Colab GPU VM."""
    return f"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path("/{REMOTE_ROOT}")
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "_ok.txt").unlink(missing_ok=True)
(OUT / "_error.txt").unlink(missing_ok=True)
(OUT / "_exit_code.txt").unlink(missing_ok=True)


def _reassemble(chunks: dict) -> None:
    for target_rel, part_rels in chunks.items():
        if len(part_rels) == 1 and part_rels[0] == target_rel:
            continue
        dst = Path("/") / target_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as out_f:
            for part_rel in part_rels:
                part = Path("/") / part_rel
                with part.open("rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)
                part.unlink()


def _install_packages(packages: list[str]) -> None:
    missing_base = [
        pkg for mod, pkg in (("yaml", "pyyaml"), ("pydantic", "pydantic"), ("cv2", "opencv-python-headless"),
                             ("timm", "timm"), ("safetensors", "safetensors"))
        if importlib.util.find_spec(mod) is None
    ]
    to_install = [*missing_base, *packages]
    if not to_install:
        return
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--system", *to_install]
        if uv
        else [sys.executable, "-m", "pip", "install", *to_install]
    )
    print("[bga-colab] installing:", " ".join(to_install), flush=True)
    subprocess.run(cmd, check=True)


exit_code = 1
try:
    meta = json.loads((ROOT / "upload" / "job.json").read_text())
    _reassemble(meta.get("chunks", {{}}))

    src_dir = ROOT / "src"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    src_tar = ROOT / "upload" / "bga_src.tar.gz"
    with tarfile.open(src_tar, "r:gz") as tf:
        tf.extractall(src_dir, filter="data")
    src_tar.unlink()

    ds_dir = ROOT / "dataset"
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds_tar = ROOT / "upload" / "dataset.tar"
    with tarfile.open(ds_tar, "r") as tf:
        tf.extractall(ds_dir, filter="data")
    ds_tar.unlink()

    _install_packages(meta.get("packages", []))

    import torch

    print(
        f"[bga-colab] torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} "
        f"device={{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Colab session")

    sys.path.insert(0, str(src_dir))
    from bga.train.dispatch import run_trainer

    init_rel = meta.get("init_rel")
    init_path = Path("/") / init_rel if init_rel else None
    workers = min(os.cpu_count() or 2, 8)
    result = run_trainer(
        meta["trainer"],
        ds_dir,
        OUT,
        meta["job"],
        num_workers=workers,
        init_path=init_path,
    )
    result["platform"] = "colab"
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    (OUT / "_ok.txt").write_text("ok\\n")
    print("[bga-colab] training finished cleanly", flush=True)
    exit_code = 0
except BaseException:
    tb = traceback.format_exc()
    (OUT / "_error.txt").write_text(tb)
    print(tb, file=sys.stderr, flush=True)
    raise
finally:
    (OUT / "_exit_code.txt").write_text(f"{{exit_code}}\\n")
""".lstrip()


def _remote_start_script() -> str:
    """Return a short snippet that spawns ``upload/run_train.py`` detached from the Jupyter kernel."""
    return f"""
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path("/{REMOTE_ROOT}")
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
for name in ("_ok.txt", "_error.txt", "_exit_code.txt", "_pid.txt"):
    (OUT / name).unlink(missing_ok=True)
log_path = ROOT / "train.log"
log_path.write_text("")
script = ROOT / "upload" / "run_train.py"
pid_path = OUT / "_pid.txt"
cmd = (
    f"nohup {{shlex.quote(sys.executable)}} -u {{shlex.quote(str(script))}} "
    f"> {{shlex.quote(str(log_path))}} 2>&1 </dev/null & echo $! > {{shlex.quote(str(pid_path))}}"
)
subprocess.run(["/bin/sh", "-c", cmd], check=True, start_new_session=True)
""".lstrip()


def _remote_poll_script(offset: int, *, max_bytes: int = POLL_LOG_CHUNK_BYTES) -> str:
    """Return a short snippet that reads new log bytes and detached-trainer status."""
    return f"""
import base64
import json
import os
from pathlib import Path

ROOT = Path("/{REMOTE_ROOT}")
OUT = ROOT / "output"
log_path = ROOT / "train.log"
offset = {offset}
max_bytes = {max_bytes}

data = b""
more = False
if log_path.exists():
    size = log_path.stat().st_size
    if size > offset:
        with log_path.open("rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
        more = size > offset + len(data)


def _pid_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{{pid}}/stat")
    if stat_path.exists():
        try:
            text = stat_path.read_text()
            state = text[text.rfind(")") + 2 : text.rfind(")") + 3]
            return state != "Z"
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


exit_path = OUT / "_exit_code.txt"
pid_path = OUT / "_pid.txt"
exit_code = None
if exit_path.exists():
    state = "done"
    try:
        exit_code = int(exit_path.read_text().strip())
    except ValueError:
        exit_code = 1
elif pid_path.exists():
    try:
        pid = int(pid_path.read_text().strip())
        state = "running" if _pid_alive(pid) else "crashed"
    except ValueError:
        state = "crashed"
else:
    state = "crashed"

payload = {{
    "state": state,
    "offset": offset + len(data),
    "data_b64": base64.b64encode(data).decode("ascii"),
    "more": more,
    "exit_code": exit_code,
}}
print("{_POLL_MARKER}" + json.dumps(payload), flush=True)
""".lstrip()


def _parse_poll_output(stdout: str) -> _PollResult:
    """Parse the ``_BGA_POLL_JSON_:`` payload emitted by :func:`_remote_poll_script`."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(_POLL_MARKER):
            continue
        raw: dict[str, Any] = json.loads(line[len(_POLL_MARKER) :])
        return _PollResult(
            state=raw["state"],
            offset=int(raw["offset"]),
            chunk=base64.b64decode(raw["data_b64"]),
            more=bool(raw["more"]),
            exit_code=int(raw["exit_code"]) if raw.get("exit_code") is not None else None,
        )
    raise RuntimeError(f"Colab status poll returned no {_POLL_MARKER} line")


def _session_alive(session: str) -> bool | None:
    """Check whether ``session`` is still listed on Colab (also refreshes cached proxy tokens).

    Returns ``True`` when listed, ``False`` when the server confirms it is gone, or ``None`` when the check
    itself failed (e.g. local network offline).

    """
    try:
        proc = _run_colab(["sessions"], capture=True, check=False, cmd_timeout=60.0)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    tag = f"[{session}]"
    return any(line.split()[:1] == [tag] for line in proc.stdout.splitlines())


def _handle_poll_error(
    session: str,
    exc: subprocess.SubprocessError,
    consecutive_errors: int,
    max_poll_errors: int,
) -> int:
    """Classify a failed poll: fail fast if the VM vanished, otherwise allow transient retries."""
    if _session_alive(session) is False:
        raise RuntimeError(f"Colab session {session} is no longer active on server") from exc
    consecutive_errors += 1
    if consecutive_errors >= max_poll_errors:
        raise RuntimeError(
            f"lost connection to Colab session {session} after {consecutive_errors} consecutive poll failures"
        ) from exc
    log.warning(
        "transient Colab poll error on %s (%d/%d); retrying in %.0fs",
        session,
        consecutive_errors,
        max_poll_errors,
        POLL_INTERVAL_S,
    )
    return consecutive_errors


def _wait_for_remote_training(
    session: str,
    *,
    timeout_s: float,
    poll_interval_s: float = POLL_INTERVAL_S,
    max_poll_errors: int = MAX_POLL_ERRORS,
) -> int:
    """Poll detached remote training until completion, streaming ``train.log`` to stdout."""
    deadline = time.time() + timeout_s
    offset = 0
    consecutive_errors = 0
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    while True:
        try:
            proc = _exec_code(session, _remote_poll_script(offset), timeout=60.0, stream=False)
            poll = _parse_poll_output(proc.stdout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            consecutive_errors = _handle_poll_error(session, exc, consecutive_errors, max_poll_errors)
            if time.time() >= deadline:
                raise TimeoutError(f"Colab training on {session} exceeded {timeout_s / 3600:.1f}h timeout") from exc
            time.sleep(poll_interval_s)
            continue

        consecutive_errors = 0
        offset = poll.offset
        if poll.chunk:
            sys.stdout.write(decoder.decode(poll.chunk))
            sys.stdout.flush()

        if poll.more:
            continue
        if poll.state == "done":
            tail = decoder.decode(b"", final=True)
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
            return 1 if poll.exit_code is None else poll.exit_code
        if poll.state == "crashed":
            tail = decoder.decode(b"", final=True)
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
            raise RuntimeError(
                f"Colab trainer process on {session} exited unexpectedly without writing an exit code "
                "(possible OOM or SIGKILL)"
            )
        if time.time() >= deadline:
            raise TimeoutError(f"Colab training on {session} exceeded {timeout_s / 3600:.1f}h timeout")
        time.sleep(poll_interval_s)


def _retry_transfer[T](
    session: str,
    label: str,
    action: Callable[[], T],
    *,
    max_attempts: int = DOWNLOAD_RETRIES,
    retry_delay_s: float = DOWNLOAD_RETRY_DELAY_S,
) -> T:
    """Retry a post-training Colab RPC or download on transient proxy/network errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt >= max_attempts:
                raise
            if _session_alive(session) is False:
                raise RuntimeError(f"Colab session {session} is no longer active on server during {label}") from exc
            delay = retry_delay_s * attempt
            log.warning(
                "transient Colab error during %s on %s (%d/%d); retrying in %.0fs",
                label,
                session,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _remote_prepare_downloads_script(filenames: Sequence[str], chunk_bytes: int) -> str:
    """Return a remote script that chunks oversized outputs and writes ``download/_manifest.json``."""
    names_json = json.dumps(list(filenames))
    return f"""
import json
import shutil
from pathlib import Path

ROOT = Path("/{REMOTE_ROOT}")
OUT = ROOT / "output"
DL = ROOT / "download"
if DL.exists():
    shutil.rmtree(DL)
DL.mkdir(parents=True, exist_ok=True)
CHUNK = {chunk_bytes}

manifest = {{"files": {{}}, "missing": [], "error": None}}
err = OUT / "_error.txt"
if err.exists():
    manifest["error"] = err.read_text()

for name in {names_json}:
    src = OUT / name
    if not src.exists():
        manifest["missing"].append(name)
        continue
    if src.stat().st_size <= CHUNK:
        manifest["files"][name] = [f"{REMOTE_ROOT}/output/{{name}}"]
        continue
    parts = []
    with src.open("rb") as f:
        idx = 0
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            part_name = f"{{name}}.part{{idx:03d}}"
            (DL / part_name).write_bytes(buf)
            parts.append(f"{REMOTE_ROOT}/download/{{part_name}}")
            idx += 1
    manifest["files"][name] = parts

(DL / "_manifest.json").write_text(json.dumps(manifest, indent=1))
""".lstrip()


def _download_one(session: str, remote_path: str, local_path: Path) -> None:
    """Download a single remote path into ``local_path``, retrying transient proxy failures."""

    def _pull() -> None:
        local_path.unlink(missing_ok=True)
        _run_colab(["download", "-s", session, remote_path, str(local_path)], cmd_timeout=180.0)

    _retry_transfer(session, f"download {remote_path}", _pull)


def download_outputs(
    session: str,
    filenames: Sequence[str],
    local_dir: Path,
    *,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict[str, Any]:
    """Download trainer outputs from Colab into ``local_dir``, chunking files larger than ``chunk_bytes``.

    Args:
        session: Active Colab session name.
        filenames: Output file names under ``/content/bga_run/output/`` to pull.
        local_dir: Local destination directory.
        chunk_bytes: Chunk threshold in raw bytes.

    Returns:
        The parsed download manifest (``files``, ``missing``, ``error``).

    Examples:
        >>> download_outputs("bga-demo", ("result.json",), Path("/tmp/out"))  # doctest: +SKIP
        {'files': {'result.json': ['content/bga_run/output/result.json']}, 'missing': [], 'error': None}

    """
    local_dir.mkdir(parents=True, exist_ok=True)
    _retry_transfer(
        session,
        "prepare downloads",
        lambda: _exec_code(session, _remote_prepare_downloads_script(filenames, chunk_bytes), timeout=120.0),
    )
    manifest_local = local_dir / "_manifest.json"
    _download_one(session, f"{REMOTE_ROOT}/download/_manifest.json", manifest_local)
    manifest: dict[str, Any] = json.loads(manifest_local.read_text())
    manifest_local.unlink(missing_ok=True)

    with _staging_tempdir(local_dir) as tmp:
        tmp_dir = Path(tmp)
        for name, remote_parts in manifest.get("files", {}).items():
            dst = local_dir / name
            if len(remote_parts) == 1:
                _download_one(session, remote_parts[0], dst)
                continue
            staged = tmp_dir / name
            with staged.open("wb") as out_f:
                for idx, part_remote in enumerate(remote_parts):
                    part_local = tmp_dir / f"{name}.part{idx:03d}"
                    _download_one(session, part_remote, part_local)
                    out_f.write(part_local.read_bytes())
                    part_local.unlink()
            staged.replace(dst)
    return manifest


def stop_session(session: str) -> None:
    """Terminate a Colab session so it stops consuming Compute Units.

    Args:
        session: Colab session name to terminate.

    Examples:
        >>> stop_session("bga-demo")  # doctest: +SKIP

    """
    try:
        _run_colab(["stop", "-s", session], capture=True, check=False)
    except Exception as exc:  # cleanup in finally must not mask training errors
        log.warning("failed to stop Colab session %s: %s", session, exc)


def _upload_inputs(session: str, spec: ColabJobSpec) -> dict[str, Any]:
    """Pack and upload ``src/bga``, dataset, optional ``init.pth``, ``run_train.py``, and ``job.json``."""
    _exec_code(
        session,
        f"from pathlib import Path\n"
        f"for sub in ('upload', 'src', 'dataset', 'output', 'download'):\n"
        f"    (Path('/{REMOTE_ROOT}') / sub).mkdir(parents=True, exist_ok=True)\n",
        timeout=60.0,
    )
    chunks: dict[str, list[str]] = {}
    with _staging_tempdir() as tmp:
        tmp_dir = Path(tmp)
        src_tar = pack_directory(ROOT / "src/bga", tmp_dir / "bga_src.tar.gz", compress=True, arcroot="bga")
        src_remote = f"{REMOTE_ROOT}/upload/bga_src.tar.gz"
        chunks[src_remote] = upload_file(session, src_tar, src_remote)

        ds_tar = pack_directory(spec.dataset_dir, tmp_dir / "dataset.tar", compress=False)
        ds_remote = f"{REMOTE_ROOT}/upload/dataset.tar"
        log.info("uploading dataset archive (%.1f MiB) to Colab session %s", ds_tar.stat().st_size / 1e6, session)
        chunks[ds_remote] = upload_file(session, ds_tar, ds_remote)

        init_rel: str | None = None
        if spec.init_file is not None:
            init_rel = f"{REMOTE_ROOT}/upload/init.pth"
            log.info("uploading warm-start weights %s to Colab session %s", spec.init_file.name, session)
            chunks[init_rel] = upload_file(session, spec.init_file, init_rel)

        script_local = tmp_dir / "run_train.py"
        script_local.write_text(_remote_train_script())
        upload_file(session, script_local, f"{REMOTE_ROOT}/upload/run_train.py")

        meta = {
            "trainer": spec.trainer_name,
            "packages": list(SETUP_PACKAGES.get(spec.trainer_name, ())),
            "chunks": chunks,
            "init_rel": init_rel,
            "job": {
                "commit": spec.commit,
                "recipe": spec.cfg.recipe.model_dump(),
                "gpu": spec.gpu,
                "base": spec.cfg.base_weights.name,
            },
        }
        job_local = tmp_dir / "job.json"
        job_local.write_text(json.dumps(meta, indent=1))
        upload_file(session, job_local, f"{REMOTE_ROOT}/upload/job.json")
    return meta


def launch_colab(spec: ColabJobSpec) -> dict[str, Any]:
    """Run a full training job on a Google Colab GPU and download outputs locally.

    Creates a Colab session, uploads code and dataset, starts the trainer detached on the VM, polls status and
    streams logs across reconnecting calls (tolerating laptop sleep or transient network drops), pulls the
    setup's output files into ``spec.output_dir``, and always terminates the session before returning.

    Args:
        spec: Job configuration, trainer name, paths, validated GPU, commit and optional warm-start weights.

    Returns:
        The trainer result dictionary augmented with ``platform="colab"``, ``ccu``, ``ccu_rate``, and
        ``ccu_balance_after`` (when available from ``colab usage``).

    Raises:
        ValueError: When the requested GPU is unsupported on Colab or ``recipe.devices > 1``.
        RuntimeError: When the ``colab`` CLI is missing, the session vanishes, or remote training fails.
        TimeoutError: When remote training exceeds ``spec.cfg.platform.timeout_h``.

    Examples:
        >>> launch_colab(spec)["platform"]  # doctest: +SKIP
        'colab'

    """
    gpu = validate_colab_gpu(spec.gpu)
    if (spec.cfg.recipe.devices or 1) > 1:
        raise ValueError("Colab runtimes provide a single GPU (recipe.devices > 1 requires --platform modal)")

    result_path = spec.output_dir / "result.json"
    result_path.unlink(missing_ok=True)

    session = f"bga-{spec.cfg.config_hash()}-{uuid.uuid4().hex[:4]}"
    usage_before = get_usage()
    usage_during: UsageInfo | None = None
    new_args = ["new", "-s", session, "--gpu", gpu]
    if gpu in HIGH_MEM_GPUS:
        new_args.append("--high-mem")
    log.info("creating Colab session %s on %s", session, gpu)
    _run_colab(new_args, capture=False, check=True)
    try:
        usage_during = get_usage()
        _upload_inputs(session, spec)
        log.info("starting %s detached on Colab session %s", spec.trainer_name, session)
        _exec_code(session, _remote_start_script(), timeout=60.0)
        _wait_for_remote_training(session, timeout_s=float(spec.cfg.platform.timeout_h) * 3600.0)
        manifest = download_outputs(session, spec.pull_files, spec.output_dir)
    finally:
        log.info("stopping Colab session %s", session)
        stop_session(session)

    if manifest.get("error"):
        raise RuntimeError(f"Colab training failed:\n{manifest['error']}")
    if not result_path.exists():
        raise RuntimeError(f"Colab training produced no result.json (missing: {manifest.get('missing')})")

    result: dict[str, Any] = json.loads(result_path.read_text())
    usage_after = get_usage()
    ccu, rate = compute_ccu(usage_before, usage_during, usage_after, float(result.get("seconds") or 0.0))
    result["platform"] = "colab"
    result["ccu"] = ccu
    if rate is not None:
        result["ccu_rate"] = rate
    if usage_after is not None:
        result["ccu_balance_after"] = usage_after.balance
    result_path.write_text(json.dumps(result, indent=1))
    return result
