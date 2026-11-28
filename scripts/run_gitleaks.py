#!/usr/bin/env python3
"""Download, SHA256-verify, cache, and run the pinned Gitleaks CLI binary."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GITLEAKS_VERSION = "8.30.1"
ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    (
        "darwin",
        "arm64",
    ): ("gitleaks_8.30.1_darwin_arm64.tar.gz", "b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5"),
    (
        "darwin",
        "x86_64",
    ): ("gitleaks_8.30.1_darwin_x64.tar.gz", "dfe101a4db2255fc85120ac7f3d25e4342c3c20cf749f2c20a18081af1952709"),
    (
        "linux",
        "aarch64",
    ): ("gitleaks_8.30.1_linux_arm64.tar.gz", "e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080"),
    (
        "linux",
        "x86_64",
    ): ("gitleaks_8.30.1_linux_x64.tar.gz", "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"),
}


def _platform_key() -> tuple[str, str]:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    elif sys_name == "linux" and machine == "arm64":
        machine = "aarch64"
    return (sys_name, machine)


def _cache_binary() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_root / "bga" / "gitleaks" / f"v{GITLEAKS_VERSION}" / "gitleaks"


def _extract_gitleaks(data: bytes) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and Path(member.name).name == "gitleaks":
                extracted = archive.extractfile(member)
                if extracted is None:
                    break
                return extracted.read()
    raise RuntimeError("gitleaks binary not found in downloaded archive")


def _install_binary(dest: Path) -> str:
    key = _platform_key()
    if key not in ASSETS:
        raise RuntimeError(f"unsupported platform for gitleaks v{GITLEAKS_VERSION}: {key!r}")
    asset, expected_sha = ASSETS[key]
    url = f"https://github.com/gitleaks/gitleaks/releases/download/v{GITLEAKS_VERSION}/{asset}"
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"refusing non-HTTPS URL: {url!r}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # fixed GitHub release HTTPS URL verified by SHA-256
            data = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"failed to download pinned gitleaks v{GITLEAKS_VERSION} from {url} to {dest}: {exc}"
        ) from exc

    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch for {asset}: got {actual_sha}, expected {expected_sha}")

    binary_bytes = _extract_gitleaks(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".gitleaks-", dir=dest.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(binary_bytes)
        tmp_path.chmod(0o755)
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return str(dest)


def _ensure_binary() -> str:
    dest = _cache_binary()
    if dest.is_file() and os.access(dest, os.X_OK):
        return str(dest)
    return _install_binary(dest)


def _build_args(args: argparse.Namespace, binary: str) -> list[str]:
    if args.staged:
        return [binary, "git", "--pre-commit", "--redact", "--staged", "--verbose"]
    if args.dir is not None:
        return [binary, "dir", "--redact", "--verbose", args.dir]
    cmd = [binary, "git", "--redact", "--verbose"]
    if args.log_opts:
        cmd.extend(["--log-opts", args.log_opts])
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pinned Gitleaks CLI.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="Scan staged git changes (pre-commit mode).")
    mode.add_argument("--dir", metavar="PATH", help="Scan files in PATH.")
    mode.add_argument("--git", action="store_true", help="Scan git history.")
    parser.add_argument("--log-opts", help="Optional git log range passed to --log-opts in --git mode.")
    args = parser.parse_args(argv)
    if args.log_opts and not args.git:
        parser.error("--log-opts requires --git")
    binary = _ensure_binary()
    cmd = _build_args(args, binary)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
