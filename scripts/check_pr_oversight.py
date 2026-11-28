#!/usr/bin/env python3
"""Audit a PR git diff for protected files, dependency changes, and security suppressions."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROTECTED_EXACT = frozenset({
    ".pre-commit-config.yaml",
    ".gitleaks.toml",
    "pyproject.toml",
    "AGENTS.md",
    "tests/test_security_policy.py",
    "scripts/check_pr_oversight.py",
    "scripts/run_gitleaks.py",
    "scripts/security_digest.py",
})
PROTECTED_PREFIXES = (".github/",)
DEPENDENCY_EXACT = frozenset({
    "pyproject.toml",
    "uv.lock",
    "src/bga/train/dispatch.py",
})
DEPENDENCY_PREFIXES = ("src/bga/remote/",)
CODE_CONFIG_SUFFIXES = frozenset({
    ".py",
    ".sh",
    ".bash",
    ".yml",
    ".yaml",
    ".toml",
    ".js",
    ".ts",
    ".html",
    ".json",
})
SUPPRESSION_SELF_EXEMPT_FILES = frozenset({
    "scripts/check_pr_oversight.py",
    "scripts/security_digest.py",
    "tests/test_pr_oversight.py",
    "tests/test_security_policy.py",
})

NOQA_RE = re.compile(r"#\s*noqa\s*:\s*([^#]+)")
SEC_CODE_RE = re.compile(r"\bS\d+\b")
OTHER_SUPPRESSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nosec", re.compile(r"#\s*nosec\b", re.IGNORECASE)),
    ("nosemgrep", re.compile(r"#\s*nosemgrep\b", re.IGNORECASE)),
    ("gitleaks:allow", re.compile(r"gitleaks:allow\b", re.IGNORECASE)),
    ("shellcheck-disable", re.compile(r"#\s*shellcheck\s+disable=", re.IGNORECASE)),
    ("zizmor-ignore", re.compile(r"#\s*zizmor\s*:", re.IGNORECASE)),
)
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class SuppressionFinding:
    """Single inline security suppression added in a diff."""

    path: str
    line: int
    kind: str
    text: str
    has_reason: bool


@dataclass(frozen=True)
class OversightReport:
    """Classified oversight findings for a PR diff."""

    protected_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    suppressions: tuple[SuppressionFinding, ...]
    unreasoned_noqa: tuple[SuppressionFinding, ...]

    @property
    def needs_oversight(self) -> bool:
        """Whether any protected file, dependency file, or suppression requires manager review."""
        return bool(self.protected_files or self.dependency_files or self.suppressions)


def _is_protected(path: str) -> bool:
    return path in PROTECTED_EXACT or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _is_dependency_file(path: str) -> bool:
    return path in DEPENDENCY_EXACT or any(path.startswith(prefix) for prefix in DEPENDENCY_PREFIXES)


def _has_noqa_reason(line: str, match: re.Match[str]) -> bool:
    codes = match.group(1)
    tail = line[match.end() :].strip()
    if tail.startswith("#") and len(tail.lstrip("#").strip()) >= 3:
        return True
    return "(" in codes and ")" in codes


def parse_name_status(output: str) -> list[str]:
    """Extract unique touched paths from ``git diff --name-status`` output."""
    paths: list[str] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        for candidate in parts[1:]:
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    return sorted(paths)


def _classify_added_line(path: str, line_no: int, text: str) -> list[SuppressionFinding]:
    if path in SUPPRESSION_SELF_EXEMPT_FILES or Path(path).suffix.lower() not in CODE_CONFIG_SUFFIXES:
        return []
    findings: list[SuppressionFinding] = []
    noqa_match = NOQA_RE.search(text)
    if noqa_match is not None and SEC_CODE_RE.search(noqa_match.group(1)):
        findings.append(
            SuppressionFinding(
                path=path,
                line=line_no,
                kind="noqa-security",
                text=text.strip(),
                has_reason=_has_noqa_reason(text, noqa_match),
            )
        )
    for kind, pattern in OTHER_SUPPRESSION_PATTERNS:
        if pattern.search(text):
            findings.append(
                SuppressionFinding(
                    path=path,
                    line=line_no,
                    kind=kind,
                    text=text.strip(),
                    has_reason=True,
                )
            )
    return findings


def parse_added_suppressions(patch_text: str) -> list[SuppressionFinding]:
    """Parse added inline security suppressions from unified ``git diff -U0`` text."""
    findings: list[SuppressionFinding] = []
    current_file: str | None = None
    next_line = 0
    for raw in patch_text.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            current_file = target[2:] if target.startswith("b/") else None if target == "/dev/null" else target
            continue
        if raw.startswith(("--- ", "diff --git ")):
            continue
        hunk = HUNK_RE.match(raw)
        if hunk is not None:
            next_line = int(hunk.group(1))
            continue
        if raw.startswith("+") and current_file is not None:
            findings.extend(_classify_added_line(current_file, next_line, raw[1:]))
            next_line += 1
    return findings


def analyze_diff(name_status_text: str, patch_text: str) -> OversightReport:
    """Classify touched files and added suppressions into an :class:`OversightReport`."""
    touched = parse_name_status(name_status_text)
    protected = tuple(p for p in touched if _is_protected(p))
    deps = tuple(p for p in touched if _is_dependency_file(p))
    suppressions = tuple(parse_added_suppressions(patch_text))
    unreasoned = tuple(item for item in suppressions if item.kind == "noqa-security" and not item.has_reason)
    return OversightReport(
        protected_files=protected,
        dependency_files=deps,
        suppressions=suppressions,
        unreasoned_noqa=unreasoned,
    )


def format_brief(report: OversightReport) -> str:
    """Format a compact Markdown Manager Oversight Brief."""
    lines = ["## Manager Oversight Brief", ""]
    if report.unreasoned_noqa:
        lines.append("**Status:** BLOCKED — unreasoned `# noqa: S*` suppression(s) added.")
    elif report.needs_oversight:
        lines.append("**Status:** Oversight required — Tomasz sign-off needed on:")
    else:
        lines.append("**Status:** Clean — no security/governance escalations.")
    lines.append("")

    if report.protected_files:
        lines.append("### Protected security / governance files touched")
        lines.extend(f"- `{path}`" for path in report.protected_files)
        lines.append("")

    if report.dependency_files:
        lines.append("### Dependency / remote-package files touched")
        lines.extend(f"- `{path}`" for path in report.dependency_files)
        lines.append("")

    if report.suppressions:
        lines.append("### Inline security suppressions added")
        for item in report.suppressions:
            tag = "MISSING RATIONALE" if item.kind == "noqa-security" and not item.has_reason else item.kind
            lines.append(f"- `{item.path}:{item.line}` (`{tag}`): `{item.text}`")
        lines.append("")

    if not report.needs_oversight:
        lines.append("- Protected files: none")
        lines.append("- Dependency files: none")
        lines.append("- Added inline suppressions: none")
        lines.append("")

    return "\n".join(lines)


def _emit_gha_annotations(report: OversightReport) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for item in report.unreasoned_noqa:
        print(f"::error file={item.path},line={item.line}::Unreasoned security # noqa suppression: {item.text}")
    for path in report.protected_files:
        print(f"::notice file={path}::Protected security/governance file modified; requires Tomasz review.")


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit PR diff for manager oversight.")
    parser.add_argument("--base", required=True, help="Base git ref/SHA.")
    parser.add_argument("--head", default="HEAD", help="Head git ref/SHA (default: HEAD).")
    parser.add_argument("--summary", help="Optional Markdown file path (e.g. $GITHUB_STEP_SUMMARY) to append to.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if unreasoned # noqa: S* suppressions are added.")
    args = parser.parse_args(argv)

    diff_range = f"{args.base}...{args.head}"
    name_status = _run_git(["diff", "--name-status", diff_range])
    patch = _run_git(["diff", "-U0", diff_range])
    report = analyze_diff(name_status, patch)
    brief = format_brief(report)
    print(brief)
    _emit_gha_annotations(report)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(brief + "\n")

    if args.strict and report.unreasoned_noqa:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
