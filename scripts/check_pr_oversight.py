#!/usr/bin/env python3
"""Audit a PR diff for protected files, dependency changes, suppressions, and commit authorship."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROTECTED_EXACT = frozenset({
    ".pre-commit-config.yaml",
    ".gitleaks.toml",
    "pyproject.toml",
    "AGENTS.md",
    "tests/test_security_policy.py",
    "scripts/agent_gh.py",
    "scripts/agent_git.py",
    "scripts/agent_identity.py",
    "scripts/check_agent_identity.py",
    "scripts/check_pr_oversight.py",
    "scripts/run_gitleaks.py",
    "scripts/security_digest.py",
    "scripts/setup_github_app.py",
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

NOQA_RE = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa(?:\s*:\s*([^#]+)|\b(?![\w:]))", re.IGNORECASE)
SEC_CODE_RE = re.compile(r"\bS\d+\b")
PAREN_REASON_RE = re.compile(r"\(([^)]+)\)")
OTHER_SUPPRESSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nosec", re.compile(r"#\s*nosec\b", re.IGNORECASE)),
    ("nosemgrep", re.compile(r"#\s*nosemgrep\b", re.IGNORECASE)),
    ("gitleaks:allow", re.compile(r"gitleaks:allow\b", re.IGNORECASE)),
    ("shellcheck-disable", re.compile(r"#\s*shellcheck\s+disable=", re.IGNORECASE)),
    ("zizmor-ignore", re.compile(r"#\s*zizmor\s*:", re.IGNORECASE)),
)
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class SuppressionFinding:
    """Single inline security suppression added in a diff."""

    path: str
    line: int
    kind: str
    text: str
    has_reason: bool


@dataclass(frozen=True)
class CommitAuthorSummary:
    """Grouped commit count for a single author string."""

    author: str
    count: int


@dataclass(frozen=True)
class OversightReport:
    """Classified oversight findings for a PR diff."""

    protected_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    suppressions: tuple[SuppressionFinding, ...]
    unreasoned_noqa: tuple[SuppressionFinding, ...]
    commit_authors: tuple[CommitAuthorSummary, ...] = ()
    suppression_approved: bool | None = None

    @property
    def needs_oversight(self) -> bool:
        """Whether any protected file, dependency file, or suppression requires manager review."""
        return bool(self.protected_files or self.dependency_files or self.suppressions)


def _is_protected(path: str) -> bool:
    return path in PROTECTED_EXACT or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _is_dependency_file(path: str) -> bool:
    return path in DEPENDENCY_EXACT or any(path.startswith(prefix) for prefix in DEPENDENCY_PREFIXES)


def _has_noqa_reason(line: str, match: re.Match[str]) -> bool:
    tail = line[match.end() :].strip()
    if tail.startswith("#") and len(tail.lstrip("#").strip()) >= 3:
        return True
    if tail.startswith("("):
        paren = PAREN_REASON_RE.match(tail)
        if paren is not None and len(paren.group(1).strip()) >= 3:
            return True
    codes_raw = match.group(1) or ""
    paren_in_codes = PAREN_REASON_RE.search(codes_raw)
    return paren_in_codes is not None and len(paren_in_codes.group(1).strip()) >= 3


def _classify_noqa(text: str) -> tuple[str, bool] | None:
    match = NOQA_RE.search(text)
    if match is None:
        return None
    codes_group = match.group(1)
    if codes_group is None:
        return ("noqa-blanket", _has_noqa_reason(text, match))
    codes_part = codes_group.split("(", 1)[0]
    if not SEC_CODE_RE.search(codes_part):
        return None
    return ("noqa-security", _has_noqa_reason(text, match))


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


def parse_commit_authors(log_text: str) -> tuple[CommitAuthorSummary, ...]:
    """Count commits per author from ``git log --format='%an <%ae>'`` output."""
    counts = Counter(line.strip() for line in log_text.splitlines() if line.strip())
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return tuple(CommitAuthorSummary(author=author, count=count) for author, count in ordered)


def _classify_added_line(path: str, line_no: int, text: str) -> list[SuppressionFinding]:
    if path in SUPPRESSION_SELF_EXEMPT_FILES or Path(path).suffix.lower() not in CODE_CONFIG_SUFFIXES:
        return []
    findings: list[SuppressionFinding] = []
    noqa_info = _classify_noqa(text)
    if noqa_info is not None:
        kind, has_reason = noqa_info
        findings.append(
            SuppressionFinding(
                path=path,
                line=line_no,
                kind=kind,
                text=text.strip(),
                has_reason=has_reason,
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


def analyze_diff(
    name_status_text: str,
    patch_text: str,
    commit_log_text: str = "",
) -> OversightReport:
    """Classify touched files, added suppressions, and commit authors into an :class:`OversightReport`."""
    touched = parse_name_status(name_status_text)
    protected = tuple(p for p in touched if _is_protected(p))
    deps = tuple(p for p in touched if _is_dependency_file(p))
    suppressions = tuple(parse_added_suppressions(patch_text))
    unreasoned = tuple(
        item for item in suppressions if item.kind in {"noqa-security", "noqa-blanket"} and not item.has_reason
    )
    authors = parse_commit_authors(commit_log_text)
    return OversightReport(
        protected_files=protected,
        dependency_files=deps,
        suppressions=suppressions,
        unreasoned_noqa=unreasoned,
        commit_authors=authors,
    )


def has_approver_approval(reviews: list[dict[str, object]], approver: str, head_sha: str) -> bool:
    """Return ``True`` when ``approver``'s latest decisive review approves ``head_sha``."""
    target = approver.strip().lower()
    decisive_states = {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
    relevant: list[dict[str, object]] = []
    for review in reviews:
        user = review.get("user")
        login = str(user.get("login", "")).strip().lower() if isinstance(user, dict) else ""
        state = str(review.get("state", "")).strip().upper()
        if login == target and state in decisive_states:
            relevant.append(review)
    if not relevant:
        return False
    latest = relevant[-1]
    return str(latest.get("state", "")).strip().upper() == "APPROVED" and str(latest.get("commit_id", "")) == head_sha


def fetch_pr_reviews(repo: str, pr_number: int, token: str) -> list[dict[str, object]]:
    """Fetch PR reviews from GitHub's REST API using HTTPS."""
    reviews: list[dict[str, object]] = []
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "bga-pr-oversight",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for page in range(1, 11):
        api_path = f"/repos/{repo}/pulls/{pr_number}/reviews?per_page=100&page={page}"
        conn = http.client.HTTPSConnection("api.github.com", timeout=30)
        try:
            conn.request("GET", api_path, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
        finally:
            conn.close()
        if resp.status >= 400:
            snippet = raw[:300].replace("\n", " ")
            raise RuntimeError(f"GitHub reviews API returned {resp.status}: {snippet}")
        batch = json.loads(raw)
        if not isinstance(batch, list):
            break
        reviews.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
    return reviews


def format_brief(report: OversightReport) -> str:
    """Format a compact Markdown Manager Oversight Brief."""
    lines = ["## Manager Oversight Brief", ""]
    if report.unreasoned_noqa:
        lines.append("**Status:** BLOCKED — unreasoned `# noqa` / `# noqa: S*` suppression(s) added.")
    elif report.suppression_approved is False:
        lines.append("**Status:** BLOCKED — inline security suppression(s) require manager approval on HEAD.")
    elif report.needs_oversight:
        lines.append("**Status:** Oversight required — Tomasz sign-off needed on:")
    else:
        lines.append("**Status:** Clean — no security/governance escalations.")
    lines.append("")

    if report.commit_authors:
        lines.append("### Commit authorship")
        for item in report.commit_authors:
            suffix = "commit" if item.count == 1 else "commits"
            lines.append(f"- `{item.author}` ({item.count} {suffix})")
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
            tag = (
                "MISSING RATIONALE"
                if item.kind in {"noqa-security", "noqa-blanket"} and not item.has_reason
                else item.kind
            )
            lines.append(f"- `{item.path}:{item.line}` (`{tag}`): `{item.text}`")
        lines.append("")

    if not report.needs_oversight:
        lines.append("- Protected files: none")
        lines.append("- Dependency files: none")
        lines.append("- Added inline suppressions: none")
        lines.append("")

    return "\n".join(lines)


def _emit_gha_annotations(report: OversightReport, approver: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for item in report.unreasoned_noqa:
        print(f"::error file={item.path},line={item.line}::Unreasoned security # noqa suppression: {item.text}")
    if report.suppression_approved is False:
        for item in report.suppressions:
            print(
                f"::error file={item.path},line={item.line}::"
                f"Inline security suppression ({item.kind}) requires @{approver} approval on HEAD."
            )
    for path in report.protected_files:
        print(f"::notice file={path}::Protected security/governance file modified; requires Tomasz review.")


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True)


def _check_suppression_approval(
    *,
    head_ref: str,
    repo: str | None,
    pr_number: int | None,
    approver: str,
) -> bool:
    if not repo or pr_number is None:
        print("Suppression approval gate requires --repo and --pr-number.", file=sys.stderr)
        return False
    head_sha = head_ref if FULL_SHA_RE.fullmatch(head_ref) else _run_git(["rev-parse", head_ref]).strip()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    try:
        reviews = fetch_pr_reviews(repo, pr_number, token)
    except RuntimeError as exc:
        print(f"Failed to verify @{approver} review status: {exc}", file=sys.stderr)
        return False
    return has_approver_approval(reviews, approver, head_sha)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PR diff for manager oversight.")
    parser.add_argument("--base", required=True, help="Base git ref/SHA.")
    parser.add_argument("--head", default="HEAD", help="Head git ref/SHA (default: HEAD).")
    parser.add_argument("--summary", help="Optional Markdown file path (e.g. $GITHUB_STEP_SUMMARY) to append to.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if unreasoned blanket or security # noqa suppressions are added.",
    )
    parser.add_argument(
        "--require-suppression-approval",
        action="store_true",
        help="Exit 1 if inline security suppressions are added without manager approval on HEAD.",
    )
    parser.add_argument("--approver", default="tk-o", help="GitHub login required to approve suppressions.")
    parser.add_argument("--repo", help="GitHub owner/repo slug for review lookup.")
    parser.add_argument("--pr-number", type=int, help="Pull request number for review lookup.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    diff_range = f"{args.base}...{args.head}"
    name_status = _run_git(["diff", "--name-status", diff_range])
    patch = _run_git(["diff", "-U0", diff_range])
    commit_log = _run_git(["log", "--right-only", "--no-merges", "--format=%an <%ae>", diff_range])
    report = analyze_diff(name_status, patch, commit_log)

    if args.require_suppression_approval and report.suppressions:
        approved = _check_suppression_approval(
            head_ref=args.head,
            repo=args.repo,
            pr_number=args.pr_number,
            approver=args.approver,
        )
        report = OversightReport(
            protected_files=report.protected_files,
            dependency_files=report.dependency_files,
            suppressions=report.suppressions,
            unreasoned_noqa=report.unreasoned_noqa,
            commit_authors=report.commit_authors,
            suppression_approved=approved,
        )

    brief = format_brief(report)
    print(brief)
    _emit_gha_annotations(report, args.approver)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(brief + "\n")

    if args.strict and report.unreasoned_noqa:
        return 1
    if report.suppression_approved is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
