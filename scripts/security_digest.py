#!/usr/bin/env python3
"""Summarize open GitHub Security alerts and active inline codebase suppressions for Tomasz."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SCAN_SUFFIXES = frozenset({".py", ".sh", ".bash", ".yml", ".yaml", ".toml", ".js", ".ts", ".html"})
SELF_EXEMPT = frozenset({
    "scripts/check_pr_oversight.py",
    "scripts/security_digest.py",
    "tests/test_pr_oversight.py",
    "tests/test_security_policy.py",
})
NOQA_RE = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa(?:\s*:\s*([^#]+)|\b(?![\w:]))", re.IGNORECASE)
SEC_CODE_RE = re.compile(r"\bS\d+\b")
SUPPRESSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nosec", re.compile(r"#\s*nosec\b", re.IGNORECASE)),
    ("nosemgrep", re.compile(r"#\s*nosemgrep\b", re.IGNORECASE)),
    ("gitleaks-allow", re.compile(r"gitleaks:allow\b", re.IGNORECASE)),
    ("shellcheck-disable", re.compile(r"#\s*shellcheck\s+disable=", re.IGNORECASE)),
    ("zizmor-ignore", re.compile(r"#\s*zizmor\s*:", re.IGNORECASE)),
)


@dataclass(frozen=True)
class SuppressionEntry:
    """Active inline or config security suppression in the repository."""

    path: str
    line: int
    kind: str
    snippet: str


@dataclass(frozen=True)
class AlertSummary:
    """Summary of one GitHub Security alert feed."""

    feed: str
    status: str
    count: int | None
    highlights: tuple[str, ...]


def collect_gitleaks_allowlists(text: str, rel: str = ".gitleaks.toml") -> list[SuppressionEntry]:
    """Parse top-level and rule-scoped allowlists from a Gitleaks TOML config."""
    data = tomllib.loads(text)
    entries: list[SuppressionEntry] = []
    if "allowlist" in data:
        entries.append(SuppressionEntry(rel, 1, "gitleaks-allowlist", f"[allowlist]: {data['allowlist']}"))
    for idx, allow in enumerate(data.get("allowlists", ()), start=1):
        entries.append(SuppressionEntry(rel, 1, "gitleaks-allowlist", f"[[allowlists]][{idx}]: {allow}"))
    for rule in data.get("rules", ()):
        rule_id = rule.get("id", "unknown")
        if "allowlist" in rule:
            entries.append(
                SuppressionEntry(rel, 1, "gitleaks-allowlist", f"rule={rule_id} [rules.allowlist]: {rule['allowlist']}")
            )
        for idx, allow in enumerate(rule.get("allowlists", ()), start=1):
            entries.append(
                SuppressionEntry(rel, 1, "gitleaks-allowlist", f"rule={rule_id} [[rules.allowlists]][{idx}]: {allow}")
            )
    return entries


def _classify_noqa_kind(line: str) -> str | None:
    match = NOQA_RE.search(line)
    if match is None:
        return None
    codes_group = match.group(1)
    if codes_group is None:
        return "noqa-blanket"
    codes_part = codes_group.split("(", 1)[0]
    if SEC_CODE_RE.search(codes_part):
        return "noqa-security"
    return None


def collect_local_suppressions(root: Path = ROOT) -> list[SuppressionEntry]:
    """Collect active inline security suppressions and Gitleaks allowlists across tracked files."""
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    entries: list[SuppressionEntry] = []
    for rel in sorted(tracked):
        if rel in SELF_EXEMPT:
            continue
        path = root / rel
        if not path.is_file():
            continue
        if rel == ".gitleaks.toml":
            entries.extend(collect_gitleaks_allowlists(path.read_text(encoding="utf-8", errors="replace"), rel))
            continue
        if path.suffix.lower() not in CODE_SCAN_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, line in enumerate(text.splitlines(), start=1):
            noqa_kind = _classify_noqa_kind(line)
            if noqa_kind is not None:
                entries.append(SuppressionEntry(rel, idx, noqa_kind, line.strip()))
            entries.extend(
                SuppressionEntry(rel, idx, kind, line.strip()) for kind, pat in SUPPRESSION_PATTERNS if pat.search(line)
            )
    return entries


def _detect_repo() -> str:
    return subprocess.check_output(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"], text=True
    ).strip()


def _flatten_alert_pages(raw: object) -> list[dict] | None:
    if not isinstance(raw, list):
        return None
    if raw and all(isinstance(page, list) for page in raw):
        return [item for page in raw for item in page]
    return [item for item in raw if isinstance(item, dict)]


def _fetch_alert_feed(repo: str, feed: str, endpoint: str) -> AlertSummary:
    proc = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", f"repos/{repo}/{endpoint}?state=open&per_page=100"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        reason = msg[-1] if msg else f"exit {proc.returncode}"
        if "403" in proc.stderr or "404" in proc.stderr or "disabled" in proc.stderr.lower():
            return AlertSummary(feed, f"disabled/inaccessible ({reason})", None, ())
        return AlertSummary(feed, f"unavailable ({reason})", None, ())

    items = _flatten_alert_pages(json.loads(proc.stdout))
    if items is None:
        return AlertSummary(feed, "unexpected response", None, ())

    highlights: list[str] = []
    for item in items[:5]:
        num = item.get("number", "?")
        rule = (
            item.get("rule", {}).get("id")
            or item.get("security_advisory", {}).get("ghsa_id")
            or item.get("secret_type_display_name")
            or "alert"
        )
        sev = (
            item.get("rule", {}).get("security_severity_level")
            or item.get("security_advisory", {}).get("severity")
            or "open"
        )
        highlights.append(f"#{num} `{rule}` ({sev})")
    return AlertSummary(feed, "enabled", len(items), tuple(highlights))


def format_digest(alerts: list[AlertSummary], suppressions: list[SuppressionEntry]) -> str:
    """Render a concise Markdown security digest for Tomasz."""
    lines = ["## Security & Oversight Digest", ""]
    if alerts:
        lines.append("### GitHub Security Feeds")
        for feed in alerts:
            count_text = "n/a" if feed.count is None else f"{feed.count} open"
            lines.append(f"- **{feed.feed}:** {feed.status} — {count_text}")
            lines.extend(f"  - {item}" for item in feed.highlights)
            if feed.count is not None and feed.count > len(feed.highlights):
                lines.append(f"  - ...and {feed.count - len(feed.highlights)} more open")
        lines.append("")

    lines.append(f"### Active Codebase Suppressions ({len(suppressions)})")
    if suppressions:
        lines.append("| Location | Kind | Line |")
        lines.append("|---|---|---|")
        for entry in suppressions:
            snippet = entry.snippet.replace("|", "\\|")
            lines.append(f"| `{entry.path}:{entry.line}` | `{entry.kind}` | `{snippet}` |")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a security & suppression digest for Tomasz.")
    parser.add_argument("--offline", action="store_true", help="Skip GitHub API queries and scan local repo only.")
    parser.add_argument("--repo", help="Optional owner/name repository override for GitHub API queries.")
    args = parser.parse_args(argv)

    suppressions = collect_local_suppressions(ROOT)
    alerts: list[AlertSummary] = []
    if not args.offline:
        repo = args.repo or _detect_repo()
        alerts = [
            _fetch_alert_feed(repo, "Code scanning (CodeQL / SARIF)", "code-scanning/alerts"),
            _fetch_alert_feed(repo, "Dependabot", "dependabot/alerts"),
            _fetch_alert_feed(repo, "Secret scanning", "secret-scanning/alerts"),
        ]
    print(format_digest(alerts, suppressions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
