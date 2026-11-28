#!/usr/bin/env python3
"""Pre-commit guard blocking AI agent shells from committing under human git identity."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.agent_identity import detect_agent_harness, load_repo_identity


def check_commit_identity(environ: Mapping[str, str] | None = None, repo_root: Path | None = None) -> int:
    """Return 0 when commit identity is valid, or 1 if an agent shell commits as the human."""
    env = os.environ if environ is None else environ
    if env.get("GITHUB_ACTIONS") == "true":
        return 0
    author_email = env.get("GIT_AUTHOR_EMAIL", "").strip().lower()
    if not author_email:
        return 0
    harness = detect_agent_harness(env)
    if harness is None or env.get("BGA_COMMIT_AS_HUMAN") == "1":
        return 0

    cfg = load_repo_identity(repo_root)
    if author_email in cfg.human_emails:
        print(
            f"Blocked commit in {harness} session using human author email ({author_email}).\n"
            "Use `uv run scripts/agent_git.py commit ...` for agent commits, or "
            "`uv run scripts/agent_git.py commit --as-human ...` solely for Tomasz's "
            "uncommitted `data/labels/` edits.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    return check_commit_identity()


if __name__ == "__main__":
    sys.exit(main())
