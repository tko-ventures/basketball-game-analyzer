#!/usr/bin/env python3
"""GitHub CLI wrapper running writes under the GitHub App installation token."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.agent_identity import (
    format_agent_label,
    load_app_credentials,
    load_repo_identity,
    mint_installation_token,
)

BODY_WRITE_SUBCOMMANDS: frozenset[tuple[str, str]] = frozenset({
    ("pr", "create"),
    ("pr", "comment"),
    ("pr", "review"),
    ("issue", "create"),
    ("issue", "comment"),
})


def _extract_wrapper_options(args: list[str]) -> tuple[str | None, bool, list[str]]:
    explicit_agent: str | None = None
    no_footer = False
    idx = 0
    while idx < len(args):
        tok = args[idx]
        if tok == "--no-footer":
            no_footer = True
            idx += 1
            continue
        if tok == "--agent" and idx + 1 < len(args):
            explicit_agent = args[idx + 1]
            idx += 2
            continue
        if tok.startswith("--agent="):
            explicit_agent = tok.split("=", 1)[1]
            idx += 1
            continue
        break
    return (explicit_agent, no_footer, args[idx:])


def append_agent_footer(body: str, agent_label: str, *, include_footer: bool = True) -> str:
    """Append a compact ``<sub>Agent: ...</sub>`` footer if absent."""
    if not include_footer or "<sub>Agent:" in body:
        return body
    stripped = body.rstrip()
    return f"{stripped}\n\n<sub>Agent: {agent_label}</sub>" if stripped else f"<sub>Agent: {agent_label}</sub>"


def _rewrite_body_flag(args: list[str], agent_label: str, *, include_footer: bool) -> list[str]:
    if not include_footer or len(args) < 2 or (args[0], args[1]) not in BODY_WRITE_SUBCOMMANDS:
        return args
    rewritten: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in {"--body", "-b"} and i + 1 < len(args):
            rewritten.extend([tok, append_agent_footer(args[i + 1], agent_label)])
            i += 2
            continue
        if tok.startswith("--body="):
            val = tok.split("=", 1)[1]
            rewritten.append(f"--body={append_agent_footer(val, agent_label)}")
            i += 1
            continue
        rewritten.append(tok)
        i += 1
    return rewritten


def _parse_pr_reply(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent_gh.py pr-reply", description="Reply to an inline PR review comment.")
    parser.add_argument("pr_number", type=int, help="Pull request number.")
    parser.add_argument("comment_id", type=int, help="Inline review comment ID to reply to.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body", "-b", help="Reply text.")
    group.add_argument("--body-file", "-F", help="Read reply text from file path.")
    return parser.parse_args(args)


def _run_pr_reply(args: list[str], *, gh_env: dict[str, str], agent_label: str, include_footer: bool) -> int:
    parsed = _parse_pr_reply(args)
    raw_body = Path(parsed.body_file).read_text(encoding="utf-8") if parsed.body_file else (parsed.body or "")
    body_text = append_agent_footer(raw_body, agent_label, include_footer=include_footer)
    cfg = load_repo_identity()
    endpoint = f"repos/{cfg.org}/{cfg.repo}/pulls/{parsed.pr_number}/comments/{parsed.comment_id}/replies"
    cmd = ["gh", "api", "-X", "POST", endpoint, "-f", f"body={body_text}"]
    return subprocess.run(cmd, env=gh_env, check=False).returncode


def _print_help() -> None:
    print(
        "Usage:\n"
        "  uv run scripts/agent_gh.py [--agent <label>] [--no-footer] <gh args...>\n"
        "  uv run scripts/agent_gh.py [--agent <label>] [--no-footer] pr-reply <pr> <comment_id> --body <text>"
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    explicit_agent, no_footer, gh_args = _extract_wrapper_options(raw_args)
    if not gh_args or gh_args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    include_footer = not no_footer
    agent_label = format_agent_label(explicit_agent)
    creds = load_app_credentials()
    entry = mint_installation_token(creds)
    gh_env = dict(os.environ)
    gh_env["GH_TOKEN"] = entry.token

    if gh_args[0] == "pr-reply":
        return _run_pr_reply(gh_args[1:], gh_env=gh_env, agent_label=agent_label, include_footer=include_footer)

    cmd_args = _rewrite_body_flag(gh_args, agent_label, include_footer=include_footer)
    return subprocess.run(["gh", *cmd_args], env=gh_env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
