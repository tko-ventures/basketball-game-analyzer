#!/usr/bin/env python3
"""Git commit and push wrapper attributing agent changes to the GitHub App bot."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.agent_identity import (
    bot_git_email,
    bot_git_name,
    format_agent_label,
    load_app_credentials,
    load_repo_identity,
    mint_installation_token,
    resolve_bot_identity,
)

PUSH_VALUE_FLAGS = frozenset({"-o", "--push-option", "--receive-pack", "--exec", "--repo"})
PROTECTED_BRANCH_REFS = frozenset({"main", "refs/heads/main"})
WILDCARD_CHARS = frozenset({"*", "?", "["})


def _extract_commit_options(args: list[str]) -> tuple[bool, str | None, list[str]]:
    as_human = False
    agent: str | None = None
    forwarded: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            forwarded.extend(args[i:])
            break
        if tok == "--as-human":
            as_human = True
            i += 1
            continue
        if tok == "--agent" and i + 1 < len(args):
            agent = args[i + 1]
            i += 2
            continue
        if tok.startswith("--agent="):
            agent = tok.split("=", 1)[1]
            i += 1
            continue
        forwarded.append(tok)
        i += 1
    return (as_human, agent, forwarded)


def _has_agent_trailer(git_args: list[str]) -> bool:
    for idx, tok in enumerate(git_args):
        if tok == "--":
            break
        if tok.startswith("--trailer=") and tok.split("=", 1)[1].lstrip().lower().startswith("agent:"):
            return True
        if tok == "--trailer" and idx + 1 < len(git_args) and git_args[idx + 1].lstrip().lower().startswith("agent:"):
            return True
    return False


def _run_commit(raw_args: list[str]) -> int:
    as_human, explicit_agent, git_args = _extract_commit_options(raw_args)
    repo_cfg = load_repo_identity()
    bot_slug, bot_user_id = resolve_bot_identity()
    bot_name = bot_git_name(bot_slug)
    bot_email = bot_git_email(bot_slug, bot_user_id)

    env = dict(os.environ)
    env["GIT_COMMITTER_NAME"] = bot_name
    env["GIT_COMMITTER_EMAIL"] = bot_email
    if as_human:
        env["GIT_AUTHOR_NAME"] = repo_cfg.human_name
        env["GIT_AUTHOR_EMAIL"] = repo_cfg.human_emails[0]
        env["BGA_COMMIT_AS_HUMAN"] = "1"
        cmd = ["git", "commit", *git_args]
    else:
        env["GIT_AUTHOR_NAME"] = bot_name
        env["GIT_AUTHOR_EMAIL"] = bot_email
        env.pop("BGA_COMMIT_AS_HUMAN", None)
        label = format_agent_label(explicit_agent, env)
        trailer_args = [] if _has_agent_trailer(git_args) else ["--trailer", f"Agent: {label}"]
        cmd = ["git", "commit", *trailer_args, *git_args]
    return subprocess.run(cmd, env=env, check=False).returncode


def _parse_push_positionals(git_args: list[str]) -> tuple[bool, list[str]]:
    positionals: list[str] = []
    has_broad_push = False
    i = 0
    while i < len(git_args):
        tok = git_args[i]
        if tok == "--":
            positionals.extend(git_args[i + 1 :])
            break
        if tok in {"--all", "--mirror"}:
            has_broad_push = True
            i += 1
            continue
        if tok in PUSH_VALUE_FLAGS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        positionals.append(tok)
        i += 1
    return (has_broad_push, positionals)


def _ref_matches_main(ref: str, current_branch: str) -> bool:
    if ref in PROTECTED_BRANCH_REFS:
        return True
    if ref == "HEAD" and current_branch in PROTECTED_BRANCH_REFS:
        return True
    if any(ch in WILDCARD_CHARS for ch in ref):
        return any(fnmatch.fnmatchcase(protected, ref) for protected in PROTECTED_BRANCH_REFS)
    return False


def is_main_push(git_args: list[str], current_branch: str) -> bool:
    """Return ``True`` when the push invocation targets ``main``."""
    has_broad_push, positionals = _parse_push_positionals(git_args)
    if has_broad_push:
        return True
    refspecs = positionals[1:]
    if not refspecs:
        return current_branch in PROTECTED_BRANCH_REFS
    for raw_spec in refspecs:
        spec = raw_spec.lstrip("+")
        if ":" in spec:
            src, dst = spec.split(":", 1)
            target = dst or src
            if _ref_matches_main(target, current_branch):
                return True
        elif _ref_matches_main(spec, current_branch):
            return True
    return False


def _current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def _write_askpass_script(directory: Path) -> Path:
    script_path = directory / "git_askpass.py"
    script_path.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        'prompt = sys.argv[1].lower() if len(sys.argv) > 1 else ""\n'
        'if "username" in prompt:\n'
        '    sys.stdout.write("x-access-token\\n")\n'
        "else:\n"
        '    sys.stdout.write(os.environ.get("BGA_GIT_ASKPASS_TOKEN", "") + "\\n")\n',
        encoding="utf-8",
    )
    script_path.chmod(0o700)
    return script_path


def _run_push(git_args: list[str]) -> int:
    branch = _current_branch()
    if is_main_push(git_args, branch):
        print(
            "Refusing to push directly to main via scripts/agent_git.py; use a feature branch and PR.",
            file=sys.stderr,
        )
        return 2

    creds = load_app_credentials()
    entry = mint_installation_token(creds)
    with tempfile.TemporaryDirectory(prefix="bga-askpass-") as tmp_dir:
        askpass = _write_askpass_script(Path(tmp_dir))
        env = dict(os.environ)
        env["GIT_ASKPASS"] = str(askpass)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["BGA_GIT_ASKPASS_TOKEN"] = entry.token
        cmd = [
            "git",
            "-c",
            "url.https://github.com/.insteadOf=git@github.com:",
            "-c",
            "url.https://github.com/.insteadOf=ssh://git@github.com/",
            "-c",
            "credential.helper=",
            "push",
            *git_args,
        ]
        return subprocess.run(cmd, env=env, check=False).returncode


def _print_help() -> None:
    print(
        "Usage:\n"
        "  uv run scripts/agent_git.py commit [--as-human] [--agent <label>] [git commit args...]\n"
        "  uv run scripts/agent_git.py push [git push args...]"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand == "commit":
        return _run_commit(rest)
    if subcommand == "push":
        return _run_push(rest)
    print(f"Unsupported agent_git subcommand: {subcommand!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
