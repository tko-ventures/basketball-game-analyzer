# Dedicated GitHub App bot identity and per-agent provenance trailers for AI agents

- **Date:** 2026-11-28
- **Status:** Proposed
- **Area:** Infra / Process
- **Decider:** Tomasz

## Context

All AI coding agent contributions to `tko-ventures/basketball-game-analyzer` currently attribute to `tk-o` (`Tomasz Kopacki <tomasz@kopacki.net>`) across four channels: `~/.gitconfig` (`git commit`), the `tk-o` SSH remote (`git push`), the local `gh` CLI keyring, and T3 Code's built-in GitHub MCP write tools (`pull_request_*`, `create_issue`, `comment_on_issue`). Beyond hiding human vs. agent authorship, this blocks formal review governance: GitHub forbids self-approval, so `tk-o`-opened agent PRs cannot receive `@tk-o` approvals or enforce `.github/CODEOWNERS` on `main` (`#62`). Tracked under Epic `#68`.

## Decision

1. **Private Organization GitHub App (`tko-ventures`, e.g. `tko-agent` → `tko-agent[bot]`):** Install a private org App on this repository, minting short-lived 1-hour installation tokens from `~/.config/bga/github-app.pem` and `~/.config/bga/github-app.env` (never committed) via zero-dependency stdlib + `openssl` helpers (`scripts/agent_identity.py`). Grant `Contents: R/W`, `Pull requests: R/W`, `Issues: R/W`, `Workflows: R/W` (gated by `.github/CODEOWNERS` + `main` branch protection), `Checks: R`, `Actions: R` (or `R/W` for reruns), `Metadata: R`, and **`Administration: No access`**.
2. **Single shared bot identity + per-agent provenance trailers and footers:** All AI agents (Claude, Antigravity, Grok, Codex) commit and push as `<bot_user_id>+<slug>[bot]@users.noreply.github.com` while appending `Agent: <harness>/<model>` trailers (preserving `Co-authored-by:`) and compact `<sub>Agent: ...</sub>` comment footers so individual agents/models stay distinguishable without managing multiple Apps. Uncommitted human edits in `data/labels/` committed by an agent per `AGENTS.md` use an explicit `--as-human` mode (`Author: Tomasz Kopacki`, `Committer: <slug>[bot]`).
3. **Mechanical guardrails & T3 Code MCP routing:** Add a local `pre-commit` guard (`scripts/check_agent_identity.py`) that blocks agent shells (detected via harness env vars) from committing as `tomasz@kopacki.net` unless explicitly flagged for human label edits, while staying silent in Tomasz's plain terminal and CI. Route git writes (`scripts/agent_git.py`) and GitHub writes (`scripts/agent_gh.py`) through the App token in `AGENTS.md` while keeping T3 Code read/link/watch tools (`link_pull_request`, `watch_pull_request`, `link_issue`, `pull_request_read`, `ci_logs`). Extend `scripts/check_pr_oversight.py` to summarize PR commit authorship.
4. **Rejected alternatives:** A dedicated machine user (e.g. reusing `tko-ventures-grokbot`, currently `admin` on this repo and scheduled for audit/downgrade in `#70`) lacks the official `[bot]` badge, relies on long-lived PATs, and fits awkwardly with GitHub's single-free-account ToS (kept only as fallback if signed feature-branch commits are ever mandated). Trailers or comment prefixes alone keep GitHub UI authorship as `tk-o` and cannot unlock `@tk-o` PR approvals.

## Consequences

- Unlocks formal `@tk-o` PR approvals and `CODEOWNERS` review enforcement on `main`, and enforces a hard GitHub permission boundary preventing agents from pushing directly to or merging `main`.
- Squash-merged agent PRs retain `<slug>[bot]` as the commit Author on `main` with GitHub's `Verified` signature; local feature-branch commits pushed via Git CLI show `Unverified` (GitHub does not attach SSH/GPG keys to `[bot]` accounts), which is acceptable while signed branch commits are not required.
- Rollout is tracked across Epic `#68` and sub-issues `#69` (ADR), `#70` (human-only App setup), `#71` (wrappers & tests), `#72` (guardrails & oversight), and `#73` (`AGENTS.md`, branch protection & E2E verification).
