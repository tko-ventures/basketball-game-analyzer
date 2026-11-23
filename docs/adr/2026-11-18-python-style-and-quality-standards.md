# Borda Python style, pre-commit hooks, and squash-merge workflow

- **Date:** 2026-11-18 *(git policy from 2026-10-29 & 2026-11-17)*
- **Status:** Accepted
- **Area:** Process
- **Decider:** Tomasz

## Context

Multiple sessions and contributors need uniform Python quality, reliable test discipline, and conflict-free git history across milestone PRs.

## Decision

1. **Python coding & testing style:** Follow Jirka Borovec's ([Borda](https://github.com/Borda)) rules from `Borda/AI-Rig` and his vision repos, enforced through pinned `.pre-commit-config.yaml` hooks and `pyproject.toml`:
   - 120-character lines, Google docstrings on all public objects, `Examples:` doctests on every public function in `src/` (run by `pytest`), full type annotations, and ruff complexity limits (McCabe ≤ 12, branches ≤ 12, statements ≤ 50, returns ≤ 6, args ≤ 12).
   - `scripts/` is exempt from annotation/docstring/naming/bandit rules (still requires a module docstring); `codespell` reports only; `src/bga/` and `tools/` get no exemptions.
   - Tests follow arrange–act–assert, one scenario per test, no loops/branches in test bodies, `@pytest.mark.parametrize`, and mocking only at process boundaries.
2. **Git & PR workflow:**
   - Work on feature branches (`feat/m<N>-<slug>` or scoped branches); commit through `uv run git commit` so hooks run.
   - Never push to `main` or merge a PR without Tomasz.
   - Tomasz merges PRs with **Squash and merge**. For stacked branches after a squash, verify tree equality against `main` and record `main` via `git merge -s ours` (no history rewrite, no force-push).

## Consequences

- `uv run pre-commit run --all-files && uv run pytest` must pass before any merge.
- Detailed style rules live in `AGENTS.md` (and `.pre-commit-config.yaml` / `pyproject.toml`).
