## Summary

<!-- 1–3 bullets explaining what changed and why. -->

## Linked Issues

<!-- Use `Part of #<n>` or `Closes #<n>` so PRs stay linked without auto-closing parent epics prematurely. -->

## Verification Gates

- [ ] `uv run pre-commit run --all-files` passed locally (incl. Ruff `S`, Gitleaks, `zizmor`, `shellcheck`)
- [ ] `uv run pytest -rs` passed locally (incl. doctests, gold leakage, and security policy tests)
- [ ] CI Security (`gitleaks`, `osv-pr`, `dependency-review`, `pr-oversight`) and CodeQL checked

## Manager Oversight (Decisions needed from Tomasz)

<!-- Agents: fill every row below explicitly; write `None` when clean. Never silence a scanner or modify protected files without documenting it here. -->

| Category | Details & 1-line rationale *(write `None` when clean)* |
|---|---|
| Protected security / governance files touched | None |
| New dependencies, git pins, or licence changes | None |
| New inline suppressions / allowlists (`# noqa: S*`, `gitleaks:allow`, etc.) | None |
| Decisions or repo-admin actions needed from Tomasz | None |
