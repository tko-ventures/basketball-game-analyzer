# Free security scanner stack and human-oversight verification contract

- **Date:** 2026-11-28
- **Status:** Proposed
- **Area:** Infra / Process
- **Decider:** Tomasz

## Context

`basketball-game-analyzer` is a public repository with frequent contributions from AI coding agents. A baseline audit (2026-11-28) found GitHub Secret Scanning, Push Protection, Dependabot Security Updates, CodeQL, and `main` required status checks disabled; `.pre-commit-config.yaml` checked only PEM private keys (`detect-private-key`); `scripts/**/*.py` blanket-ignored all Ruff `S` (`flake8-bandit`) rules; and nothing flagged when an agent weakened checks, added unvetted dependencies, or inserted inline suppressions to make a failing check pass.

## Decision

Adopt the **Lean V1** free-for-public-repo security stack and an explicit agent-to-manager oversight contract:

1. **Secrets:** Enable GitHub Secret Scanning, Push Protection, non-provider patterns, and validity checks, backed by the MIT **Gitleaks** CLI in `.pre-commit-config.yaml` and `.github/workflows/security.yml` (verified by SHA-256). Explicitly reject `gitleaks/gitleaks-action`, which requires a paid license for GitHub Organizations (`tko-ventures`).
2. **Supply chain & licences:** Enable Dependabot Security Updates with `.github/dependabot.yml` (`uv`, `github-actions`), **OSV-Scanner** (`google/osv-scanner-action` PR delta blocking + push/weekly SARIF), **Dependency Review** (`actions/dependency-review-action` blocking `high`+ CVEs and `AGPL-3.0*` / `GPL-3.0*` licences per [2026-11-18](2026-11-18-open-weights-and-permissive-licence-assumptions.md)), and policy tests enforcing pinned remote Modal/Colab dependencies.
3. **SAST, Workflows, Shell & AI-footgun rules:** Run GitHub **CodeQL** (`python`, `javascript-typescript`, `actions`), narrow `scripts/**/*.py` Ruff ignores from `"S"` to explicit benign rules (`S101`, `S311`, `S603`, `S607`), enforce **zizmor** and **shellcheck**, SHA-pin all workflow actions with `persist-credentials: false`, and enforce repo policy invariants in `tests/test_security_policy.py` (`torch.load(..., weights_only=True)`, forbidden packages/models `ultralytics`/`yolo26`/`sam3`, remote package pins, Trojan Source Unicode guard, reasoned `# noqa: S*`). Semgrep OSS is deferred unless custom AST tests prove insufficient.
4. **Verification-phase human oversight ("agents managing their manager"):** Protect governance/security paths via `.github/CODEOWNERS`, require a **Manager Oversight** summary box via `.github/pull_request_template.md`, audit PR diffs in CI (`scripts/check_pr_oversight.py`), roll up alerts and active suppressions via `scripts/security_digest.py`, and codify anti-bypass rules in `AGENTS.md`.
5. **Severity & gating policy:** Hard-fail PRs on leaked secrets, newly introduced `high`+ CVEs, forbidden licences, unpinned remote deps, unreasoned `# noqa: S*` suppressions, and high-severity SAST/workflow findings. Surface pre-existing transitive advisories and medium/low alerts in GitHub Security and the digest for Tomasz's triage.

## Consequences

- Agents must never bypass hooks (`--no-verify`) or add inline suppressions, allowlists, or ignore rules without an inline rationale and explicit escalation in the PR's Manager Oversight section.
- Repository-admin settings (Secret Scanning, Push Protection, Dependabot updates, and `main` branch protection required checks) remain human-only actions for Tomasz (`#62`).
