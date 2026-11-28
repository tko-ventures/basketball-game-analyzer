"""Tests for the verification-phase PR oversight, digest, and Gitleaks wrapper scripts."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import pytest

from scripts.check_pr_oversight import analyze_diff, fetch_pr_reviews, format_brief, has_approver_approval, main
from scripts.run_gitleaks import _install_binary
from scripts.security_digest import _fetch_alert_feed, collect_gitleaks_allowlists


class _DummyResp:
    def read(self) -> bytes:
        return b"not-a-valid-tarball"

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _install_review_https_stub(
    monkeypatch: pytest.MonkeyPatch,
    pages: dict[int, tuple[int, dict[str, str], object]],
) -> list[str]:
    requested_paths: list[str] = []

    class _FakeHTTPSConnection:
        def __init__(self, host: str, timeout: int = 30) -> None:
            self.host = host
            self.timeout = timeout
            self._status = 500
            self._headers: dict[str, str] = {}
            self._payload = b"[]"

        def request(self, method: str, path: str, headers: dict[str, str] | None = None) -> None:
            requested_paths.append(path)
            page_num = int(path.rsplit("page=", 1)[1])
            status, resp_headers, payload = pages[page_num]
            self._status = status
            self._headers = resp_headers
            self._payload = json.dumps(payload).encode("utf-8")

        def getresponse(self):
            return SimpleNamespace(
                status=self._status,
                getheader=lambda key, default=None: self._headers.get(key, default),
                read=lambda: self._payload,
            )

        def close(self) -> None:
            return

    monkeypatch.setattr("http.client.HTTPSConnection", _FakeHTTPSConnection)
    return requested_paths


class TestAnalyzeDiff:
    def test_clean_pipeline_edit_needs_no_oversight(self):
        name_status = "M\tsrc/bga/cli.py\n"
        patch = "diff --git a/src/bga/cli.py b/src/bga/cli.py\n+++ b/src/bga/cli.py\n@@ -10,0 +11,1 @@\n+print('ok')\n"
        report = analyze_diff(name_status, patch)
        assert not report.needs_oversight
        assert report.protected_files == ()
        assert report.dependency_files == ()
        assert report.suppressions == ()

    @pytest.mark.parametrize(
        "touched_path",
        [
            pytest.param(".github/workflows/ci.yml", id="workflow"),
            pytest.param(".pre-commit-config.yaml", id="pre-commit"),
            pytest.param(".gitleaks.toml", id="gitleaks-config"),
            pytest.param("AGENTS.md", id="agents-md"),
            pytest.param("scripts/run_gitleaks.py", id="gitleaks-runner"),
            pytest.param("scripts/agent_identity.py", id="agent-identity"),
            pytest.param("scripts/agent_git.py", id="agent-git"),
            pytest.param("scripts/agent_gh.py", id="agent-gh"),
            pytest.param("scripts/check_agent_identity.py", id="check-agent-identity"),
            pytest.param("scripts/setup_github_app.py", id="setup-github-app"),
        ],
    )
    def test_flags_protected_governance_files(self, touched_path: str):
        report = analyze_diff(f"M\t{touched_path}\n", "")
        assert report.needs_oversight
        assert touched_path in report.protected_files

    @pytest.mark.parametrize(
        "touched_path",
        [
            pytest.param("uv.lock", id="uv-lock"),
            pytest.param("src/bga/remote/modal_train.py", id="modal-train"),
            pytest.param("src/bga/train/dispatch.py", id="dispatch"),
        ],
    )
    def test_flags_dependency_and_remote_package_files(self, touched_path: str):
        report = analyze_diff(f"M\t{touched_path}\n", "")
        assert report.needs_oversight
        assert touched_path in report.dependency_files

    @pytest.mark.parametrize(
        ("line_text", "expected_kind"),
        [
            pytest.param(
                "urllib.request.urlopen(req)  # noqa: S310  # scheme validated above", "noqa-security", id="inline"
            ),
            pytest.param("# ruff: noqa: S310  # scheme validated for entire script", "noqa-security", id="file-level"),
            pytest.param("x = 1  # noqa  # legacy module", "noqa-blanket", id="blanket-reasoned"),
        ],
    )
    def test_reasoned_noqa_forms_trigger_oversight_without_blocking(self, line_text: str, expected_kind: str):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -20,0 +21,1 @@\n"
            f"+{line_text}\n"
        )
        report = analyze_diff("M\tscripts/example.py\n", patch)
        assert report.needs_oversight
        assert len(report.suppressions) == 1
        assert report.suppressions[0].kind == expected_kind
        assert report.unreasoned_noqa == ()

    @pytest.mark.parametrize(
        "line_text",
        [
            pytest.param("urllib.request.urlopen(req)  # noqa: S310", id="inline-security"),
            pytest.param("# ruff: noqa: S310", id="file-level-security"),
            pytest.param("x = eval(user_input)  # noqa", id="blanket-inline"),
            pytest.param("# ruff: noqa", id="blanket-file-level"),
            pytest.param("# flake8: noqa", id="flake8-blanket"),
        ],
    )
    def test_unreasoned_security_or_blanket_noqa_is_flagged_as_blocking(self, line_text: str):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -20,0 +21,1 @@\n"
            f"+{line_text}\n"
        )
        report = analyze_diff("M\tscripts/example.py\n", patch)
        assert len(report.unreasoned_noqa) == 1
        assert report.unreasoned_noqa[0].line == 21

    def test_non_security_file_noqa_is_not_flagged(self):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -1,0 +1,1 @@\n"
            "+# ruff: noqa: E501  (sentences kept whole)\n"
        )
        report = analyze_diff("M\tscripts/example.py\n", patch)
        assert report.suppressions == ()
        assert report.unreasoned_noqa == ()

    @pytest.mark.parametrize(
        ("rel_path", "line_text", "expected_kind"),
        [
            pytest.param("scripts/run.sh", "# shellcheck disable=SC2086", "shellcheck-disable", id="shellcheck"),
            pytest.param("src/bga/foo.py", "val = 'x'  # gitleaks:allow", "gitleaks:allow", id="gitleaks-allow"),
            pytest.param(".github/workflows/ci.yml", "# zizmor: ignore[artipacked]", "zizmor-ignore", id="zizmor"),
            pytest.param("src/bga/foo.py", "subprocess.run(cmd)  # nosec", "nosec", id="nosec"),
        ],
    )
    def test_detects_other_inline_suppressions(self, rel_path: str, line_text: str, expected_kind: str):
        patch = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n@@ -1,0 +2,1 @@\n+{line_text}\n"
        report = analyze_diff(f"M\t{rel_path}\n", patch)
        assert len(report.suppressions) == 1
        assert report.suppressions[0].kind == expected_kind

    def test_markdown_docs_and_self_exempt_files_do_not_trigger_suppression_false_positives(self):
        patch = (
            "diff --git a/docs/notes.md b/docs/notes.md\n"
            "+++ b/docs/notes.md\n"
            "@@ -1,0 +2,1 @@\n"
            "+Never add `# noqa: S310` without reason.\n"
            "diff --git a/tests/test_pr_oversight.py b/tests/test_pr_oversight.py\n"
            "+++ b/tests/test_pr_oversight.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+line = '# noqa: S310'\n"
        )
        report = analyze_diff("M\tdocs/notes.md\nM\ttests/test_pr_oversight.py\n", patch)
        assert report.suppressions == ()
        assert report.unreasoned_noqa == ()

    def test_groups_commit_authors_in_descending_order(self):
        log_text = (
            "tko-agent[bot] <1+tko-agent[bot]@users.noreply.github.com>\n"
            "Tomasz Kopacki <tomasz@kopacki.net>\n"
            "tko-agent[bot] <1+tko-agent[bot]@users.noreply.github.com>\n"
        )
        report = analyze_diff("M\tsrc/bga/cli.py\n", "", log_text)
        assert len(report.commit_authors) == 2
        assert report.commit_authors[0].author.startswith("tko-agent[bot]")
        assert report.commit_authors[0].count == 2
        assert report.commit_authors[1].count == 1


class TestSuppressionApprovalGate:
    @pytest.mark.parametrize(
        ("reviews", "expected"),
        [
            pytest.param([], False, id="no-reviews"),
            pytest.param(
                [{"user": {"login": "other-user"}, "state": "APPROVED", "commit_id": "a" * 40}],
                False,
                id="wrong-approver",
            ),
            pytest.param(
                [{"user": {"login": "tk-o"}, "state": "APPROVED", "commit_id": "b" * 40}],
                False,
                id="stale-commit-approval",
            ),
            pytest.param(
                [
                    {"user": {"login": "tk-o"}, "state": "APPROVED", "commit_id": "a" * 40},
                    {"user": {"login": "tk-o"}, "state": "CHANGES_REQUESTED", "commit_id": "a" * 40},
                ],
                False,
                id="changes-requested-after-approval",
            ),
            pytest.param(
                [
                    {"user": {"login": "tk-o"}, "state": "APPROVED", "commit_id": "a" * 40},
                    {"user": {"login": "tk-o"}, "state": "DISMISSED", "commit_id": "a" * 40},
                ],
                False,
                id="dismissed-after-approval",
            ),
            pytest.param(
                [
                    {"user": {"login": "tk-o"}, "state": "APPROVED", "commit_id": "a" * 40},
                    {"user": {"login": "tk-o"}, "state": "COMMENTED", "commit_id": "a" * 40},
                ],
                True,
                id="comment-after-approval-stays-approved",
            ),
            pytest.param(
                [{"user": {"login": "tk-o"}, "state": "APPROVED", "commit_id": "a" * 40}],
                True,
                id="approved-on-head",
            ),
        ],
    )
    def test_has_approver_approval(self, reviews: list[dict[str, object]], expected: bool):
        assert has_approver_approval(reviews, "tk-o", "a" * 40) is expected

    def test_fetch_pr_reviews_paginates_until_last_page(self, monkeypatch: pytest.MonkeyPatch):
        page_one = [{"id": idx} for idx in range(100)]
        page_two = [{"id": 100}]
        _install_review_https_stub(
            monkeypatch,
            {
                1: (200, {"Link": '<https://api.github.com/next>; rel="next"'}, page_one),
                2: (200, {}, page_two),
            },
        )
        reviews = fetch_pr_reviews("tko-ventures/basketball-game-analyzer", 75, "")
        assert len(reviews) == 101

    def test_fetch_pr_reviews_raises_when_max_pages_exceeded(self, monkeypatch: pytest.MonkeyPatch):
        full_page = [{"id": idx} for idx in range(100)]
        _install_review_https_stub(
            monkeypatch,
            {
                1: (200, {"Link": '<https://api.github.com/next>; rel="next"'}, full_page),
                2: (200, {"Link": '<https://api.github.com/next>; rel="next"'}, full_page),
            },
        )
        with pytest.raises(RuntimeError, match="exceeded 2 review pages"):
            fetch_pr_reviews("tko-ventures/basketball-game-analyzer", 75, "", max_pages=2)


class TestFormatAndCli:
    def test_format_brief_clean_message_and_commit_authorship(self):
        log_text = "tko-agent[bot] <1+tko-agent[bot]@users.noreply.github.com>\n"
        report = analyze_diff("M\tsrc/bga/cli.py\n", "", log_text)
        text = format_brief(report)
        assert "Clean — no security/governance escalations." in text
        assert "### Commit authorship" in text
        assert "tko-agent[bot]" in text

    def test_strict_cli_exits_nonzero_on_unreasoned_noqa(self, monkeypatch, tmp_path):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+eval(x)  # noqa: S307\n"
        )
        responses = iter(["M\tscripts/example.py\n", patch, "tko-agent[bot] <1@users.noreply.github.com>\n"])
        monkeypatch.setattr(
            "scripts.check_pr_oversight.subprocess.check_output", lambda _cmd, text=True: next(responses)
        )
        summary = tmp_path / "summary.md"
        rc = main(["--base", "main", "--head", "HEAD", "--summary", str(summary), "--strict"])
        assert rc == 1
        assert "BLOCKED" in summary.read_text(encoding="utf-8")

    def test_strict_cli_allows_reasoned_noqa_with_oversight_notice(self, monkeypatch, tmp_path):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+urllib.request.urlopen(req)  # noqa: S310  # scheme validated above\n"
        )
        responses = iter(["M\tscripts/example.py\n", patch, ""])
        monkeypatch.setattr(
            "scripts.check_pr_oversight.subprocess.check_output", lambda _cmd, text=True: next(responses)
        )
        summary = tmp_path / "summary.md"
        rc = main(["--base", "main", "--head", "HEAD", "--summary", str(summary), "--strict"])
        assert rc == 0
        assert "Oversight required" in summary.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        ("reviews", "expected_rc"),
        [
            pytest.param([], 1, id="unapproved-suppression-blocked"),
            pytest.param([{"user": {"login": "tk-o"}, "state": "APPROVED", "commit_id": "c" * 40}], 0, id="approved"),
        ],
    )
    def test_require_suppression_approval_enforces_manager_signoff(
        self, monkeypatch: pytest.MonkeyPatch, reviews: list[dict[str, object]], expected_rc: int
    ):
        patch = (
            "diff --git a/src/bga/foo.py b/src/bga/foo.py\n"
            "+++ b/src/bga/foo.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+subprocess.run(cmd)  # noqa: S603  # fixed argv\n"
        )
        responses = iter(["M\tsrc/bga/foo.py\n", patch, ""])
        monkeypatch.setattr(
            "scripts.check_pr_oversight.subprocess.check_output", lambda _cmd, text=True: next(responses)
        )
        _install_review_https_stub(monkeypatch, {1: (200, {}, reviews)})
        rc = main([
            "--base",
            "main",
            "--head",
            "c" * 40,
            "--strict",
            "--require-suppression-approval",
            "--repo",
            "tko-ventures/basketball-game-analyzer",
            "--pr-number",
            "99",
        ])
        assert rc == expected_rc


class TestSecurityDigest:
    def test_paginated_alert_feed_counts_all_pages(self, monkeypatch):
        page_1 = [{"number": 1, "rule": {"id": "py/test", "security_severity_level": "high"}}] * 100
        page_2 = [{"number": 101, "rule": {"id": "py/test", "security_severity_level": "medium"}}]
        monkeypatch.setattr(
            "scripts.security_digest.subprocess.run",
            lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=json.dumps([page_1, page_2]), stderr=""),
        )
        summary = _fetch_alert_feed("owner/repo", "Code scanning", "code-scanning/alerts")
        assert summary.count == 101
        assert len(summary.highlights) == 5

    def test_collect_gitleaks_allowlists_includes_rule_scoped_tables(self):
        toml_text = (
            '[allowlist]\npaths = ["docs/"]\n[[rules]]\nid = "custom-rule"\n[[rules.allowlists]]\nregexes = ["dummy"]\n'
        )
        entries = collect_gitleaks_allowlists(toml_text)
        assert len(entries) == 2
        assert any("custom-rule" in entry.snippet and "rules.allowlists" in entry.snippet for entry in entries)


class TestRunGitleaks:
    def test_download_failure_raises_without_path_fallback(self, monkeypatch, tmp_path):
        def _fail(*_a, **_kw):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr("scripts.run_gitleaks.urllib.request.urlopen", _fail)
        with pytest.raises(RuntimeError, match="failed to download pinned gitleaks"):
            _install_binary(tmp_path / "gitleaks")

    def test_checksum_mismatch_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("scripts.run_gitleaks.urllib.request.urlopen", lambda *_a, **_kw: _DummyResp())
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            _install_binary(tmp_path / "gitleaks")
