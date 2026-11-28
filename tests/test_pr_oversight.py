"""Tests for the verification-phase PR oversight audit script."""

from __future__ import annotations

import pytest

from scripts.check_pr_oversight import analyze_diff, format_brief, main


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

    def test_reasoned_security_noqa_triggers_oversight_without_unreasoned_flag(self):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -20,0 +21,1 @@\n"
            "+urllib.request.urlopen(req)  # noqa: S310  # scheme validated above\n"
        )
        report = analyze_diff("M\tscripts/example.py\n", patch)
        assert report.needs_oversight
        assert len(report.suppressions) == 1
        assert report.suppressions[0].kind == "noqa-security"
        assert report.unreasoned_noqa == ()

    def test_unreasoned_security_noqa_is_flagged_as_blocking(self):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -20,0 +21,1 @@\n"
            "+urllib.request.urlopen(req)  # noqa: S310\n"
        )
        report = analyze_diff("M\tscripts/example.py\n", patch)
        assert len(report.unreasoned_noqa) == 1
        assert report.unreasoned_noqa[0].line == 21

    @pytest.mark.parametrize(
        ("rel_path", "line_text", "expected_kind"),
        [
            pytest.param("scripts/run.sh", "# shellcheck disable=SC2086", "shellcheck-disable", id="shellcheck"),
            pytest.param("src/bga/foo.py", "token = 'x'  # gitleaks:allow", "gitleaks:allow", id="gitleaks-allow"),
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


class TestFormatAndCli:
    def test_format_brief_clean_message(self):
        report = analyze_diff("M\tsrc/bga/cli.py\n", "")
        text = format_brief(report)
        assert "Clean — no security/governance escalations." in text

    def test_strict_cli_exits_nonzero_on_unreasoned_noqa(self, monkeypatch, tmp_path):
        patch = (
            "diff --git a/scripts/example.py b/scripts/example.py\n"
            "+++ b/scripts/example.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+eval(x)  # noqa: S307\n"
        )
        responses = iter(["M\tscripts/example.py\n", patch])
        monkeypatch.setattr("scripts.check_pr_oversight._run_git", lambda _args: next(responses))
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
        responses = iter(["M\tscripts/example.py\n", patch])
        monkeypatch.setattr("scripts.check_pr_oversight._run_git", lambda _args: next(responses))
        summary = tmp_path / "summary.md"
        rc = main(["--base", "main", "--head", "HEAD", "--summary", str(summary), "--strict"])
        assert rc == 0
        assert "Oversight required" in summary.read_text(encoding="utf-8")
