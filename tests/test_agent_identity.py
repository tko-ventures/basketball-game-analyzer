"""Tests for GitHub App agent identity, git/gh wrappers, pre-commit guard, and manifest setup."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import agent_gh, agent_git, agent_identity, check_agent_identity, setup_github_app


class TestAgentIdentityHelpers:
    @pytest.mark.parametrize(
        ("environ", "expected"),
        [
            pytest.param({}, None, id="plain-human-shell"),
            pytest.param({"ANTIGRAVITY_AGENT": "1", "T3CODE_THREAD_ID": "t1"}, "antigravity", id="antigravity"),
            pytest.param({"CLAUDECODE": "1", "T3CODE_THREAD_ID": "t1"}, "claude", id="claude-code"),
            pytest.param({"CLAUDE_AGENT_SDK_VERSION": "0.1", "T3CODE_THREAD_ID": "t1"}, "claude", id="claude-sdk"),
            pytest.param({"CODEX_SANDBOX": "seatbelt"}, "codex", id="codex"),
            pytest.param({"GROK_AGENT": "1"}, "grok", id="grok"),
            pytest.param({"T3CODE_THREAD_ID": "t1"}, "t3-agent", id="t3-fallback"),
        ],
    )
    def test_detects_harness_from_environment(self, environ: dict[str, str], expected: str | None):
        assert agent_identity.detect_agent_harness(environ) == expected

    def test_formats_bot_git_name_and_noreply_email(self):
        assert agent_identity.bot_git_name("tko-agent") == "tko-agent[bot]"
        assert agent_identity.bot_git_email("tko-agent", 987654) == "987654+tko-agent[bot]@users.noreply.github.com"

    def test_format_agent_label_prefers_explicit_then_env_then_harness(self):
        assert agent_identity.format_agent_label("antigravity/gemini-4", {"CLAUDECODE": "1"}) == "antigravity/gemini-4"
        assert agent_identity.format_agent_label(None, {"BGA_AGENT_MODEL": "claude/opus-5.5"}) == "claude/opus-5.5"
        assert agent_identity.format_agent_label(None, {"CLAUDECODE": "1"}) == "claude"
        assert agent_identity.format_agent_label(None, {}) == "ai-agent"

    def test_write_secret_file_sets_0600_mode(self, tmp_path: Path):
        target = tmp_path / "nested" / "secret.env"
        agent_identity.write_secret_file(target, "KEY=value\n")
        assert target.read_text(encoding="utf-8") == "KEY=value\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_mint_installation_token_uses_valid_cache_without_network(self, tmp_path: Path, monkeypatch):
        pem_path = tmp_path / "app.pem"
        pem_path.write_text("dummy", encoding="utf-8")
        cache_path = tmp_path / "cache.json"
        expected = f"cached-{tmp_path.name}"
        cache_path.write_text(
            json.dumps({
                "app_id": "111",
                "installation_id": "222",
                "token": expected,
                "expires_at": 2000.0,
            }),
            encoding="utf-8",
        )
        creds = agent_identity.AppCredentials("111", "222", "tko-agent", 333, pem_path)
        monkeypatch.setattr("scripts.agent_identity.github_api_json", lambda *_a, **_kw: pytest.fail("unexpected API"))
        entry = agent_identity.mint_installation_token(creds, cache_path=cache_path, now=1000.0)
        assert entry == agent_identity.TokenCacheEntry(token=expected, expires_at=2000.0)

    def test_mint_installation_token_refreshes_expired_cache_and_writes_0600(self, tmp_path: Path, monkeypatch):
        pem_path = tmp_path / "app.pem"
        pem_path.write_text("dummy", encoding="utf-8")
        cache_path = tmp_path / "cache.json"
        expected = f"fresh-{tmp_path.name}"
        creds = agent_identity.AppCredentials("111", "222", "tko-agent", 333, pem_path)
        monkeypatch.setattr("scripts.agent_identity.sign_app_jwt", lambda *_a, **_kw: "signed.jwt.value")
        monkeypatch.setattr(
            "scripts.agent_identity.github_api_json",
            lambda *_a, **_kw: {"token": expected, "expires_at": "2026-11-28T13:00:00Z"},
        )
        entry = agent_identity.mint_installation_token(creds, cache_path=cache_path, now=1000.0)
        assert entry.token == expected
        assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert saved["token"] == expected


class TestAgentGitWrapper:
    @pytest.mark.parametrize(
        ("push_args", "current_branch", "expected_blocked"),
        [
            pytest.param([], "main", True, id="bare-push-on-main"),
            pytest.param(["origin"], "main", True, id="origin-on-main"),
            pytest.param(["origin", "HEAD"], "main", True, id="head-on-main"),
            pytest.param(["origin", "main"], "feat/x", True, id="explicit-main"),
            pytest.param(["origin", "+feat/x:refs/heads/main"], "feat/x", True, id="force-refspec-to-main"),
            pytest.param(["--all"], "feat/x", True, id="push-all"),
            pytest.param(["origin", "feat/x"], "feat/x", False, id="feature-branch"),
            pytest.param(["-u", "origin", "HEAD"], "feat/x", False, id="head-on-feature"),
            pytest.param(["origin", "HEAD:feat/y"], "main", False, id="push-to-non-main-refspec"),
        ],
    )
    def test_is_main_push_detects_protected_target(
        self, push_args: list[str], current_branch: str, expected_blocked: bool
    ):
        assert agent_git.is_main_push(push_args, current_branch) is expected_blocked

    def test_commit_sets_bot_identity_and_agent_trailer(self, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_run(cmd, env=None, check=False):
            captured["cmd"] = cmd
            captured["env"] = env
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("scripts.agent_git.resolve_bot_identity", lambda: ("tko-agent", 555))
        monkeypatch.setattr("scripts.agent_git.subprocess.run", _fake_run)
        monkeypatch.setenv("ANTIGRAVITY_AGENT", "1")
        monkeypatch.delenv("BGA_AGENT_MODEL", raising=False)
        monkeypatch.delenv("BGA_AGENT_NAME", raising=False)

        rc = agent_git.main(["commit", "-m", "feat: add test"])
        assert rc == 0
        assert captured["cmd"] == ["git", "commit", "--trailer", "Agent: antigravity", "-m", "feat: add test"]
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["GIT_AUTHOR_NAME"] == "tko-agent[bot]"
        assert env["GIT_AUTHOR_EMAIL"] == "555+tko-agent[bot]@users.noreply.github.com"
        assert env["GIT_COMMITTER_EMAIL"] == "555+tko-agent[bot]@users.noreply.github.com"
        assert "BGA_COMMIT_AS_HUMAN" not in env

    def test_commit_as_human_attributes_author_to_human_and_committer_to_bot(self, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_run(cmd, env=None, check=False):
            captured["cmd"] = cmd
            captured["env"] = env
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("scripts.agent_git.resolve_bot_identity", lambda: ("tko-agent", 555))
        monkeypatch.setattr("scripts.agent_git.subprocess.run", _fake_run)

        rc = agent_git.main(["commit", "--as-human", "-m", "data: Tomasz labels"])
        assert rc == 0
        assert captured["cmd"] == ["git", "commit", "-m", "data: Tomasz labels"]
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["GIT_AUTHOR_NAME"] == "Tomasz Kopacki"
        assert env["GIT_AUTHOR_EMAIL"] == "tomasz@kopacki.net"
        assert env["GIT_COMMITTER_NAME"] == "tko-agent[bot]"
        assert env["BGA_COMMIT_AS_HUMAN"] == "1"

    def test_push_uses_ephemeral_askpass_and_keeps_token_out_of_argv(self, tmp_path: Path, monkeypatch):
        pem_path = tmp_path / "app.pem"
        pem_path.write_text("dummy", encoding="utf-8")
        expected = f"ephemeral-{tmp_path.name}"
        captured: dict[str, object] = {}

        def _fake_run(cmd, env=None, check=False):
            captured["cmd"] = cmd
            captured["env"] = env
            captured["askpass_text"] = Path(env["GIT_ASKPASS"]).read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("scripts.agent_git._current_branch", lambda: "feat/test")
        monkeypatch.setattr(
            "scripts.agent_git.load_app_credentials",
            lambda: agent_identity.AppCredentials("111", "222", "tko-agent", 555, pem_path),
        )
        monkeypatch.setattr(
            "scripts.agent_git.mint_installation_token",
            lambda _c: agent_identity.TokenCacheEntry(expected, 9999999999.0),
        )
        monkeypatch.setattr("scripts.agent_git.subprocess.run", _fake_run)

        rc = agent_git.main(["push", "-u", "origin", "feat/test"])
        assert rc == 0
        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert not any(expected in part for part in cmd)
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["BGA_GIT_ASKPASS_TOKEN"] == expected
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert "BGA_GIT_ASKPASS_TOKEN" in str(captured["askpass_text"])

    def test_push_refuses_main_without_minting_token(self, monkeypatch, capsys):
        monkeypatch.setattr("scripts.agent_git._current_branch", lambda: "main")
        monkeypatch.setattr(
            "scripts.agent_git.load_app_credentials", lambda: pytest.fail("should not load creds on main push")
        )
        rc = agent_git.main(["push", "origin", "main"])
        assert rc == 2
        assert "Refusing to push directly to main" in capsys.readouterr().err


class TestAgentGhWrapper:
    def test_pr_comment_injects_gh_token_and_agent_footer(self, tmp_path: Path, monkeypatch):
        pem_path = tmp_path / "app.pem"
        pem_path.write_text("dummy", encoding="utf-8")
        expected = f"gh-{tmp_path.name}"
        captured: dict[str, object] = {}

        def _fake_run(cmd, env=None, check=False):
            captured["cmd"] = cmd
            captured["env"] = env
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(
            "scripts.agent_gh.load_app_credentials",
            lambda: agent_identity.AppCredentials("111", "222", "tko-agent", 555, pem_path),
        )
        monkeypatch.setattr(
            "scripts.agent_gh.mint_installation_token",
            lambda _c: agent_identity.TokenCacheEntry(expected, 9999999999.0),
        )
        monkeypatch.setattr("scripts.agent_gh.subprocess.run", _fake_run)

        rc = agent_gh.main(["--agent", "claude/opus-5.5", "pr", "comment", "74", "--body", "Looks good"])
        assert rc == 0
        assert captured["cmd"] == [
            "gh",
            "pr",
            "comment",
            "74",
            "--body",
            "Looks good\n\n<sub>Agent: claude/opus-5.5</sub>",
        ]
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["GH_TOKEN"] == expected

    def test_pr_reply_targets_review_comment_replies_endpoint(self, tmp_path: Path, monkeypatch):
        pem_path = tmp_path / "app.pem"
        pem_path.write_text("dummy", encoding="utf-8")
        captured: dict[str, object] = {}

        def _fake_run(cmd, env=None, check=False):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(
            "scripts.agent_gh.load_app_credentials",
            lambda: agent_identity.AppCredentials("111", "222", "tko-agent", 555, pem_path),
        )
        monkeypatch.setattr(
            "scripts.agent_gh.mint_installation_token",
            lambda _c: agent_identity.TokenCacheEntry("gh-ephemeral-val", 9999999999.0),
        )
        monkeypatch.setattr("scripts.agent_gh.subprocess.run", _fake_run)

        rc = agent_gh.main(["--agent", "antigravity", "pr-reply", "74", "12345", "--body", "Fixed"])
        assert rc == 0
        assert captured["cmd"] == [
            "gh",
            "api",
            "-X",
            "POST",
            "repos/tko-ventures/basketball-game-analyzer/pulls/74/comments/12345/replies",
            "-f",
            "body=Fixed\n\n<sub>Agent: antigravity</sub>",
        ]


class TestCheckAgentIdentityGuard:
    @pytest.mark.parametrize(
        ("environ", "expected_rc"),
        [
            pytest.param({"GIT_AUTHOR_EMAIL": "tomasz@kopacki.net"}, 0, id="human-shell-passes"),
            pytest.param(
                {"GITHUB_ACTIONS": "true", "ANTIGRAVITY_AGENT": "1", "GIT_AUTHOR_EMAIL": "tomasz@kopacki.net"},
                0,
                id="ci-passes",
            ),
            pytest.param({"ANTIGRAVITY_AGENT": "1"}, 0, id="unset-author-email-passes"),
            pytest.param(
                {"ANTIGRAVITY_AGENT": "1", "GIT_AUTHOR_EMAIL": "555+tko-agent[bot]@users.noreply.github.com"},
                0,
                id="bot-email-passes",
            ),
            pytest.param(
                {
                    "ANTIGRAVITY_AGENT": "1",
                    "GIT_AUTHOR_EMAIL": "tomasz@kopacki.net",
                    "BGA_COMMIT_AS_HUMAN": "1",
                },
                0,
                id="explicit-as-human-passes",
            ),
            pytest.param(
                {"ANTIGRAVITY_AGENT": "1", "GIT_AUTHOR_EMAIL": "tomasz@kopacki.net"},
                1,
                id="agent-with-human-email-blocked",
            ),
            pytest.param(
                {"CLAUDECODE": "1", "GIT_AUTHOR_EMAIL": "Tomasz@Kopacki.net"},
                1,
                id="case-insensitive-human-email-blocked",
            ),
        ],
    )
    def test_precommit_identity_guard(self, environ: dict[str, str], expected_rc: int):
        assert check_agent_identity.check_commit_identity(environ) == expected_rc


class TestSetupGithubAppManifest:
    def test_manifest_excludes_administration_and_disables_webhooks(self):
        manifest = setup_github_app.build_app_manifest(
            "tko-ventures",
            "basketball-game-analyzer",
            "tko-agent",
            "http://127.0.0.1:43210/callback",
        )
        assert manifest["public"] is False
        assert manifest["hook_attributes"] == {
            "url": "https://github.com/tko-ventures/basketball-game-analyzer",
            "active": False,
        }
        perms = manifest["default_permissions"]
        assert isinstance(perms, dict)
        assert "administration" not in perms
        assert perms["contents"] == "write"
        assert perms["pull_requests"] == "write"
        assert perms["workflows"] == "write"

    def test_render_setup_page_escapes_html_fields(self, tmp_path: Path):
        state = setup_github_app.ManifestSetupState(
            org="tko-ventures",
            repo="basketball-game-analyzer",
            name='bot<script>"xss',
            config_dir=tmp_path,
            identity_file=tmp_path / "agent-identity.json",
            csrf_state="csrf123",
            bound_port=8080,
        )
        page = setup_github_app.render_setup_page(state)
        assert "<script>" not in page
        assert "bot&lt;script&gt;" in page

    def test_callback_validates_state_saves_0600_pem_and_returns_install_url(self, tmp_path: Path, monkeypatch):
        state = setup_github_app.ManifestSetupState(
            org="tko-ventures",
            repo="basketball-game-analyzer",
            name="tko-agent",
            config_dir=tmp_path,
            identity_file=tmp_path / "agent-identity.json",
            csrf_state="expected-csrf",
            bound_port=8080,
        )
        with pytest.raises(ValueError, match="state"):
            setup_github_app.handle_manifest_callback(state, "code=abc123&state=wrong")

        def _fake_api(method: str, path: str, **_kw):
            if path.startswith("/app-manifests/"):
                return {"id": 4242, "slug": "tko-agent", "pem": "DUMMY-PEM-PAYLOAD"}
            return {"id": 989898}

        monkeypatch.setattr("scripts.setup_github_app.github_api_json", _fake_api)
        install_url = setup_github_app.handle_manifest_callback(state, "code=validCode_1&state=expected-csrf")
        assert install_url == "https://github.com/organizations/tko-ventures/settings/apps/tko-agent/installations"
        pem_path = tmp_path / "github-app.pem"
        assert pem_path.read_text(encoding="utf-8") == "DUMMY-PEM-PAYLOAD"
        assert stat.S_IMODE(pem_path.stat().st_mode) == 0o600
        assert state.app_id == "4242"
        assert state.bot_user_id == 989898

    def test_write_app_env_and_update_identity_json(self, tmp_path: Path):
        pem_path = tmp_path / "github-app.pem"
        env_path = setup_github_app.write_app_env_file(
            tmp_path,
            app_id="4242",
            installation_id="777",
            bot_slug="tko-agent",
            bot_user_id=989898,
            pem_path=pem_path,
        )
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
        parsed = agent_identity.parse_env_file(env_path)
        assert parsed["BGA_GITHUB_APP_ID"] == "4242"
        assert parsed["BGA_GITHUB_INSTALLATION_ID"] == "777"
        assert parsed["BGA_GITHUB_BOT_USER_ID"] == "989898"

        id_file = tmp_path / "agent-identity.json"
        setup_github_app.update_repo_identity_file(id_file, "tko-agent", 989898)
        updated = json.loads(id_file.read_text(encoding="utf-8"))
        assert updated["bot_slug"] == "tko-agent"
        assert updated["bot_user_id"] == 989898
