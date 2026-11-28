"""Tests for GitHub App agent identity, git/gh wrappers, pre-commit guard, and manifest setup."""

from __future__ import annotations

import io
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import agent_gh, agent_git, agent_identity, check_agent_identity, setup_github_app

GITHUB_ENV_VARS = (
    "BGA_GITHUB_APP_ID",
    "BGA_GITHUB_INSTALLATION_ID",
    "BGA_GITHUB_BOT_SLUG",
    "BGA_GITHUB_BOT_USER_ID",
    "BGA_GITHUB_APP_PEM",
    "BGA_AGENT_MODEL",
    "BGA_AGENT_NAME",
    "BGA_COMMIT_AS_HUMAN",
    "ANTIGRAVITY_AGENT",
    "CLAUDECODE",
    "CLAUDE_AGENT_SDK_VERSION",
    "CODEX_SANDBOX",
    "CODEX_CI",
    "GROK_AGENT",
    "CURSOR_AGENT",
    "OPENCODE",
    "T3CODE_THREAD_ID",
)


def _install_https_stub(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[tuple[str, str], tuple[int, object]],
) -> list[tuple[str, str, dict[str, str]]]:
    calls: list[tuple[str, str, dict[str, str]]] = []

    class _FakeHTTPSConnection:
        def __init__(self, host: str, timeout: int = 30) -> None:
            self.host = host
            self.timeout = timeout
            self._status = 500
            self._payload: bytes = b"{}"

        def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None):
            calls.append((method, path, dict(headers or {})))
            status, payload = routes[(method, path)]
            self._status = status
            self._payload = json.dumps(payload).encode("utf-8")

        def getresponse(self):
            return SimpleNamespace(status=self._status, read=lambda: self._payload)

        def close(self) -> None:
            return

    monkeypatch.setattr("http.client.HTTPSConnection", _FakeHTTPSConnection)
    return calls


def _install_subprocess_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch: str = "feat/test",
    sig_bytes: bytes = b"sig-bytes",
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def _fake_run(cmd, input=None, env=None, capture_output=False, check=False):  # noqa: A002 (mirrors subprocess.run signature)
        if cmd and cmd[0] == "openssl":
            return SimpleNamespace(returncode=0, stdout=sig_bytes, stderr=b"")
        entry: dict[str, object] = {"cmd": list(cmd), "env": dict(env or {})}
        if env and "GIT_ASKPASS" in env:
            entry["askpass_text"] = Path(env["GIT_ASKPASS"]).read_text(encoding="utf-8")
        if "--body-file" in cmd:
            body_idx = cmd.index("--body-file")
            if body_idx + 1 < len(cmd):
                entry["body_file_text"] = Path(cmd[body_idx + 1]).read_text(encoding="utf-8")
        captured.append(entry)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("subprocess.check_output", lambda *_a, **_kw: f"{branch}\n")
    return captured


@pytest.fixture
def configured_app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    for var_name in GITHUB_ENV_VARS:
        monkeypatch.delenv(var_name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cfg_dir = tmp_path / ".config" / "bga"
    cfg_dir.mkdir(parents=True)
    pem_path = cfg_dir / "github-app.pem"
    pem_path.write_text("DUMMY-PEM-BYTES", encoding="utf-8")
    env_path = setup_github_app.write_app_env_file(
        cfg_dir,
        app_id="4242",
        installation_id="777",
        bot_slug="test-bot",
        bot_user_id=555000,
        pem_path=pem_path,
    )
    return {"home": tmp_path, "cfg_dir": cfg_dir, "pem_path": pem_path, "env_path": env_path}


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

    def test_mint_installation_token_uses_valid_cache_without_network(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cache_path = tmp_path / "cache.json"
        expected = f"cached-{tmp_path.name}"
        cache_path.write_text(
            json.dumps({
                "app_id": "4242",
                "installation_id": "777",
                "token": expected,
                "expires_at": 2000.0,
            }),
            encoding="utf-8",
        )
        api_calls = _install_https_stub(monkeypatch, {})
        creds = agent_identity.load_app_credentials()
        entry = agent_identity.mint_installation_token(creds, cache_path=cache_path, now=1000.0)
        assert entry == agent_identity.TokenCacheEntry(token=expected, expires_at=2000.0)
        assert api_calls == []

    def test_mint_installation_token_signs_jwt_and_caches_0600_response(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cache_path = tmp_path / "cache" / "bga" / "github-app-token.json"
        expected = f"fresh-{tmp_path.name}"
        _install_subprocess_stub(monkeypatch)
        api_calls = _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2026-11-28T13:00:00Z"},
                )
            },
        )
        creds = agent_identity.load_app_credentials()
        entry = agent_identity.mint_installation_token(creds, cache_path=cache_path, now=1000.0)
        assert entry.token == expected
        assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert saved["token"] == expected
        assert api_calls[0][2]["Authorization"].startswith("Bearer ")


class TestAgentGitWrapper:
    @pytest.mark.parametrize(
        ("push_args", "current_branch", "expected_blocked"),
        [
            pytest.param([], "main", True, id="bare-push-on-main"),
            pytest.param(["origin"], "main", True, id="origin-on-main"),
            pytest.param(["origin", "HEAD"], "main", True, id="head-on-main"),
            pytest.param(["origin", "main"], "feat/x", True, id="explicit-main"),
            pytest.param(["origin", "+feat/x:refs/heads/main"], "feat/x", True, id="force-refspec-to-main"),
            pytest.param(["origin", "refs/heads/*:refs/heads/*"], "feat/x", True, id="wildcard-all-heads"),
            pytest.param(["origin", "refs/heads/*"], "feat/x", True, id="wildcard-bare-heads"),
            pytest.param(["origin", ":refs/heads/*"], "feat/x", True, id="wildcard-delete-heads"),
            pytest.param(["origin", "refs/heads/m*:refs/heads/m*"], "feat/x", True, id="wildcard-m-prefix"),
            pytest.param(["--delete", "origin", "main"], "feat/x", True, id="delete-main"),
            pytest.param(["--all"], "feat/x", True, id="push-all"),
            pytest.param(["origin", "feat/x"], "feat/x", False, id="feature-branch"),
            pytest.param(["origin", "refs/heads/feat/*:refs/heads/feat/*"], "feat/x", False, id="safe-feat-wildcard"),
            pytest.param(["-u", "origin", "HEAD"], "feat/x", False, id="head-on-feature"),
            pytest.param(["origin", "HEAD:feat/y"], "main", False, id="push-to-non-main-refspec"),
        ],
    )
    def test_is_main_push_detects_protected_and_wildcard_targets(
        self, push_args: list[str], current_branch: str, expected_blocked: bool
    ):
        assert agent_git.is_main_push(push_args, current_branch) is expected_blocked

    def test_commit_resolves_bot_identity_and_appends_agent_trailer(
        self, configured_app_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ):
        captured = _install_subprocess_stub(monkeypatch)
        monkeypatch.setenv("ANTIGRAVITY_AGENT", "1")

        rc = agent_git.main(["commit", "-m", "feat: add test"])
        assert rc == 0
        assert captured[0]["cmd"] == ["git", "commit", "--trailer", "Agent: antigravity", "-m", "feat: add test"]
        env = captured[0]["env"]
        assert isinstance(env, dict)
        assert env["GIT_AUTHOR_NAME"] == "test-bot[bot]"
        assert env["GIT_AUTHOR_EMAIL"] == "555000+test-bot[bot]@users.noreply.github.com"
        assert env["GIT_COMMITTER_EMAIL"] == "555000+test-bot[bot]@users.noreply.github.com"
        assert "BGA_COMMIT_AS_HUMAN" not in env

    def test_commit_as_human_attributes_author_to_human_and_committer_to_bot(
        self, configured_app_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ):
        captured = _install_subprocess_stub(monkeypatch)

        rc = agent_git.main(["commit", "--as-human", "-m", "data: Tomasz labels"])
        assert rc == 0
        assert captured[0]["cmd"] == ["git", "commit", "-m", "data: Tomasz labels"]
        env = captured[0]["env"]
        assert isinstance(env, dict)
        assert env["GIT_AUTHOR_NAME"] == "Tomasz Kopacki"
        assert env["GIT_AUTHOR_EMAIL"] == "tomasz@kopacki.net"
        assert env["GIT_COMMITTER_NAME"] == "test-bot[bot]"
        assert env["BGA_COMMIT_AS_HUMAN"] == "1"

    def test_push_mints_token_via_https_and_uses_ephemeral_askpass(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        expected = f"ephemeral-{tmp_path.name}"
        captured = _install_subprocess_stub(monkeypatch, branch="feat/test")
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2099-11-28T13:00:00Z"},
                )
            },
        )

        rc = agent_git.main(["push", "-u", "origin", "feat/test"])
        assert rc == 0
        cmd = captured[0]["cmd"]
        assert isinstance(cmd, list)
        assert not any(expected in str(part) for part in cmd)
        env = captured[0]["env"]
        assert isinstance(env, dict)
        assert env["BGA_GIT_ASKPASS_TOKEN"] == expected
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert "BGA_GIT_ASKPASS_TOKEN" in str(captured[0]["askpass_text"])

    def test_push_refuses_main_without_calling_github_api(
        self, configured_app_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys
    ):
        _install_subprocess_stub(monkeypatch, branch="main")
        api_calls = _install_https_stub(monkeypatch, {})
        rc = agent_git.main(["push", "origin", "main"])
        assert rc == 2
        assert api_calls == []
        assert "Refusing to push directly to main" in capsys.readouterr().err


class TestAgentGhWrapper:
    def test_pr_comment_injects_minted_gh_token_and_agent_footer(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        expected = f"gh-{tmp_path.name}"
        captured = _install_subprocess_stub(monkeypatch)
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2099-11-28T13:00:00Z"},
                )
            },
        )

        rc = agent_gh.main(["--agent", "claude/opus-5.5", "pr", "comment", "74", "--body", "Looks good"])
        assert rc == 0
        assert captured[0]["cmd"] == [
            "gh",
            "pr",
            "comment",
            "74",
            "--body",
            "Looks good\n\n<sub>Agent: claude/opus-5.5</sub>",
        ]
        env = captured[0]["env"]
        assert isinstance(env, dict)
        assert env["GH_TOKEN"] == expected

    def test_pr_create_rewrites_body_file_without_mutating_source_file(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        expected = f"gh-file-{tmp_path.name}"
        src_body = tmp_path / "pr_body.md"
        src_body.write_text("## Summary\n\n- Fixed bug.\n", encoding="utf-8")
        captured = _install_subprocess_stub(monkeypatch)
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2099-11-28T13:00:00Z"},
                )
            },
        )

        rc = agent_gh.main(["--agent", "antigravity", "pr", "create", "--title", "Fix", "--body-file", str(src_body)])
        assert rc == 0
        assert src_body.read_text(encoding="utf-8") == "## Summary\n\n- Fixed bug.\n"
        assert captured[0]["body_file_text"] == "## Summary\n\n- Fixed bug.\n\n<sub>Agent: antigravity</sub>"

    def test_pr_comment_rewrites_stdin_body_file(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        expected = f"gh-stdin-{tmp_path.name}"
        captured = _install_subprocess_stub(monkeypatch)
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2099-11-28T13:00:00Z"},
                )
            },
        )
        monkeypatch.setattr("sys.stdin", io.StringIO("From stdin\n"))

        rc = agent_gh.main(["--agent", "grok/4.7", "pr", "comment", "75", "-F", "-"])
        assert rc == 0
        assert captured[0]["body_file_text"] == "From stdin\n\n<sub>Agent: grok/4.7</sub>"

    def test_pr_comment_no_footer_leaves_body_file_untouched(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        expected = f"gh-nofooter-{tmp_path.name}"
        src_body = tmp_path / "comment.md"
        src_body.write_text("Plain body\n", encoding="utf-8")
        captured = _install_subprocess_stub(monkeypatch)
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2099-11-28T13:00:00Z"},
                )
            },
        )

        rc = agent_gh.main(["--no-footer", "pr", "comment", "75", "--body-file", str(src_body)])
        assert rc == 0
        assert captured[0]["cmd"] == ["gh", "pr", "comment", "75", "--body-file", str(src_body)]
        assert captured[0]["body_file_text"] == "Plain body\n"

    def test_pr_reply_targets_review_comment_replies_endpoint(
        self, configured_app_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        expected = f"gh-reply-{tmp_path.name}"
        captured = _install_subprocess_stub(monkeypatch)
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app/installations/777/access_tokens"): (
                    201,
                    {"token": expected, "expires_at": "2099-11-28T13:00:00Z"},
                )
            },
        )

        rc = agent_gh.main(["--agent", "antigravity", "pr-reply", "74", "12345", "--body", "Fixed"])
        assert rc == 0
        assert captured[0]["cmd"] == [
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

    def test_render_setup_and_redirect_pages_escape_html_fields(self, tmp_path: Path):
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

        redirect = setup_github_app.render_redirect_page(
            "https://github.com/organizations/tko-ventures/settings/apps/tko-agent/installations",
            'tko-agent"<>',
        )
        assert "<meta http-equiv='refresh'" in redirect
        assert "tko-agent&quot;&lt;&gt;" in redirect

    def test_callback_validates_state_slug_saves_0600_pem_and_returns_install_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
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

        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app-manifests/validCode_1/conversions"): (
                    201,
                    {"id": 4242, "slug": "tko-agent", "pem": "DUMMY-PEM-PAYLOAD"},
                ),
                ("GET", "/users/tko-agent%5Bbot%5D"): (200, {"id": 989898}),
            },
        )
        install_url = setup_github_app.handle_manifest_callback(state, "code=validCode_1&state=expected-csrf")
        assert install_url == "https://github.com/organizations/tko-ventures/settings/apps/tko-agent/installations"
        pem_path = tmp_path / "github-app.pem"
        assert pem_path.read_text(encoding="utf-8") == "DUMMY-PEM-PAYLOAD"
        assert stat.S_IMODE(pem_path.stat().st_mode) == 0o600
        assert state.app_id == "4242"
        assert state.bot_user_id == 989898

    @pytest.mark.parametrize(
        "bad_slug",
        [
            pytest.param("evil\r\nSet-Cookie: a=b", id="crlf-injection"),
            pytest.param("evil/path", id="path-slash"),
            pytest.param("-leading-dash", id="leading-dash"),
        ],
    )
    def test_callback_rejects_malformed_slug(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_slug: str):
        state = setup_github_app.ManifestSetupState(
            org="tko-ventures",
            repo="basketball-game-analyzer",
            name="tko-agent",
            config_dir=tmp_path,
            identity_file=tmp_path / "agent-identity.json",
            csrf_state="expected-csrf",
            bound_port=8080,
        )
        _install_https_stub(
            monkeypatch,
            {
                ("POST", "/app-manifests/validCode_1/conversions"): (
                    201,
                    {"id": 4242, "slug": bad_slug, "pem": "DUMMY-PEM-PAYLOAD"},
                ),
            },
        )
        with pytest.raises(ValueError, match="slug"):
            setup_github_app.handle_manifest_callback(state, "code=validCode_1&state=expected-csrf")

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
