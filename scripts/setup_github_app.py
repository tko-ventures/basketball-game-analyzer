#!/usr/bin/env python3
"""Two-click GitHub App Manifest setup helper storing credentials in ~/.config/bga/."""

from __future__ import annotations

import argparse
import html
import json
import re
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.agent_identity import (
    AppCredentials,
    default_config_dir,
    find_repo_root,
    github_api_json,
    mint_installation_token,
    parse_env_file,
    sign_app_jwt,
    write_secret_file,
)

CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SLUG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?$")
ORG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?$")
DEFAULT_PERMISSIONS: dict[str, str] = {
    "contents": "write",
    "pull_requests": "write",
    "issues": "write",
    "workflows": "write",
    "checks": "read",
    "actions": "write",
    "metadata": "read",
}


@dataclass
class ManifestSetupState:
    """In-memory state shared with the localhost OAuth/manifest callback handler."""

    org: str
    repo: str
    name: str
    config_dir: Path
    identity_file: Path
    csrf_state: str
    bound_port: int = 0
    app_id: str | None = None
    bot_slug: str | None = None
    bot_user_id: int | None = None
    error: str | None = None
    callback_done: threading.Event = field(default_factory=threading.Event)


def build_app_manifest(org: str, repo: str, name: str, redirect_url: str) -> dict[str, object]:
    """Build the GitHub App manifest payload matching ADR 2026-11-28 permissions."""
    repo_url = f"https://github.com/{org}/{repo}"
    return {
        "name": name,
        "url": repo_url,
        "hook_attributes": {"url": repo_url, "active": False},
        "redirect_url": redirect_url,
        "public": False,
        "default_permissions": dict(DEFAULT_PERMISSIONS),
        "default_events": [],
    }


def render_setup_page(state: ManifestSetupState) -> str:
    """Render the minimal HTML page submitting the pre-filled manifest to GitHub."""
    redirect_url = f"http://127.0.0.1:{state.bound_port}/callback"
    manifest = build_app_manifest(state.org, state.repo, state.name, redirect_url)
    manifest_json = html.escape(json.dumps(manifest), quote=True)
    action_url = html.escape(
        f"https://github.com/organizations/{urllib.parse.quote(state.org, safe='')}/settings/apps/new"
        f"?state={urllib.parse.quote(state.csrf_state, safe='')}",
        quote=True,
    )
    app_name = html.escape(state.name)
    repo_slug = html.escape(f"{state.org}/{state.repo}")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Create GitHub App ({app_name})</title></head>"
        "<body style='font-family:system-ui,sans-serif;max-width:640px;margin:48px auto;padding:0 16px'>"
        f"<h2>Step 1 of 2: Create <code>{app_name}</code> on <code>{repo_slug}</code></h2>"
        "<p>Permissions pre-filled per ADR 2026-11-28: Contents, Pull requests, Issues, Workflows, Actions (R/W); "
        "Checks &amp; Metadata (R); Administration (none); Webhooks disabled.</p>"
        f"<form method='post' action='{action_url}'>"
        f"<input type='hidden' name='manifest' value='{manifest_json}'>"
        f"<button type='submit' style='font-size:16px;padding:10px 18px'>Create {app_name} on GitHub</button>"
        "</form></body></html>"
    )


def render_redirect_page(install_url: str, bot_slug: str) -> str:
    """Render a safe HTML meta-refresh page directing the browser to the install page."""
    escaped_url = html.escape(install_url, quote=True)
    escaped_slug = html.escape(bot_slug)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='refresh' content='0;url={escaped_url}'>"
        f"<title>Redirecting to install {escaped_slug}[bot]</title></head>"
        "<body style='font-family:system-ui,sans-serif;max-width:640px;margin:48px auto;padding:0 16px'>"
        f"<h2>Step 2 of 2: Install <code>{escaped_slug}[bot]</code></h2>"
        f"<p>If you are not redirected automatically, <a href='{escaped_url}'>click here to install the App</a>.</p>"
        "</body></html>"
    )


def _fetch_bot_user_id(bot_slug: str) -> int:
    login = urllib.parse.quote(f"{bot_slug}[bot]", safe="")
    user_payload = github_api_json("GET", f"/users/{login}")
    if not isinstance(user_payload, dict) or "id" not in user_payload:
        raise RuntimeError(f"Could not look up bot user ID for {bot_slug}[bot].")
    return int(user_payload["id"])


def handle_manifest_callback(state: ManifestSetupState, query_string: str) -> str:
    """Validate callback params, exchange manifest code, save PEM, and return install URL."""
    if not ORG_RE.fullmatch(state.org):
        raise ValueError(f"Invalid organization login: {state.org!r}")
    params = urllib.parse.parse_qs(query_string, keep_blank_values=False)
    got_state = params.get("state", [""])[0]
    code = params.get("code", [""])[0]
    if not got_state or not secrets.compare_digest(got_state, state.csrf_state):
        raise ValueError("Invalid OAuth/manifest state token.")
    if not CODE_RE.fullmatch(code):
        raise ValueError("Invalid manifest conversion code.")

    conversion = github_api_json("POST", f"/app-manifests/{code}/conversions", body={})
    if not isinstance(conversion, dict) or not conversion.get("id") or not conversion.get("pem"):
        raise RuntimeError("Unexpected /app-manifests conversion response.")

    app_id = str(conversion["id"])
    bot_slug = str(conversion.get("slug") or state.name)
    if not SLUG_RE.fullmatch(bot_slug):
        raise ValueError(f"Invalid GitHub App slug returned by GitHub: {bot_slug!r}")
    pem_path = state.config_dir / "github-app.pem"
    write_secret_file(pem_path, str(conversion["pem"]))
    bot_user_id = _fetch_bot_user_id(bot_slug)

    state.app_id = app_id
    state.bot_slug = bot_slug
    state.bot_user_id = bot_user_id
    org_quoted = urllib.parse.quote(state.org, safe="")
    slug_quoted = urllib.parse.quote(bot_slug, safe="")
    return f"https://github.com/organizations/{org_quoted}/settings/apps/{slug_quoted}/installations"


def _make_handler(state: ManifestSetupState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/":
                body = render_setup_page(state).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/callback":
                try:
                    install_url = handle_manifest_callback(state, parsed.query)
                except Exception as exc:
                    state.error = str(exc)
                    state.callback_done.set()
                    msg = html.escape(f"Setup error: {exc}").encode("utf-8")
                    self.send_response(HTTPStatus.BAD_REQUEST)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                body = render_redirect_page(install_url, state.bot_slug or state.name).encode("utf-8")
                state.callback_done.set()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return _Handler


def find_org_installation_id(app_id: str, pem_path: Path, org: str) -> str | None:
    """Return the GitHub App installation ID for ``org``, or ``None`` if not yet installed."""
    jwt_token = sign_app_jwt(app_id, pem_path)
    payload = github_api_json("GET", "/app/installations", bearer=jwt_token)
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        account = item.get("account")
        if isinstance(account, dict) and str(account.get("login", "")).lower() == org.lower():
            return str(item["id"])
    return None


def poll_for_installation(
    app_id: str,
    pem_path: Path,
    org: str,
    *,
    timeout_s: int = 600,
    poll_interval_s: float = 3.0,
) -> str:
    """Poll GitHub until the App is installed on ``org`` and return its installation ID."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found = find_org_installation_id(app_id, pem_path, org)
        if found is not None:
            return found
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for App {app_id} installation on {org}.")


def update_repo_identity_file(identity_file: Path, bot_slug: str, bot_user_id: int) -> None:
    """Update ``.github/agent-identity.json`` with the confirmed ``bot_slug`` and ``bot_user_id``."""
    existing = json.loads(identity_file.read_text(encoding="utf-8")) if identity_file.is_file() else {}
    payload = {
        "org": existing.get("org", "tko-ventures"),
        "repo": existing.get("repo", "basketball-game-analyzer"),
        "bot_slug": bot_slug,
        "bot_user_id": bot_user_id,
        "human_name": existing.get("human_name", "Tomasz Kopacki"),
        "human_emails": list(existing.get("human_emails", ["tomasz@kopacki.net"])),
    }
    identity_file.parent.mkdir(parents=True, exist_ok=True)
    identity_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_app_env_file(
    config_dir: Path,
    *,
    app_id: str,
    installation_id: str,
    bot_slug: str,
    bot_user_id: int,
    pem_path: Path,
) -> Path:
    """Persist local App identifiers into ``~/.config/bga/github-app.env`` with ``0600`` mode."""
    env_path = config_dir / "github-app.env"
    lines = [
        f"BGA_GITHUB_APP_ID={app_id}",
        f"BGA_GITHUB_INSTALLATION_ID={installation_id}",
        f"BGA_GITHUB_BOT_SLUG={bot_slug}",
        f"BGA_GITHUB_BOT_USER_ID={bot_user_id}",
        f"BGA_GITHUB_APP_PEM={pem_path}",
    ]
    write_secret_file(env_path, "\n".join(lines) + "\n")
    return env_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description="Two-click GitHub App Manifest setup helper.")
    parser.add_argument("--org", default="tko-ventures", help="GitHub organization login.")
    parser.add_argument("--repo", default="basketball-game-analyzer", help="Repository name.")
    parser.add_argument("--name", default="tko-agent", help="GitHub App name/slug.")
    parser.add_argument("--port", type=int, default=0, help="Localhost callback port (default: ephemeral).")
    parser.add_argument("--config-dir", type=Path, default=default_config_dir(), help="Local secret dir.")
    parser.add_argument(
        "--identity-file",
        type=Path,
        default=repo_root / ".github" / "agent-identity.json",
        help="Path to committed .github/agent-identity.json.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for browser confirmation.")
    parser.add_argument("--no-browser", action="store_true", help="Print URL without opening a browser tab.")
    parser.add_argument(
        "--complete-install",
        action="store_true",
        help="Skip manifest creation and complete installation polling from existing .pem.",
    )
    parser.add_argument("--app-id", help="App ID when using --complete-install without prior env file.")
    parser.add_argument("--slug", help="App slug when using --complete-install without prior env file.")
    return parser.parse_args(argv)


def _finalize_installation(
    *,
    app_id: str,
    bot_slug: str,
    bot_user_id: int,
    org: str,
    config_dir: Path,
    identity_file: Path,
    timeout_s: int,
) -> int:
    pem_path = config_dir / "github-app.pem"
    print(f"Waiting for installation of {bot_slug}[bot] on {org}...", flush=True)
    installation_id = poll_for_installation(app_id, pem_path, org, timeout_s=timeout_s)
    env_path = write_app_env_file(
        config_dir,
        app_id=app_id,
        installation_id=installation_id,
        bot_slug=bot_slug,
        bot_user_id=bot_user_id,
        pem_path=pem_path,
    )
    update_repo_identity_file(identity_file, bot_slug, bot_user_id)
    creds = AppCredentials(
        app_id=app_id,
        installation_id=installation_id,
        bot_slug=bot_slug,
        bot_user_id=bot_user_id,
        pem_path=pem_path,
    )
    entry = mint_installation_token(creds)
    print(
        f"Setup complete! App ID={app_id}, Installation ID={installation_id}, "
        f"Bot User ID={bot_user_id}, token expires_at={int(entry.expires_at)}. "
        f"Saved env to {env_path} and updated {identity_file}.",
        flush=True,
    )
    return 0


def _run_complete_install(args: argparse.Namespace) -> int:
    existing = parse_env_file(args.config_dir / "github-app.env")
    app_id = args.app_id or existing.get("BGA_GITHUB_APP_ID", "")
    bot_slug = args.slug or existing.get("BGA_GITHUB_BOT_SLUG") or args.name
    if not app_id:
        print("Missing --app-id (or BGA_GITHUB_APP_ID in github-app.env) for --complete-install.", file=sys.stderr)
        return 2
    raw_uid = existing.get("BGA_GITHUB_BOT_USER_ID", "")
    bot_user_id = int(raw_uid) if raw_uid else _fetch_bot_user_id(bot_slug)
    return _finalize_installation(
        app_id=app_id,
        bot_slug=bot_slug,
        bot_user_id=bot_user_id,
        org=args.org,
        config_dir=args.config_dir,
        identity_file=args.identity_file,
        timeout_s=args.timeout,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.complete_install:
        return _run_complete_install(args)

    state = ManifestSetupState(
        org=args.org,
        repo=args.repo,
        name=args.name,
        config_dir=args.config_dir,
        identity_file=args.identity_file,
        csrf_state=secrets.token_urlsafe(24),
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(state))
    server.daemon_threads = True
    state.bound_port = int(server.server_address[1])
    setup_url = f"http://127.0.0.1:{state.bound_port}/"
    print(f"Setup URL: {setup_url}", flush=True)
    if not args.no_browser:
        webbrowser.open(setup_url)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = state.callback_done.wait(timeout=args.timeout)
    finally:
        server.shutdown()
        server.server_close()

    if not completed:
        print(f"Timed out after {args.timeout}s waiting for browser manifest creation.", file=sys.stderr)
        return 1
    if state.error or not state.app_id or not state.bot_slug or state.bot_user_id is None:
        print(f"Manifest setup failed: {state.error}", file=sys.stderr)
        return 1

    print(
        f"App created ({state.bot_slug}[bot], app_id={state.app_id}, bot_user_id={state.bot_user_id}).",
        flush=True,
    )
    return _finalize_installation(
        app_id=state.app_id,
        bot_slug=state.bot_slug,
        bot_user_id=state.bot_user_id,
        org=args.org,
        config_dir=args.config_dir,
        identity_file=args.identity_file,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
