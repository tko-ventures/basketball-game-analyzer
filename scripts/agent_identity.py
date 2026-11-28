"""Shared GitHub App identity detection, JWT signing, and token minting helpers."""

from __future__ import annotations

import base64
import http.client
import json
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: Ordered harness env markers (specific harnesses checked before generic T3 Code catch-all).
AGENT_ENV_MARKERS: tuple[tuple[str, str], ...] = (
    ("ANTIGRAVITY_AGENT", "antigravity"),
    ("CLAUDECODE", "claude"),
    ("CLAUDE_AGENT_SDK_VERSION", "claude"),
    ("CODEX_SANDBOX", "codex"),
    ("CODEX_CI", "codex"),
    ("GROK_AGENT", "grok"),
    ("CURSOR_AGENT", "cursor"),
    ("OPENCODE", "opencode"),
    ("T3CODE_THREAD_ID", "t3-agent"),
)

GITHUB_API_HOST = "api.github.com"
JWT_LIFETIME_S = 540
JWT_CLOCK_SKEW_S = 60
TOKEN_MIN_REMAINING_S = 600


@dataclass(frozen=True)
class RepoIdentityConfig:
    """Repository-committed non-secret identity metadata."""

    org: str
    repo: str
    bot_slug: str
    bot_user_id: int | None
    human_name: str
    human_emails: tuple[str, ...]


@dataclass(frozen=True)
class AppCredentials:
    """Local GitHub App credentials resolved from config and env files."""

    app_id: str
    installation_id: str
    bot_slug: str
    bot_user_id: int
    pem_path: Path


@dataclass(frozen=True)
class TokenCacheEntry:
    """Cached short-lived GitHub App installation access token."""

    token: str
    expires_at: float


def default_config_dir() -> Path:
    """Return the local user config directory for bga secrets outside the repo."""
    return Path.home() / ".config" / "bga"


def default_env_path() -> Path:
    """Return the default GitHub App metadata env file path."""
    return default_config_dir() / "github-app.env"


def default_pem_path() -> Path:
    """Return the default GitHub App private key file path."""
    return default_config_dir() / "github-app.pem"


def default_cache_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the user-only token cache file path."""
    env = os.environ if environ is None else environ
    xdg_cache = env.get("XDG_CACHE_HOME", "").strip()
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "bga" / "github-app-token.json"


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root containing ``.github/agent-identity.json`` or ``.git``."""
    anchor = Path(__file__).resolve() if start is None else start.resolve()
    candidates = [anchor, *anchor.parents]
    for parent in candidates:
        if (parent / ".github" / "agent-identity.json").is_file():
            return parent
    for parent in candidates:
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def load_repo_identity(repo_root: Path | None = None) -> RepoIdentityConfig:
    """Load non-secret identity metadata from ``.github/agent-identity.json``."""
    root = find_repo_root() if repo_root is None else repo_root
    path = root / ".github" / "agent-identity.json"
    if not path.is_file():
        return RepoIdentityConfig(
            org="tko-ventures",
            repo="basketball-game-analyzer",
            bot_slug="tko-agent",
            bot_user_id=None,
            human_name="Tomasz Kopacki",
            human_emails=("tomasz@kopacki.net",),
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_user_id = raw.get("bot_user_id")
    bot_user_id = int(raw_user_id) if raw_user_id is not None else None
    emails = tuple(str(item).strip().lower() for item in raw.get("human_emails", []) if str(item).strip())
    return RepoIdentityConfig(
        org=str(raw.get("org", "tko-ventures")),
        repo=str(raw.get("repo", "basketball-game-analyzer")),
        bot_slug=str(raw.get("bot_slug", "tko-agent")),
        bot_user_id=bot_user_id,
        human_name=str(raw.get("human_name", "Tomasz Kopacki")),
        human_emails=emails or ("tomasz@kopacki.net",),
    )


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` pairs from an optional local env file."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        val = raw_val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        values[key.strip()] = val
    return values


def _resolve_setting(
    key: str,
    environ: Mapping[str, str],
    env_values: Mapping[str, str],
    fallback: str = "",
) -> str:
    env_val = environ.get(key, "").strip()
    if env_val:
        return env_val
    file_val = env_values.get(key, "").strip()
    return file_val or fallback


def resolve_bot_identity(
    repo_root: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, int]:
    """Resolve ``(bot_slug, bot_user_id)`` without requiring private key or API credentials."""
    env = os.environ if environ is None else environ
    cfg = load_repo_identity(repo_root)
    env_values = parse_env_file(default_env_path() if env_file is None else env_file)
    bot_slug = _resolve_setting("BGA_GITHUB_BOT_SLUG", env, env_values, cfg.bot_slug)
    raw_user_id = _resolve_setting(
        "BGA_GITHUB_BOT_USER_ID",
        env,
        env_values,
        str(cfg.bot_user_id) if cfg.bot_user_id is not None else "",
    )
    if not bot_slug:
        raise RuntimeError("Missing bot slug: set bot_slug in .github/agent-identity.json or BGA_GITHUB_BOT_SLUG.")
    if not raw_user_id:
        raise RuntimeError(
            "Missing bot user ID: run `uv run scripts/setup_github_app.py` to populate "
            ".github/agent-identity.json and ~/.config/bga/github-app.env."
        )
    try:
        return (bot_slug, int(raw_user_id))
    except ValueError as exc:
        raise RuntimeError(f"Invalid BGA_GITHUB_BOT_USER_ID: {raw_user_id!r}") from exc


def load_app_credentials(
    repo_root: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppCredentials:
    """Load GitHub App credentials from environ, ``~/.config/bga/github-app.env``, and repo config."""
    env = os.environ if environ is None else environ
    resolved_env_file = default_env_path() if env_file is None else env_file
    env_values = parse_env_file(resolved_env_file)
    bot_slug, bot_user_id = resolve_bot_identity(repo_root=repo_root, env_file=resolved_env_file, environ=env)
    app_id = _resolve_setting("BGA_GITHUB_APP_ID", env, env_values)
    installation_id = _resolve_setting("BGA_GITHUB_INSTALLATION_ID", env, env_values)
    pem_raw = _resolve_setting("BGA_GITHUB_APP_PEM", env, env_values, str(default_pem_path()))
    pem_path = Path(pem_raw).expanduser()

    missing = [
        name
        for name, val in (
            ("BGA_GITHUB_APP_ID", app_id),
            ("BGA_GITHUB_INSTALLATION_ID", installation_id),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing GitHub App configuration ({', '.join(missing)}) in {resolved_env_file}. "
            "Run `uv run scripts/setup_github_app.py` first."
        )
    if not pem_path.is_file():
        raise RuntimeError(
            f"GitHub App private key not found at {pem_path}. Run `uv run scripts/setup_github_app.py` first."
        )
    return AppCredentials(
        app_id=app_id,
        installation_id=installation_id,
        bot_slug=bot_slug,
        bot_user_id=bot_user_id,
        pem_path=pem_path,
    )


def bot_git_name(bot_slug: str) -> str:
    """Return the canonical Git author/committer name for a GitHub App bot."""
    return f"{bot_slug}[bot]"


def bot_git_email(bot_slug: str, bot_user_id: int) -> str:
    """Return the canonical GitHub noreply email that renders the ``[bot]`` badge."""
    return f"{bot_user_id}+{bot_slug}[bot]@users.noreply.github.com"


def detect_agent_harness(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the active AI agent harness name from environment markers, or ``None``."""
    env = os.environ if environ is None else environ
    for var_name, harness in AGENT_ENV_MARKERS:
        if env.get(var_name, "").strip():
            return harness
    return None


def is_agent_session(environ: Mapping[str, str] | None = None) -> bool:
    """Return ``True`` when running inside a recognized AI agent shell."""
    return detect_agent_harness(environ) is not None


def format_agent_label(explicit: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Resolve the provenance label for git trailers and comment footers."""
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env = os.environ if environ is None else environ
    for var_name in ("BGA_AGENT_MODEL", "BGA_AGENT_NAME"):
        val = env.get(var_name, "").strip()
        if val:
            return val
    return detect_agent_harness(env) or "ai-agent"


def write_secret_file(path: Path, content: str) -> None:
    """Write ``content`` atomically with ``0600`` permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        tmp_path.chmod(0o600)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_app_jwt(app_id: str, pem_path: Path, now: int | None = None) -> str:
    """Sign a short-lived RS256 GitHub App JWT using ``openssl``."""
    issued = int(time.time()) if now is None else int(now)
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    payload = _b64url(
        json.dumps(
            {
                "iat": issued - JWT_CLOCK_SKEW_S,
                "exp": issued + JWT_LIFETIME_S,
                "iss": str(app_id),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signing_input = f"{header}.{payload}"
    proc = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(pem_path)],
        input=signing_input.encode("ascii"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"openssl JWT signing failed (exit {proc.returncode}): {err}")
    return f"{signing_input}.{_b64url(proc.stdout)}"


def github_api_json(
    method: str,
    api_path: str,
    *,
    bearer: str | None = None,
    body: dict[str, object] | None = None,
    timeout: int = 30,
) -> object:
    """Execute an HTTPS request against ``api.github.com`` and decode JSON."""
    if not api_path.startswith("/"):
        raise ValueError(f"GitHub API path must start with '/': {api_path!r}")
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "bga-agent-identity",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    payload_bytes: bytes | None = None
    if body is not None:
        payload_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    conn = http.client.HTTPSConnection(GITHUB_API_HOST, timeout=timeout)
    try:
        conn.request(method, api_path, body=payload_bytes, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
    finally:
        conn.close()

    if resp.status >= 400:
        snippet = raw[:400].replace("\n", " ")
        raise RuntimeError(f"GitHub API {method} {api_path} returned {resp.status}: {snippet}")
    return json.loads(raw) if raw.strip() else {}


def parse_iso_timestamp(value: str) -> float:
    """Parse an ISO-8601 UTC timestamp such as ``2026-11-28T13:00:00Z`` into epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _read_cached_token(cache_path: Path, app_id: str, installation_id: str, now_ts: float) -> TokenCacheEntry | None:
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if str(cached.get("app_id", "")) != app_id or str(cached.get("installation_id", "")) != installation_id:
        return None
    token_val = str(cached.get("token", "")).strip()
    expires_at = cached.get("expires_at")
    if not token_val or not isinstance(expires_at, (int, float)):
        return None
    if float(expires_at) - now_ts < TOKEN_MIN_REMAINING_S:
        return None
    return TokenCacheEntry(token=token_val, expires_at=float(expires_at))


def mint_installation_token(
    creds: AppCredentials,
    *,
    cache_path: Path | None = None,
    now: float | None = None,
) -> TokenCacheEntry:
    """Return a valid cached or freshly minted GitHub App installation access token."""
    now_ts = time.time() if now is None else float(now)
    resolved_cache = default_cache_path() if cache_path is None else cache_path
    cached = _read_cached_token(resolved_cache, creds.app_id, creds.installation_id, now_ts)
    if cached is not None:
        return cached

    jwt_token = sign_app_jwt(creds.app_id, creds.pem_path, now=int(now_ts))
    payload = github_api_json(
        "POST",
        f"/app/installations/{creds.installation_id}/access_tokens",
        bearer=jwt_token,
        body={},
    )
    if not isinstance(payload, dict) or not payload.get("token") or not payload.get("expires_at"):
        raise RuntimeError("Unexpected GitHub access_tokens response payload.")
    entry = TokenCacheEntry(
        token=str(payload["token"]),
        expires_at=parse_iso_timestamp(str(payload["expires_at"])),
    )
    cache_doc = json.dumps(
        {
            "app_id": creds.app_id,
            "installation_id": creds.installation_id,
            "token": entry.token,
            "expires_at": entry.expires_at,
        },
        indent=2,
    )
    write_secret_file(resolved_cache, cache_doc + "\n")
    return entry
