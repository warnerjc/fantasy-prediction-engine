"""Repo-wide runtime configuration, read from a `.env` file at the repo root.

Any layer can `import config` and pull a credential/setting by name. Real
environment variables always win over `.env` (`override=False`), so containers
and CI can inject secrets without a file. Copy `.env.example` to `.env` and fill
it in — `.env` is git-ignored, `.env.example` is the tracked list of every
property name.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env", override=False)


def get(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Return env var ``name``. Blank counts as unset. ``required`` raises a
    message pointing at `.env.example` rather than returning ``None``."""
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        if required:
            raise RuntimeError(
                f"missing required env var {name!r}. Copy .env.example to .env "
                f"(at {REPO_ROOT}) and set it."
            )
        return default
    return val


def _path(name: str, default: Path) -> Path:
    raw = get(name)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def yahoo_config() -> dict:
    """Yahoo Fantasy API settings. ``client_id`` / ``client_secret`` are only
    read (and required) when actually talking to Yahoo — see ``data.yahoo``."""
    return {
        "client_id": get("YAHOO_CLIENT_ID"),
        "client_secret": get("YAHOO_CLIENT_SECRET"),
        "redirect_uri": get("YAHOO_REDIRECT_URI", "https://localhost:8000/callback"),
        # Yahoo Fantasy access is set by the app's API Permissions, not an OAuth
        # scope string — leave this empty. (Only set it for the OIDC `openid` flow.)
        "scope": get("YAHOO_SCOPE", ""),
        "token_path": _path("YAHOO_TOKEN_PATH", REPO_ROOT / "data" / "cache" / "yahoo_token.json"),
        "league_id": get("YAHOO_LEAGUE_ID", "236625"),
    }
