"""Runtime paths and settings for the Live-Document backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root (Live-Document/) so backend can import modules/...
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BACKEND_DIR.parent
FRONTEND_DIST = WEB_DIR / "frontend" / "dist"

DATA_DIR = Path(os.getenv("LIVE_DOC_DATA_DIR", BACKEND_DIR / "data")).resolve()
OUTPUTS_DIR = DATA_DIR / "outputs"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "live_document.db"

HOST = os.getenv("LIVE_DOC_HOST", "127.0.0.1")
PORT = int(os.getenv("LIVE_DOC_PORT", "8000"))

# Comma-separated list of allowed CORS origins; "*" allows all.
CORS_ORIGINS = [
    o.strip() for o in os.getenv("LIVE_DOC_CORS_ORIGINS", "*").split(",") if o.strip()
]

# When set, serve the built frontend (web/frontend/dist) from the same port.
SERVE_FRONTEND = os.getenv("LIVE_DOC_SERVE_FRONTEND", "0") == "1"

for _d in (DATA_DIR, OUTPUTS_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Authentication (required for public deployment) ----
# Radeon Cloud's rc-tunnel assigns a public URL reachable from the internet;
# per the official guide, the app itself must enforce login. All /api/*
# endpoints except /api/v1/auth/login require the access token below.
AUTH_TOKEN_FILE = DATA_DIR / "auth_token.txt"
AUTH_ENABLED = os.getenv("LIVE_DOC_AUTH_DISABLED", "0") != "1"


def _load_or_create_auth_token() -> str:
    """Token from env, else persisted file, else generate & persist."""
    env_token = os.getenv("LIVE_DOC_AUTH_TOKEN", "").strip()
    if env_token:
        return env_token
    if AUTH_TOKEN_FILE.exists():
        existing = AUTH_TOKEN_FILE.read_text().strip()
        if existing:
            return existing
    import secrets

    token = secrets.token_urlsafe(24)
    AUTH_TOKEN_FILE.write_text(token + "\n")
    try:
        AUTH_TOKEN_FILE.chmod(0o600)
    except OSError:
        pass
    return token


AUTH_TOKEN = _load_or_create_auth_token()
