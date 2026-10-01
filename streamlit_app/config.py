"""Configuration settings for Nirnaya Streamlit Showcase."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


def _load_env_file(env_path: Path) -> None:
    """Lightweight key=value parser for .env files."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val


# Attempt to discover and load .env
search_paths = [
    Path(".env"),
    Path("nirnaya_server/.env"),
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / ".env",
]

for p in search_paths:
    if p.exists():
        _load_env_file(p)
        break


@dataclass
class StreamlitSettings:
    """Settings for the Streamlit test & demo UI."""

    # Backend API connection
    api_url: str = os.getenv("NIRNAYA_API_URL", "http://localhost:8000").rstrip("/")
    api_key: str = os.getenv("NIRNAYA_API_KEY", "nir_live_root_secret_key_change_me")

    # Timeouts (in seconds) - Configurable to prevent timeouts on long CPU inference
    api_timeout_seconds: float = float(os.getenv("NIRNAYA_API_TIMEOUT_SECONDS", "300"))
    connect_timeout_seconds: float = float(os.getenv("NIRNAYA_CONNECT_TIMEOUT_SECONDS", "15"))

    # Authentication credentials
    admin_username: str = os.getenv("STREAMLIT_AUTH_USERNAME", "admin")
    admin_password: str = os.getenv("STREAMLIT_AUTH_PASSWORD", "nirnaya")
    extra_users_str: str = os.getenv("STREAMLIT_EXTRA_USERS", "")

    # UI preferences
    app_title: str = os.getenv("STREAMLIT_APP_TITLE", "Nirnaya Decision Engine UI")

    # Derived users dict
    users: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.users = {self.admin_username: self.admin_password}
        if self.extra_users_str:
            for pair in self.extra_users_str.split(","):
                if ":" in pair:
                    u, p = pair.split(":", 1)
                    self.users[u.strip()] = p.strip()

    def verify_credentials(self, username: str, password: str) -> bool:
        """Verifies username and password against configured store."""
        if not username or not password:
            return False
        expected_pass = self.users.get(username)
        return expected_pass is not None and expected_pass == password


settings = StreamlitSettings()
