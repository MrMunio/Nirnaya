"""Configuration management for Nirnaya Server."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _load_env_file(env_path: Path) -> None:
    """Lightweight .env file parser if python-dotenv is not installed."""
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


# Attempt to load .env from current directory or parent directory
for candidate in [Path(".env"), Path("nirnaya_server/.env"), Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent.parent.parent / ".env"]:
    if candidate.exists():
        _load_env_file(candidate)
        break


class Settings:
    """Application settings loaded from environment variables."""

    # Server binding
    HOST: str = os.getenv("NIRNAYA_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("NIRNAYA_PORT", "8000"))
    ENV: str = os.getenv("NIRNAYA_ENV", "production")

    # Model Engine Configuration
    # Defaults to qwen3-0.6b-gguf for lightweight deployment & testing
    DEFAULT_MODEL: str = os.getenv("NIRNAYA_DEFAULT_MODEL", "qwen3-0.6b-gguf")
    MODELS_CONFIG_PATH: str = os.getenv(
        "NIRNAYA_MODELS_CONFIG",
        str(Path(__file__).resolve().parent / "core" / "models_config.json")
    )
    WARMUP_ON_STARTUP: bool = os.getenv("WARMUP_ON_STARTUP", "false").lower() in ("true", "1", "yes")

    # Cyclic Permutation Pooling for Choice Questions
    # Set to 1 to cap/disable permutations (single evaluation pass, no cyclic rotation)
    # If not set, defaults to the model's configured n_perms (e.g., 3)
    N_PERMS: Optional[int] = (
        int(os.getenv("NIRNAYA_N_PERMS"))
        if os.getenv("NIRNAYA_N_PERMS") is not None and os.getenv("NIRNAYA_N_PERMS").strip() != ""
        else None
    )

    # Security & SQLite Database
    DB_PATH: str = os.getenv("NIRNAYA_DB_PATH", "nirnaya_server/data/nirnaya.db")
    INITIAL_ADMIN_KEY: str = os.getenv("INITIAL_ADMIN_KEY", "nir_live_root_secret_key_change_me")
    REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "true").lower() in ("true", "1", "yes")

    @classmethod
    def get_resolved_db_path(cls) -> Path:
        p = Path(cls.DB_PATH)
        if not p.is_absolute():
            # Resolve relative to project root or server dir
            base_dir = Path(__file__).resolve().parent.parent
            if p.parts and p.parts[0] == "nirnaya_server":
                p = base_dir / Path(*p.parts[1:])
            else:
                p = base_dir / p
        return p


settings = Settings()
