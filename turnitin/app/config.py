"""
Environment & API Configuration Loader for VeriScholar v3.
Automatically reads turnitin/.env and parent workspace .env without ever logging secrets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return values
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                values[k] = v
    except Exception:
        pass
    return values


def load_env_defaults() -> None:
    """Populate os.environ from workspace/.env and turnitin/.env if not already set."""
    for candidate in (WORKSPACE_DIR / ".env", PROJECT_DIR / ".env"):
        for k, v in _parse_env_file(candidate).items():
            if k not in os.environ or not os.environ[k].strip():
                os.environ[k] = v


def get_gemini_api_key(override: Optional[str] = None) -> Optional[str]:
    if override and override.strip():
        return override.strip()
    load_env_defaults()
    val = (os.environ.get("GEMINI_API_KEY") or "").strip()
    return val or None


def get_optional_key(name: str) -> Optional[str]:
    load_env_defaults()
    val = (os.environ.get(name) or "").strip()
    return val or None


def get_groq_api_key(override: Optional[str] = None) -> Optional[str]:
    if override and override.strip():
        return override.strip()
    return get_optional_key("GROQ_API_KEY") or get_optional_key("GROQ_KEY")


def get_mistral_api_key(override: Optional[str] = None) -> Optional[str]:
    if override and override.strip():
        return override.strip()
    return get_optional_key("MISTRAL_API_KEY") or get_optional_key("MISTRAL_KEY")


def get_serper_api_key() -> Optional[str]:
    return get_optional_key("SERPER_API_KEY") or get_optional_key("SERPER_KEY")


def get_tavily_api_key() -> Optional[str]:
    return get_optional_key("TAVILY_API_KEY") or get_optional_key("TAVILY_KEY")


def get_core_api_key() -> Optional[str]:
    return get_optional_key("CORE_API_KEY") or get_optional_key("CORE_KEY")


# Auto-load on import
load_env_defaults()
