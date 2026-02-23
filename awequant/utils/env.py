
"""Environment variable helpers used across AweQuant."""

from __future__ import annotations

import os


TRUTHFUL = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: str | bool | None = "0") -> bool:
    """Return ``True`` when an env var is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        if default is None:
            return False
        if isinstance(default, bool):
            return default
        value = default
    return str(value).strip().lower() in TRUTHFUL


__all__ = ["env_flag"]
