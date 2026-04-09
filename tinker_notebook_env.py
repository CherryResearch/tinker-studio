from __future__ import annotations

import os
import platform
from typing import Any

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback
    winreg = None


TINKER_API_KEY_ENV = "TINKER_API_KEY"


def ensure_tinker_api_key(required: bool = True) -> dict[str, Any]:
    process_value = os.environ.get(TINKER_API_KEY_ENV)
    if process_value:
        return _build_result(process_value, "process")

    registry_value, source = _read_windows_env_var(TINKER_API_KEY_ENV)
    if registry_value:
        os.environ[TINKER_API_KEY_ENV] = registry_value
        return _build_result(registry_value, source)

    result = _build_result(None, None)
    if required:
        raise RuntimeError(
            "Could not find TINKER_API_KEY in the process environment or Windows User/Machine environment."
        )
    return result


def describe_tinker_api_key(info: dict[str, Any] | None = None) -> str:
    info = info or ensure_tinker_api_key(required=False)
    if not info.get("present"):
        return "Tinker API key: not found"
    return (
        f"Tinker API key: found via {info['source']} environment "
        f"({info['masked_value']})"
    )


def _read_windows_env_var(name: str) -> tuple[str | None, str | None]:
    if platform.system() != "Windows" or winreg is None:
        return None, None

    lookup_order = [
        (winreg.HKEY_CURRENT_USER, r"Environment", "user"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            "machine",
        ),
    ]
    for hive, path, source in lookup_order:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        if isinstance(value, str) and value:
            return value, source
    return None, None


def _build_result(value: str | None, source: str | None) -> dict[str, Any]:
    return {
        "env_var": TINKER_API_KEY_ENV,
        "present": bool(value),
        "source": source,
        "masked_value": _mask_secret(value),
    }


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
