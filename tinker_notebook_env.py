from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback
    winreg = None


TINKER_API_KEY_ENV = "TINKER_API_KEY"
TINKER_API_KEY_DOTENV_PATH_ENV = "TINKER_API_KEY_DOTENV_PATH"
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DOTENV_PATH = PROJECT_ROOT / ".env"


def load_project_dotenv(dotenv_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(dotenv_path).resolve() if dotenv_path else PROJECT_DOTENV_PATH
    exists = resolved_path.is_file()
    had_api_key = bool(os.environ.get(TINKER_API_KEY_ENV))
    loaded = (
        load_dotenv(dotenv_path=resolved_path, override=False, encoding="utf-8-sig")
        if exists
        else False
    )
    if not had_api_key and os.environ.get(TINKER_API_KEY_ENV):
        os.environ[TINKER_API_KEY_DOTENV_PATH_ENV] = str(resolved_path)
    return {
        "path": str(resolved_path),
        "exists": exists,
        "loaded": bool(loaded),
    }


def ensure_tinker_api_key(
    required: bool = True,
    dotenv_path: str | Path | None = None,
) -> dict[str, Any]:
    had_process_value = bool(os.environ.get(TINKER_API_KEY_ENV))
    dotenv_info = load_project_dotenv(dotenv_path)
    process_value = os.environ.get(TINKER_API_KEY_ENV)
    if process_value:
        dotenv_source_path = os.environ.get(TINKER_API_KEY_DOTENV_PATH_ENV)
        source = "project .env" if dotenv_source_path else "process"
        source_path = dotenv_source_path or (None if had_process_value else str(dotenv_info["path"]))
        return _build_result(process_value, source, source_path=source_path)

    registry_value, source = _read_windows_env_var(TINKER_API_KEY_ENV)
    if registry_value:
        os.environ[TINKER_API_KEY_ENV] = registry_value
        return _build_result(registry_value, source)

    result = _build_result(None, None)
    if required:
        raise RuntimeError(
            "Could not find TINKER_API_KEY in the process environment, project .env, "
            "or Windows User/Machine environment."
        )
    return result


def describe_tinker_api_key(info: dict[str, Any] | None = None) -> str:
    info = info or ensure_tinker_api_key(required=False)
    if not info.get("present"):
        return "Tinker API key: not found"
    source = info["source"]
    if source in {"process", "user", "machine"}:
        source = f"{source} environment"
    elif info.get("source_path"):
        source = f"{source} ({info['source_path']})"
    return (
        f"Tinker API key: found via {source} "
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


def _build_result(
    value: str | None,
    source: str | None,
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    return {
        "env_var": TINKER_API_KEY_ENV,
        "present": bool(value),
        "source": source,
        "source_path": source_path,
        "masked_value": _mask_secret(value),
    }


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
