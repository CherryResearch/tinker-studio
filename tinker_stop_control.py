from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STOP_SIGNAL_FILENAME = ".tinker_stop_request.json"


def default_stop_signal_path(workspace_root: str | Path) -> Path:
    return Path(workspace_root).resolve() / STOP_SIGNAL_FILENAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_stop(path: str | Path, *, reason: str = "user_requested") -> dict[str, Any]:
    stop_path = Path(path).resolve()
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested": True,
        "requested_at_utc": utc_now_iso(),
        "reason": reason,
        "hostname": socket.gethostname(),
        "path": str(stop_path),
    }
    stop_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def clear_stop_request(path: str | Path) -> bool:
    stop_path = Path(path).resolve()
    if not stop_path.exists():
        return False
    stop_path.unlink()
    return True


def read_stop_request(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stop_path = Path(path).resolve()
    if not stop_path.exists():
        return None
    try:
        payload = json.loads(stop_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "requested": True,
            "requested_at_utc": None,
            "reason": "unparseable_stop_signal",
            "hostname": None,
            "path": str(stop_path),
        }
    if "path" not in payload:
        payload["path"] = str(stop_path)
    return payload


def format_stop_request(path: str | Path | None) -> str:
    payload = read_stop_request(path)
    if not payload:
        return "No stop request is currently pending."
    requested_at = payload.get("requested_at_utc") or "unknown time"
    reason = payload.get("reason") or "unspecified"
    return f"Stop requested at {requested_at} ({reason})."
