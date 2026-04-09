from __future__ import annotations

import argparse
from pathlib import Path

from tinker_stop_control import (
    clear_stop_request,
    default_stop_signal_path,
    format_stop_request,
    request_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the cooperative stop signal for Tinker notebook runs.")
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Workspace root that should contain the stop signal file.",
    )
    parser.add_argument(
        "--action",
        choices=["request", "clear", "status"],
        default="status",
        help="Action to perform.",
    )
    parser.add_argument(
        "--reason",
        default="cli_request",
        help="Reason string to store when requesting a stop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal_path = default_stop_signal_path(args.workspace)

    if args.action == "request":
        payload = request_stop(signal_path, reason=args.reason)
        print(f"Stop requested: {payload['path']}")
        print(format_stop_request(signal_path))
        return 0

    if args.action == "clear":
        removed = clear_stop_request(signal_path)
        if removed:
            print(f"Cleared stop request: {signal_path}")
        else:
            print(f"No stop request existed: {signal_path}")
        return 0

    print(f"Stop signal path: {signal_path}")
    print(format_stop_request(signal_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
