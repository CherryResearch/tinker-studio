from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def can_control_sleep_state() -> bool:
    return os.name == "nt" and hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32")


@contextmanager
def keep_system_awake_context(*, enabled: bool, reason: str = "training") -> None:
    if not enabled or not can_control_sleep_state():
        yield
        return

    kernel32 = ctypes.windll.kernel32
    request_flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    previous_state = kernel32.SetThreadExecutionState(request_flags)
    if previous_state == 0:
        print(f"[AWAKE] failed to request keep-awake mode for {reason}")
        yield
        return

    print(f"[AWAKE] keep-awake enabled for {reason}")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print(f"[AWAKE] keep-awake released for {reason}")
