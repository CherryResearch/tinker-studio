from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tinker

from tinker_notebook_env import ensure_tinker_api_key
from tinker_stop_control import default_stop_signal_path, format_stop_request


ACTIVE_SECONDS_THRESHOLD = 30.0
LOCAL_RUNNING_STALE_SECONDS = 5 * 60
RUN_OUTPUTS_DIRNAME = "run_outputs"
LATEST_RUN_FILENAME = "latest_active_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Tinker training runs from the API.")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Training run ID to track. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=6,
        help="Number of recent runs to show when --run-id is not provided.",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=15,
        help="Seconds between refreshes.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of refreshes to perform. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Workspace root used to locate the stop signal file.",
    )
    return parser.parse_args()


def session_id_from_run_id(run_id: str | None) -> str | None:
    if not run_id:
        return None
    if ":train:" in run_id:
        return run_id.split(":train:", 1)[0]
    if ":sample:" in run_id:
        return run_id.split(":sample:", 1)[0]
    return None


def unique_preserving_order(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def classify_status(seconds_since_last_request: float | None) -> str:
    if seconds_since_last_request is None:
        return "UNKNOWN"
    if seconds_since_last_request <= ACTIVE_SECONDS_THRESHOLD:
        return "ACTIVE"
    return "IDLE"


def extract_resume_checkpoint(payload: dict[str, Any]) -> str | None:
    latest_checkpoint = payload.get("latest_checkpoint_path")
    if isinstance(latest_checkpoint, str) and latest_checkpoint.strip():
        return latest_checkpoint

    summary = payload.get("summary")
    if isinstance(summary, dict):
        checkpoint_paths = summary.get("checkpoint_paths")
        if isinstance(checkpoint_paths, list):
            for checkpoint_path in reversed(checkpoint_paths):
                if isinstance(checkpoint_path, str) and checkpoint_path.strip():
                    return checkpoint_path
        final_state_path = summary.get("final_state_path")
        if isinstance(final_state_path, str) and final_state_path.strip():
            return final_state_path

    return None


def payload_status(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("status"):
        return str(summary["status"])
    if payload.get("status"):
        return str(payload["status"])
    return "unknown"


def parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    stripped = value.strip()
    try:
        return datetime.strptime(stripped, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def seconds_since_utc_timestamp(value: Any, *, now: datetime | None = None) -> float | None:
    timestamp = parse_utc_timestamp(value)
    if timestamp is None:
        return None
    current_time = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    return max(0.0, (current_time - timestamp).total_seconds())


def format_elapsed_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def last_event_age_seconds(payload: dict[str, Any]) -> float | None:
    return seconds_since_utc_timestamp(payload.get("last_event_at_utc") or payload.get("started_at_utc"))


def running_payload_is_stale(payload: dict[str, Any]) -> bool:
    if payload_status(payload) != "running":
        return False
    age_seconds = last_event_age_seconds(payload)
    return age_seconds is None or age_seconds >= LOCAL_RUNNING_STALE_SECONDS


def running_payload_appears_active(payload: dict[str, Any]) -> bool:
    if payload_status(payload) != "running":
        return False
    age_seconds = last_event_age_seconds(payload)
    return age_seconds is not None and age_seconds < LOCAL_RUNNING_STALE_SECONDS


def describe_payload_status(payload: dict[str, Any]) -> str:
    status = payload_status(payload)
    if status != "running":
        return status
    age_seconds = last_event_age_seconds(payload)
    if running_payload_is_stale(payload):
        if age_seconds is None:
            return "running (heartbeat age unknown; likely interrupted)"
        return f"running (stale heartbeat, last update {format_elapsed_seconds(age_seconds)} ago)"
    if age_seconds is None:
        return "running (heartbeat age unknown)"
    return f"running (heartbeat {format_elapsed_seconds(age_seconds)} ago)"


def is_payload_auto_resumable(payload: dict[str, Any]) -> bool:
    status = payload_status(payload)
    if status in {"error", "stopped"}:
        return extract_resume_checkpoint(payload) is not None
    if status == "running":
        return running_payload_is_stale(payload) and extract_resume_checkpoint(payload) is not None
    return False


def load_latest_local_run_payload(workspace_root: str | Path) -> dict[str, Any] | None:
    latest_path = Path(workspace_root).resolve() / RUN_OUTPUTS_DIRNAME / LATEST_RUN_FILENAME
    if not latest_path.exists():
        return None
    try:
        data = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def list_recent_training_runs_df(rest_client: Any, *, limit: int) -> pd.DataFrame:
    response = rest_client.list_training_runs(limit=limit).result()
    training_runs = getattr(response, "training_runs", None) or getattr(response, "runs", None) or []
    now_utc = pd.Timestamp.now(tz="UTC")

    rows: list[dict[str, Any]] = []
    for run in training_runs:
        training_run_id = getattr(run, "training_run_id", None) or getattr(run, "id", None)
        last_request_time = getattr(run, "last_request_time", None)
        age_seconds = None
        if last_request_time is not None:
            age_seconds = round((now_utc - pd.Timestamp(last_request_time)).total_seconds(), 1)

        rows.append(
            {
                "training_run_id": training_run_id,
                "session_id": session_id_from_run_id(training_run_id),
                "base_model": getattr(run, "base_model", None) or getattr(run, "base_model_name", None),
                "lora_rank": getattr(run, "lora_rank", None) or getattr(run, "rank", None),
                "corrupted": getattr(run, "corrupted", None),
                "last_request_time_utc": str(last_request_time),
                "seconds_since_last_request": age_seconds,
                "status": classify_status(age_seconds),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            by=["seconds_since_last_request", "training_run_id"],
            na_position="last",
        ).reset_index(drop=True)
    return frame


def get_run_monitor_df(rest_client: Any, training_run_ids: list[str]) -> pd.DataFrame:
    rows = []
    now_utc = pd.Timestamp.now(tz="UTC")

    for training_run_id in unique_preserving_order(training_run_ids):
        run = rest_client.get_training_run(training_run_id).result()
        checkpoints = rest_client.list_checkpoints(training_run_id).result()
        last_request_time = getattr(run, "last_request_time", None)
        seconds_since_last_request = None
        if last_request_time is not None:
            seconds_since_last_request = round(
                (now_utc - pd.Timestamp(last_request_time)).total_seconds(),
                1,
            )

        rows.append(
            {
                "training_run_id": run.training_run_id,
                "session_id": session_id_from_run_id(run.training_run_id),
                "base_model": run.base_model,
                "lora_rank": run.lora_rank,
                "corrupted": run.corrupted,
                "last_request_time_utc": str(last_request_time),
                "seconds_since_last_request": seconds_since_last_request,
                "status": classify_status(seconds_since_last_request),
                "num_checkpoints": len(checkpoints.checkpoints),
                "last_checkpoint": getattr(run.last_checkpoint, "checkpoint_id", None),
                "last_sampler_checkpoint": getattr(run.last_sampler_checkpoint, "checkpoint_id", None),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            by=["seconds_since_last_request", "training_run_id"],
            na_position="last",
        ).reset_index(drop=True)
    return frame


def render_local_status(payload: dict[str, Any] | None) -> None:
    print("Local workspace status")
    print("----------------------")
    if payload is None:
        print("No local run record found yet.")
        print()
        return

    print(f"Run name: {payload.get('run_name') or 'unknown'}")
    print(f"Status: {describe_payload_status(payload)}")
    training_run_id = payload.get("training_run_id")
    if training_run_id:
        print(f"Training run id: {training_run_id}")
    last_event = payload.get("last_event")
    if isinstance(last_event, str) and last_event.strip():
        print(f"Last event: {last_event}")
    last_event_at_utc = payload.get("last_event_at_utc")
    if isinstance(last_event_at_utc, str) and last_event_at_utc.strip():
        print(
            f"Last update: {last_event_at_utc} ({format_elapsed_seconds(last_event_age_seconds(payload))} ago)"
        )
    completed_steps = payload.get("completed_steps")
    planned_steps = payload.get("planned_steps")
    if isinstance(completed_steps, int):
        if isinstance(planned_steps, int):
            print(f"Progress: {completed_steps}/{planned_steps} steps")
        else:
            print(f"Completed steps: {completed_steps}")
    waiting_on = payload.get("waiting_on")
    if isinstance(waiting_on, str) and waiting_on.strip():
        print(f"Waiting on: {waiting_on}")
    wait_elapsed_seconds = payload.get("wait_elapsed_seconds")
    if isinstance(wait_elapsed_seconds, (int, float)):
        print(f"Current wait: {format_elapsed_seconds(float(wait_elapsed_seconds))}")
    in_flight_step = payload.get("in_flight_step")
    if isinstance(in_flight_step, int):
        epoch = payload.get("in_flight_epoch")
        batch_index = payload.get("in_flight_batch_index")
        batch_count = payload.get("in_flight_batch_count")
        details = ""
        if isinstance(epoch, int) and isinstance(batch_index, int) and isinstance(batch_count, int):
            details = f" (epoch {epoch}, batch {batch_index}/{batch_count})"
        print(f"In-flight step: {in_flight_step}{details}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        final_state_path = summary.get("final_state_path")
        if isinstance(final_state_path, str) and final_state_path.strip():
            print(f"Final state: {final_state_path}")
    if is_payload_auto_resumable(payload):
        checkpoint_path = extract_resume_checkpoint(payload)
        print(f"Latest checkpoint: {checkpoint_path or 'none recorded'}")
        resume_hint = payload.get("auto_resume_hint") or payload.get("resume_hint")
        if checkpoint_path and isinstance(resume_hint, str) and resume_hint.strip():
            print(f"Resume hint: {resume_hint}")
    elif running_payload_appears_active(payload):
        print("Resume hint: not shown because the latest local record still looks active")
    else:
        print("Resume hint: not shown because the latest local record is not resumable")
    print()


def render_once(rest_client: Any, run_ids: list[str], recent: int, *, stop_signal_path: str, workspace_root: str) -> None:
    render_local_status(load_latest_local_run_payload(workspace_root))

    if run_ids:
        monitor_df = get_run_monitor_df(rest_client, run_ids)
        if monitor_df.empty:
            print("No matching runs found.")
        else:
            print("API status")
            print("----------")
            print(monitor_df.to_string(index=False))
    else:
        recent_df = list_recent_training_runs_df(rest_client, limit=recent)
        if recent_df.empty:
            print("No recent runs returned by the API.")
        else:
            print("Recent API runs")
            print("---------------")
            print(recent_df.to_string(index=False))

    print()
    print(f"Refreshed at {pd.Timestamp.now(tz='UTC').isoformat()}")
    print(format_stop_request(stop_signal_path))
    print(
        f"Status rule: ACTIVE if last request <= {int(ACTIVE_SECONDS_THRESHOLD)}s ago, else IDLE."
    )


def main() -> int:
    ensure_tinker_api_key(required=True)
    args = parse_args()
    rest_client = tinker.ServiceClient().create_rest_client()
    stop_signal_path = str(default_stop_signal_path(args.workspace))

    iteration = 0
    while True:
        print("\033[2J\033[H", end="")
        render_once(
            rest_client,
            run_ids=args.run_id,
            recent=args.recent,
            stop_signal_path=stop_signal_path,
            workspace_root=args.workspace,
        )
        iteration += 1

        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(max(1, args.refresh))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
