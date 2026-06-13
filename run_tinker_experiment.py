from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker

from tinker_experiment_manager import build_experiment_dataset_variants
from tinker_notebook_env import describe_tinker_api_key, ensure_tinker_api_key
from tinker_notebook_workflows import run_training_loop
from tinker_stop_control import clear_stop_request, default_stop_signal_path
from tinker_training_utils import (
    find_dataset_root,
    load_dataset_bundle,
    resolve_model_names,
    select_renderer_name,
    slugify_name,
)


MODEL_OVERRIDES = {
    "gpt-oss-20b": {"learning_rate": 1e-4, "renderer_name": None},
    "openai/gpt-oss-20b": {"learning_rate": 1e-4, "renderer_name": None},
}
AUTO_RESUME_STALE_SECONDS = 5 * 60
RUN_OUTPUTS_DIRNAME = "run_outputs"
LATEST_RUN_FILENAME = "latest_active_run.json"
WAIT_STATE_KEYS = (
    "waiting_on",
    "wait_elapsed_seconds",
    "in_flight_step",
    "in_flight_epoch",
    "in_flight_batch_index",
    "in_flight_batch_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Tinker experiment to completion from the CLI.")
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Workspace root containing the dataset and tinker_env.",
    )
    parser.add_argument(
        "--run-name",
        default="essay_recent_r16",
        help="Experiment run name to execute.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the notebook's smoke-test settings instead of the full run.",
    )
    parser.add_argument(
        "--keep-system-awake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the Windows machine awake while training is active.",
    )
    parser.add_argument(
        "--clear-stop-request",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear any existing cooperative stop request before starting.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List the available experiment run names and exit.",
    )
    parser.add_argument(
        "--describe-latest",
        action="store_true",
        help="Show the latest locally recorded run for --run-name and exit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the selected run from the latest locally recorded checkpoint.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume the latest interrupted local run if possible; otherwise start fresh.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume from an explicit Tinker checkpoint path.",
    )
    parser.add_argument(
        "--resume-with-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, restore optimizer state as well as weights.",
    )
    return parser.parse_args()


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def get_current_session_id(client: Any) -> str | None:
    holder = getattr(client, "holder", None)
    return holder.get_session_id() if holder else None


def get_sampling_session_id(sampling_client: Any) -> str | None:
    return getattr(sampling_client, "_sampling_session_id", None)


def get_model_override(requested_name: str, resolved_name: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(MODEL_OVERRIDES.get(requested_name, {}))
    merged.update(MODEL_OVERRIDES.get(resolved_name, {}))
    return merged


def get_experiment_specs(smoke_test: bool) -> list[dict[str, Any]]:
    num_epochs = 1 if smoke_test else 3
    batch_size = 8
    max_length = 512
    return [
        {
            "run_name": "essay_recent_r16",
            "model_alias": "gpt-oss-20b",
            "dataset_variant": "recent_posts_plus_essays",
            "lora_rank": 16,
            "learning_rate": 1e-4,
            "batch_size": batch_size,
            "max_length": max_length,
            "num_epochs": num_epochs,
            "train_example_limit": 96 if smoke_test else None,
            "max_steps_per_model": 12 if smoke_test else None,
            "eval_example_limit": 5,
            "sample_temperature": 0.71,
            "sample_max_tokens": 96,
            "print_every_steps": 1 if smoke_test else 5,
            "save_state_every_steps": 4 if smoke_test else 25,
            "run_post_train_eval": False,
            "run_post_train_sampling": True,
            "notes": "Original training posts plus later posts and chunked essays. Eval is disabled by default for this variant.",
        },
        {
            "run_name": "initial_r32_lr7e5_b6",
            "model_alias": "gpt-oss-20b",
            "dataset_variant": "initial_posts",
            "lora_rank": 32,
            "learning_rate": 7e-5,
            "batch_size": 6,
            "max_length": max_length,
            "num_epochs": num_epochs,
            "train_example_limit": 96 if smoke_test else None,
            "max_steps_per_model": 12 if smoke_test else None,
            "eval_example_limit": 5,
            "sample_temperature": 0.71,
            "sample_max_tokens": 96,
            "print_every_steps": 1 if smoke_test else 5,
            "save_state_every_steps": 4 if smoke_test else 25,
            "run_post_train_eval": True,
            "run_post_train_sampling": True,
            "notes": "Initial post-only split with higher LoRA rank, lower learning rate, and smaller batch size.",
        },
        {
            "run_name": "essay_recent_interview_r16",
            "model_alias": "gpt-oss-20b",
            "dataset_variant": "recent_posts_essays_interview",
            "lora_rank": 16,
            "learning_rate": 1e-4,
            "batch_size": batch_size,
            "max_length": max_length,
            "num_epochs": num_epochs,
            "train_example_limit": 96 if smoke_test else None,
            "max_steps_per_model": 12 if smoke_test else None,
            "eval_example_limit": 5,
            "sample_temperature": 0.71,
            "sample_max_tokens": 96,
            "print_every_steps": 1 if smoke_test else 5,
            "save_state_every_steps": 4 if smoke_test else 25,
            "run_post_train_eval": False,
            "run_post_train_sampling": True,
            "notes": "Recent posts plus essays plus a balanced interview-derived corpus with direct Q&A and distilled post continuations.",
        },
        {
            "run_name": "essay_recent_interview_r32_lr7e5_b6",
            "model_alias": "gpt-oss-20b",
            "dataset_variant": "recent_posts_essays_interview",
            "lora_rank": 32,
            "learning_rate": 7e-5,
            "batch_size": 6,
            "max_length": max_length,
            "num_epochs": num_epochs,
            "train_example_limit": 96 if smoke_test else None,
            "max_steps_per_model": 12 if smoke_test else None,
            "eval_example_limit": 5,
            "sample_temperature": 0.71,
            "sample_max_tokens": 96,
            "print_every_steps": 1 if smoke_test else 5,
            "save_state_every_steps": 4 if smoke_test else 20,
            "run_post_train_eval": False,
            "run_post_train_sampling": True,
            "notes": "Interview-mix run with higher LoRA rank, gentler learning rate, and smaller batch size for a slightly more ambitious fit.",
        },
        {
            "run_name": "personal_sources_r16",
            "model_alias": "gpt-oss-20b",
            "dataset_variant": "personal_sources_mix",
            "lora_rank": 16,
            "learning_rate": 1e-4,
            "batch_size": batch_size,
            "max_length": max_length,
            "num_epochs": num_epochs,
            "train_example_limit": 96 if smoke_test else None,
            "max_steps_per_model": 12 if smoke_test else None,
            "eval_example_limit": 5,
            "sample_temperature": 0.71,
            "sample_max_tokens": 96,
            "print_every_steps": 1 if smoke_test else 5,
            "save_state_every_steps": 4 if smoke_test else 25,
            "run_post_train_eval": False,
            "run_post_train_sampling": True,
            "notes": "Mixed corpus plus local imported notes, poetry, and longform sources.",
        },
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def parse_step_from_checkpoint_path(checkpoint_path: str | None) -> int | None:
    if not checkpoint_path:
        return None
    match = re.search(r"step-(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def payload_status(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("status"):
        return str(summary["status"])
    if payload.get("status"):
        return str(payload["status"])
    return "unknown"


def last_event_age_seconds(payload: dict[str, Any]) -> float | None:
    return seconds_since_utc_timestamp(payload.get("last_event_at_utc") or payload.get("started_at_utc"))


def running_payload_is_stale(
    payload: dict[str, Any],
    *,
    stale_after_seconds: float = AUTO_RESUME_STALE_SECONDS,
) -> bool:
    if payload_status(payload) != "running":
        return False
    age_seconds = last_event_age_seconds(payload)
    return age_seconds is None or age_seconds >= stale_after_seconds


def running_payload_appears_active(
    payload: dict[str, Any],
    *,
    stale_after_seconds: float = AUTO_RESUME_STALE_SECONDS,
) -> bool:
    if payload_status(payload) != "running":
        return False
    age_seconds = last_event_age_seconds(payload)
    return age_seconds is not None and age_seconds < stale_after_seconds


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


def iter_run_payloads(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not run_dir.exists():
        return payloads
    for path in sorted(run_dir.glob("*.json"), reverse=True):
        if path.name == LATEST_RUN_FILENAME:
            continue
        payload = read_json(path)
        if payload is not None:
            payload["_path"] = str(path)
            payloads.append(payload)
    payloads.sort(key=lambda item: str(item.get("started_at_utc") or ""), reverse=True)
    return payloads


def find_latest_payload(run_dir: Path, *, run_name: str) -> dict[str, Any] | None:
    for payload in iter_run_payloads(run_dir):
        if str(payload.get("run_name") or "") == run_name:
            return payload
    return None


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


def extract_resume_step(payload: dict[str, Any], checkpoint_path: str | None) -> int | None:
    if checkpoint_path:
        latest_checkpoint_path = payload.get("latest_checkpoint_path")
        latest_checkpoint_step = payload.get("latest_checkpoint_step")
        if checkpoint_path == latest_checkpoint_path and isinstance(latest_checkpoint_step, int):
            return latest_checkpoint_step

    if isinstance(payload.get("completed_steps"), int):
        return int(payload["completed_steps"])

    summary = payload.get("summary")
    if isinstance(summary, dict):
        if checkpoint_path and checkpoint_path == summary.get("final_state_path"):
            train_steps = summary.get("train_steps")
            if isinstance(train_steps, int):
                return train_steps
        train_steps = summary.get("train_steps")
        if isinstance(train_steps, int):
            parsed = parse_step_from_checkpoint_path(checkpoint_path)
            return train_steps if parsed is None else parsed

    return parse_step_from_checkpoint_path(checkpoint_path)


def is_payload_auto_resumable(payload: dict[str, Any]) -> bool:
    status = payload_status(payload)
    if status in {"error", "stopped"}:
        return extract_resume_checkpoint(payload) is not None
    if status == "running":
        return running_payload_is_stale(payload) and extract_resume_checkpoint(payload) is not None
    if status not in {"running", "error", "stopped"}:
        return False
    return False


def find_resume_payload(run_dir: Path, *, run_name: str) -> dict[str, Any] | None:
    payload = find_latest_payload(run_dir, run_name=run_name)
    if payload is not None and is_payload_auto_resumable(payload):
        return payload
    return None


def find_payload_for_checkpoint(run_dir: Path, checkpoint_path: str) -> dict[str, Any] | None:
    for payload in iter_run_payloads(run_dir):
        if extract_resume_checkpoint(payload) == checkpoint_path:
            return payload
        summary = payload.get("summary")
        if isinstance(summary, dict):
            checkpoint_paths = summary.get("checkpoint_paths") or []
            if checkpoint_path in checkpoint_paths:
                return payload
            if checkpoint_path == summary.get("final_state_path"):
                return payload
    return None


def build_resume_command(run_name: str) -> str:
    return f".\\launch_tinker_experiment.bat {run_name} --resume"


def build_auto_resume_command(run_name: str) -> str:
    return f".\\launch_tinker_experiment.bat {run_name} --auto-resume"


def build_active_run_message(payload: dict[str, Any], *, run_name: str) -> str:
    age_seconds = last_event_age_seconds(payload)
    age_text = format_elapsed_seconds(age_seconds)
    return (
        f"Latest local record for {run_name} still looks active "
        f"(last update {age_text} ago). Refusing to start a duplicate run."
    )


def is_probable_connection_issue(exc: Exception) -> bool:
    if isinstance(exc, tinker.APIConnectionError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "no progress made",
        "requests appear to be stuck",
        "connection",
        "timed out",
        "timeout",
        "temporarily unavailable",
    )
    return any(marker in text for marker in markers)


def describe_payload(payload: dict[str, Any], *, run_name: str) -> list[str]:
    lines: list[str] = []
    lines.append(f"run_name: {payload.get('run_name') or run_name}")
    lines.append(f"status: {describe_payload_status(payload)}")
    lines.append(f"started_at_utc: {payload.get('started_at_utc') or 'unknown'}")
    training_run_id = payload.get("training_run_id")
    if training_run_id:
        lines.append(f"training_run_id: {training_run_id}")
    last_event = payload.get("last_event")
    if isinstance(last_event, str) and last_event.strip():
        lines.append(f"last_event: {last_event}")
    last_event_at_utc = payload.get("last_event_at_utc")
    if isinstance(last_event_at_utc, str) and last_event_at_utc.strip():
        lines.append(f"last_event_at_utc: {last_event_at_utc}")
        lines.append(f"last_event_age: {format_elapsed_seconds(last_event_age_seconds(payload))}")
    completed_steps = payload.get("completed_steps")
    planned_steps = payload.get("planned_steps")
    if isinstance(completed_steps, int):
        if isinstance(planned_steps, int):
            lines.append(f"progress: {completed_steps}/{planned_steps} steps")
        else:
            lines.append(f"completed_steps: {completed_steps}")
    waiting_on = payload.get("waiting_on")
    if isinstance(waiting_on, str) and waiting_on.strip():
        lines.append(f"waiting_on: {waiting_on}")
    wait_elapsed_seconds = payload.get("wait_elapsed_seconds")
    if isinstance(wait_elapsed_seconds, (int, float)):
        lines.append(f"wait_elapsed: {format_elapsed_seconds(float(wait_elapsed_seconds))}")
    in_flight_step = payload.get("in_flight_step")
    if isinstance(in_flight_step, int):
        epoch = payload.get("in_flight_epoch")
        batch_index = payload.get("in_flight_batch_index")
        batch_count = payload.get("in_flight_batch_count")
        lines.append(
            "in_flight_step: "
            f"{in_flight_step}"
            + (
                f" (epoch {epoch}, batch {batch_index}/{batch_count})"
                if isinstance(epoch, int) and isinstance(batch_index, int) and isinstance(batch_count, int)
                else ""
            )
        )
    summary = payload.get("summary")
    if isinstance(summary, dict):
        final_state_path = summary.get("final_state_path")
        if isinstance(final_state_path, str) and final_state_path.strip():
            lines.append(f"final_state: {final_state_path}")
    if is_payload_auto_resumable(payload):
        checkpoint_path = extract_resume_checkpoint(payload)
        if checkpoint_path:
            lines.append(f"latest_checkpoint: {checkpoint_path}")
            lines.append(f"auto_resume_command: {build_auto_resume_command(run_name)}")
            lines.append(f"resume_command: {build_resume_command(run_name)}")
        else:
            lines.append("latest_checkpoint: none recorded")
    elif running_payload_appears_active(payload):
        lines.append("resume_command: not suggested because the latest local record still looks active")
    else:
        lines.append("resume_command: not suggested because the latest local record is not resumable")
    source_path = payload.get("_path")
    if source_path:
        lines.append(f"record_path: {source_path}")
    return lines


def print_available_runs(experiment_specs: dict[str, dict[str, Any]]) -> None:
    print("Available runs:")
    for run_name, spec in experiment_specs.items():
        print(f"  - {run_name}")
        print(
            f"    model={spec['model_alias']} dataset={spec['dataset_variant']} "
            f"rank={spec['lora_rank']} lr={spec['learning_rate']:.2e} "
            f"epochs={spec['num_epochs']} batch_size={spec['batch_size']}"
        )
        print(f"    notes={spec['notes']}")


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace).resolve()
    run_dir = workspace_root / RUN_OUTPUTS_DIRNAME
    os.chdir(workspace_root)

    experiment_specs = {item["run_name"]: item for item in get_experiment_specs(args.smoke_test)}
    if args.list_runs:
        print_available_runs(experiment_specs)
        return 0

    if args.run_name not in experiment_specs:
        available = ", ".join(sorted(experiment_specs))
        raise KeyError(f"Unknown run name: {args.run_name}. Available run names: {available}")

    if args.describe_latest:
        payload = find_latest_payload(run_dir, run_name=args.run_name)
        if payload is None:
            print(f"No local run record found for {args.run_name}.")
            return 0
        for line in describe_payload(payload, run_name=args.run_name):
            print(line)
        return 0

    if sum(bool(value) for value in (args.resume, args.auto_resume, args.resume_from_checkpoint)) > 1:
        raise ValueError(
            "Use only one of --resume, --auto-resume, or --resume-from-checkpoint for a single run."
        )

    key_info = ensure_tinker_api_key(required=True)
    print(describe_tinker_api_key(key_info))

    stop_signal_path = default_stop_signal_path(workspace_root)
    if args.clear_stop_request and clear_stop_request(stop_signal_path):
        print(f"[STOP] cleared stale stop request: {stop_signal_path}")

    dataset_root = find_dataset_root(workspace_root)
    bundle = load_dataset_bundle(dataset_root)
    dataset_variants = build_experiment_dataset_variants(bundle)

    experiment_spec = experiment_specs[args.run_name]
    dataset_variant = dataset_variants[experiment_spec["dataset_variant"]]

    resume_checkpoint_path: str | None = None
    resume_payload: dict[str, Any] | None = None
    resume_start_step = 0
    resume_source_label = "fresh run"
    latest_payload = find_latest_payload(run_dir, run_name=args.run_name)
    if args.resume_from_checkpoint:
        resume_checkpoint_path = args.resume_from_checkpoint.strip()
        resume_payload = find_payload_for_checkpoint(run_dir, resume_checkpoint_path)
        resume_start_step = extract_resume_step(resume_payload, resume_checkpoint_path) if resume_payload else 0
        resume_start_step = 0 if resume_start_step is None else resume_start_step
        resume_source_label = "explicit checkpoint"
    elif args.resume:
        if latest_payload is not None and running_payload_appears_active(latest_payload):
            raise RuntimeError(build_active_run_message(latest_payload, run_name=args.run_name))
        resume_payload = find_resume_payload(run_dir, run_name=args.run_name)
        if resume_payload is None:
            raise RuntimeError(
                f"No resumable local checkpoint found for {args.run_name}. "
                f"Use --describe-latest first or pass --resume-from-checkpoint."
            )
        resume_checkpoint_path = extract_resume_checkpoint(resume_payload)
        if not resume_checkpoint_path:
            raise RuntimeError(f"Latest resumable record for {args.run_name} does not include a checkpoint path.")
        resume_start_step = extract_resume_step(resume_payload, resume_checkpoint_path) or 0
        resume_source_label = f"latest local checkpoint from {resume_payload.get('started_at_utc') or 'unknown run'}"
    elif args.auto_resume:
        if latest_payload is not None and running_payload_appears_active(latest_payload):
            raise RuntimeError(build_active_run_message(latest_payload, run_name=args.run_name))
        resume_payload = find_resume_payload(run_dir, run_name=args.run_name)
        if resume_payload is None:
            print(f"[RESUME] no interrupted local run found for {args.run_name}; starting fresh")
        else:
            resume_checkpoint_path = extract_resume_checkpoint(resume_payload)
            if not resume_checkpoint_path:
                print(
                    f"[RESUME] latest interrupted run for {args.run_name} did not record a checkpoint; "
                    "starting fresh"
                )
                resume_payload = None
            else:
                resume_start_step = extract_resume_step(resume_payload, resume_checkpoint_path) or 0
                resume_source_label = (
                    "auto-resume from latest interrupted local checkpoint "
                    f"({resume_payload.get('started_at_utc') or 'unknown run'})"
                )

    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    supported_models = [model.model_name for model in capabilities.supported_models]
    resolutions = resolve_model_names([experiment_spec["model_alias"]], supported_models)
    resolution = next((item for item in resolutions if item.resolved_name), None)
    if resolution is None or not resolution.resolved_name:
        raise RuntimeError(f"Could not resolve model alias: {experiment_spec['model_alias']}")

    requested_name = resolution.requested_name
    resolved_name = resolution.resolved_name
    override = get_model_override(requested_name, resolved_name)
    renderer_name = select_renderer_name(
        resolved_name,
        override=experiment_spec.get("renderer_name", override.get("renderer_name")),
    )

    run_slug = slugify_name(experiment_spec["run_name"]) or "run"
    timestamp = now_utc_compact()
    run_info_path = run_dir / f"{timestamp}-{run_slug}.json"
    active_run_path = run_dir / LATEST_RUN_FILENAME

    print(
        f"[START] run_name={experiment_spec['run_name']} requested={requested_name} resolved={resolved_name}"
    )
    print(
        f"[START] dataset_variant={experiment_spec['dataset_variant']} train_examples={len(dataset_variant.train_examples)} "
        f"validation_examples={len(dataset_variant.validation_examples)} test_examples={len(dataset_variant.test_examples)}"
    )
    print(
        f"[START] rank={experiment_spec['lora_rank']} lr={experiment_spec['learning_rate']:.2e} "
        f"batch_size={experiment_spec['batch_size']} epochs={experiment_spec['num_epochs']}"
    )
    print(f"[START] notes={experiment_spec['notes']}")

    session_id = get_current_session_id(service_client)
    if resume_checkpoint_path:
        print(
            f"[RESUME] source={resume_source_label} checkpoint={resume_checkpoint_path} "
            f"starting_step={resume_start_step} with_optimizer={bool(args.resume_with_optimizer)}"
        )
        if args.resume_with_optimizer:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                resume_checkpoint_path
            )
        else:
            training_client = service_client.create_training_client_from_state(resume_checkpoint_path)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=resolved_name,
            rank=int(experiment_spec["lora_rank"]),
        )
    session_id = get_current_session_id(service_client) or session_id
    training_run_id = training_client.model_id

    run_metadata = {
        "started_at_utc": timestamp,
        "workspace_root": str(workspace_root),
        "dataset_root": str(dataset_root),
        "run_name": experiment_spec["run_name"],
        "dataset_variant": experiment_spec["dataset_variant"],
        "model_alias": experiment_spec["model_alias"],
        "requested_name": requested_name,
        "resolved_name": resolved_name,
        "training_run_id": training_run_id,
        "session_id": session_id,
        "stop_signal_path": str(stop_signal_path),
        "status": "running",
        "last_event": "started",
        "last_event_at_utc": timestamp,
        "pid": os.getpid(),
        "starting_step": resume_start_step,
        "completed_steps": resume_start_step,
        "resume_mode": (
            "explicit_checkpoint"
            if args.resume_from_checkpoint
            else "latest_local_checkpoint"
            if args.resume
            else "auto_resumed_latest_local_checkpoint"
            if args.auto_resume and resume_checkpoint_path
            else "auto_resume_fresh_start"
            if args.auto_resume
            else "fresh_run"
        ),
        "resumed_from_checkpoint": resume_checkpoint_path,
        "resume_with_optimizer": bool(args.resume_with_optimizer) if resume_checkpoint_path else None,
        "resume_source_run_id": None if not resume_payload else resume_payload.get("training_run_id"),
        "resume_hint": build_resume_command(experiment_spec["run_name"]),
        "auto_resume_hint": build_auto_resume_command(experiment_spec["run_name"]),
    }

    runtime_state = dict(run_metadata)

    def update_runtime_state(event: dict[str, Any]) -> None:
        event_name = event.get("event")
        runtime_state["last_event"] = event_name
        runtime_state["last_event_at_utc"] = now_utc_compact()
        completed_steps = event.get("completed_steps")
        if isinstance(completed_steps, int):
            runtime_state["completed_steps"] = completed_steps
        planned_steps = event.get("planned_steps")
        if isinstance(planned_steps, int):
            runtime_state["planned_steps"] = planned_steps
        checkpoint_path = event.get("checkpoint_path")
        latest_checkpoint_path = event.get("latest_checkpoint_path")
        checkpoint_path_value = None
        if isinstance(checkpoint_path, str) and checkpoint_path.strip():
            checkpoint_path_value = checkpoint_path
        elif isinstance(latest_checkpoint_path, str) and latest_checkpoint_path.strip():
            checkpoint_path_value = latest_checkpoint_path
        if checkpoint_path_value:
            runtime_state["latest_checkpoint_path"] = checkpoint_path_value
        latest_checkpoint_step = event.get("latest_checkpoint_step")
        if isinstance(latest_checkpoint_step, int):
            runtime_state["latest_checkpoint_step"] = latest_checkpoint_step
        elif checkpoint_path_value and isinstance(completed_steps, int):
            runtime_state["latest_checkpoint_step"] = completed_steps
        sampler_model_path = event.get("sampler_model_path")
        if isinstance(sampler_model_path, str) and sampler_model_path.strip():
            runtime_state["sampler_model_path"] = sampler_model_path
        sampler_id = event.get("sampler_id")
        if isinstance(sampler_id, str) and sampler_id.strip():
            runtime_state["sampler_id"] = sampler_id
        wait_state_updated = False
        for key in WAIT_STATE_KEYS:
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                runtime_state[key] = value
                wait_state_updated = True
            elif isinstance(value, (int, float)):
                runtime_state[key] = value
                wait_state_updated = True
        if not wait_state_updated and event_name != "heartbeat":
            for key in WAIT_STATE_KEYS:
                runtime_state.pop(key, None)
        event_status = event.get("status")
        if isinstance(event_status, str) and event_status.strip():
            runtime_state["status"] = event_status
        write_json(run_info_path, runtime_state)
        write_json(active_run_path, runtime_state)

    write_json(run_info_path, runtime_state)
    write_json(active_run_path, runtime_state)

    print(f"[READY] session_id={session_id}")
    print(f"[READY] training_run_id={training_run_id}")
    print(f"[READY] metadata_path={run_info_path}")
    print(f"[READY] resume_hint={build_resume_command(experiment_spec['run_name'])}")
    print(f"[READY] auto_resume_hint={build_auto_resume_command(experiment_spec['run_name'])}")

    try:
        result = run_training_loop(
            training_client=training_client,
            requested_name=requested_name,
            resolved_name=resolved_name,
            renderer_name=renderer_name,
            learning_rate=float(experiment_spec["learning_rate"]),
            training_run_label=run_slug,
            additional_steps=experiment_spec.get("max_steps_per_model"),
            train_examples=dataset_variant.train_examples,
            validation_examples=dataset_variant.validation_examples,
            test_examples=dataset_variant.test_examples,
            train_example_limit=experiment_spec.get("train_example_limit"),
            shuffle_seed=7,
            starting_step=resume_start_step,
            batch_size=int(experiment_spec["batch_size"]),
            max_length=int(experiment_spec["max_length"]),
            num_epochs=int(experiment_spec["num_epochs"]),
            print_every_steps=int(experiment_spec["print_every_steps"]),
            save_state_every_steps=int(experiment_spec["save_state_every_steps"]),
            eval_example_limit=int(experiment_spec["eval_example_limit"]),
            sample_temperature=float(experiment_spec["sample_temperature"]),
            sample_max_tokens=int(experiment_spec["sample_max_tokens"]),
            run_post_train_eval=bool(experiment_spec["run_post_train_eval"]),
            run_post_train_sampling=bool(experiment_spec["run_post_train_sampling"]),
            keep_system_awake=bool(args.keep_system_awake),
            stop_signal_path=stop_signal_path,
            state_checkpoint_prefix="tinker-studio-state",
            sampler_checkpoint_prefix="tinker-studio-sampler",
            slugify_name=slugify_name,
            get_sampling_session_id=get_sampling_session_id,
            progress_callback=update_runtime_state,
        )
    except Exception as exc:
        error_payload = dict(runtime_state)
        error_payload.update(
            {
                "status": "error",
                "completed_at_utc": now_utc_compact(),
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(run_info_path, error_payload)
        write_json(active_run_path, error_payload)
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        if is_probable_connection_issue(exc):
            print(
                "[ERROR] lost connection to Tinker or stopped making progress long enough to continue."
            )
            print(
                f"[RECOVER] details were saved to {run_info_path}"
            )
        else:
            print(traceback.format_exc())
        latest_checkpoint_path = error_payload.get("latest_checkpoint_path")
        if latest_checkpoint_path:
            print(f"[RECOVER] latest_checkpoint_path={latest_checkpoint_path}")
            print(f"[RECOVER] resume_hint={build_resume_command(experiment_spec['run_name'])}")
            print(f"[RECOVER] auto_resume_hint={build_auto_resume_command(experiment_spec['run_name'])}")
        return 1

    summary = dict(result["summary"])
    completed_payload = dict(runtime_state)
    completed_payload.update(
        {
            "status": summary.get("status"),
            "completed_at_utc": now_utc_compact(),
            "summary": summary,
            "history_rows": len(result["history"]),
            "sample_rows": len(result["samples"]),
        }
    )
    write_json(run_info_path, completed_payload)
    write_json(active_run_path, completed_payload)

    print("[DONE] run completed")
    print(
        f"[DONE] status={summary.get('status')} train_steps={summary.get('train_steps')} "
        f"latest_checkpoint={completed_payload.get('latest_checkpoint_path') or summary.get('final_state_path')}"
    )
    print(f"[DONE] monitor_hint=.\\launch_tinker_monitor.bat --run-id {training_run_id}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
