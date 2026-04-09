from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tinker import AdamParams
from tinker_cookbook.renderers import TrainOnWhat

from tinker_system_awake import keep_system_awake_context
from tinker_stop_control import read_stop_request
from tinker_training_utils import (
    build_batches,
    build_datums,
    build_eval_prompts,
    compute_batch_loss,
    evaluate_cross_entropy,
    maybe_take_examples,
    sample_generations,
)


ProgressCallback = Callable[[dict[str, Any]], None]
WaitForFutureResult = Callable[[Any, str, dict[str, Any] | None], Any]
FUTURE_WAIT_POLL_SECONDS = 15.0
STALL_LOG_EVERY_SECONDS = 60.0


@dataclass
class SubmittedTrainingStep:
    step: int
    epoch: int
    batch_index: int
    epoch_batch_count: int
    learning_rate: float
    submitted_at: float
    batch: list[Any]
    forward_future: Any
    optim_future: Any


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(value), 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def notify_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(dict(payload))
    except Exception as exc:
        print(f"[WARN] progress update failed: {exc!r}")


def save_named_state_checkpoint(
    training_client: Any,
    *,
    resolved_name: str,
    state_checkpoint_prefix: str,
    slugify_name: Any,
    step_label: str,
    wait_for_future: WaitForFutureResult | None = None,
    wait_context: dict[str, Any] | None = None,
) -> str | None:
    checkpoint_name = f"{state_checkpoint_prefix}-{slugify_name(resolved_name)}-{step_label}"
    checkpoint_future = training_client.save_state(checkpoint_name)
    response = (
        checkpoint_future.result()
        if wait_for_future is None
        else wait_for_future(checkpoint_future, "save_state", wait_context)
    )
    checkpoint_path = getattr(response, "path", None)
    print(f"[CHECKPOINT] saved training state: {checkpoint_path}")
    return checkpoint_path


def save_named_sampler_checkpoint(
    training_client: Any,
    *,
    resolved_name: str,
    sampler_checkpoint_prefix: str,
    slugify_name: Any,
    wait_for_future: WaitForFutureResult | None = None,
    wait_context: dict[str, Any] | None = None,
) -> tuple[str | None, Any]:
    sampler_name = f"{sampler_checkpoint_prefix}-{slugify_name(resolved_name)}-{int(time.time())}"
    sampler_future = training_client.save_weights_for_sampler(sampler_name)
    response = (
        sampler_future.result()
        if wait_for_future is None
        else wait_for_future(sampler_future, "save_sampler_weights", wait_context)
    )
    sampler_model_path = getattr(response, "path", None)
    print(f"[SAMPLER] saved sampler weights: {sampler_model_path}")
    sampling_client = training_client.create_sampling_client(sampler_model_path)
    return sampler_model_path, sampling_client


def run_training_loop(
    *,
    training_client: Any,
    requested_name: str,
    resolved_name: str,
    renderer_name: str,
    learning_rate: float,
    training_run_label: str,
    additional_steps: int | None,
    train_examples: list[Any],
    validation_examples: list[Any],
    test_examples: list[Any],
    train_example_limit: int | None,
    shuffle_seed: int,
    starting_step: int,
    batch_size: int,
    max_length: int,
    num_epochs: int,
    print_every_steps: int,
    save_state_every_steps: int,
    eval_example_limit: int,
    sample_temperature: float,
    sample_max_tokens: int,
    run_post_train_eval: bool,
    run_post_train_sampling: bool,
    keep_system_awake: bool,
    stop_signal_path: str | Path | None,
    state_checkpoint_prefix: str,
    sampler_checkpoint_prefix: str,
    slugify_name: Any,
    get_sampling_session_id: Any,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    tokenizer = training_client.get_tokenizer()
    model_train_examples = maybe_take_examples(
        train_examples,
        limit=train_example_limit,
        seed=shuffle_seed,
    )

    renderer, train_datums = build_datums(
        model_train_examples,
        tokenizer,
        resolved_name,
        renderer_name=renderer_name,
        max_length=max_length,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    _, validation_datums = build_datums(
        validation_examples,
        tokenizer,
        resolved_name,
        renderer_name=renderer_name,
        max_length=max_length,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    _, test_datums = build_datums(
        test_examples,
        tokenizer,
        resolved_name,
        renderer_name=renderer_name,
        max_length=max_length,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    train_batches = build_batches(train_datums, batch_size, shuffle=True, seed=shuffle_seed)
    validation_batches = build_batches(
        validation_datums,
        max(1, min(batch_size, len(validation_datums))),
        shuffle=False,
    )
    test_batches = build_batches(
        test_datums,
        max(1, min(batch_size, len(test_datums))),
        shuffle=False,
    )

    if not train_batches:
        raise ValueError(
            "No training batches were created. Check the dataset contents and max_length setting."
        )

    total_epoch_steps = max(1, num_epochs * len(train_batches))
    if additional_steps is None:
        total_planned_steps = total_epoch_steps
        stop_mode = f"full_epochs ({num_epochs} epoch(s))"
    else:
        total_planned_steps = min(max(1, additional_steps), total_epoch_steps)
        stop_mode = f"step_cap ({total_planned_steps} step(s))"
    if starting_step < 0:
        raise ValueError(f"starting_step must be >= 0, got {starting_step}")
    if starting_step >= total_planned_steps:
        raise ValueError(
            f"starting_step={starting_step} is already at or past the planned step cap "
            f"({total_planned_steps})."
        )
    print(
        f"[DATA] train_examples={len(model_train_examples)} validation_examples={len(validation_examples)} "
        f"test_examples={len(test_examples)}"
    )
    print(
        f"[DATA] train_batches={len(train_batches)} validation_batches={len(validation_batches)} "
        f"test_batches={len(test_batches)}"
    )
    print(
        f"[PLAN] label={training_run_label} planned_steps={total_planned_steps} "
        f"max_length={max_length} batch_size={batch_size} stop_mode={stop_mode} "
        f"starting_step={starting_step}"
    )
    notify_progress(
        progress_callback,
        event="planned",
        planned_steps=total_planned_steps,
        completed_steps=starting_step,
        train_examples=len(model_train_examples),
        validation_examples=len(validation_examples),
        test_examples=len(test_examples),
        train_batches=len(train_batches),
        validation_batches=len(validation_batches),
        test_batches=len(test_batches),
        stop_mode=stop_mode,
    )

    history_rows: list[dict[str, Any]] = []
    checkpoint_paths: list[str | None] = []
    completed_steps = starting_step
    submitted_steps = starting_step
    global_start_time = time.time()
    final_state_path = None
    latest_checkpoint_path = None
    latest_checkpoint_step = None
    post_validation = {"mean_nll": float("nan")}
    post_test = {"mean_nll": float("nan")}
    post_eval_error = None
    sampler_id = None
    sampler_model_path = None
    sample_rows: list[dict[str, str]] = []
    sample_error = None
    stop_request: dict[str, Any] | None = None
    pending_step: SubmittedTrainingStep | None = None

    def build_wait_context(
        *,
        submitted: SubmittedTrainingStep | None = None,
        completed_steps_override: int | None = None,
    ) -> dict[str, Any]:
        context = {
            "completed_steps": (
                completed_steps
                if completed_steps_override is None
                else completed_steps_override
            ),
            "planned_steps": total_planned_steps,
        }
        if submitted is not None:
            context.update(
                {
                    "in_flight_step": submitted.step,
                    "in_flight_epoch": submitted.epoch + 1,
                    "in_flight_batch_index": submitted.batch_index,
                    "in_flight_batch_count": submitted.epoch_batch_count,
                }
            )
        return context

    def wait_for_future_result(
        future: Any,
        waiting_on: str,
        wait_context: dict[str, Any] | None = None,
    ) -> Any:
        context = dict(wait_context or {})
        wait_started_at = time.time()
        next_stall_log_at = STALL_LOG_EVERY_SECONDS
        printed_stall_log = False

        while True:
            try:
                result = future.result(timeout=FUTURE_WAIT_POLL_SECONDS)
            except FutureTimeoutError:
                wait_elapsed_seconds = time.time() - wait_started_at
                heartbeat_payload = dict(context)
                heartbeat_payload.update(
                    {
                        "event": "heartbeat",
                        "waiting_on": waiting_on,
                        "wait_elapsed_seconds": round(wait_elapsed_seconds, 1),
                    }
                )
                if latest_checkpoint_path:
                    heartbeat_payload["latest_checkpoint_path"] = latest_checkpoint_path
                if latest_checkpoint_step is not None:
                    heartbeat_payload["latest_checkpoint_step"] = latest_checkpoint_step
                notify_progress(progress_callback, **heartbeat_payload)

                if wait_elapsed_seconds >= next_stall_log_at:
                    in_flight_step = context.get("in_flight_step")
                    in_flight_epoch = context.get("in_flight_epoch")
                    in_flight_batch_index = context.get("in_flight_batch_index")
                    in_flight_batch_count = context.get("in_flight_batch_count")
                    location = (
                        f"step {in_flight_step}/{total_planned_steps} "
                        f"(epoch {in_flight_epoch}/{num_epochs}, batch {in_flight_batch_index}/{in_flight_batch_count})"
                        if isinstance(in_flight_step, int)
                        else f"after {context.get('completed_steps', completed_steps)}/{total_planned_steps} completed steps"
                    )
                    print(
                        f"[WAIT] {location} has been waiting on {waiting_on} for "
                        f"{format_seconds(wait_elapsed_seconds)}. Process still alive; likely "
                        "Tinker capacity pressure or a local network interruption."
                    )
                    printed_stall_log = True
                    next_stall_log_at += STALL_LOG_EVERY_SECONDS
                continue

            if printed_stall_log:
                recovered_after = time.time() - wait_started_at
                in_flight_step = context.get("in_flight_step")
                location = (
                    f"step {in_flight_step}/{total_planned_steps}"
                    if isinstance(in_flight_step, int)
                    else f"{context.get('completed_steps', completed_steps)}/{total_planned_steps} completed steps"
                )
                print(
                    f"[WAIT] {location} recovered after {format_seconds(recovered_after)} "
                    f"waiting on {waiting_on}."
                )
            return result

    def submit_training_step(
        *,
        step_num: int,
        epoch: int,
        batch_index: int,
        epoch_batch_count: int,
        batch: list[Any],
    ) -> SubmittedTrainingStep:
        lr_scale = 1.0 - ((step_num - 1) / max(total_planned_steps - 1, 1))
        current_lr = learning_rate * max(0.1, lr_scale)
        submitted_at = time.time()
        forward_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(
            AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )
        )
        return SubmittedTrainingStep(
            step=step_num,
            epoch=epoch,
            batch_index=batch_index,
            epoch_batch_count=epoch_batch_count,
            learning_rate=current_lr,
            submitted_at=submitted_at,
            batch=batch,
            forward_future=forward_future,
            optim_future=optim_future,
        )

    def finish_training_step(submitted: SubmittedTrainingStep) -> int:
        nonlocal latest_checkpoint_path, latest_checkpoint_step

        wait_context = build_wait_context(
            submitted=submitted,
            completed_steps_override=max(completed_steps, submitted.step - 1),
        )
        forward_result = wait_for_future_result(
            submitted.forward_future,
            "forward_backward",
            wait_context,
        )
        optim_result = wait_for_future_result(
            submitted.optim_future,
            "optim_step",
            wait_context,
        )
        batch_loss, batch_weight = compute_batch_loss(forward_result.loss_fn_outputs, submitted.batch)
        optim_metrics = getattr(optim_result, "metrics", None) or {}
        if not isinstance(optim_metrics, dict):
            optim_metrics = {}

        step_elapsed_seconds = time.time() - submitted.submitted_at
        total_elapsed_seconds = time.time() - global_start_time

        history_rows.append(
            {
                "step": submitted.step,
                "epoch": submitted.epoch,
                "train_loss": batch_loss,
                "batch_weight": batch_weight,
                "learning_rate": submitted.learning_rate,
                "elapsed_seconds": step_elapsed_seconds,
                "total_elapsed_seconds": total_elapsed_seconds,
                **optim_metrics,
            }
        )
        notify_progress(
            progress_callback,
            event="step_completed",
            completed_steps=submitted.step,
            planned_steps=total_planned_steps,
            epoch=submitted.epoch + 1,
            batch_index=submitted.batch_index,
            epoch_batch_count=submitted.epoch_batch_count,
            train_loss=batch_loss,
            batch_weight=batch_weight,
            learning_rate=submitted.learning_rate,
            step_elapsed_seconds=round(step_elapsed_seconds, 3),
            total_elapsed_seconds=round(total_elapsed_seconds, 3),
        )

        should_print_step = (
            submitted.step == 1
            or submitted.step == total_planned_steps
            or submitted.step % print_every_steps == 0
        )
        if should_print_step:
            print(
                f"[STEP {submitted.step}/{total_planned_steps}] epoch={submitted.epoch + 1}/{num_epochs} "
                f"batch={submitted.batch_index}/{submitted.epoch_batch_count} loss={batch_loss:.4f} "
                f"lr={submitted.learning_rate:.2e} batch_weight={batch_weight:.1f} "
                f"step_time={format_seconds(step_elapsed_seconds)} "
                f"total_time={format_seconds(total_elapsed_seconds)}"
            )

        if save_state_every_steps and submitted.step % save_state_every_steps == 0:
            checkpoint_path = save_named_state_checkpoint(
                training_client,
                resolved_name=resolved_name,
                state_checkpoint_prefix=state_checkpoint_prefix,
                slugify_name=slugify_name,
                step_label=f"{training_run_label}-step-{submitted.step:04d}",
                wait_for_future=wait_for_future_result,
                wait_context=build_wait_context(
                    submitted=submitted,
                    completed_steps_override=submitted.step,
                ),
            )
            checkpoint_paths.append(checkpoint_path)
            if checkpoint_path:
                latest_checkpoint_path = checkpoint_path
                latest_checkpoint_step = submitted.step
            notify_progress(
                progress_callback,
                event="checkpoint_saved",
                completed_steps=submitted.step,
                planned_steps=total_planned_steps,
                checkpoint_path=checkpoint_path,
                latest_checkpoint_step=latest_checkpoint_step,
            )

        return submitted.step

    awake_context = (
        keep_system_awake_context(
            enabled=keep_system_awake,
            reason=f"{training_run_label} ({resolved_name})",
        )
        if keep_system_awake
        else nullcontext()
    )

    with awake_context:
        for epoch in range(num_epochs):
            epoch_batches = build_batches(
                train_datums,
                batch_size,
                shuffle=True,
                seed=shuffle_seed + epoch,
            )
            print(f"[EPOCH {epoch + 1}/{num_epochs}] {len(epoch_batches)} batch(es) queued")

            for batch_index, batch in enumerate(epoch_batches, start=1):
                stop_request = read_stop_request(stop_signal_path)
                if stop_request:
                    print(
                        "[STOP] external stop requested; ending training after the last completed step. "
                        f"{stop_request.get('requested_at_utc') or ''} {stop_request.get('reason') or ''}".strip()
                    )
                    notify_progress(
                        progress_callback,
                        event="stop_requested",
                        completed_steps=completed_steps,
                        planned_steps=total_planned_steps,
                        requested_at_utc=stop_request.get("requested_at_utc"),
                        reason=stop_request.get("reason"),
                    )
                    break

                if pending_step is not None and save_state_every_steps and (
                    pending_step.step % save_state_every_steps == 0
                ):
                    completed_steps = finish_training_step(pending_step)
                    pending_step = None

                if submitted_steps >= total_planned_steps:
                    break

                submitted_steps += 1
                current_step = submit_training_step(
                    step_num=submitted_steps,
                    epoch=epoch,
                    batch_index=batch_index,
                    epoch_batch_count=len(epoch_batches),
                    batch=batch,
                )

                if pending_step is not None:
                    completed_steps = finish_training_step(pending_step)
                pending_step = current_step

            if stop_request:
                print(f"[STOP] stop request acknowledged for {training_run_label}")
                break

            if submitted_steps >= total_planned_steps:
                print(f"[STOP] reached planned steps for {training_run_label}: {total_planned_steps}")
                break

        if pending_step is not None:
            completed_steps = finish_training_step(pending_step)
            pending_step = None

        final_step_label = (
            f"{training_run_label}-stopped-{int(time.time())}"
            if stop_request
            else f"{training_run_label}-final-{int(time.time())}"
        )
        final_state_path = save_named_state_checkpoint(
            training_client,
            resolved_name=resolved_name,
            state_checkpoint_prefix=state_checkpoint_prefix,
            slugify_name=slugify_name,
            step_label=final_step_label,
            wait_for_future=wait_for_future_result,
            wait_context=build_wait_context(completed_steps_override=completed_steps),
        )
        if final_state_path:
            latest_checkpoint_path = final_state_path
            latest_checkpoint_step = completed_steps
        notify_progress(
            progress_callback,
            event="final_checkpoint_saved",
            completed_steps=completed_steps,
            planned_steps=total_planned_steps,
            checkpoint_path=final_state_path,
            latest_checkpoint_step=latest_checkpoint_step,
        )

        if stop_request:
            print("[STOP] skipping post-train eval and sampling because stop was requested")
        elif run_post_train_eval:
            print("[EVAL] computing post-train validation/test loss")
            try:
                post_validation = evaluate_cross_entropy(training_client, validation_batches)
                post_test = evaluate_cross_entropy(training_client, test_batches)
                print(
                    f"[EVAL] validation_nll={post_validation['mean_nll']:.4f} "
                    f"test_nll={post_test['mean_nll']:.4f}"
                )
            except Exception as exc:
                post_eval_error = repr(exc)
                print(f"[EVAL] failed: {post_eval_error}")

        if stop_request:
            pass
        elif run_post_train_sampling:
            print("[SAMPLER] exporting named sampler weights and creating sampling client")
            try:
                sampler_model_path, sampling_client = save_named_sampler_checkpoint(
                    training_client,
                    resolved_name=resolved_name,
                    sampler_checkpoint_prefix=sampler_checkpoint_prefix,
                    slugify_name=slugify_name,
                    wait_for_future=wait_for_future_result,
                    wait_context=build_wait_context(completed_steps_override=completed_steps),
                )
                sampler_id = get_sampling_session_id(sampling_client)
                notify_progress(
                    progress_callback,
                    event="sampler_saved",
                    completed_steps=completed_steps,
                    planned_steps=total_planned_steps,
                    sampler_model_path=sampler_model_path,
                    sampler_id=sampler_id,
                )
                sample_rows = sample_generations(
                    sampling_client,
                    renderer,
                    build_eval_prompts(test_examples, limit=eval_example_limit),
                    max_tokens=sample_max_tokens,
                    temperature=sample_temperature,
                )
                print(f"[SAMPLER] generated {len(sample_rows)} sample(s)")
            except Exception as exc:
                sample_error = repr(exc)
                print(f"[SAMPLER] failed: {sample_error}")

    summary = {
        "requested_name": requested_name,
        "resolved_name": resolved_name,
        "renderer_name": renderer_name,
        "learning_rate": learning_rate,
        "train_examples": len(model_train_examples),
        "train_steps": completed_steps,
        "post_validation_nll": post_validation["mean_nll"],
        "post_test_nll": post_test["mean_nll"],
        "post_eval_error": post_eval_error,
        "sample_error": sample_error,
        "checkpoint_paths": checkpoint_paths,
        "final_state_path": final_state_path,
        "sampler_id": sampler_id,
        "sampler_model_path": sampler_model_path,
        "stop_requested": bool(stop_request),
        "stop_requested_at_utc": None if not stop_request else stop_request.get("requested_at_utc"),
        "stop_request_reason": None if not stop_request else stop_request.get("reason"),
        "stop_signal_path": None if stop_signal_path is None else str(Path(stop_signal_path).resolve()),
        "status": (
            "stopped"
            if stop_request
            else "ok"
            if sample_error is None and post_eval_error is None
            else "completed_with_followup_errors"
        ),
    }
    notify_progress(
        progress_callback,
        event="finished",
        completed_steps=completed_steps,
        planned_steps=total_planned_steps,
        status=summary["status"],
        final_state_path=final_state_path,
        sampler_id=sampler_id,
        sampler_model_path=sampler_model_path,
    )

    return {
        "summary": summary,
        "history": pd.DataFrame(history_rows),
        "samples": pd.DataFrame(sample_rows),
        "run_monitor": pd.DataFrame(),
        "session_monitor": pd.DataFrame(),
        "sampler_monitor": pd.DataFrame(),
    }
