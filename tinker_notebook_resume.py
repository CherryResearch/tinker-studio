from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import display


ACTIVE_SECONDS_THRESHOLD = 30.0


@dataclass(frozen=True)
class ResumeSelection:
    training_run_id: str
    checkpoint_id: str
    tinker_path: str
    status: str
    seconds_since_last_request: float | None
    num_checkpoints: int
    base_model: str | None
    lora_rank: int | None


def classify_status(seconds_since_last_request: float | None) -> str:
    if seconds_since_last_request is None:
        return "UNKNOWN"
    if seconds_since_last_request <= ACTIVE_SECONDS_THRESHOLD:
        return "ACTIVE"
    return "IDLE"


def session_id_from_run_id(run_id: str | None) -> str | None:
    if not run_id:
        return None
    if ":train:" in run_id:
        return run_id.split(":train:", 1)[0]
    if ":sample:" in run_id:
        return run_id.split(":sample:", 1)[0]
    return None


def list_resumable_runs_df(rest_client: Any, *, limit: int = 20) -> pd.DataFrame:
    response = rest_client.list_training_runs(limit=limit).result()
    training_runs = getattr(response, "training_runs", None) or getattr(response, "runs", None) or []
    now_utc = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, Any]] = []

    for run in training_runs:
        training_run_id = getattr(run, "training_run_id", None) or getattr(run, "id", None)
        checkpoints_response = rest_client.list_checkpoints(training_run_id).result()
        checkpoints = list(getattr(checkpoints_response, "checkpoints", []) or [])
        last_request_time = getattr(run, "last_request_time", None)
        seconds_since_last_request = None
        if last_request_time is not None:
            seconds_since_last_request = round(
                (now_utc - pd.Timestamp(last_request_time)).total_seconds(),
                1,
            )

        last_checkpoint_id = getattr(getattr(run, "last_checkpoint", None), "checkpoint_id", None)
        if not last_checkpoint_id and checkpoints:
            last_checkpoint_id = getattr(checkpoints[-1], "checkpoint_id", None)

        rows.append(
            {
                "training_run_id": training_run_id,
                "session_id": session_id_from_run_id(training_run_id),
                "base_model": getattr(run, "base_model", None) or getattr(run, "base_model_name", None),
                "lora_rank": getattr(run, "lora_rank", None) or getattr(run, "rank", None),
                "corrupted": getattr(run, "corrupted", None),
                "last_request_time_utc": str(last_request_time),
                "seconds_since_last_request": seconds_since_last_request,
                "status": classify_status(seconds_since_last_request),
                "num_checkpoints": len(checkpoints),
                "last_checkpoint_id": last_checkpoint_id,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.sort_values(
        by=["num_checkpoints", "seconds_since_last_request", "training_run_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return frame


def list_checkpoints_df(rest_client: Any, training_run_id: str) -> pd.DataFrame:
    checkpoints_response = rest_client.list_checkpoints(training_run_id).result()
    checkpoints = list(getattr(checkpoints_response, "checkpoints", []) or [])
    rows = []
    for checkpoint in checkpoints:
        checkpoint_id = getattr(checkpoint, "checkpoint_id", None)
        rows.append(
            {
                "training_run_id": training_run_id,
                "checkpoint_id": checkpoint_id,
                "tinker_path": (
                    f"tinker://{training_run_id}/{checkpoint_id}" if checkpoint_id else None
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_default_run_id(runs_df: pd.DataFrame) -> str | None:
    if runs_df.empty:
        return None

    resumable = runs_df[runs_df["num_checkpoints"] > 0]
    if resumable.empty:
        return None

    active = resumable[resumable["status"] == "ACTIVE"]
    target = active if not active.empty else resumable
    return str(target.iloc[0]["training_run_id"])


def build_resume_selection(
    rest_client: Any,
    *,
    training_run_id: str | None = None,
    checkpoint_id: str | None = None,
    limit: int = 20,
) -> ResumeSelection | None:
    runs_df = list_resumable_runs_df(rest_client, limit=limit)
    if runs_df.empty:
        return None

    run_id = training_run_id or choose_default_run_id(runs_df)
    if not run_id:
        return None

    row = runs_df.loc[runs_df["training_run_id"] == run_id]
    if row.empty:
        return None
    row_data = row.iloc[0]

    checkpoints_df = list_checkpoints_df(rest_client, run_id)
    if checkpoints_df.empty:
        return None

    checkpoint_row = None
    if checkpoint_id:
        matching = checkpoints_df.loc[checkpoints_df["checkpoint_id"] == checkpoint_id]
        if not matching.empty:
            checkpoint_row = matching.iloc[0]
    if checkpoint_row is None:
        checkpoint_row = checkpoints_df.iloc[-1]

    return ResumeSelection(
        training_run_id=run_id,
        checkpoint_id=str(checkpoint_row["checkpoint_id"]),
        tinker_path=str(checkpoint_row["tinker_path"]),
        status=str(row_data["status"]),
        seconds_since_last_request=(
            None
            if pd.isna(row_data["seconds_since_last_request"])
            else float(row_data["seconds_since_last_request"])
        ),
        num_checkpoints=int(row_data["num_checkpoints"]),
        base_model=None if pd.isna(row_data["base_model"]) else str(row_data["base_model"]),
        lora_rank=None if pd.isna(row_data["lora_rank"]) else int(row_data["lora_rank"]),
    )


def display_resume_selector(rest_client: Any, *, limit: int = 20) -> dict[str, Any]:
    runs_df = list_resumable_runs_df(rest_client, limit=limit)
    resumable_df = runs_df[runs_df["num_checkpoints"] > 0].reset_index(drop=True)

    title = widgets.HTML("<h3 style='margin:0'>Resume Tinker Run</h3>")
    help_text = widgets.HTML(
        "<div>Select a resumable run and checkpoint. "
        "The newest active run with checkpoints is recommended automatically.</div>"
    )

    output = widgets.Output()

    if resumable_df.empty:
        empty_message = widgets.HTML(
            "<div><b>No resumable runs found.</b> None of the recent runs has a saved checkpoint yet.</div>"
        )
        display(widgets.VBox([title, help_text, empty_message]))
        return {
            "runs_df": runs_df,
            "resumable_df": resumable_df,
            "selection": None,
            "output": output,
        }

    run_options = []
    for _, row in resumable_df.iterrows():
        seconds_value = row["seconds_since_last_request"]
        age_text = "n/a" if pd.isna(seconds_value) else f"{float(seconds_value):.1f}s ago"
        label = (
            f"{row['status']} | {row['training_run_id']} | "
            f"{row['num_checkpoints']} ckpt | {age_text}"
        )
        run_options.append((label, str(row["training_run_id"])))

    default_run_id = choose_default_run_id(resumable_df)
    run_dropdown = widgets.Dropdown(
        options=run_options,
        value=default_run_id,
        description="Run:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "70px"},
    )

    checkpoint_dropdown = widgets.Dropdown(
        options=[],
        description="Checkpoint:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "70px"},
    )

    selection_state: dict[str, Any] = {
        "selection": None,
        "runs_df": runs_df,
        "resumable_df": resumable_df,
        "run_dropdown": run_dropdown,
        "checkpoint_dropdown": checkpoint_dropdown,
        "output": output,
    }

    def refresh_checkpoint_options(*_: Any) -> None:
        selected_run_id = run_dropdown.value
        checkpoints_df = list_checkpoints_df(rest_client, selected_run_id)
        checkpoint_options = [
            (str(row["checkpoint_id"]), str(row["checkpoint_id"]))
            for _, row in checkpoints_df.iterrows()
        ]
        checkpoint_dropdown.options = checkpoint_options
        if checkpoint_options:
            checkpoint_dropdown.value = checkpoint_options[-1][1]
        refresh_output()

    def refresh_output(*_: Any) -> None:
        with output:
            output.clear_output(wait=True)
            selection = build_resume_selection(
                rest_client,
                training_run_id=run_dropdown.value,
                checkpoint_id=checkpoint_dropdown.value,
                limit=limit,
            )
            selection_state["selection"] = selection
            if selection is None:
                print("No resume selection is available yet.")
                return

            row_df = pd.DataFrame(
                [
                    {
                        "training_run_id": selection.training_run_id,
                        "checkpoint_id": selection.checkpoint_id,
                        "tinker_path": selection.tinker_path,
                        "status": selection.status,
                        "seconds_since_last_request": selection.seconds_since_last_request,
                        "num_checkpoints": selection.num_checkpoints,
                        "base_model": selection.base_model,
                        "lora_rank": selection.lora_rank,
                    }
                ]
            )
            display(row_df)
            print("selected_resume_config = {")
            print(f"    'training_run_id': '{selection.training_run_id}',")
            print(f"    'checkpoint_id': '{selection.checkpoint_id}',")
            print(f"    'tinker_path': '{selection.tinker_path}',")
            print("}")

    run_dropdown.observe(refresh_checkpoint_options, names="value")
    checkpoint_dropdown.observe(refresh_output, names="value")
    refresh_checkpoint_options()

    display(
        widgets.VBox(
            [
                title,
                help_text,
                run_dropdown,
                checkpoint_dropdown,
                output,
            ]
        )
    )
    return selection_state
