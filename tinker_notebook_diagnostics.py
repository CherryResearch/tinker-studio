from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


RUN_ID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27}:(?:train|sample):\d+\b")


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


def extract_notebook_artifacts(notebook_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(notebook_path).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))

    artifact_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        source_lines = [str(item) for item in cell.get("source", [])]
        source_head = next((line.strip() for line in source_lines if str(line).strip()), "")
        source_text = "\n".join(source_lines)
        output_text_chunks: list[str] = []
        for output in cell.get("outputs", []):
            output_text_chunks.extend(_flatten_output_text(output))

            error_type = output.get("ename")
            error_value = output.get("evalue")
            if error_type or error_value:
                error_rows.append(
                    {
                        "cell_index": cell_index,
                        "record_origin": "saved_output",
                        "error_type": error_type or "NotebookError",
                        "message": str(error_value or "").strip(),
                        "source_head": source_head[:120],
                    }
                )

        for artifact_id in sorted(set(RUN_ID_PATTERN.findall(source_text))):
            artifact_rows.append(
                {
                    "cell_index": cell_index,
                    "record_origin": "cell_source",
                    "artifact_kind": "sampler_id" if ":sample:" in artifact_id else "training_run_id",
                    "artifact_id": artifact_id,
                    "session_id": session_id_from_run_id(artifact_id),
                    "source_head": source_head[:120],
                }
            )

        output_text = "\n".join(output_text_chunks)
        for artifact_id in sorted(set(RUN_ID_PATTERN.findall(output_text))):
            artifact_rows.append(
                {
                    "cell_index": cell_index,
                    "record_origin": "saved_output",
                    "artifact_kind": "sampler_id" if ":sample:" in artifact_id else "training_run_id",
                    "artifact_id": artifact_id,
                    "session_id": session_id_from_run_id(artifact_id),
                    "source_head": source_head[:120],
                }
            )

        for line in output_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Failed: "):
                error_rows.append(
                    {
                        "cell_index": cell_index,
                        "record_origin": "saved_output",
                        "error_type": "PrintedFailure",
                        "message": stripped.removeprefix("Failed: ").strip(),
                        "source_head": source_head[:120],
                    }
                )

    artifact_df = pd.DataFrame(artifact_rows)
    if not artifact_df.empty:
        artifact_df = artifact_df.drop_duplicates().sort_values(
            by=["artifact_kind", "artifact_id", "cell_index"]
        ).reset_index(drop=True)

    error_df = pd.DataFrame(error_rows)
    if not error_df.empty:
        error_df = error_df.drop_duplicates().sort_values(
            by=["cell_index", "error_type", "message"]
        ).reset_index(drop=True)

    return artifact_df, error_df


def list_recent_training_runs_df(rest_client: Any, *, limit: int = 20) -> pd.DataFrame:
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
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            by=["seconds_since_last_request", "training_run_id"],
            na_position="last",
        ).reset_index(drop=True)
    return frame


def recovered_tracking_state(notebook_path: str | Path) -> dict[str, list[str]]:
    artifact_df, _ = extract_notebook_artifacts(notebook_path)
    if artifact_df.empty:
        return {
            "training_run_ids": [],
            "sampler_ids": [],
            "session_ids": [],
        }

    training_run_ids = artifact_df.loc[
        artifact_df["artifact_kind"] == "training_run_id", "artifact_id"
    ].tolist()
    sampler_ids = artifact_df.loc[
        artifact_df["artifact_kind"] == "sampler_id", "artifact_id"
    ].tolist()
    session_ids = artifact_df["session_id"].dropna().tolist()
    return {
        "training_run_ids": unique_preserving_order(training_run_ids),
        "sampler_ids": unique_preserving_order(sampler_ids),
        "session_ids": unique_preserving_order(session_ids),
    }


def _flatten_output_text(output: dict[str, Any]) -> list[str]:
    chunks: list[str] = []

    text_value = output.get("text")
    if isinstance(text_value, list):
        chunks.extend(str(item) for item in text_value)
    elif text_value is not None:
        chunks.append(str(text_value))

    data_value = output.get("data") or {}
    for mime_type, payload in data_value.items():
        if not str(mime_type).startswith("text/"):
            continue
        if isinstance(payload, list):
            chunks.extend(str(item) for item in payload)
        else:
            chunks.append(str(payload))

    traceback_value = output.get("traceback")
    if isinstance(traceback_value, list):
        chunks.extend(str(item) for item in traceback_value)

    return chunks
