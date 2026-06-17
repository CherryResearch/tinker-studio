from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd
import streamlit as st

from monitor_tinker_runs import (
    describe_payload_status,
    extract_resume_checkpoint,
    format_elapsed_seconds,
    last_event_age_seconds,
    list_recent_training_runs_df,
    load_latest_local_run_payload,
    payload_status,
    seconds_since_utc_timestamp,
)
from tinker_stop_control import default_stop_signal_path, format_stop_request
from tinker_notebook_env import ensure_tinker_api_key


WORKSPACE_ROOT = Path(__file__).resolve().parent


def resolve_dataset_root() -> Path:
    for env_var in ("TINKER_STUDIO_DATASET_ROOT", "TINKER_DATASET_ROOT"):
        configured = os.environ.get(env_var)
        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = WORKSPACE_ROOT / configured_path
            return configured_path.resolve()
    default_root = WORKSPACE_ROOT / "data" / "training_data_cerise"
    if (default_root / "tinker" / "dataset_manifest.json").exists():
        return default_root
    manifests = sorted(WORKSPACE_ROOT.glob("**/tinker/dataset_manifest.json"))
    if manifests:
        return manifests[0].parent.parent
    return default_root


DATASET_ROOT = resolve_dataset_root()
POSTS_CSV_PATH = DATASET_ROOT / "processed" / "posts.csv"
MANIFEST_PATH = DATASET_ROOT / "tinker" / "dataset_manifest.json"
DATASET_BUILDER_PATH = DATASET_ROOT / "build_bluesky_finetune_dataset.py"
RENTRY_PAGES_PATH = DATASET_ROOT / "processed" / "rentry_pages.jsonl"
IMPORTED_SOURCES_PATH = DATASET_ROOT / "processed" / "imported_sources.jsonl"
SOURCE_IMPORTER_PATH = WORKSPACE_ROOT / "tinker_source_imports.py"
ENDPOINT_BASE_URL = "http://localhost:8765/v1"
ENDPOINT_PORT = 8765
IMPORTABLE_SOURCE_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
SOURCE_EXPORTS_ENV = "TINKER_STUDIO_SOURCE_EXPORTS_JSON"


st.set_page_config(
    page_title="Tinker Studio",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #15110c;
            --paper: #f6efe4;
            --clay: #ba5632;
            --moss: #6d7d46;
            --blue: #2d657c;
            --sand: #dbc7a6;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(186, 86, 50, 0.18), transparent 28rem),
                radial-gradient(circle at 82% 12%, rgba(45, 101, 124, 0.14), transparent 24rem),
                linear-gradient(135deg, #fbf4e9 0%, #efe0c8 44%, #f7efe2 100%);
        }
        [data-testid="stSidebar"] {
            background: rgba(29, 25, 19, 0.91);
        }
        [data-testid="stSidebar"] * {
            color: #f8eddb;
        }
        .stApp,
        .stApp p,
        .stApp li,
        .stApp label {
            color: var(--ink);
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #f8eddb;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 250, 240, 0.78);
            border: 1px solid rgba(21, 17, 12, 0.12);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 18px 42px rgba(79, 54, 24, 0.08);
        }
        div[data-testid="stMetricValue"] {
            color: var(--clay) !important;
            font-weight: 800;
        }
        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stMetricLabel"] p {
            color: rgba(21, 17, 12, 0.66) !important;
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] div {
            color: var(--clay) !important;
        }
        div[data-testid="stCodeBlock"],
        div[data-testid="stCodeBlock"] *,
        .stCode,
        .stCode *,
        pre,
        code {
            color: #f7ead4 !important;
        }
        div[data-testid="stCodeBlock"],
        .stCode pre {
            background: #111820 !important;
            border-radius: 12px;
        }
        div[data-baseweb="tab-list"] button p {
            color: var(--ink) !important;
            font-weight: 750;
            white-space: nowrap;
        }
        div[data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: thin;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: var(--ink) !important;
            background: rgba(255, 250, 240, 0.88) !important;
        }
        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {
            color: rgba(21, 17, 12, 0.46) !important;
            opacity: 1 !important;
        }
        div[data-baseweb="select"] > div {
            color: var(--ink) !important;
            background: rgba(255, 250, 240, 0.9) !important;
            border-color: rgba(21, 17, 12, 0.46) !important;
        }
        div[data-baseweb="select"] * {
            color: var(--ink) !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {
            color: rgba(21, 17, 12, 0.64) !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: rgba(248, 237, 219, 0.78) !important;
        }
        [data-testid="stSidebar"] input {
            color: var(--ink) !important;
        }
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] * {
            color: var(--ink) !important;
        }
        .status-card {
            padding: 1rem 1.15rem;
            border-radius: 18px;
            background: rgba(255, 250, 240, 0.78);
            border: 1px solid rgba(21, 17, 12, 0.12);
            min-height: 116px;
            overflow-wrap: anywhere;
        }
        .status-label {
            color: rgba(21, 17, 12, 0.56);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .status-value {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 780;
            margin-top: 0.35rem;
        }
        .status-detail {
            color: rgba(21, 17, 12, 0.62);
            margin-top: 0.45rem;
        }
        .source-pill {
            display: inline-block;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            background: rgba(109, 125, 70, 0.16);
            color: var(--ink);
            border: 1px solid rgba(109, 125, 70, 0.26);
            margin: 0.15rem 0.2rem 0.15rem 0;
            font-size: 0.86rem;
            font-weight: 700;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
            gap: 0.75rem;
            margin: 0.9rem 0 1rem;
        }
        .metric-card {
            padding: 0.85rem 0.9rem;
            border-radius: 8px;
            background: rgba(255, 250, 240, 0.78);
            border: 1px solid rgba(21, 17, 12, 0.12);
            box-shadow: 0 10px 28px rgba(79, 54, 24, 0.08);
            min-width: 0;
        }
        .metric-card-label {
            color: rgba(21, 17, 12, 0.62);
            font-size: 0.84rem;
            font-weight: 760;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .metric-card-value {
            color: var(--clay);
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 850;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        .metric-card-detail {
            color: rgba(21, 17, 12, 0.56);
            font-size: 0.78rem;
            line-height: 1.3;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 8px;
            background:
                linear-gradient(120deg, rgba(21, 17, 12, 0.94), rgba(69, 55, 39, 0.92)),
                radial-gradient(circle at top right, rgba(186, 86, 50, 0.35), transparent 18rem);
            color: #fff4df;
            box-shadow: 0 24px 70px rgba(54, 39, 19, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            color: #fff4df;
            font-size: 2.6rem;
            line-height: 1;
        }
        .hero p {
            color: rgba(255, 244, 223, 0.76);
            margin: 0.65rem 0 0;
            max-width: 58rem;
        }
        .soft-panel {
            padding: 0.85rem 1rem;
            border: 1px solid rgba(21, 17, 12, 0.12);
            border-radius: 16px;
            background: rgba(255, 250, 240, 0.58);
            margin: 0.7rem 0 1rem;
        }
        .reader-meta {
            color: rgba(21, 17, 12, 0.62);
            font-size: 0.9rem;
            overflow-wrap: anywhere;
        }
        @media (max-width: 720px) {
            .hero {
                padding: 1rem;
            }
            .hero h1 {
                font-size: 2rem;
            }
            .hero p {
                font-size: 0.95rem;
            }
            .metric-grid {
                grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
                gap: 0.55rem;
            }
            .metric-card {
                padding: 0.7rem;
            }
            .metric-card-value {
                font-size: 1.55rem;
            }
        }
        .soft-panel strong {
            color: var(--ink);
        }
        .mono-line {
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 0.88rem;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30)
def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@st.cache_data(ttl=30)
def load_posts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "created_at" in frame.columns:
        frame["created_at_ts"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    if "text_char_count" not in frame.columns and "text" in frame.columns:
        frame["text_char_count"] = frame["text"].fillna("").astype(str).str.len()
    if "text_word_count" not in frame.columns and "text" in frame.columns:
        frame["text_word_count"] = frame["text"].fillna("").astype(str).str.split().str.len()
    context_columns = [
        col
        for col in ["reply_context_text", "parent_text", "root_text"]
        if col in frame.columns
    ]
    if context_columns:
        frame["has_reply_context"] = (
            frame[context_columns]
            .fillna("")
            .astype(str)
            .apply(lambda row: any(value.strip() for value in row), axis=1)
        )
    return frame


@st.cache_data(ttl=30)
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


@st.cache_data(ttl=30)
def inspect_source_export(path_string: str, recursive: bool) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        return {
            "exists": False,
            "importable_files": 0,
            "markdown_files": 0,
            "pdf_files": 0,
            "latest_modified": "-",
        }
    candidates = path.rglob("*") if recursive and path.is_dir() else path.glob("*") if path.is_dir() else [path]
    importable_files = 0
    markdown_files = 0
    pdf_files = 0
    latest_mtime = 0.0
    for candidate in candidates:
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in IMPORTABLE_SOURCE_EXTENSIONS:
            importable_files += 1
        if suffix in {".md", ".markdown"}:
            markdown_files += 1
        if suffix == ".pdf":
            pdf_files += 1
        try:
            latest_mtime = max(latest_mtime, candidate.stat().st_mtime)
        except OSError:
            pass
    latest_modified = "-"
    if latest_mtime:
        latest_modified = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")
    return {
        "exists": True,
        "importable_files": importable_files,
        "markdown_files": markdown_files,
        "pdf_files": pdf_files,
        "latest_modified": latest_modified,
    }


def configured_source_exports() -> list[dict[str, Any]]:
    raw_value = os.environ.get(SOURCE_EXPORTS_ENV, "").strip()
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    exports: list[dict[str, Any]] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        path_value = str(item.get("path") or "").strip()
        if not path_value:
            continue
        exports.append(
            {
                "name": str(item.get("name") or f"Source export {index}"),
                "path": Path(path_value).expanduser(),
                "source_type": str(item.get("source_type") or "auto"),
                "recursive": bool(item.get("recursive", True)),
                "exclude_names": str(item.get("exclude_names") or ""),
            }
        )
    return exports


def known_source_export_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in configured_source_exports():
        path = source["path"]
        recursive = bool(source["recursive"])
        summary = inspect_source_export(str(path), recursive)
        rows.append(
            {
                "source": source["name"],
                "path": str(path),
                "exists": summary["exists"],
                "importable_files": summary["importable_files"],
                "markdown_files": summary["markdown_files"],
                "pdf_files": summary["pdf_files"],
                "latest_modified": summary["latest_modified"],
                "recommended_source_type": source["source_type"],
                "recursive": recursive,
                "exclude_names": source["exclude_names"],
            }
        )
    return rows


@st.cache_data(ttl=20)
def load_local_payload(workspace_root: Path) -> dict[str, Any] | None:
    return load_latest_local_run_payload(workspace_root)


@st.cache_data(ttl=20)
def load_recent_api_runs(limit: int) -> pd.DataFrame:
    try:
        ensure_tinker_api_key(required=False)
    except Exception:
        pass
    if not os.environ.get("TINKER_API_KEY"):
        return pd.DataFrame()
    try:
        import tinker

        rest_client = tinker.ServiceClient().create_rest_client()
        return list_recent_training_runs_df(rest_client, limit=limit)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_git_status() -> str:
    return run_local_command(["git", "status", "--short", "--ignored=matching"])


@st.cache_data(ttl=30)
def notebook_outputs_summary(path: Path) -> dict[str, int | bool]:
    if not path.exists():
        return {"exists": False, "cells": 0, "cells_with_outputs": 0, "execution_count_cells": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "cells": 0, "cells_with_outputs": -1, "execution_count_cells": -1}
    cells = data.get("cells") if isinstance(data, dict) else []
    if not isinstance(cells, list):
        cells = []
    cells_with_outputs = 0
    execution_count_cells = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("outputs"):
            cells_with_outputs += 1
        if cell.get("execution_count") is not None:
            execution_count_cells += 1
    return {
        "exists": True,
        "cells": len(cells),
        "cells_with_outputs": cells_with_outputs,
        "execution_count_cells": execution_count_cells,
    }


@st.cache_data(ttl=60)
def load_evaluation_rows(variant_name: str, split_name: str, limit: int) -> list[dict[str, Any]]:
    from tinker_experiment_manager import build_experiment_dataset_variants
    from tinker_training_utils import build_eval_prompts, find_dataset_root, load_dataset_bundle

    dataset_root = find_dataset_root(WORKSPACE_ROOT)
    bundle = load_dataset_bundle(dataset_root)
    variants = build_experiment_dataset_variants(bundle)
    if variant_name not in variants:
        return []
    variant = variants[variant_name]
    split_examples = {
        "train": variant.train_examples,
        "validation": variant.validation_examples,
        "test": variant.test_examples,
    }[split_name]
    examples = build_eval_prompts(split_examples, limit=max(1, int(limit)))
    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "example_id": example.example_id,
                "opening_text": example.opening_text,
                "target_text": example.target_text,
                "source_kind": example.metadata.get("source_kind") or ("reply" if example.metadata.get("is_reply") else "post"),
                "created_at": example.metadata.get("created_at"),
                "reply_context_text": example.metadata.get("reply_context_text") or "",
            }
        )
    return rows


@st.cache_data(ttl=20)
def load_sampler_run_records(workspace_root: Path) -> list[dict[str, Any]]:
    run_outputs = workspace_root / "run_outputs"
    if not run_outputs.exists():
        return []
    paths = [run_outputs / "latest_active_run.json"]
    paths.extend(
        sorted(
            (path for path in run_outputs.glob("*.json") if path.name != "latest_active_run.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    records: list[dict[str, Any]] = []
    seen_checkpoints: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        sampler_checkpoint = extract_sampler_checkpoint_from_payload(payload)
        if not sampler_checkpoint or sampler_checkpoint in seen_checkpoints:
            continue
        seen_checkpoints.add(sampler_checkpoint)
        records.append(
            {
                "display_name": run_display_name(payload),
                "run_name": str(payload.get("run_name") or "tinker-studio"),
                "dataset_variant": str(payload.get("dataset_variant") or ""),
                "model_alias": str(payload.get("model_alias") or payload.get("requested_name") or ""),
                "status": payload_status(payload),
                "sampler_checkpoint": sampler_checkpoint,
                "payload": payload,
                "source_path": str(path),
            }
        )
    return records


def format_timestamp(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)
    if pd.isna(timestamp):
        return "-"
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def parse_run_timestamp(value: Any) -> datetime | None:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        timestamp = value.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
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


def format_run_date(value: Any) -> str:
    timestamp = parse_run_timestamp(value)
    return timestamp.strftime("%Y-%m-%d") if timestamp else "undated"


def format_run_datetime(value: Any) -> str:
    timestamp = parse_run_timestamp(value)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC") if timestamp else "-"


def format_learning_rate(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
        if number != number:
            return ""
        return f"{number:.0e}"
    except (TypeError, ValueError):
        return str(value)


def extract_rank_from_run_name(run_name: str) -> str:
    match = re.search(r"(?:^|_)r(\d+)(?:_|$)", run_name)
    return match.group(1) if match else ""


def short_model_name(value: str) -> str:
    return value.rsplit("/", 1)[-1] if value else "model"


def humanize_variant_name(value: str) -> str:
    return value.replace("_", " ").strip() or "dataset"


def run_display_name(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "unknown local run"
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    run_name = str(payload.get("run_name") or "unknown")
    model_name = str(
        payload.get("model_alias")
        or payload.get("requested_name")
        or summary.get("requested_name")
        or "model"
    )
    variant = str(payload.get("dataset_variant") or "dataset")
    rank = str(payload.get("lora_rank") or summary.get("lora_rank") or extract_rank_from_run_name(run_name))
    learning_rate = format_learning_rate(payload.get("learning_rate") or summary.get("learning_rate"))
    date_label = format_run_date(payload.get("started_at_utc") or payload.get("completed_at_utc"))
    parts = [date_label, short_model_name(model_name)]
    if rank:
        parts.append(f"r{rank}")
    if learning_rate:
        parts.append(f"lr{learning_rate}")
    parts.append(humanize_variant_name(variant))
    return " ".join(parts)


def api_run_display_name(row: pd.Series) -> str:
    timestamp = format_run_date(row.get("last_request_time_utc"))
    model = short_model_name(str(row.get("base_model") or "model"))
    rank = str(row.get("lora_rank") or "").strip()
    run_id = str(row.get("training_run_id") or "")
    short_id = run_id.split(":", 1)[0][:8] if run_id else "unknown"
    parts = [timestamp, model]
    if rank and rank.lower() != "nan":
        parts.append(f"r{rank}")
    parts.append(short_id)
    return " ".join(parts)


def decorate_api_runs(api_runs: pd.DataFrame) -> pd.DataFrame:
    if api_runs.empty:
        return api_runs
    frame = api_runs.copy()
    frame.insert(0, "run_label", frame.apply(api_run_display_name, axis=1))
    if "status" in frame.columns:
        frame.insert(1, "state", frame["status"].fillna("UNKNOWN").astype(str).str.upper())
    return frame



def last_update_label(payload: dict[str, Any]) -> str:
    status = payload_status(payload)
    if status != "running":
        timestamp = payload.get("completed_at_utc") or payload.get("last_event_at_utc")
        if timestamp:
            age = format_elapsed_seconds(seconds_since_utc_timestamp(timestamp))
            return f"finished {age} ago"
        return status
    heartbeat_age = format_elapsed_seconds(last_event_age_seconds(payload))
    return f"heartbeat {heartbeat_age} ago"


def manifest_summary_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    derived_sources = manifest.get("derived_sources") if isinstance(manifest.get("derived_sources"), list) else []
    return [
        {"field": "Collected", "value": str(manifest.get("collected_at_utc") or "unknown")},
        {"field": "Profile", "value": str(manifest.get("profile_url") or "unknown")},
        {"field": "Posts", "value": f"{int(counts.get('all_post_rows') or 0):,}"},
        {"field": "Training rows", "value": f"{int(counts.get('non_empty_training_rows') or 0):,}"},
        {"field": "Replies with context", "value": f"{int(counts.get('reply_rows_with_context') or 0):,}"},
        {"field": "Derived sources", "value": ", ".join(str(source) for source in derived_sources) or "-"},
    ]


def render_status_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{escape(label)}</div>
            <div class="status-value">{escape(value)}</div>
            <div class="status-detail">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[dict[str, str]]) -> None:
    cards = []
    for metric in metrics:
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-card-label">{escape(str(metric.get("label") or ""))}</div>'
            f'<div class="metric-card-value">{escape(str(metric.get("value") or ""))}</div>'
            f'<div class="metric-card-detail">{escape(str(metric.get("detail") or ""))}</div>'
            "</div>"
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def normalize_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = re.split(r"[,;\n]", value)
    elif isinstance(value, list):
        raw_values = []
        for item in value:
            if isinstance(item, dict):
                raw_values.append(str(item.get("name") or item.get("label") or ""))
            else:
                raw_values.append(str(item))
    else:
        raw_values = [str(value)]
    tags: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        tag = raw.strip()
        if not tag:
            continue
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def row_tags(row: dict[str, Any], *, defaults: list[str] | None = None) -> list[str]:
    tags = normalize_tag_list(row.get("tags") or row.get("labels"))
    source_type = str(row.get("source_type") or "").strip()
    fallback_tags = [] if tags else [source_type]
    for tag in [*(defaults or []), *fallback_tags]:
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def source_tag_counts(
    posts: pd.DataFrame,
    rentry_rows: list[dict[str, Any]],
    imported_rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(tag: str, count: int = 1) -> None:
        if tag:
            counts[tag] = counts.get(tag, 0) + count

    if not posts.empty:
        add("bluesky", len(posts))
        if "is_reply" in posts.columns:
            reply_mask = posts["is_reply"].astype(str).str.lower().isin(["true", "1"])
            reply_count = int(reply_mask.sum())
            add("reply", reply_count)
            add("post", max(0, len(posts) - reply_count))
        else:
            add("post", len(posts))

    for _ in rentry_rows:
        add("writing")
        add("markdown")
        add("longform")
    for row in imported_rows:
        for tag in row_tags(row):
            add(tag)
    return dict(sorted(counts.items()))


def readable_source_rows(rentry_rows: list[dict[str, Any]], imported_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(rentry_rows):
        text = str(row.get("rendered_text") or row.get("text") or "").strip()
        if not text:
            continue
        title = str(row.get("title") or f"Markdown source {index + 1}").strip()
        tags = row_tags(row, defaults=["writing", "markdown", "longform"])
        rows.append(
            {
                "key": f"markdown::{row.get('url') or title}",
                "kind": "writing",
                "title": title,
                "text": text,
                "tags": tags,
                "word_count": row.get("word_count") or len(text.split()),
                "path": "processed markdown source",
                "editable": False,
                "row": row,
            }
        )
    for index, row in enumerate(imported_rows):
        text = str(row.get("text") or row.get("rendered_text") or "").strip()
        if not text:
            continue
        title = str(row.get("title") or f"Imported source {index + 1}").strip()
        tags = row_tags(row)
        rows.append(
            {
                "key": str(row.get("id") or f"imported::{index}::{title}"),
                "kind": "writing",
                "title": title,
                "text": text,
                "tags": tags,
                "labels": normalize_tag_list(row.get("labels")),
                "word_count": row.get("word_count") or len(text.split()),
                "path": str(row.get("source_path") or ""),
                "editable": True,
                "row": row,
            }
        )
    return rows


def write_imported_source_labels(row_id: str, labels: list[str]) -> bool:
    if not IMPORTED_SOURCES_PATH.exists():
        return False
    rows = load_jsonl(IMPORTED_SOURCES_PATH)
    updated = False
    for row in rows:
        if str(row.get("id") or "") == row_id:
            row["labels"] = labels
            updated = True
            break
    if not updated:
        return False
    with IMPORTED_SOURCES_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return True


def dataset_age(manifest: dict[str, Any]) -> str:
    collected_at = manifest.get("collected_at_utc")
    if not collected_at:
        return "unknown"
    try:
        timestamp = datetime.fromisoformat(str(collected_at).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    age_seconds = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    return format_elapsed_seconds(age_seconds)


def default_bluesky_handle(manifest: dict[str, Any]) -> str:
    profile_url = str(manifest.get("profile_url") or "").rstrip("/")
    if profile_url:
        return profile_url.rsplit("/", 1)[-1]
    return os.environ.get("TINKER_STUDIO_BLUESKY_HANDLE", "example.bsky.social")


def run_dataset_refresh(handle: str) -> tuple[int, str]:
    command = [
        sys.executable,
        str(DATASET_BUILDER_PATH),
        "--handle",
        handle,
        "--outdir",
        str(DATASET_ROOT),
    ]
    completed = subprocess.run(
        command,
        cwd=str(DATASET_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    return completed.returncode, output.strip()


def run_local_command(command: list[str], *, timeout: int = 12) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return str(exc)
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    return output.strip()


def run_powershell(command: str, *, timeout: int = 12) -> str:
    return run_local_command(["powershell", "-NoProfile", "-Command", command], timeout=timeout)


def split_exclude_names(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n;]", value) if item.strip()]


def run_source_import(
    input_path: str,
    source_type: str,
    mode: str,
    label: str,
    recursive: bool,
    exclude_names: list[str],
) -> tuple[int, str]:
    command = [
        sys.executable,
        str(SOURCE_IMPORTER_PATH),
        "--input",
        input_path,
        "--dataset-root",
        str(DATASET_ROOT),
        "--source-type",
        source_type,
        f"--{mode}",
        "--recursive" if recursive else "--no-recursive",
    ]
    for name in exclude_names:
        command.extend(["--exclude-name", name])
    if label.strip():
        command.extend(["--label", label.strip()])
    completed = subprocess.run(
        command,
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    return completed.returncode, output.strip()


def run_source_preview(
    input_path: str,
    source_type: str,
    mode: str,
    label: str,
    recursive: bool,
    exclude_names: list[str],
) -> tuple[int, str]:
    command = [
        sys.executable,
        str(SOURCE_IMPORTER_PATH),
        "--input",
        input_path,
        "--dataset-root",
        str(DATASET_ROOT),
        "--source-type",
        source_type,
        f"--{mode}",
        "--recursive" if recursive else "--no-recursive",
        "--preview",
    ]
    for name in exclude_names:
        command.extend(["--exclude-name", name])
    if label.strip():
        command.extend(["--label", label.strip()])
    completed = subprocess.run(
        command,
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    return completed.returncode, output.strip()


def start_endpoint_bridge(
    run_name: str,
    port: int,
    *,
    model_id: str = "",
    sampler_checkpoint: str | None = None,
) -> tuple[bool, str]:
    log_path = WORKSPACE_ROOT / "run_logs" / "tinker_endpoint.out.log"
    err_path = WORKSPACE_ROOT / "run_logs" / "tinker_endpoint.err.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "serve_tinker_endpoint.py"),
        "--workspace",
        str(WORKSPACE_ROOT),
        "--run-name",
        run_name,
        "--port",
        str(port),
    ]
    if model_id.strip():
        command.extend(["--model-id", model_id.strip()])
    if sampler_checkpoint:
        command.extend(["--sampler-checkpoint", sampler_checkpoint])
    try:
        with log_path.open("ab") as stdout, err_path.open("ab") as stderr:
            subprocess.Popen(
                command,
                cwd=str(WORKSPACE_ROOT),
                stdout=stdout,
                stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0),
            )
    except Exception as exc:
        return False, str(exc)
    return True, f"Started endpoint bridge on port {port}. Logs: {log_path}"


def stop_process_on_port(port: int) -> str:
    script = (
        "$processIds = Get-NetTCPConnection -LocalPort "
        + str(int(port))
        + " -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; "
        + "if (-not $processIds) { 'no process bound to port'; exit 0 }; "
        + "foreach ($processId in $processIds) { Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue; \"stopped $processId\" }"
    )
    return run_powershell(script)


def extract_sampler_checkpoint_from_payload(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("sampler_model_path", "sampler_checkpoint"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    summary = payload.get("summary")
    if isinstance(summary, dict):
        value = summary.get("sampler_model_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def endpoint_root(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return clean[:-3] if clean.endswith("/v1") else clean


def endpoint_health(base_url: str) -> tuple[bool, str]:
    try:
        with request.urlopen(f"{endpoint_root(base_url)}/health", timeout=2) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status == 200, body
    except Exception as exc:
        return False, str(exc)


def call_chat_endpoint(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"endpoint returned {exc.code}: {detail}") from exc
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("endpoint response did not include choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("endpoint response did not include a message")
    return str(message.get("content") or "")


def render_header(manifest: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>Tinker Studio</h1>
            <p>Corpus readiness, source coverage, and recent training run telemetry in one local control surface.</p>
            <p>Dataset snapshot age: <strong>{dataset_age(manifest)}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_overview(
    posts: pd.DataFrame,
    manifest: dict[str, Any],
    rentry_rows: list[dict[str, Any]],
    imported_rows: list[dict[str, Any]],
) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    latest_post = "-"
    latest_post_metric = "-"
    earliest_post = "-"
    if not posts.empty and "created_at_ts" in posts.columns:
        latest_ts = posts["created_at_ts"].max()
        latest_post = format_timestamp(latest_ts)
        latest_post_metric = latest_ts.strftime("%Y-%m-%d") if not pd.isna(latest_ts) else "-"
        earliest_post = format_timestamp(posts["created_at_ts"].min())

    render_metric_grid(
        [
            {"label": "Media rows", "value": f"{int(counts.get('all_post_rows') or len(posts)):,}", "detail": "currently Bluesky"},
            {"label": "Trainable rows", "value": f"{int(counts.get('non_empty_training_rows') or 0):,}", "detail": "base split"},
            {"label": "Reply context", "value": f"{int(counts.get('reply_rows_with_context') or 0):,}", "detail": "conversation tags"},
            {"label": "Source docs", "value": f"{len(rentry_rows) + len(imported_rows):,}", "detail": "markdown + imports"},
            {"label": "Tags", "value": f"{len(source_tag_counts(posts, rentry_rows, imported_rows)):,}", "detail": "computed + user labels"},
            {"label": "Latest media", "value": latest_post_metric, "detail": "snapshot timestamp"},
        ]
    )

    st.caption(f"Corpus range: {earliest_post} to {latest_post}")
    st.markdown(
        """
        <div class="soft-panel">
            <strong>Corpus model:</strong> sources are organized into three families: <strong>posts</strong>
            with reply structure and metadata, <strong>writing</strong> from markdown documents with variable
            user tags, and future <strong>traces</strong> for curated tool/reasoning workflows. Traces are a
            planned corpus family, not random local chat import.
        </div>
        """,
        unsafe_allow_html=True,
    )

    known_export_rows = known_source_export_rows()
    detected_importable = sum(int(row["importable_files"]) for row in known_export_rows if row["exists"])
    if detected_importable and not imported_rows:
        st.warning(
            "Local exported writing folders were detected, but no imported source rows are loaded. "
            "The personal source training mix will omit those markdowns until they are imported."
        )

    if posts.empty:
        st.warning(f"No posts found at {POSTS_CSV_PATH}")
        return


def render_sources_overview(posts: pd.DataFrame, rentry_rows: list[dict[str, Any]], imported_rows: list[dict[str, Any]]) -> None:
    st.markdown("### Sources")
    tag_counts = source_tag_counts(posts, rentry_rows, imported_rows)
    st.markdown(
        "".join(
            f'<span class="source-pill">{escape(tag)}: {count}</span>'
            for tag, count in tag_counts.items()
        ),
        unsafe_allow_html=True,
    )
    known_exports = known_source_export_rows()
    detected_importable = sum(int(row["importable_files"]) for row in known_exports if row["exists"])
    if detected_importable and not imported_rows:
        st.warning(
            "Detected local markdown exports that are not represented in imported_sources.jsonl. "
            "Import them before using personal_sources_mix for a larger run."
        )
    elif imported_rows:
        st.success(f"Loaded {len(imported_rows):,} imported source rows from {IMPORTED_SOURCES_PATH}.")

    source_tabs = st.tabs(["Inventory", "Reader", "Import"])
    with source_tabs[0]:
        render_source_inventory(posts, rentry_rows, imported_rows, known_exports)
    with source_tabs[1]:
        render_source_reader(rentry_rows, imported_rows)
    with source_tabs[2]:
        render_source_import_controls()


def render_source_inventory(
    posts: pd.DataFrame,
    rentry_rows: list[dict[str, Any]],
    imported_rows: list[dict[str, Any]],
    known_exports: list[dict[str, Any]],
) -> None:
    render_inventory_reader_shortcut(rentry_rows, imported_rows)
    inventory_tabs = st.tabs(["Posts", "Writing", "Local Exports"])
    with inventory_tabs[0]:
        render_post_inventory(posts)
    with inventory_tabs[1]:
        render_writing_inventory(rentry_rows, imported_rows)
    with inventory_tabs[2]:
        st.dataframe(pd.DataFrame(known_exports), width="stretch", hide_index=True)


def render_inventory_reader_shortcut(rentry_rows: list[dict[str, Any]], imported_rows: list[dict[str, Any]]) -> None:
    readable_rows = sorted(
        readable_source_rows(rentry_rows, imported_rows),
        key=lambda row: (0 if row["editable"] else 1, str(row["title"]).lower()),
    )
    if not readable_rows:
        return
    labels = [
        f"{row['title']} | {', '.join(row['tags'])}"
        for row in readable_rows
    ]
    selector_col, button_col = st.columns([3, 1])
    selected_label = selector_col.selectbox("Open writing source in reader", labels, key="inventory_reader_shortcut")
    selected_row = readable_rows[labels.index(selected_label)]
    if button_col.button("Open In Reader", width="stretch", key="inventory_reader_shortcut_button"):
        st.session_state.source_reader_selected_key = selected_row["key"]
        st.success("Reader selection updated.")


def render_post_inventory(posts: pd.DataFrame) -> None:
    st.markdown("#### Posts")
    if posts.empty:
        st.info("No post rows found.")
        return
    query = st.text_input(
        "Search post text",
        placeholder="filter by phrase, tag, or topic",
        key="inventory_post_search",
    )
    filter_col, context_col = st.columns(2)
    post_scope = filter_col.selectbox(
        "Scope",
        ["all", "posts", "replies"],
        index=0,
        key="inventory_post_scope",
    )
    show_reply_context = context_col.toggle(
        "Show reply context",
        value=True,
        key="inventory_show_reply_context",
        help="Shows the parent/root text captured for replies when available.",
    )
    explorer = posts.copy()
    if query and "text" in explorer.columns:
        explorer = explorer[explorer["text"].fillna("").astype(str).str.contains(query, case=False, na=False)]
    if post_scope != "all" and "is_reply" in explorer.columns:
        reply_mask = explorer["is_reply"].astype(str).str.lower().isin(["true", "1"])
        explorer = explorer[reply_mask] if post_scope == "replies" else explorer[~reply_mask]
    if "created_at_ts" in explorer.columns:
        explorer = explorer.sort_values("created_at_ts", ascending=False, na_position="last")
    base_columns = [
        col
        for col in ["created_at", "text", "is_reply", "has_reply_context", "like_count", "reply_count", "repost_count", "quote_count", "uri"]
        if col in explorer.columns
    ]
    context_columns = [
        col
        for col in ["reply_context_text", "parent_author", "parent_text", "root_author", "root_text"]
        if col in explorer.columns
    ]
    visible_columns = base_columns + (context_columns if show_reply_context else [])
    st.caption(f"Showing {min(len(explorer), 100):,} of {len(explorer):,} matching rows.")
    st.dataframe(explorer[visible_columns].head(100), width="stretch", hide_index=True)


def render_writing_inventory(rentry_rows: list[dict[str, Any]], imported_rows: list[dict[str, Any]]) -> None:
    st.markdown("#### Writing")
    inventory_rows: list[dict[str, Any]] = []
    readable_rows = sorted(
        readable_source_rows(rentry_rows, imported_rows),
        key=lambda row: (0 if row["editable"] else 1, str(row["title"]).lower()),
    )
    for row in readable_rows:
        inventory_rows.append(
            {
                "title": row["title"],
                "kind": row["kind"],
                "tags": ", ".join(row["tags"]),
                "word_count": row["word_count"],
                "path": row["path"],
            }
        )
    if not inventory_rows:
        st.info("No readable document sources found yet.")
        return
    option_labels = [
        f"{row['title']} | {row['kind']} | {', '.join(row['tags'])}"
        for row in readable_rows
    ]
    selected_label = st.selectbox("Open source", option_labels, key="inventory_reader_link_select")
    selected_row = readable_rows[option_labels.index(selected_label)]
    action_col, meta_col = st.columns([1, 3])
    if action_col.button("Open In Reader", width="stretch", key="inventory_open_reader_button"):
        st.session_state.source_reader_selected_key = selected_row["key"]
        st.success("Reader selection updated below.")
    meta_col.markdown(
        f'<div class="reader-meta">{escape(selected_row["kind"])} | {int(selected_row["word_count"]):,} words | '
        f'{escape(", ".join(selected_row["tags"]) or "untagged")}</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(inventory_rows), width="stretch", hide_index=True)


def render_source_reader(rentry_rows: list[dict[str, Any]], imported_rows: list[dict[str, Any]]) -> None:
    rows = sorted(
        readable_source_rows(rentry_rows, imported_rows),
        key=lambda row: (0 if row["editable"] else 1, str(row["title"]).lower()),
    )
    if not rows:
        st.info("No markdown or long-form sources are available to read.")
        return

    all_tags = sorted({tag for row in rows for tag in row["tags"]})
    filters_left, filters_right = st.columns([2, 3])
    query = filters_left.text_input("Find source", placeholder="title, tag, or path", key="source_reader_query")
    selected_tags = filters_right.multiselect("Filter tags", all_tags, key="source_reader_tags")

    filtered_rows = rows
    if query.strip():
        needle = query.strip().lower()
        filtered_rows = [
            row
            for row in filtered_rows
            if needle in row["title"].lower()
            or needle in row["path"].lower()
            or any(needle in tag.lower() for tag in row["tags"])
        ]
    if selected_tags:
        selected_set = set(selected_tags)
        filtered_rows = [row for row in filtered_rows if selected_set.intersection(row["tags"])]
    if not filtered_rows:
        st.info("No sources match the current filters.")
        return

    labels = [
        f"{row['title']} | {row['kind']} | {', '.join(row['tags'])}"
        for row in filtered_rows
    ]
    selected_key = str(st.session_state.get("source_reader_selected_key") or "")
    selected_index = 0
    if selected_key:
        for index, row in enumerate(filtered_rows):
            if row["key"] == selected_key:
                selected_index = index
                break
    selected_label = st.selectbox("Source", labels, index=selected_index, key="source_reader_selected")
    selected = filtered_rows[labels.index(selected_label)]
    st.session_state.source_reader_selected_key = selected["key"]

    st.markdown(f"#### {selected['title']}")
    st.markdown(
        f'<div class="reader-meta">{escape(selected["kind"])} | {int(selected["word_count"]):,} words | '
        f'{escape(", ".join(selected["tags"]) or "untagged")} | {escape(selected["path"])}</div>',
        unsafe_allow_html=True,
    )

    if selected["editable"]:
        tag_text = st.text_input(
            "Tags / labels",
            value=", ".join(selected.get("labels") or []),
            key=f"source_reader_tags_{selected['key']}",
            help="Comma, semicolon, or newline separated. These are user-facing corpus tags; source type remains prompt routing.",
        )
        if st.button("Save Tags", width="stretch", key=f"source_reader_save_tags_{selected['key']}"):
            labels_to_save = normalize_tag_list(tag_text)
            if write_imported_source_labels(selected["key"], labels_to_save):
                st.cache_data.clear()
                st.success("Tags saved.")
                st.rerun()
            else:
                st.error("Could not save tags for this source.")

    rendered_tab, raw_tab = st.tabs(["Rendered Markdown", "Raw Text"])
    with rendered_tab:
        st.markdown(selected["text"])
    with raw_tab:
        st.code(selected["text"], language="markdown")


def render_source_import_controls() -> None:
    st.caption("Imports are written into the ignored dataset folder so private notes stay local by default.")
    configured_exports = configured_source_exports()
    default_export = configured_exports[0] if configured_exports else {}
    input_path = st.text_input(
        "Input file or folder",
        value=str(default_export.get("path") or ""),
        placeholder=r"C:\Users\you\Takeout\Keep",
        key="source_import_input_path",
    )
    recursive = st.toggle(
        "Include subfolders",
        value=bool(default_export.get("recursive", True)),
        key="source_import_recursive",
    )
    exclude_names_text = st.text_area(
        "Exclude file/folder names",
        value=str(default_export.get("exclude_names") or ""),
        key="source_import_exclude_names",
        height=82,
    )
    source_type_options = ["auto", "google_keep", "poetry", "notes", "longform"]
    default_source_type = str(default_export.get("source_type") or "auto")
    source_type_index = source_type_options.index(default_source_type) if default_source_type in source_type_options else 0
    source_type = st.selectbox(
        "Prompt routing",
        source_type_options,
        index=source_type_index,
        help="Prompt routing is internal. Use tags/labels below for user-facing corpus organization.",
        key="source_import_type",
    )
    label = st.text_input("Tags / labels", placeholder="drive-writing, longform, draft", key="source_import_label")
    mode = "append" if st.toggle("Append to existing imports", value=True, key="source_import_append") else "replace"
    exclude_names = split_exclude_names(exclude_names_text)
    preview_col, import_col = st.columns(2)
    if preview_col.button("Preview Import", width="stretch", key="source_preview_button"):
        if not input_path.strip():
            st.error("Choose an input file or folder first.")
        elif not SOURCE_IMPORTER_PATH.exists():
            st.error(f"Importer not found: {SOURCE_IMPORTER_PATH}")
        else:
            with st.spinner("Previewing local sources..."):
                try:
                    returncode, output = run_source_preview(
                        input_path,
                        source_type,
                        mode,
                        label,
                        recursive,
                        exclude_names,
                    )
                except subprocess.TimeoutExpired:
                    st.error("Preview timed out after 120 seconds.")
                    return
            if returncode == 0:
                st.success("Preview complete. Nothing was written.")
                if output:
                    st.code(output[-4000:], language="text")
            else:
                st.error(f"Preview failed with exit code {returncode}.")
                if output:
                    st.code(output[-2400:], language="text")
    if import_col.button("Import Local Sources", width="stretch", key="source_import_button"):
        if not input_path.strip():
            st.error("Choose an input file or folder first.")
        elif not SOURCE_IMPORTER_PATH.exists():
            st.error(f"Importer not found: {SOURCE_IMPORTER_PATH}")
        else:
            with st.spinner("Importing local sources..."):
                try:
                    returncode, output = run_source_import(
                        input_path,
                        source_type,
                        mode,
                        label,
                        recursive,
                        exclude_names,
                    )
                except subprocess.TimeoutExpired:
                    st.error("Import timed out after 120 seconds.")
                    return
            if returncode == 0:
                st.cache_data.clear()
                st.success("Sources imported.")
                if output:
                    st.code(output[-2200:], language="text")
                st.rerun()
            else:
                st.error(f"Import failed with exit code {returncode}.")
                if output:
                    st.code(output[-2400:], language="text")


def render_training_overview() -> None:
    payload = load_local_payload(WORKSPACE_ROOT)
    stop_signal_path = default_stop_signal_path(WORKSPACE_ROOT)
    api_runs = load_recent_api_runs(8)

    st.markdown("### Runs")
    if payload:
        status = payload_status(payload)
        update_label = last_update_label(payload)
        checkpoint = extract_resume_checkpoint(payload) or "-"
        status_detail = "complete" if status == "ok" else describe_payload_status(payload)
        render_metric_grid(
            [
                {"label": "Current state", "value": status.upper(), "detail": status_detail},
                {"label": "Run label", "value": run_display_name(payload), "detail": str(payload.get("run_name") or "-")},
                {"label": "Last update", "value": update_label, "detail": "local heartbeat / completion"},
                {"label": "Checkpoint", "value": "found" if checkpoint != "-" else "missing", "detail": Path(checkpoint).name if checkpoint != "-" else "-"},
            ]
        )
        with st.expander("Run identifiers and resume detail"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"field": "Display label", "value": run_display_name(payload)},
                        {"field": "State", "value": status.upper()},
                        {"field": "Run name", "value": str(payload.get("run_name") or "-")},
                        {"field": "Dataset variant", "value": str(payload.get("dataset_variant") or "-")},
                        {"field": "Training run ID", "value": str(payload.get("training_run_id") or "-")},
                        {"field": "Sampler ID", "value": str(payload.get("sampler_id") or "-")},
                        {"field": "Started", "value": format_run_datetime(payload.get("started_at_utc"))},
                        {"field": "Completed", "value": format_run_datetime(payload.get("completed_at_utc"))},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    else:
        render_metric_grid(
            [
                {"label": "Current state", "value": "NONE", "detail": "no local run record"},
                {"label": "Run label", "value": "-", "detail": "start a run to populate telemetry"},
                {"label": "Last update", "value": "-", "detail": ""},
                {"label": "Checkpoint", "value": "-", "detail": ""},
            ]
        )

    st.markdown("#### Stop Control")
    stop_detail = format_stop_request(str(stop_signal_path))
    if "No stop request" in stop_detail:
        st.success("No stop request is currently pending.")
    else:
        st.warning("A stop request is pending.")
        st.code(stop_detail, language="text")

    st.markdown("#### Recent Tinker API Runs")
    if api_runs.empty:
        st.info("No Tinker API run data available. Set TINKER_API_KEY before launching Streamlit to enable this panel.")
    else:
        decorated_runs = decorate_api_runs(api_runs)
        preferred_columns = [
            column
            for column in [
                "run_label",
                "state",
                "training_run_id",
                "base_model",
                "lora_rank",
                "last_request_time_utc",
                "seconds_since_last_request",
                "corrupted",
            ]
            if column in decorated_runs.columns
        ]
        st.dataframe(decorated_runs[preferred_columns], width="stretch", hide_index=True)


def render_endpoint_chat() -> None:
    payload = load_local_payload(WORKSPACE_ROOT)
    sampler_records = load_sampler_run_records(WORKSPACE_ROOT)
    selected_record: dict[str, Any] | None = None
    if sampler_records:
        option_map = {
            f"{record['display_name']} | {record['run_name']}": record
            for record in sampler_records
        }
        selected_label = st.selectbox(
            "Trained sampler",
            list(option_map),
            key="endpoint_sampler_select",
            help="A local tag generated from date, model, rank, learning rate, and dataset variant.",
        )
        selected_record = option_map[selected_label]
        selected_payload = selected_record.get("payload") if isinstance(selected_record.get("payload"), dict) else payload
    else:
        selected_payload = payload

    sampler_checkpoint = (
        str(selected_record.get("sampler_checkpoint"))
        if selected_record and selected_record.get("sampler_checkpoint")
        else extract_sampler_checkpoint_from_payload(selected_payload)
    )
    run_name = str((selected_payload or {}).get("run_name") or "tinker-studio")

    st.markdown(
        """
        <div class="soft-panel">
            <strong>How this bridge works:</strong> Streamlit starts a localhost OpenAI-compatible server.
            Clients use the base URL below and send the model id in `/v1/chat/completions`; the bridge routes
            that request to the selected Tinker sampler checkpoint.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_id = st.text_input(
        "Model id advertised by the bridge",
        value=run_name,
        key="endpoint_model_id",
        help="Use this model value from Float, curl, or any OpenAI-compatible client.",
    )
    base_url = st.text_input(
        "OpenAI-compatible base URL",
        value=ENDPOINT_BASE_URL,
        key="endpoint_base_url",
        help="Paste this into clients that support custom OpenAI-compatible endpoints.",
    )
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05, key="endpoint_temperature")
    max_tokens = st.slider("Max tokens", min_value=24, max_value=512, value=128, step=8, key="endpoint_max_tokens")

    bridge_ok, bridge_detail = endpoint_health(base_url)
    left, middle, right = st.columns(3)
    left.metric("Sampler checkpoint", "FOUND" if sampler_checkpoint else "MISSING")
    middle.metric("Bridge", "ONLINE" if bridge_ok else "OFFLINE")
    right.metric("Port", endpoint_root(base_url).rsplit(":", 1)[-1])

    if sampler_checkpoint:
        render_status_card("Selected sampler", run_display_name(selected_payload), Path(sampler_checkpoint).name)
        with st.expander("Checkpoint URI"):
            st.code(sampler_checkpoint, language="text")
    else:
        st.warning(
            "No sampler checkpoint found in run_outputs/latest_active_run.json. "
            "Run post-train sampling or pass --sampler-checkpoint to the endpoint server."
        )

    st.markdown("#### Launch Bridge")
    command_preview = f'.\\launch_tinker_endpoint.bat --run-name "{run_name}" --port {ENDPOINT_PORT} --model-id "{model_id}"'
    if sampler_checkpoint:
        command_preview += ' --sampler-checkpoint "<selected checkpoint URI>"'
    st.code(
        command_preview,
        language="powershell",
    )
    start_col, stop_col, clear_col = st.columns(3)
    if start_col.button("Start Endpoint", width="stretch", key="endpoint_start_button"):
        ok, detail = start_endpoint_bridge(
            run_name,
            ENDPOINT_PORT,
            model_id=model_id,
            sampler_checkpoint=sampler_checkpoint,
        )
        if ok:
            st.success(detail)
            st.cache_data.clear()
        else:
            st.error(detail)
    if stop_col.button("Stop Endpoint", width="stretch", key="endpoint_stop_button"):
        st.info(stop_process_on_port(ENDPOINT_PORT) or "No endpoint process found.")
        st.cache_data.clear()
    if clear_col.button("Clear Chat", width="stretch", key="endpoint_clear_chat_button"):
        st.session_state.endpoint_chat_messages = []
        st.rerun()
    st.markdown("#### Float / OpenAI-Compatible Config")
    st.code(
        json.dumps(
            {
                "base_url": base_url,
                "model": model_id,
                "api_key": "local-dev",
            },
            indent=2,
        ),
        language="json",
    )

    if not bridge_ok:
        st.info(f"Bridge is not reachable yet: {bridge_detail}")
        return
    with st.expander("Bridge health response"):
        st.code(bridge_detail, language="json")

    st.markdown("#### Chat Smoke Test")
    if "endpoint_chat_messages" not in st.session_state:
        st.session_state.endpoint_chat_messages = []
    for message in st.session_state.endpoint_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    prompt = st.chat_input("Send a message to the local trained endpoint")
    if prompt:
        st.session_state.endpoint_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Sampling..."):
                try:
                    response_text = call_chat_endpoint(
                        base_url=base_url,
                        model=model_id,
                        messages=st.session_state.endpoint_chat_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    response_text = f"Endpoint error: {exc}"
                st.write(response_text)
        st.session_state.endpoint_chat_messages.append({"role": "assistant", "content": response_text})


def render_evaluation() -> None:
    st.markdown("### Evaluation")
    st.markdown(
        """
        <div class="soft-panel">
            Evaluation compares held-out openings against the real target text. If the local endpoint bridge is online,
            this page can also generate model continuations for side-by-side review.
        </div>
        """,
        unsafe_allow_html=True,
    )
    variant_name = st.selectbox(
        "Dataset variant",
        ["initial_posts", "recent_posts_plus_essays", "recent_posts_essays_interview", "personal_sources_mix"],
        index=2,
        key="eval_dataset_variant",
    )
    split_name = st.selectbox("Split", ["test", "validation", "train"], index=0, key="eval_split")
    limit = st.slider("Held-out examples", min_value=1, max_value=12, value=5, key="eval_limit")
    settings = (variant_name, split_name, int(limit))
    if st.session_state.get("eval_settings") != settings:
        st.session_state.eval_loaded = False
        st.session_state.eval_settings = settings

    if not st.session_state.get("eval_loaded"):
        st.info("Held-out examples are loaded on demand because the Tinker dataset helpers are relatively heavy.")
        if not st.button("Load Held-Out Examples", width="stretch", key="eval_load_button"):
            return
        st.session_state.eval_loaded = True

    try:
        with st.spinner("Building held-out examples..."):
            rows = load_evaluation_rows(variant_name, split_name, limit)
    except Exception as exc:
        st.error(f"Could not build evaluation rows: {exc}")
        return
    if not rows:
        st.warning("No evaluation rows found for this selection.")
        return
    eval_df = pd.DataFrame(rows)
    st.dataframe(eval_df, width="stretch", hide_index=True)

    base_url = st.text_input("Evaluation endpoint", value=ENDPOINT_BASE_URL, key="eval_base_url")
    model_id = st.text_input(
        "Evaluation model id",
        value=str(st.session_state.get("endpoint_model_id") or variant_name),
        key="eval_model_id",
        help="This is the model id sent to the local OpenAI-compatible bridge.",
    )
    bridge_ok, bridge_detail = endpoint_health(base_url)
    if not bridge_ok:
        st.info(f"Endpoint offline: {bridge_detail}")
        return
    if st.button("Generate Evaluation Samples", width="stretch", key="eval_generate_button"):
        generated_rows = []
        with st.spinner("Generating held-out comparisons..."):
            for row in rows:
                try:
                    generated = call_chat_endpoint(
                        base_url=base_url,
                        model=model_id,
                        messages=[{"role": "user", "content": str(row["opening_text"])}],
                        temperature=0.7,
                        max_tokens=128,
                    )
                except Exception as exc:
                    generated = f"ERROR: {exc}"
                generated_rows.append({**row, "generated_text": generated})
        st.dataframe(pd.DataFrame(generated_rows), width="stretch", hide_index=True)


def render_diagnostics(manifest: dict[str, Any]) -> None:
    st.markdown("### Diagnostics")
    endpoint_ok, endpoint_detail = endpoint_health(ENDPOINT_BASE_URL)
    notebook_summary = notebook_outputs_summary(WORKSPACE_ROOT / "tinker_train_and_eval.ipynb")
    left, middle, right = st.columns(3)
    left.metric("Dashboard", "ONLINE")
    middle.metric("Endpoint", "ONLINE" if endpoint_ok else "OFFLINE")
    right.metric("Notebook outputs", str(notebook_summary.get("cells_with_outputs", "?")))

    st.markdown("#### Paths")
    st.dataframe(
        pd.DataFrame(
            [
                {"name": "workspace", "path": str(WORKSPACE_ROOT), "exists": WORKSPACE_ROOT.exists()},
                {"name": "dataset", "path": str(DATASET_ROOT), "exists": DATASET_ROOT.exists()},
                {"name": "manifest", "path": str(MANIFEST_PATH), "exists": MANIFEST_PATH.exists()},
                {"name": "imported sources", "path": str(IMPORTED_SOURCES_PATH), "exists": IMPORTED_SOURCES_PATH.exists()},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Dataset")
    st.dataframe(
        pd.DataFrame(manifest_summary_rows(manifest)),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Raw manifest JSON"):
        st.code(json.dumps(manifest, indent=2, default=str), language="json")

    st.markdown("#### Publish Safety")
    safety_rows = [
        {
            "check": "Notebook outputs stripped",
            "status": "PASS" if notebook_summary.get("cells_with_outputs") == 0 else "REVIEW",
            "detail": json.dumps(notebook_summary, default=str),
        },
        {
            "check": "Generated dataset ignored",
            "status": "PASS" if run_local_command(["git", "check-ignore", str(DATASET_ROOT)], timeout=5) else "REVIEW",
            "detail": str(DATASET_ROOT),
        },
        {
            "check": "Run outputs ignored",
            "status": "PASS" if run_local_command(["git", "check-ignore", "run_outputs"], timeout=5) else "REVIEW",
            "detail": "run_outputs",
        },
    ]
    st.dataframe(pd.DataFrame(safety_rows), width="stretch", hide_index=True)
    with st.expander("Git status"):
        st.code(load_git_status() or "clean", language="text")
    with st.expander("Endpoint detail"):
        st.code(endpoint_detail, language="text")

    button_cols = st.columns(4)
    if button_cols[0].button("Clear Streamlit Cache", width="stretch", key="diag_clear_cache_button"):
        st.cache_data.clear()
        st.rerun()
    if button_cols[1].button("Stop Endpoint", width="stretch", key="diag_stop_endpoint_button"):
        st.info(stop_process_on_port(ENDPOINT_PORT) or "No endpoint process found.")
    if button_cols[2].button("Refresh Dataset", width="stretch", key="diag_refresh_dataset_button"):
        st.cache_data.clear()
        st.rerun()
    if button_cols[3].button("Show Suggestions", width="stretch", key="diag_show_suggestions_button"):
        st.session_state.show_suggestions = True

    if st.session_state.get("show_suggestions"):
        st.markdown("#### Suggested Next Improvements")
        st.write(
            "- Add a real queue for long-running imports and Bluesky refreshes.\n"
            "- Add source weights per Keep color, label, poetry, notes, and essays.\n"
            "- Add a public/private publish report that emits exact files safe to commit.\n"
            "- Add full Bluesky thread expansion as an optional mode with max depth and max chars.\n"
            "- Add endpoint auth for non-local use; keep current bridge localhost-only by default."
        )


def render_refresh_controls(manifest: dict[str, Any]) -> None:
    st.sidebar.header("Post Source")
    handle = st.sidebar.text_input("Bluesky handle", value=default_bluesky_handle(manifest), key="sidebar_bluesky_handle")
    st.sidebar.caption(f"Current snapshot: {manifest.get('collected_at_utc', 'unknown')}")

    if st.sidebar.button("Refresh Bluesky Snapshot", type="primary", width="stretch", key="sidebar_pull_posts_button"):
        if not DATASET_BUILDER_PATH.exists():
            st.sidebar.error(f"Dataset builder not found: {DATASET_BUILDER_PATH}")
            return
        with st.spinner("Pulling posts and rebuilding dataset files..."):
            try:
                returncode, output = run_dataset_refresh(handle)
            except subprocess.TimeoutExpired:
                st.sidebar.error("Post refresh timed out after 180 seconds.")
                return
        if returncode == 0:
            st.cache_data.clear()
            st.sidebar.success("Dataset refreshed.")
            if output:
                st.sidebar.code(output[-1800:], language="text")
            st.rerun()
        else:
            st.sidebar.error(f"Refresh failed with exit code {returncode}.")
            if output:
                st.sidebar.code(output[-2400:], language="text")

    st.sidebar.divider()
    st.sidebar.caption(f"Workspace: {WORKSPACE_ROOT}")


def main() -> None:
    apply_theme()
    manifest = load_manifest(MANIFEST_PATH)
    posts = load_posts(POSTS_CSV_PATH)
    rentry_rows = load_jsonl(RENTRY_PAGES_PATH)
    imported_rows = load_jsonl(IMPORTED_SOURCES_PATH)

    render_refresh_controls(manifest)
    render_header(manifest)

    corpus_tab, evaluation_tab, training_tab, chat_tab, diagnostics_tab, files_tab = st.tabs(
        ["Corpus", "Evaluation", "Training", "Chat / Endpoint", "Diagnostics", "Files"]
    )
    with corpus_tab:
        render_dataset_overview(posts, manifest, rentry_rows, imported_rows)
        render_sources_overview(posts, rentry_rows, imported_rows)
    with evaluation_tab:
        render_evaluation()
    with training_tab:
        render_training_overview()
    with chat_tab:
        render_endpoint_chat()
    with diagnostics_tab:
        render_diagnostics(manifest)
    with files_tab:
        st.dataframe(
            pd.DataFrame(
                [
                    {"artifact": "Posts CSV", "path": str(POSTS_CSV_PATH), "exists": POSTS_CSV_PATH.exists()},
                    {"artifact": "Dataset manifest", "path": str(MANIFEST_PATH), "exists": MANIFEST_PATH.exists()},
                    {"artifact": "Long-form seed docs", "path": str(RENTRY_PAGES_PATH), "exists": RENTRY_PAGES_PATH.exists()},
                    {"artifact": "Imported sources", "path": str(IMPORTED_SOURCES_PATH), "exists": IMPORTED_SOURCES_PATH.exists()},
                    {"artifact": "Dataset builder", "path": str(DATASET_BUILDER_PATH), "exists": DATASET_BUILDER_PATH.exists()},
                    {"artifact": "Source importer", "path": str(SOURCE_IMPORTER_PATH), "exists": SOURCE_IMPORTER_PATH.exists()},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if manifest:
            st.markdown("#### Dataset Manifest Summary")
            st.dataframe(pd.DataFrame(manifest_summary_rows(manifest)), width="stretch", hide_index=True)
            with st.expander("Raw manifest JSON"):
                st.code(json.dumps(manifest, indent=2, default=str), language="json")


if __name__ == "__main__":
    main()
