from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import tinker
from tinker_cookbook.renderers import get_renderer

from run_tinker_experiment import find_latest_payload
from tinker_notebook_env import ensure_tinker_api_key
from tinker_training_utils import (
    COMPLETION_SYSTEM_PROMPT,
    CONVERSATIONAL_SYSTEM_PROMPT,
    ConversationExample,
    sample_generations,
    select_renderer_name,
)


CHAT_HISTORY_DIRNAME = "endpoint_chat_history"
DEFAULT_CHAT_MAX_TOKENS = 192
DEFAULT_CHAT_TEMPERATURE = 0.4
DEFAULT_CHAT_MODE = "chat"
CHAT_MODES = {"chat", "completion"}
RUN_RECORD_LIMIT = 10
PROMPT_HISTORY_MESSAGE_LIMIT = 12
PROMPT_HISTORY_CHAR_LIMIT = 7000
CONTROL_TOKEN_RE = re.compile(r"<\|[^>|]{1,80}\|>")
HARMONY_ASSISTANT_HEADER_RE = re.compile(
    r"<\|start\|>\s*assistant\s*<\|channel\|>\s*(analysis|commentary|final)\s*<\|message\|>",
    re.IGNORECASE,
)
HARMONY_CHANNEL_HEADER_RE = re.compile(
    r"<\|channel\|>\s*(analysis|commentary|final)\s*<\|message\|>",
    re.IGNORECASE,
)


def extract_sampler_checkpoint(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    sampler_model_path = payload.get("sampler_model_path")
    if isinstance(sampler_model_path, str) and sampler_model_path.strip():
        return sampler_model_path
    summary = payload.get("summary")
    if isinstance(summary, dict):
        sampler_model_path = summary.get("sampler_model_path")
        if isinstance(sampler_model_path, str) and sampler_model_path.strip():
            return sampler_model_path
    return None


def run_key_for_checkpoint(sampler_checkpoint: str) -> str:
    return hashlib.sha1(sampler_checkpoint.encode("utf-8")).hexdigest()[:12]


def run_payload_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is not None:
        return value
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary.get(key)
    return None


def run_sort_timestamp(payload: dict[str, Any], path: Path) -> str:
    for key in ("completed_at_utc", "last_event_at_utc", "started_at_utc"):
        value = run_payload_value(payload, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_record(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    sampler_checkpoint = extract_sampler_checkpoint(payload)
    if not sampler_checkpoint:
        return None
    run_name = str(run_payload_value(payload, "run_name") or path.stem).strip()
    resolved_name = str(run_payload_value(payload, "resolved_name") or "").strip()
    model_alias = str(run_payload_value(payload, "model_alias") or "").strip()
    base_model = str(run_payload_value(payload, "base_model") or resolved_name or model_alias or "").strip()
    started_at = str(run_payload_value(payload, "started_at_utc") or "").strip()
    completed_at = str(run_payload_value(payload, "completed_at_utc") or "").strip()
    return {
        "run_key": run_key_for_checkpoint(sampler_checkpoint),
        "run_name": run_name,
        "status": str(run_payload_value(payload, "status") or "").strip(),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "sort_timestamp": run_sort_timestamp(payload, path),
        "model_alias": model_alias,
        "resolved_name": resolved_name,
        "base_model": base_model,
        "dataset_variant": str(run_payload_value(payload, "dataset_variant") or "").strip(),
        "training_run_id": str(run_payload_value(payload, "training_run_id") or "").strip(),
        "sampler_id": str(run_payload_value(payload, "sampler_id") or "").strip(),
        "sampler_checkpoint": sampler_checkpoint,
        "learning_rate": run_payload_value(payload, "learning_rate"),
        "lora_rank": run_payload_value(payload, "lora_rank"),
        "batch_size": run_payload_value(payload, "batch_size"),
        "num_epochs": run_payload_value(payload, "num_epochs"),
        "include_tags": run_payload_value(payload, "include_tags") or [],
        "exclude_tags": run_payload_value(payload, "exclude_tags") or [],
        "record_path": str(path),
    }


def load_recent_run_records(run_dir: Path, *, limit: int = RUN_RECORD_LIMIT) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_checkpoints: set[str] = set()
    paths = sorted(run_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        record = build_run_record(path, payload)
        if record is None:
            continue
        checkpoint = record["sampler_checkpoint"]
        if checkpoint in seen_checkpoints:
            continue
        seen_checkpoints.add(checkpoint)
        records.append(record)
        if len(records) >= limit:
            break
    records.sort(key=lambda item: str(item.get("sort_timestamp") or ""), reverse=True)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local OpenAI-compatible chat bridge for a Tinker sampler checkpoint.")
    parser.add_argument("--workspace", default=str(Path.cwd()), help="Workspace root.")
    parser.add_argument("--run-name", default="essay_recent_r16", help="Run name used to infer the sampler checkpoint.")
    parser.add_argument("--sampler-checkpoint", help="Explicit sampler checkpoint path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument("--model-id", default="", help="Model id advertised through /v1/models.")
    parser.add_argument(
        "--default-mode",
        choices=sorted(CHAT_MODES),
        default=DEFAULT_CHAT_MODE,
        help="Default browser/API mode: chat preserves conversation turns; completion finishes the user's text.",
    )
    return parser.parse_args()


class TinkerEndpoint:
    def __init__(
        self,
        *,
        workspace: Path,
        run_name: str,
        sampler_checkpoint: str | None,
        model_id: str,
        default_mode: str = DEFAULT_CHAT_MODE,
    ) -> None:
        ensure_tinker_api_key(required=True)
        self.workspace = workspace
        self.run_dir = workspace / "run_outputs"
        self.run_name = run_name
        self.history_dir = workspace / "run_outputs" / CHAT_HISTORY_DIRNAME
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id or run_name
        self.default_mode = normalize_chat_mode(default_mode)
        self._sampler_lock = threading.Lock()
        self.service_client = tinker.ServiceClient()
        self.rest_client = self.service_client.create_rest_client()
        self.sampler_checkpoint = ""
        self.base_model = "unknown"
        self.sampling_client = None
        self.tokenizer = None
        self.renderer_name = ""
        self.renderer = None
        initial_checkpoint = sampler_checkpoint or self._resolve_sampler_checkpoint()
        initial_record = self.find_run_record(sampler_checkpoint=initial_checkpoint)
        self.active_run = {}
        self.load_sampler(initial_checkpoint, run_record=initial_record, model_id=model_id or run_name)

    def _resolve_sampler_checkpoint(self) -> str:
        payload = find_latest_payload(self.run_dir, run_name=self.run_name)
        checkpoint = extract_sampler_checkpoint(payload)
        if not checkpoint:
            raise RuntimeError(
                f"No sampler checkpoint found for {self.run_name}. "
                "Pass --sampler-checkpoint or run post-train sampling first."
            )
        return checkpoint

    def load_sampler(
        self,
        sampler_checkpoint: str,
        *,
        run_record: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        checkpoint_info = self.rest_client.get_weights_info_by_tinker_path(sampler_checkpoint).result()
        base_model = str(getattr(checkpoint_info, "base_model", None) or "unknown")
        sampling_client = self.service_client.create_sampling_client(sampler_checkpoint)
        tokenizer = sampling_client.get_tokenizer()
        renderer_name = select_renderer_name(base_model)
        active_record = self.build_active_run_record(
            sampler_checkpoint=sampler_checkpoint,
            run_record=run_record,
            base_model=base_model,
        )
        self.sampler_checkpoint = sampler_checkpoint
        self.base_model = base_model
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.renderer_name = renderer_name
        self.renderer = None
        self.active_run = active_record
        self.run_name = str(active_record.get("run_name") or self.run_name)
        self.model_id = str(model_id or active_record.get("run_name") or self.run_name)
        return dict(self.active_run)

    def build_active_run_record(
        self,
        *,
        sampler_checkpoint: str,
        run_record: dict[str, Any] | None,
        base_model: str,
    ) -> dict[str, Any]:
        record = dict(run_record or {})
        record.setdefault("run_key", run_key_for_checkpoint(sampler_checkpoint))
        record.setdefault("run_name", self.run_name)
        record.setdefault("sampler_checkpoint", sampler_checkpoint)
        record["base_model"] = base_model or record.get("base_model") or "unknown"
        return record

    def find_run_record(
        self,
        *,
        run_key: str | None = None,
        sampler_checkpoint: str | None = None,
        model: str | None = None,
        search_limit: int = 50,
    ) -> dict[str, Any] | None:
        key = str(run_key or "").strip()
        checkpoint = str(sampler_checkpoint or "").strip()
        model_id = str(model or "").strip()
        for record in load_recent_run_records(self.run_dir, limit=search_limit):
            if key and key == record.get("run_key"):
                return record
            if checkpoint and checkpoint == record.get("sampler_checkpoint"):
                return record
            if model_id and model_id in {record.get("run_name"), record.get("model_alias"), record.get("resolved_name")}:
                return record
        return None

    def select_run(
        self,
        *,
        run_key: str | None = None,
        sampler_checkpoint: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        record = self.find_run_record(run_key=run_key, sampler_checkpoint=sampler_checkpoint, model=model)
        checkpoint = str(sampler_checkpoint or "").strip()
        if record is not None:
            checkpoint = str(record.get("sampler_checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("No matching sampler run found.")
        with self._sampler_lock:
            if checkpoint != self.sampler_checkpoint:
                return self.load_sampler(checkpoint, run_record=record, model_id=str((record or {}).get("run_name") or model or ""))
            if record is not None:
                self.active_run = self.build_active_run_record(
                    sampler_checkpoint=checkpoint,
                    run_record=record,
                    base_model=self.base_model,
                )
            return dict(self.active_run)

    def active_run_record(self) -> dict[str, Any]:
        record = dict(self.active_run or {})
        record["run_key"] = record.get("run_key") or run_key_for_checkpoint(self.sampler_checkpoint)
        record["run_name"] = record.get("run_name") or self.run_name
        record["base_model"] = record.get("base_model") or self.base_model
        record["sampler_checkpoint"] = record.get("sampler_checkpoint") or self.sampler_checkpoint
        return record

    def list_run_records(self, *, limit: int = RUN_RECORD_LIMIT) -> list[dict[str, Any]]:
        records = load_recent_run_records(self.run_dir, limit=limit)
        active_checkpoint = self.sampler_checkpoint
        if active_checkpoint and all(record.get("sampler_checkpoint") != active_checkpoint for record in records):
            records = [self.active_run_record(), *records[: max(0, limit - 1)]]
        output: list[dict[str, Any]] = []
        for record in records[:limit]:
            item = dict(record)
            item["is_active"] = bool(active_checkpoint and item.get("sampler_checkpoint") == active_checkpoint)
            output.append(item)
        return output

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        mode: str | None = None,
        run_key: str | None = None,
        model: str | None = None,
    ) -> str:
        if run_key:
            self.select_run(run_key=run_key)
        elif model and model != self.model_id:
            record = self.find_run_record(model=model)
            if record is not None:
                self.select_run(run_key=str(record.get("run_key") or ""))
        mode_name = normalize_chat_mode(mode or self.default_mode)
        example = build_endpoint_example(messages, mode=mode_name)
        if self.sampling_client is None:
            raise RuntimeError("No active sampling client is loaded.")
        if self.renderer is None:
            self.renderer = get_renderer(self.renderer_name, self.tokenizer)
        rows = sample_generations(
            self.sampling_client,
            self.renderer,
            [example],
            max_tokens=max(1, int(max_tokens)),
            temperature=float(temperature),
        )
        return sanitize_assistant_output(str(rows[0].get("generated_text") or ""))

    def list_chat_sessions(self, *, limit: int = 25) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in sorted(self.history_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue
            messages = coerce_chat_messages(payload.get("messages"))
            title = str(payload.get("title") or "").strip() or title_from_messages(messages)
            sessions.append(
                {
                    "session_id": path.stem,
                    "title": title or "Untitled chat",
                    "message_count": len(messages),
                    "updated_at_utc": payload.get("updated_at_utc"),
                    "model": payload.get("model") or self.model_id,
                    "run_name": payload.get("run_name") or self.run_name,
                }
            )
            if len(sessions) >= limit:
                break
        return sessions

    def load_chat_history(self, session_id: str) -> dict[str, Any]:
        normalized_id = normalize_session_id(session_id)
        path = self.history_path(normalized_id)
        payload = read_json(path) if path.exists() else None
        if not isinstance(payload, dict):
            payload = {}
        messages = sanitize_chat_history(coerce_chat_messages(payload.get("messages")))
        return {
            "session_id": normalized_id,
            "title": str(payload.get("title") or "").strip() or title_from_messages(messages),
            "messages": messages,
            "created_at_utc": payload.get("created_at_utc"),
            "updated_at_utc": payload.get("updated_at_utc"),
            "model": payload.get("model") or self.model_id,
            "run_name": payload.get("run_name") or self.run_name,
        }

    def save_chat_history(
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        title: str | None = None,
    ) -> dict[str, Any]:
        normalized_id = normalize_session_id(session_id)
        now = now_utc_iso()
        path = self.history_path(normalized_id)
        existing = read_json(path) if path.exists() else None
        existing_created_at = existing.get("created_at_utc") if isinstance(existing, dict) else None
        clean_messages = sanitize_chat_history(coerce_chat_messages(messages))
        payload = {
            "session_id": normalized_id,
            "title": (title or title_from_messages(clean_messages)).strip(),
            "model": self.model_id,
            "run_name": self.run_name,
            "base_model": self.base_model,
            "run_key": self.active_run_record().get("run_key"),
            "sampler_checkpoint": self.sampler_checkpoint,
            "created_at_utc": existing_created_at or now,
            "updated_at_utc": now,
            "messages": clean_messages,
        }
        write_json(path, payload)
        return payload

    def history_path(self, session_id: str) -> Path:
        return self.history_dir / f"{normalize_session_id(session_id)}.json"


def build_endpoint_example(messages: list[dict[str, str]], *, mode: str) -> ConversationExample:
    clean_messages = sanitize_chat_history(messages)
    user_text = last_user_text(clean_messages)
    if not user_text:
        raise ValueError("messages must include at least one non-empty user message")
    if mode == "completion":
        prompt_messages = [{"role": "user", "content": build_completion_prompt(clean_messages)}]
    else:
        prompt_messages = build_conversation_prompt_messages(clean_messages)
    return ConversationExample(
        example_id=f"{mode}-request",
        opening_text=user_text,
        target_text="",
        messages=[*prompt_messages, {"role": "assistant", "content": ""}],
        metadata={"source": "endpoint_chat", "mode": mode},
    )


def build_conversation_prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system_text = "\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system" and str(message.get("content") or "").strip()
    )
    system_prompt = CONVERSATIONAL_SYSTEM_PROMPT
    if system_text:
        system_prompt = f"{system_prompt}\n\nAdditional guidance:\n{system_text}"
    conversation_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    return [
        {"role": "system", "content": system_prompt},
        *trim_chat_history(conversation_messages),
    ]


def build_completion_prompt(messages: list[dict[str, str]]) -> str:
    user_text = last_user_text(messages)
    system_text = "\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system" and str(message.get("content") or "").strip()
    )
    instructions = [
        COMPLETION_SYSTEM_PROMPT,
    ]
    if system_text:
        instructions.append(f"Additional guidance: {system_text}")
    return " ".join(instructions) + "\n\nOpening:\n" + user_text


def trim_chat_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    total_chars = 0
    for message in reversed(messages):
        if len(selected) >= PROMPT_HISTORY_MESSAGE_LIMIT:
            break
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        remaining_chars = PROMPT_HISTORY_CHAR_LIMIT - total_chars
        if remaining_chars <= 0:
            break
        if len(content) > remaining_chars:
            if selected:
                break
            content = content[-remaining_chars:]
        selected.append({"role": message["role"], "content": content})
        total_chars += len(content)
    return list(reversed(selected))


def sanitize_chat_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    clean_messages: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        if role == "assistant":
            had_control_tokens = bool(CONTROL_TOKEN_RE.search(text))
            was_repetitive = is_repetition_dominated(text)
            text = sanitize_assistant_output(text)
            if had_control_tokens or was_repetitive:
                continue
        else:
            text = strip_control_tokens(text)
        if text:
            clean_messages.append({"role": role, "content": text})
    return clean_messages


def sanitize_assistant_output(text: str) -> str:
    cleaned = strip_control_tokens(text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = cap_repeated_sentences(cleaned, max_repeats=2)
    if is_repetition_dominated(cleaned):
        cleaned = first_words(cleaned, max_words=80).rstrip(",.;:") + "..."
    return cleaned.strip()


def strip_control_tokens(text: str) -> str:
    cleaned = HARMONY_ASSISTANT_HEADER_RE.sub("", str(text))
    cleaned = HARMONY_CHANNEL_HEADER_RE.sub("", cleaned)
    cleaned = CONTROL_TOKEN_RE.sub("", cleaned)
    cleaned = cleaned.replace("<|end|>", "").replace("<|start|>", "")
    return cleaned.strip()


def cap_repeated_sentences(text: str, *, max_repeats: int) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if len(sentences) < max_repeats + 2:
        return text
    seen: dict[str, int] = {}
    kept: list[str] = []
    for sentence in sentences:
        key = normalize_repetition_unit(sentence)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_repeats:
            kept.append(sentence)
    return " ".join(kept).strip() if kept else text


def is_repetition_dominated(text: str) -> bool:
    stripped = strip_control_tokens(text)
    words = re.findall(r"\w+", stripped.lower())
    if len(words) < 24:
        return False
    sentences = [normalize_repetition_unit(part) for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
    if len(sentences) >= 4:
        sentence_counts = {sentence: sentences.count(sentence) for sentence in set(sentences)}
        if max(sentence_counts.values(), default=0) >= 4:
            return True
    for ngram_size in range(3, 8):
        if len(words) < ngram_size * 4:
            continue
        ngrams = [" ".join(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1)]
        counts = {ngram: ngrams.count(ngram) for ngram in set(ngrams)}
        top_count = max(counts.values(), default=0)
        if top_count >= 6 and top_count * ngram_size >= len(words) * 0.4:
            return True
    return False


def normalize_repetition_unit(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def first_words(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def normalize_chat_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"complete", "completion", "post", "continue"}:
        return "completion"
    if mode in {"chat", "conversation", "conversational"}:
        return "chat"
    raise ValueError(f"Unsupported chat mode: {value!r}. Expected one of: {', '.join(sorted(CHAT_MODES))}")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_session_id(value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return f"chat-{uuid.uuid4().hex[:12]}"
    safe_chars = []
    for character in candidate[:80]:
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        else:
            safe_chars.append("-")
    normalized = "".join(safe_chars).strip("-_")
    return normalized or f"chat-{uuid.uuid4().hex[:12]}"


def coerce_chat_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        content = item.get("content")
        if content is None:
            content = item.get("text")
        text = str(content or "").strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def title_from_messages(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(str(message.get("content") or "").split())
            if text:
                return text[:80]
    return ""


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = str(message.get("content") or "").strip()
            if text:
                return text
    return ""


def render_chat_page(endpoint: TinkerEndpoint) -> str:
    config = json.dumps(
        {
            "model": endpoint.model_id,
            "apiBase": "/v1",
            "chatUrl": "/v1/chat/completions",
            "healthUrl": "/health",
            "historyUrl": "/v1/chat/history",
            "sessionsUrl": "/v1/chat/sessions",
            "runsUrl": "/v1/runs",
            "selectRunUrl": "/v1/runs/select",
            "defaultMaxTokens": DEFAULT_CHAT_MAX_TOKENS,
            "defaultTemperature": DEFAULT_CHAT_TEMPERATURE,
            "defaultMode": endpoint.default_mode,
            "activeRun": endpoint.active_run_record(),
        },
        ensure_ascii=True,
    )
    title = html.escape(f"Tinker Studio Chat - {endpoint.model_id}")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #172126;
      --muted: #647078;
      --line: #dce3e6;
      --accent: #116a5b;
      --accent-strong: #0c4f45;
      --assistant: #e9f2f0;
      --user: #eaf0fb;
      --error: #9f2525;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .topbar {{
      width: min(1040px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 18px 0;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }}
    h1 {{
      margin: 0;
      font-size: 1.15rem;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fbfcfc;
      white-space: nowrap;
    }}
    main {{
      width: min(1040px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(280px, 1.8fr) minmax(130px, 0.7fr) minmax(220px, 1.2fr) 120px 130px auto auto;
      gap: 10px;
      align-items: end;
    }}
    .run-detail {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfc;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .run-detail strong {{
      color: var(--ink);
      font-weight: 650;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 600;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      font: inherit;
      color: var(--ink);
      background: var(--panel);
    }}
    input:focus, select:focus, textarea:focus {{
      outline: 2px solid rgba(17, 106, 91, 0.2);
      border-color: var(--accent);
    }}
    select {{
      min-height: 42px;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      padding: 11px 16px;
      font: inherit;
      font-weight: 650;
      color: #ffffff;
      background: var(--accent);
      cursor: pointer;
      min-height: 42px;
    }}
    button:hover {{ background: var(--accent-strong); }}
    button:disabled {{
      opacity: 0.55;
      cursor: wait;
    }}
    .secondary {{
      color: var(--ink);
      background: #edf1f2;
      border: 1px solid var(--line);
    }}
    .secondary:hover {{ background: #e2e8ea; }}
    #conversation {{
      min-height: 42vh;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding-bottom: 4px;
    }}
    .empty {{
      margin: auto;
      color: var(--muted);
      text-align: center;
    }}
    .message {{
      max-width: 78%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      white-space: pre-wrap;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .message.user {{
      align-self: flex-end;
      background: var(--user);
    }}
    .message.assistant {{
      align-self: flex-start;
      background: var(--assistant);
    }}
    .message.error {{
      align-self: flex-start;
      background: #fff0f0;
      color: var(--error);
      border-color: #efc7c7;
    }}
    form {{
      border-top: 1px solid var(--line);
      background: var(--panel);
    }}
    .composer {{
      width: min(1040px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 14px 0 16px;
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: end;
    }}
    textarea {{
      min-height: 52px;
      max-height: 180px;
      resize: vertical;
    }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    @media (max-width: 760px) {{
      .topbar, .controls, .composer {{
        grid-template-columns: 1fr;
      }}
      .meta {{
        justify-content: flex-start;
      }}
      .message {{
        max-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="topbar">
        <h1>Tinker Studio Chat</h1>
        <div class="meta" aria-live="polite">
          <span class="pill" id="status">Checking endpoint</span>
          <span class="pill" id="model-pill"></span>
          <span class="pill">API base /v1</span>
        </div>
      </div>
    </header>
    <main>
      <section class="controls" aria-label="Sampling controls">
        <label>Run
          <select id="run-list">
            <option value="">Loading runs</option>
          </select>
        </label>
        <input id="model" type="hidden">
        <label>Mode
          <select id="mode">
            <option value="chat">Chat</option>
            <option value="completion">Completion</option>
          </select>
        </label>
        <label>History
          <select id="session-list">
            <option value="">Current chat</option>
          </select>
        </label>
        <label>Temperature
          <input id="temperature" type="number" min="0" max="2" step="0.05">
        </label>
        <label>Max tokens
          <input id="max-tokens" type="number" min="1" max="4096" step="1">
        </label>
        <button type="button" class="secondary" id="new-chat">New</button>
        <button type="button" class="secondary" id="refresh">Refresh</button>
      </section>
      <div class="run-detail" id="run-detail"></div>
      <section id="conversation" aria-live="polite">
        <div class="empty" id="empty">Send a message to the selected local sampler.</div>
      </section>
    </main>
    <form id="chat-form">
      <div class="composer">
        <label class="sr-only" for="prompt">Message</label>
        <textarea id="prompt" placeholder="Message the local Tinker endpoint" required></textarea>
        <button type="submit" id="send">Send</button>
        <button type="button" class="secondary" id="clear">Clear</button>
      </div>
    </form>
  </div>
  <script>
    const config = {config};
    const storageKey = "tinkerEndpointChatState";
    const sessionIdKey = "tinkerEndpointChatSessionId";
    const state = {{
      sessionId: getStoredSessionId(),
      mode: config.defaultMode || "chat",
      runKey: config.activeRun?.run_key || "",
      runs: [],
      messages: [],
    }};
    const statusEl = document.getElementById("status");
    const modelEl = document.getElementById("model");
    const runListEl = document.getElementById("run-list");
    const runDetailEl = document.getElementById("run-detail");
    const modeEl = document.getElementById("mode");
    const modelPillEl = document.getElementById("model-pill");
    const sessionListEl = document.getElementById("session-list");
    const conversationEl = document.getElementById("conversation");
    const emptyEl = document.getElementById("empty");
    const promptEl = document.getElementById("prompt");
    const sendEl = document.getElementById("send");
    const temperatureEl = document.getElementById("temperature");
    const maxTokensEl = document.getElementById("max-tokens");

    modelEl.value = config.model;
    modeEl.value = state.mode;
    temperatureEl.value = String(config.defaultTemperature || 0.4);
    maxTokensEl.value = String(config.defaultMaxTokens || 192);
    modelPillEl.textContent = `Model ${{config.model}}`;

    function generateSessionId() {{
      return `chat-${{Date.now().toString(36)}}-${{Math.random().toString(16).slice(2, 10)}}`;
    }}

    function getStoredSessionId() {{
      try {{
        return localStorage.getItem(sessionIdKey) || generateSessionId();
      }} catch (_error) {{
        return generateSessionId();
      }}
    }}

    function loadLocalState() {{
      try {{
        const cached = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
        if (cached && cached.sessionId === state.sessionId && Array.isArray(cached.messages)) {{
          if (["chat", "completion"].includes(cached.mode)) state.mode = cached.mode;
          if (typeof cached.runKey === "string" && cached.runKey.trim()) state.runKey = cached.runKey;
          state.messages = cached.messages.filter((message) =>
            message && ["system", "user", "assistant"].includes(message.role) && String(message.content || "").trim()
          );
          modeEl.value = state.mode;
        }}
      }} catch (_error) {{
        state.messages = [];
      }}
    }}

    function persistLocalState() {{
      try {{
        localStorage.setItem(sessionIdKey, state.sessionId);
        localStorage.setItem(storageKey, JSON.stringify({{
          sessionId: state.sessionId,
          mode: state.mode,
          runKey: state.runKey,
          messages: state.messages,
        }}));
      }} catch (_error) {{
        // Local storage is a convenience; server history is the durable copy.
      }}
    }}

    function setStatus(text, ok) {{
      statusEl.textContent = text;
      statusEl.style.borderColor = ok ? "#9ccbc3" : "#efc7c7";
      statusEl.style.color = ok ? "#0c4f45" : "#9f2525";
    }}

    function escapeHtml(value) {{
      return String(value || "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }}[char]));
    }}

    function parseRunTime(value) {{
      const raw = String(value || "").trim();
      let match = /^(\\d{{4}})(\\d{{2}})(\\d{{2}})T(\\d{{2}})(\\d{{2}})(\\d{{2}})Z$/.exec(raw);
      if (match) {{
        return new Date(Date.UTC(
          Number(match[1]),
          Number(match[2]) - 1,
          Number(match[3]),
          Number(match[4]),
          Number(match[5]),
          Number(match[6])
        ));
      }}
      const parsed = raw ? new Date(raw) : null;
      return parsed && !Number.isNaN(parsed.getTime()) ? parsed : null;
    }}

    function formatRunTime(value) {{
      const date = parseRunTime(value);
      if (!date) return "";
      return date.toLocaleString(undefined, {{
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      }});
    }}

    function runBaseModel(run) {{
      return run?.base_model || run?.resolved_name || run?.model_alias || "";
    }}

    function runOptionLabel(run) {{
      const time = formatRunTime(run?.completed_at_utc || run?.started_at_utc);
      const base = runBaseModel(run);
      return [run?.run_name || "unnamed run", time, base].filter(Boolean).join(" - ");
    }}

    function updateRunDetail(run) {{
      if (!run) {{
        runDetailEl.textContent = "No sampler run selected.";
        return;
      }}
      const completed = formatRunTime(run.completed_at_utc);
      const started = formatRunTime(run.started_at_utc);
      const parts = [
        completed ? `completed ${{completed}}` : (started ? `started ${{started}}` : ""),
        runBaseModel(run) ? `base ${{runBaseModel(run)}}` : "",
        run.dataset_variant ? `dataset ${{run.dataset_variant}}` : "",
        run.status ? `status ${{run.status}}` : "",
        run.learning_rate ? `lr ${{run.learning_rate}}` : "",
      ].filter(Boolean);
      runDetailEl.innerHTML = `<strong>${{escapeHtml(run.run_name || "unnamed run")}}</strong> ${{escapeHtml(parts.join(" - "))}}`;
      modelEl.value = run.run_name || config.model;
      modelPillEl.textContent = `Model ${{run.run_name || config.model}}`;
    }}

    async function loadRuns() {{
      try {{
        const response = await fetch(config.runsUrl, {{ cache: "no-store" }});
        const data = await response.json();
        if (!response.ok) throw new Error(data?.error?.message || response.statusText);
        state.runs = Array.isArray(data.runs) ? data.runs : [];
        const activeRun = data.active_run || config.activeRun || null;
        if (!state.runKey && activeRun?.run_key) state.runKey = activeRun.run_key;
        if (state.runKey && !state.runs.some((run) => run.run_key === state.runKey) && activeRun?.run_key === state.runKey) {{
          state.runs.unshift(activeRun);
        }}
        let selectedRun = state.runs.find((run) => run.run_key === state.runKey) || null;
        if (!selectedRun) {{
          state.runKey = activeRun?.run_key || state.runs[0]?.run_key || "";
          selectedRun = state.runs.find((run) => run.run_key === state.runKey) || activeRun || null;
        }}
        runListEl.innerHTML = "";
        state.runs.forEach((run) => {{
          const option = document.createElement("option");
          option.value = run.run_key || "";
          option.textContent = runOptionLabel(run);
          runListEl.appendChild(option);
        }});
        if (!state.runs.length) {{
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No sampler runs";
          runListEl.appendChild(option);
        }}
        runListEl.value = state.runKey;
        updateRunDetail(selectedRun);
        persistLocalState();
      }} catch (error) {{
        runListEl.innerHTML = '<option value="">Runs unavailable</option>';
        runDetailEl.textContent = `Runs unavailable: ${{error.message}}`;
      }}
    }}

    async function selectRun(runKey) {{
      if (!runKey) return;
      const localRun = state.runs.find((run) => run.run_key === runKey);
      state.runKey = runKey;
      updateRunDetail(localRun);
      persistLocalState();
      setStatus("Loading run", true);
      const response = await fetch(config.selectRunUrl, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ run_key: runKey }}),
      }});
      const data = await response.json().catch(() => ({{}}));
      if (!response.ok) throw new Error(data?.error?.message || response.statusText);
      if (data.active_run) {{
        const index = state.runs.findIndex((run) => run.run_key === data.active_run.run_key);
        if (index >= 0) state.runs[index] = data.active_run;
        updateRunDetail(data.active_run);
      }}
      setStatus("Endpoint online", true);
    }}

    async function refreshHealth() {{
      try {{
        const response = await fetch(config.healthUrl, {{ cache: "no-store" }});
        const data = await response.json();
        if (!response.ok) throw new Error(data?.error?.message || response.statusText);
        setStatus("Endpoint online", true);
        if (data.model) {{
          modelEl.value = data.model;
          modelPillEl.textContent = `Model ${{data.model}}`;
        }}
        if (data.active_run?.run_key) {{
          state.runKey = data.active_run.run_key;
          updateRunDetail(data.active_run);
        }}
      }} catch (error) {{
        setStatus(`Endpoint offline: ${{error.message}}`, false);
      }}
    }}

    function renderMessages() {{
      conversationEl.querySelectorAll(".message").forEach((node) => node.remove());
      emptyEl.hidden = state.messages.length > 0;
      state.messages.forEach((message) => addMessage(message.role, message.content, false));
    }}

    function addMessage(role, content) {{
      emptyEl.hidden = true;
      const message = document.createElement("div");
      message.className = `message ${{role}}`;
      message.textContent = content;
      conversationEl.appendChild(message);
      message.scrollIntoView({{ block: "end", behavior: "smooth" }});
      return message;
    }}

    async function loadSessions() {{
      try {{
        const response = await fetch(config.sessionsUrl, {{ cache: "no-store" }});
        const data = await response.json();
        if (!response.ok) throw new Error(data?.error?.message || response.statusText);
        const sessions = Array.isArray(data.sessions) ? data.sessions : [];
        sessionListEl.innerHTML = "";
        const currentOption = document.createElement("option");
        currentOption.value = state.sessionId;
        currentOption.textContent = "Current chat";
        sessionListEl.appendChild(currentOption);
        sessions.forEach((session) => {{
          if (!session?.session_id || session.session_id === state.sessionId) return;
          const option = document.createElement("option");
          option.value = session.session_id;
          const count = Number(session.message_count || 0);
          option.textContent = `${{session.title || "Untitled chat"}} (${{count}})`;
          sessionListEl.appendChild(option);
        }});
        sessionListEl.value = state.sessionId;
      }} catch (_error) {{
        sessionListEl.innerHTML = `<option value="${{state.sessionId}}">Current chat</option>`;
      }}
    }}

    async function loadHistory(sessionId) {{
      const url = `${{config.historyUrl}}?session_id=${{encodeURIComponent(sessionId)}}`;
      const response = await fetch(url, {{ cache: "no-store" }});
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error?.message || response.statusText);
      state.sessionId = data.session_id || sessionId;
      const serverMessages = Array.isArray(data.messages) ? data.messages : [];
      if (serverMessages.length === 0 && state.messages.length > 0) {{
        persistLocalState();
        await saveHistory();
        await loadSessions();
        return;
      }}
      state.messages = serverMessages;
      persistLocalState();
      renderMessages();
      await loadSessions();
    }}

    async function saveHistory() {{
      const payload = {{ session_id: state.sessionId, messages: state.messages }};
      const response = await fetch(config.historyUrl, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      if (!response.ok) {{
        const data = await response.json().catch(() => ({{}}));
        throw new Error(data?.error?.message || response.statusText);
      }}
    }}

    async function sendMessage(event) {{
      event.preventDefault();
      const prompt = promptEl.value.trim();
      if (!prompt) return;
      promptEl.value = "";
      state.messages.push({{ role: "user", content: prompt }});
      persistLocalState();
      addMessage("user", prompt);
      const pending = addMessage("assistant", "Sampling...");
      sendEl.disabled = true;
      try {{
        const payload = {{
          session_id: state.sessionId,
          model: modelEl.value.trim() || config.model,
          run_key: state.runKey,
          messages: state.messages,
          mode: modeEl.value || state.mode || config.defaultMode || "chat",
          temperature: Number(temperatureEl.value || config.defaultTemperature || 0.4),
          max_tokens: Number(maxTokensEl.value || config.defaultMaxTokens || 192),
        }};
        const response = await fetch(config.chatUrl, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        const data = await response.json().catch(() => ({{}}));
        if (!response.ok) throw new Error(data?.error?.message || response.statusText);
        const reply = data?.choices?.[0]?.message?.content || "";
        pending.textContent = reply || "(empty response)";
        if (Array.isArray(data?.messages)) {{
          state.messages = data.messages;
        }} else {{
          state.messages.push({{ role: "assistant", content: reply }});
        }}
        if (data?.session_id) state.sessionId = data.session_id;
        if (data?.active_run?.run_key) {{
          state.runKey = data.active_run.run_key;
          updateRunDetail(data.active_run);
        }}
        persistLocalState();
        await loadSessions();
      }} catch (error) {{
        pending.className = "message error";
        pending.textContent = `Endpoint error: ${{error.message}}`;
      }} finally {{
        sendEl.disabled = false;
        promptEl.focus();
      }}
    }}

    document.getElementById("chat-form").addEventListener("submit", sendMessage);
    document.getElementById("refresh").addEventListener("click", refreshHealth);
    modeEl.addEventListener("change", () => {{
      state.mode = modeEl.value || "chat";
      persistLocalState();
    }});
    runListEl.addEventListener("change", () => {{
      selectRun(runListEl.value).catch((error) => setStatus(`Run load failed: ${{error.message}}`, false));
    }});
    document.getElementById("new-chat").addEventListener("click", async () => {{
      state.sessionId = generateSessionId();
      state.messages = [];
      persistLocalState();
      renderMessages();
      await loadSessions();
      promptEl.focus();
    }});
    document.getElementById("clear").addEventListener("click", () => {{
      state.messages = [];
      conversationEl.querySelectorAll(".message").forEach((node) => node.remove());
      emptyEl.hidden = false;
      persistLocalState();
      saveHistory().catch(() => {{}});
      promptEl.focus();
    }});
    sessionListEl.addEventListener("change", () => {{
      if (sessionListEl.value) loadHistory(sessionListEl.value).catch((error) => setStatus(`History load failed: ${{error.message}}`, false));
    }});
    promptEl.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && !event.shiftKey) {{
        event.preventDefault();
        document.getElementById("chat-form").requestSubmit();
      }}
    }});
    loadLocalState();
    renderMessages();
    loadRuns().catch((error) => setStatus(`Run load failed: ${{error.message}}`, false));
    loadHistory(state.sessionId).catch(() => loadSessions());
    refreshHealth();
  </script>
</body>
</html>
"""


def make_handler(endpoint: TinkerEndpoint):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TinkerStudioEndpoint/0.1"

        def do_GET(self) -> None:
            path = route_path(self.path)
            if path in {"/", "/chat"} or (path == "/v1" and self.accepts_html()):
                self.write_html(render_chat_page(endpoint))
                return
            if path == "/v1":
                self.write_json(
                    {
                        "status": "ok",
                        "chat_url": "/chat",
                        "api_base": "/v1",
                        "model": endpoint.model_id,
                        "active_run": endpoint.active_run_record(),
                        "default_mode": endpoint.default_mode,
                    }
                )
                return
            if path == "/favicon.ico":
                self.write_bytes(b"", status=204, content_type="image/x-icon")
                return
            if path in {"/health", "/v1/health"}:
                self.write_json(
                    {
                        "status": "ok",
                        "model": endpoint.model_id,
                        "base_model": endpoint.base_model,
                        "sampler_checkpoint": endpoint.sampler_checkpoint,
                        "active_run": endpoint.active_run_record(),
                        "default_mode": endpoint.default_mode,
                    }
                )
                return
            if path in {"/v1/runs", "/runs"}:
                self.write_json({"runs": endpoint.list_run_records(), "active_run": endpoint.active_run_record()})
                return
            if path == "/v1/chat/sessions":
                self.write_json({"sessions": endpoint.list_chat_sessions()})
                return
            if path == "/v1/chat/history":
                session_id = first_query_value(self.path, "session_id") or first_query_value(self.path, "sessionId")
                self.write_json(endpoint.load_chat_history(session_id or ""))
                return
            if path == "/v1/models":
                self.write_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": str(record.get("run_name") or record.get("run_key") or endpoint.model_id),
                                "object": "model",
                                "created": 0,
                                "owned_by": "local-tinker",
                                "base_model": record.get("base_model") or record.get("resolved_name") or "",
                                "run_key": record.get("run_key") or "",
                            }
                            for record in endpoint.list_run_records()
                        ],
                    }
                )
                return
            self.write_error(404, "not found")

        def do_POST(self) -> None:
            path = route_path(self.path)
            if path == "/v1/runs/select":
                try:
                    payload = self.read_json()
                    active_run = endpoint.select_run(
                        run_key=str(payload.get("run_key") or payload.get("runKey") or ""),
                        sampler_checkpoint=str(payload.get("sampler_checkpoint") or ""),
                        model=str(payload.get("model") or ""),
                    )
                    self.write_json({"status": "selected", "active_run": active_run})
                except (ValueError, json.JSONDecodeError) as exc:
                    self.write_error(400, str(exc))
                except Exception as exc:
                    self.write_error(500, str(exc))
                return
            if path == "/v1/chat/history":
                try:
                    payload = self.read_json()
                    session_id = str(payload.get("session_id") or payload.get("sessionId") or "")
                    messages = coerce_chat_messages(payload.get("messages"))
                    saved = endpoint.save_chat_history(session_id=session_id, messages=messages)
                    self.write_json({"status": "saved", "session_id": saved["session_id"]})
                except (ValueError, json.JSONDecodeError) as exc:
                    self.write_error(400, str(exc))
                except Exception as exc:
                    self.write_error(500, str(exc))
                return
            if path != "/v1/chat/completions":
                self.write_error(404, "not found")
                return
            try:
                payload = self.read_json()
                messages = sanitize_chat_history(coerce_chat_messages(payload.get("messages")))
                if not messages:
                    raise ValueError("messages must be a list")
                mode = normalize_chat_mode(payload.get("mode") or payload.get("chat_mode") or endpoint.default_mode)
                output = endpoint.chat(
                    messages=messages,
                    temperature=float(payload.get("temperature", DEFAULT_CHAT_TEMPERATURE)),
                    max_tokens=int(payload.get("max_tokens", DEFAULT_CHAT_MAX_TOKENS)),
                    mode=mode,
                    run_key=str(payload.get("run_key") or payload.get("runKey") or ""),
                    model=str(payload.get("model") or ""),
                )
                created = int(time.time())
                session_id = str(payload.get("session_id") or payload.get("sessionId") or "")
                response_messages = [*messages, {"role": "assistant", "content": output}]
                saved = endpoint.save_chat_history(session_id=session_id, messages=response_messages)
                self.write_json(
                    {
                        "id": f"chatcmpl-tinker-{created}",
                        "object": "chat.completion",
                        "created": created,
                        "model": payload.get("model") or endpoint.model_id,
                        "active_run": endpoint.active_run_record(),
                        "session_id": saved["session_id"],
                        "mode": mode,
                        "messages": saved["messages"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": output},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self.write_error(400, str(exc))
            except Exception as exc:
                self.write_error(500, str(exc))

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            data = self.rfile.read(length).decode("utf-8")
            payload = json.loads(data) if data else {}
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def accepts_html(self) -> bool:
            accept = self.headers.get("Accept", "")
            return "text/html" in accept

        def write_html(self, html_text: str, status: int = 200) -> None:
            self.write_bytes(html_text.encode("utf-8"), status=status, content_type="text/html; charset=utf-8")

        def write_bytes(self, body: bytes, *, status: int, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def write_error(self, status: int, message: str) -> None:
            self.write_json({"error": {"message": message}}, status=status)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[endpoint] {self.address_string()} - {format % args}")

    return Handler


def route_path(raw_path: str) -> str:
    path = urlsplit(raw_path).path.rstrip("/")
    return path or "/"


def first_query_value(raw_path: str, key: str) -> str | None:
    values = parse_qs(urlsplit(raw_path).query).get(key)
    if not values:
        return None
    value = str(values[0] or "").strip()
    return value or None


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    endpoint = TinkerEndpoint(
        workspace=workspace,
        run_name=args.run_name,
        sampler_checkpoint=args.sampler_checkpoint,
        model_id=args.model_id,
        default_mode=args.default_mode,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(endpoint))
    print(f"Browser chat: http://{args.host}:{args.port}/chat")
    print(f"OpenAI-compatible base URL: http://{args.host}:{args.port}/v1")
    print(f"Default mode: {endpoint.default_mode}")
    print(f"Sampler checkpoint: {endpoint.sampler_checkpoint}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
