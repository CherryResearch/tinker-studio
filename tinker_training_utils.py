from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat, get_renderer, get_text_content
from tinker_cookbook.supervised.data import conversation_to_datum


MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_ROOT_ENV_VARS = ("TINKER_STUDIO_DATASET_ROOT", "TINKER_DATASET_ROOT")
DEFAULT_DATASET_ROOT = Path("data") / "training_data"


@dataclass(frozen=True)
class DatasetBundle:
    root: Path
    manifest: dict[str, Any]
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ConversationExample:
    example_id: str
    opening_text: str
    target_text: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelResolution:
    requested_name: str
    resolved_name: str | None
    match_type: str | None
    note: str


def find_dataset_root(search_start: str | Path) -> Path:
    search_root = Path(search_start).resolve()
    for env_var in DATASET_ROOT_ENV_VARS:
        configured = os.environ.get(env_var)
        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = search_root / configured_path
            configured_root = configured_path.resolve()
            manifest_path = configured_root / "tinker" / MANIFEST_FILENAME
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"{env_var} points to {configured_root}, but {manifest_path} does not exist."
                )
            return configured_root

    default_root = search_root / DEFAULT_DATASET_ROOT
    if (default_root / "tinker" / MANIFEST_FILENAME).exists():
        return default_root

    matches = sorted(
        path
        for path in search_root.rglob(MANIFEST_FILENAME)
        if path.parent.name == "tinker"
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find a Tinker dataset manifest below {search_root}."
        )
    return matches[0].parent.parent


def load_dataset_bundle(dataset_root: str | Path) -> DatasetBundle:
    root = Path(dataset_root).resolve()
    manifest_path = root / "tinker" / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetBundle(
        root=root,
        manifest=manifest,
        train_rows=_load_jsonl(root / "tinker" / "train.jsonl"),
        validation_rows=_load_jsonl(root / "tinker" / "validation.jsonl"),
        test_rows=_load_jsonl(root / "tinker" / "test.jsonl"),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    char_counts = [len(str(row.get("text") or "")) for row in rows]
    word_counts = [len(str(row.get("text") or "").split()) for row in rows]
    reply_count = sum(bool(row.get("is_reply")) for row in rows)
    hashtag_count = sum(len(row.get("hashtags") or []) for row in rows)
    return {
        "rows": len(rows),
        "mean_chars": round(statistics.mean(char_counts), 2) if char_counts else 0.0,
        "median_chars": round(statistics.median(char_counts), 2) if char_counts else 0.0,
        "mean_words": round(statistics.mean(word_counts), 2) if word_counts else 0.0,
        "median_words": round(statistics.median(word_counts), 2) if word_counts else 0.0,
        "reply_rate": round(reply_count / len(rows), 4) if rows else 0.0,
        "avg_hashtags": round(hashtag_count / len(rows), 4) if rows else 0.0,
    }


def build_post_examples(
    rows: Sequence[dict[str, Any]],
    *,
    opening_ratio: float = 0.4,
    min_opening_words: int = 3,
    max_opening_words: int = 18,
    min_completion_words: int = 3,
) -> list[ConversationExample]:
    examples: list[ConversationExample] = []
    for index, row in enumerate(rows):
        target_text = str(row.get("text") or "").strip()
        if not target_text:
            continue
        opening_text = _pick_opening(
            target_text,
            opening_ratio=opening_ratio,
            min_opening_words=min_opening_words,
            max_opening_words=max_opening_words,
            min_completion_words=min_completion_words,
        )
        prompt = _build_user_prompt(opening_text=opening_text, row=row)
        example_id = str(row.get("id") or row.get("post_id") or f"row-{index}")
        examples.append(
            ConversationExample(
                example_id=example_id,
                opening_text=opening_text,
                target_text=target_text,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_text},
                ],
                metadata={
                    "created_at": row.get("created_at"),
                    "is_reply": bool(row.get("is_reply")),
                    "hashtags": list(row.get("hashtags") or []),
                    "reply_context_text": row.get("reply_context_text") or row.get("parent_text") or "",
                },
            )
        )
    return examples


def _pick_opening(
    text: str,
    *,
    opening_ratio: float,
    min_opening_words: int,
    max_opening_words: int,
    min_completion_words: int,
) -> str:
    words = re.findall(r"\S+", text.strip())
    if len(words) <= 1:
        return text.strip()
    proposed = math.ceil(len(words) * opening_ratio)
    opening_words = max(min_opening_words, min(max_opening_words, proposed))
    opening_words = min(opening_words, max(1, len(words) - min_completion_words))
    if opening_words >= len(words):
        opening_words = max(1, len(words) // 2)
    return " ".join(words[:opening_words]).strip()


def _build_user_prompt(opening_text: str, row: dict[str, Any]) -> str:
    instructions = [
        "Write a finished Bluesky post in the target author's voice.",
        "Keep the opening exactly as given.",
        "Match the tone, brevity, formatting, and internet-native style of the author's posts.",
        "Return only the final post text.",
    ]
    if row.get("is_reply"):
        instructions.append("Make it read naturally as a reply.")
    reply_context = str(
        row.get("reply_context_text")
        or row.get("parent_text")
        or row.get("root_text")
        or ""
    ).strip()
    hashtags = list(row.get("hashtags") or [])
    if hashtags:
        instructions.append("If hashtags fit naturally, keep them concise.")
    context_block = f"\n\nReply context:\n{reply_context}" if reply_context else ""
    return textwrap.dedent(
        f"""
        {" ".join(instructions)}
        {context_block}

        Opening:
        {opening_text}
        """
    ).strip()


def resolve_model_names(
    requested_models: Sequence[str],
    supported_models: Sequence[str],
) -> list[ModelResolution]:
    supported = list(supported_models)
    supported_by_norm = {_normalize_name(model_name): model_name for model_name in supported}
    supported_suffix_map = {
        _normalize_name(model_name.split("/", 1)[-1]): model_name for model_name in supported
    }
    resolutions: list[ModelResolution] = []

    for requested in requested_models:
        requested_norm = _normalize_name(requested)
        if requested_norm in supported_by_norm:
            resolved = supported_by_norm[requested_norm]
            resolutions.append(
                ModelResolution(requested, resolved, "exact", "Matched an exact server model name.")
            )
            continue
        if requested_norm in supported_suffix_map:
            resolved = supported_suffix_map[requested_norm]
            resolutions.append(
                ModelResolution(
                    requested,
                    resolved,
                    "providerless",
                    "Matched the provider-qualified server name.",
                )
            )
            continue

        substring_matches = [
            model_name
            for model_name in supported
            if requested_norm in _normalize_name(model_name)
            or requested_norm in _normalize_name(model_name.split("/", 1)[-1])
        ]
        if len(substring_matches) == 1:
            resolutions.append(
                ModelResolution(
                    requested,
                    substring_matches[0],
                    "substring",
                    "Matched a single supported model by substring.",
                )
            )
            continue
        if len(substring_matches) > 1:
            resolutions.append(
                ModelResolution(
                    requested,
                    None,
                    None,
                    f"Ambiguous request. Matching models: {', '.join(substring_matches)}",
                )
            )
            continue

        resolutions.append(
            ModelResolution(
                requested,
                None,
                None,
                "No supported model matched this request.",
            )
        )

    return resolutions


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    normalized = re.sub(r"[^a-z0-9./:-]+", "-", normalized)
    return re.sub(r"-{2,}", "-", normalized).strip("-")


def select_renderer_name(model_name: str, override: str | None = None) -> str:
    if override:
        return override
    try:
        return model_info.get_recommended_renderer_name(model_name)
    except Exception as exc:  # pragma: no cover - notebook fallback path
        raise ValueError(
            f"Could not infer a renderer for {model_name}. "
            "Set a renderer_name override for this model in the notebook config."
        ) from exc


def build_datums(
    examples: Sequence[ConversationExample],
    tokenizer: Any,
    model_name: str,
    *,
    renderer_name: str,
    max_length: int,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
):
    renderer = get_renderer(renderer_name, tokenizer)
    datums = [
        conversation_to_datum(
            example.messages,
            renderer,
            max_length=max_length,
            train_on_what=train_on_what,
        )
        for example in examples
    ]
    return renderer, datums


def build_batches(
    items: Sequence[Any],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> list[list[Any]]:
    batch_size = max(1, int(batch_size))
    items_list = list(items)
    if shuffle:
        random.Random(seed).shuffle(items_list)
    return [
        items_list[start : start + batch_size]
        for start in range(0, len(items_list), batch_size)
        if items_list[start : start + batch_size]
    ]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist())
    return np.asarray(value)


def compute_batch_loss(loss_fn_outputs: Sequence[dict[str, Any]], batch: Sequence[Any]) -> tuple[float, float]:
    logprobs = np.concatenate([_to_numpy(output["logprobs"]) for output in loss_fn_outputs])
    weights = np.concatenate([_to_numpy(datum.loss_fn_inputs["weights"]) for datum in batch])
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return math.nan, 0.0
    loss = float(-np.dot(logprobs, weights) / weight_sum)
    return loss, weight_sum


def evaluate_cross_entropy(training_client: Any, batches: Sequence[Sequence[Any]]) -> dict[str, float | int]:
    total_weight = 0.0
    weighted_loss_sum = 0.0
    batch_losses: list[float] = []
    sequence_count = 0

    for batch in batches:
        if not batch:
            continue
        result = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
        batch_loss, batch_weight = compute_batch_loss(result.loss_fn_outputs, batch)
        if math.isnan(batch_loss):
            continue
        batch_losses.append(batch_loss)
        weighted_loss_sum += batch_loss * batch_weight
        total_weight += batch_weight
        sequence_count += len(batch)

    return {
        "mean_nll": (weighted_loss_sum / total_weight) if total_weight else math.nan,
        "mean_batch_loss": statistics.mean(batch_losses) if batch_losses else math.nan,
        "num_batches": len(batch_losses),
        "num_sequences": sequence_count,
        "weight_sum": total_weight,
    }


def build_eval_prompts(
    examples: Sequence[ConversationExample],
    *,
    limit: int = 5,
) -> list[ConversationExample]:
    if limit <= 0 or not examples:
        return []
    if len(examples) <= limit:
        return list(examples)
    if limit == 1:
        return [examples[len(examples) // 2]]
    step = (len(examples) - 1) / (limit - 1)
    selected: list[ConversationExample] = []
    used_indices: set[int] = set()
    for item_index in range(limit):
        index = round(item_index * step)
        if index not in used_indices:
            used_indices.add(index)
            selected.append(examples[index])
    return selected


def sample_generations(
    sampling_client: Any,
    renderer: Any,
    prompts: Sequence[ConversationExample],
    *,
    max_tokens: int = 96,
    temperature: float = 0.7,
) -> list[dict[str, str]]:
    import tinker

    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )

    rows: list[dict[str, str]] = []
    for example in prompts:
        prompt = renderer.build_generation_prompt(example.messages[:-1])
        result_future = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )
        result = result_future.result()
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        rows.append(
            {
                "example_id": example.example_id,
                "opening_text": example.opening_text,
                "target_text": example.target_text,
                "generated_text": get_text_content(response),
            }
        )
    return rows


def slugify_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return re.sub(r"-{2,}", "-", cleaned).strip("-")


def maybe_take_examples(
    examples: Sequence[ConversationExample],
    *,
    limit: int | None = None,
    seed: int = 0,
) -> list[ConversationExample]:
    items = list(examples)
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, k=limit)
