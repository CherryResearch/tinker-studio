from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tinker_interview_data import (
    build_interview_examples,
    evenly_limit_examples,
    load_processed_interview_rows,
)
from tinker_training_utils import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    ConversationExample,
    DatasetBundle,
    build_post_examples,
    slugify_name,
    text_sha256,
)


PROCESSED_POSTS_PATH = Path("processed") / "posts.jsonl"
RENTRY_PAGES_PATH = Path("processed") / "rentry_pages.jsonl"
IMPORTED_SOURCES_PATH = Path("processed") / "imported_sources.jsonl"
SYNTHETIC_SOURCES_PATH = Path("processed") / "synthetic_sources.jsonl"


@dataclass(frozen=True)
class DatasetVariant:
    name: str
    train_examples: list[ConversationExample]
    validation_examples: list[ConversationExample]
    test_examples: list[ConversationExample]
    notes: str
    allow_post_train_eval: bool = True
    source_counts: dict[str, int] = field(default_factory=dict)


def build_experiment_dataset_variants(
    bundle: DatasetBundle,
    *,
    longform_chunk_words: int = 220,
    longform_chunk_overlap_words: int = 60,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
) -> dict[str, DatasetVariant]:
    initial_train_examples = build_post_examples(bundle.train_rows)
    initial_validation_examples = build_post_examples(bundle.validation_rows)
    initial_test_examples = build_post_examples(bundle.test_rows)

    processed_post_rows = load_processed_post_rows(bundle.root)
    recent_post_rows = select_recent_post_rows(processed_post_rows, bundle.train_rows)
    recent_post_examples = build_post_examples(recent_post_rows)

    rentry_rows = load_rentry_rows(bundle.root)
    longform_examples = build_longform_examples(
        rentry_rows,
        chunk_words=longform_chunk_words,
        chunk_overlap_words=longform_chunk_overlap_words,
    )
    interview_rows = load_processed_interview_rows(bundle.root)
    interview_qa_examples, interview_post_examples = build_interview_examples(interview_rows)
    imported_rows = load_imported_source_rows(bundle.root)
    imported_examples = build_imported_source_examples(imported_rows)
    synthetic_rows = load_synthetic_source_rows(bundle.root)
    synthetic_examples = build_synthetic_source_examples(synthetic_rows)
    conversational_examples = build_conversational_voice_examples(
        initial_post_examples=initial_train_examples,
        recent_post_examples=recent_post_examples,
        longform_examples=longform_examples,
        interview_qa_examples=interview_qa_examples,
        interview_post_examples=interview_post_examples,
        imported_examples=imported_examples,
        synthetic_examples=synthetic_examples,
    )
    conversational_validation_examples = build_conversational_eval_examples(
        initial_validation_examples,
        split_name="validation",
    )
    conversational_test_examples = build_conversational_eval_examples(
        initial_test_examples,
        split_name="test",
    )

    mixed_train_examples = dedupe_examples(
        initial_train_examples + recent_post_examples + longform_examples
    )
    interview_balance_cap = max(8, len(mixed_train_examples) // 6)
    balanced_interview_examples = dedupe_examples(
        evenly_limit_examples(interview_qa_examples, limit=interview_balance_cap)
        + evenly_limit_examples(interview_post_examples, limit=interview_balance_cap)
    )
    mixed_with_interview_examples = dedupe_examples(
        mixed_train_examples + balanced_interview_examples
    )
    personal_source_examples = dedupe_examples(mixed_with_interview_examples + imported_examples)

    variants = {
        "initial_posts": DatasetVariant(
            name="initial_posts",
            train_examples=initial_train_examples,
            validation_examples=initial_validation_examples,
            test_examples=initial_test_examples,
            notes="Original chronological post-only train split with clean held-out validation/test splits.",
            allow_post_train_eval=True,
            source_counts={"initial_posts": len(initial_train_examples)},
        ),
        "recent_posts_plus_essays": DatasetVariant(
            name="recent_posts_plus_essays",
            train_examples=mixed_train_examples,
            validation_examples=initial_validation_examples,
            test_examples=initial_test_examples,
            notes=(
                "Original train split plus later posts and chunked Rentry essays. "
                "Because the later posts come from the original validation/test horizon, "
                "post-train eval is disabled by default to avoid misleading metrics."
            ),
            allow_post_train_eval=False,
            source_counts={
                "initial_posts": len(initial_train_examples),
                "recent_posts": len(recent_post_examples),
                "essay_chunks": len(longform_examples),
            },
        ),
        "recent_posts_essays_interview": DatasetVariant(
            name="recent_posts_essays_interview",
            train_examples=mixed_with_interview_examples,
            validation_examples=initial_validation_examples,
            test_examples=initial_test_examples,
            notes=(
                "Original train split plus later posts, chunked Rentry essays, and a balanced "
                "interview-derived corpus with both direct Q&A and post-style continuation examples. "
                "Post-train eval stays disabled because recent posts still overlap the original validation/test horizon."
            ),
            allow_post_train_eval=False,
            source_counts={
                "initial_posts": len(initial_train_examples),
                "recent_posts": len(recent_post_examples),
                "essay_chunks": len(longform_examples),
                "interview_qa": len(interview_qa_examples),
                "interview_post": len(interview_post_examples),
                "interview_balanced_mix": len(balanced_interview_examples),
            },
        ),
        "personal_sources_mix": DatasetVariant(
            name="personal_sources_mix",
            train_examples=personal_source_examples,
            validation_examples=initial_validation_examples,
            test_examples=initial_test_examples,
            notes=(
                "Recent posts, essays, interview rows, and local imported source rows. "
                "Imported notes and poetry are treated with source-specific prompts. "
                "Post-train eval stays disabled because this is a blended personal-source corpus."
            ),
            allow_post_train_eval=False,
            source_counts={
                "initial_posts": len(initial_train_examples),
                "recent_posts": len(recent_post_examples),
                "essay_chunks": len(longform_examples),
                "interview_balanced_mix": len(balanced_interview_examples),
                "imported_sources": len(imported_examples),
            },
        ),
        "conversational_voice_mix": DatasetVariant(
            name="conversational_voice_mix",
            train_examples=conversational_examples,
            validation_examples=conversational_validation_examples,
            test_examples=conversational_test_examples,
            notes=(
                "Conversational voice corpus built from the same personal sources, but formatted as chat instead "
                "of opening completion. Interview Q&A is kept direct, replies use reply context, short posts are "
                "capped, and longform/imported sources become reflective chat answers. Completion-style variants "
                "remain available separately."
            ),
            allow_post_train_eval=False,
            source_counts=count_source_kinds(conversational_examples),
        ),
    }
    return apply_dataset_variant_tag_filters(
        variants,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )


def build_dataset_variant_summary_df(
    dataset_variants: dict[str, DatasetVariant],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_name, variant in dataset_variants.items():
        rows.append(
            {
                "dataset_variant": variant_name,
                "train_examples": len(variant.train_examples),
                "validation_examples": len(variant.validation_examples),
                "test_examples": len(variant.test_examples),
                "allow_post_train_eval": variant.allow_post_train_eval,
                "source_counts": format_source_counts(variant.source_counts),
                "notes": variant.notes,
            }
        )
    return pd.DataFrame(rows)


def dataset_variant_split_examples(variant: DatasetVariant, split_name: str) -> list[ConversationExample]:
    split = str(split_name or "train").strip().lower()
    if split == "train":
        return variant.train_examples
    if split == "validation":
        return variant.validation_examples
    if split == "test":
        return variant.test_examples
    raise ValueError(f"Unknown split_name: {split_name!r}")


def build_training_example_preview_rows(
    variant: DatasetVariant,
    *,
    dataset_variant_name: str | None = None,
    split_name: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    examples = dataset_variant_split_examples(variant, split_name)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    rows: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples, start=1):
        metadata = dict(example.metadata or {})
        training_format = str(
            metadata.get("training_format")
            or metadata.get("conversation_format")
            or infer_training_format(example)
        )
        source_path = str(metadata.get("source_path") or metadata.get("url") or "")
        prompt_messages = example.messages[:-1] if example.messages and example.messages[-1].get("role") == "assistant" else example.messages
        rows.append(
            {
                "dataset_variant": dataset_variant_name or variant.name,
                "split": split_name,
                "example_index": example_index,
                "example_id": example.example_id,
                "training_format": training_format,
                "conversation_format": metadata.get("conversation_format") or training_format,
                "transform": metadata.get("transform") or metadata.get("source_kind") or "",
                "raw_transform": metadata.get("raw_transform") or "",
                "source_kind": metadata.get("source_kind") or "",
                "raw_source_kind": metadata.get("raw_source_kind") or "",
                "raw_source_id": metadata.get("raw_source_id") or metadata.get("source_id") or example.example_id,
                "source_path": source_path,
                "raw_text_sha256": metadata.get("raw_text_sha256") or "",
                "tags": example_tags(example),
                "message_roles": [message.get("role") for message in example.messages],
                "prompt_messages": prompt_messages,
                "messages": example.messages,
                "opening_text": example.opening_text,
                "target_text": example.target_text,
                "prompt_chars": sum(len(str(message.get("content") or "")) for message in prompt_messages),
                "target_chars": len(example.target_text),
                "metadata": metadata,
            }
        )
    return rows


def infer_training_format(example: ConversationExample) -> str:
    roles = [str(message.get("role") or "") for message in example.messages]
    if "system" in roles or len([role for role in roles if role == "user"]) > 1:
        return "chat"
    return "completion"


def write_training_example_preview_jsonl(
    path: str | Path,
    dataset_variants: dict[str, DatasetVariant],
    *,
    split_names: Sequence[str] = ("train",),
    limit_per_split: int | None = None,
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for variant_name, variant in dataset_variants.items():
            for split_name in split_names:
                for row in build_training_example_preview_rows(
                    variant,
                    dataset_variant_name=variant_name,
                    split_name=split_name,
                    limit=limit_per_split,
                ):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
    return count


def build_experiment_plan_df(
    experiment_specs: Sequence[dict[str, Any]],
    dataset_variants: dict[str, DatasetVariant],
    *,
    default_config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in experiment_specs:
        variant_name = str(spec.get("dataset_variant") or "initial_posts")
        if variant_name not in dataset_variants:
            raise KeyError(f"Unknown dataset_variant: {variant_name}")
        variant = dataset_variants[variant_name]
        run_post_train_eval = bool(
            spec.get(
                "run_post_train_eval",
                bool(default_config.get("run_post_train_eval", True) and variant.allow_post_train_eval),
            )
        )
        rows.append(
            {
                "run_name": spec["run_name"],
                "model_alias": spec.get("model_alias", default_config.get("model_alias")),
                "dataset_variant": variant_name,
                "train_examples": len(variant.train_examples),
                "validation_examples": len(variant.validation_examples),
                "test_examples": len(variant.test_examples),
                "lora_rank": int(spec.get("lora_rank", default_config.get("lora_rank", 16))),
                "learning_rate": float(
                    spec.get("learning_rate", default_config.get("learning_rate", 1e-4))
                ),
                "batch_size": int(spec.get("batch_size", default_config.get("batch_size", 8))),
                "max_length": int(spec.get("max_length", default_config.get("max_length", 512))),
                "num_epochs": int(spec.get("num_epochs", default_config.get("num_epochs", 1))),
                "max_steps_per_model": spec.get(
                    "max_steps_per_model",
                    default_config.get("max_steps_per_model"),
                ),
                "run_post_train_eval": run_post_train_eval,
                "run_post_train_sampling": bool(
                    spec.get(
                        "run_post_train_sampling",
                        default_config.get("run_post_train_sampling", True),
                    )
                ),
                "notes": spec.get("notes", variant.notes),
            }
        )
    return pd.DataFrame(rows)


def resolve_selected_experiment_specs(
    experiment_specs: Sequence[dict[str, Any]],
    run_names: Sequence[str] | None,
) -> list[dict[str, Any]]:
    specs_by_name = {str(spec["run_name"]): dict(spec) for spec in experiment_specs}
    if not run_names:
        return list(specs_by_name.values())

    selected_specs: list[dict[str, Any]] = []
    for run_name in run_names:
        if run_name not in specs_by_name:
            raise KeyError(f"Unknown run_name in RUN_EXPERIMENT_NAMES: {run_name}")
        selected_specs.append(specs_by_name[run_name])
    return selected_specs


def load_processed_post_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(Path(dataset_root) / PROCESSED_POSTS_PATH)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        if not normalized.get("id"):
            normalized["id"] = normalized.get("post_id")
        normalized_rows.append(normalized)
    return [row for row in normalized_rows if str(row.get("text") or "").strip()]


def load_rentry_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_root) / RENTRY_PAGES_PATH
    if not path.exists():
        return []
    return load_jsonl_rows(path)


def load_imported_source_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_root) / IMPORTED_SOURCES_PATH
    if not path.exists():
        return []
    return [row for row in load_jsonl_rows(path) if str(row.get("text") or "").strip()]


def load_synthetic_source_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_root) / SYNTHETIC_SOURCES_PATH
    if not path.exists():
        return []
    return [row for row in load_jsonl_rows(path) if str(row.get("text") or "").strip()]


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_recent_post_rows(
    processed_post_rows: Sequence[dict[str, Any]],
    initial_train_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_cutoff = max(
        (str(row.get("created_at") or "") for row in initial_train_rows if row.get("created_at")),
        default="",
    )
    train_ids = {str(row.get("id") or "") for row in initial_train_rows if row.get("id")}
    recent_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in processed_post_rows:
        row_id = str(row.get("id") or "")
        created_at = str(row.get("created_at") or "")
        is_recent = bool(created_at and created_at > train_cutoff)
        if not is_recent and row_id and row_id not in train_ids:
            is_recent = True
        if not is_recent:
            continue
        dedupe_key = row_id or f"created_at::{created_at}::{hash(str(row.get('text') or ''))}"
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        recent_rows.append(dict(row))
    return recent_rows


def build_conversational_voice_examples(
    *,
    initial_post_examples: Sequence[ConversationExample],
    recent_post_examples: Sequence[ConversationExample],
    longform_examples: Sequence[ConversationExample],
    interview_qa_examples: Sequence[ConversationExample],
    interview_post_examples: Sequence[ConversationExample],
    imported_examples: Sequence[ConversationExample],
    synthetic_examples: Sequence[ConversationExample],
) -> list[ConversationExample]:
    post_pool = dedupe_examples([*initial_post_examples, *recent_post_examples])
    reply_pool = [
        example
        for example in post_pool
        if example.metadata.get("is_reply")
        and str(example.metadata.get("reply_context_text") or "").strip()
    ]
    standalone_post_pool = [
        example
        for example in post_pool
        if not example.metadata.get("is_reply")
    ]

    examples: list[ConversationExample] = []
    examples.extend(build_conversational_interview_qa_examples(interview_qa_examples))
    examples.extend(
        build_conversational_reflection_examples(
            evenly_limit_examples(interview_post_examples, limit=72),
            source_kind="interview_reflection",
            prompt_family="interview",
        )
    )
    examples.extend(
        build_conversational_reply_examples(
            evenly_limit_examples(reply_pool, limit=120),
            source_kind="reply_chat",
        )
    )
    examples.extend(
        build_conversational_post_examples(
            evenly_limit_examples(standalone_post_pool, limit=80),
            source_kind="post_chat",
        )
    )
    examples.extend(
        build_conversational_reflection_examples(
            evenly_limit_examples(longform_examples, limit=32),
            source_kind="longform_reflection",
            prompt_family="longform",
        )
    )
    examples.extend(
        build_conversational_reflection_examples(
            evenly_limit_examples(imported_examples, limit=72),
            source_kind="imported_reflection",
            prompt_family="imported",
        )
    )
    examples.extend(synthetic_examples)
    return dedupe_examples(examples)


def build_conversational_eval_examples(
    examples: Sequence[ConversationExample],
    *,
    split_name: str,
    limit: int = 40,
) -> list[ConversationExample]:
    selected = evenly_limit_examples(examples, limit=limit)
    converted: list[ConversationExample] = []
    for index, example in enumerate(selected, start=1):
        if example.metadata.get("is_reply") and str(example.metadata.get("reply_context_text") or "").strip():
            converted.extend(
                build_conversational_reply_examples(
                    [example],
                    source_kind=f"{split_name}_reply_chat",
                    id_prefix=f"{split_name}-{index:03d}",
                )
            )
        else:
            converted.extend(
                build_conversational_post_examples(
                    [example],
                    source_kind=f"{split_name}_post_chat",
                    id_prefix=f"{split_name}-{index:03d}",
                )
            )
    return converted


def build_conversational_interview_qa_examples(
    examples: Sequence[ConversationExample],
) -> list[ConversationExample]:
    converted: list[ConversationExample] = []
    for index, example in enumerate(examples, start=1):
        question = normalize_prompt_text(example.opening_text)
        if not question or not example.target_text.strip():
            continue
        theme = normalize_prompt_text(str(example.metadata.get("theme") or ""))
        prompt = question
        if theme:
            prompt = f"{question}\n\nWe were talking about: {theme}."
        converted.append(
            make_conversational_example(
                example=example,
                example_id=f"chat-interview-qa-{example_id_suffix(example, index)}",
                user_prompt=prompt,
                source_kind="interview_chat",
            )
        )
    return converted


def build_conversational_reply_examples(
    examples: Sequence[ConversationExample],
    *,
    source_kind: str,
    id_prefix: str = "chat-reply",
) -> list[ConversationExample]:
    converted: list[ConversationExample] = []
    for index, example in enumerate(examples, start=1):
        reply_context = normalize_prompt_text(str(example.metadata.get("reply_context_text") or ""))
        if not reply_context or not example.target_text.strip():
            continue
        prompt = (
            "A friend says:\n"
            f"{limit_words(reply_context, max_words=90)}\n\n"
            "How would you answer them naturally, in your own voice?"
        )
        converted.append(
            make_conversational_example(
                example=example,
                example_id=f"{id_prefix}-{example_id_suffix(example, index)}",
                user_prompt=prompt,
                source_kind=source_kind,
            )
        )
    return converted


def build_conversational_post_examples(
    examples: Sequence[ConversationExample],
    *,
    source_kind: str,
    id_prefix: str = "chat-post",
) -> list[ConversationExample]:
    prompts = [
        "What is something you would actually say to a friend right now?",
        "Give me the small, honest version of what is on your mind.",
        "Say the thought naturally, like we are just talking.",
        "What is the line you would send without making it a performance?",
    ]
    converted: list[ConversationExample] = []
    for index, example in enumerate(examples, start=1):
        if not example.target_text.strip():
            continue
        prompt = prompts[(index - 1) % len(prompts)]
        hashtags = [
            str(tag).strip()
            for tag in example.metadata.get("hashtags", [])
            if str(tag).strip()
        ]
        if hashtags:
            prompt = f"{prompt}\n\nThe loose topic is: {', '.join(hashtags[:3])}."
        converted.append(
            make_conversational_example(
                example=example,
                example_id=f"{id_prefix}-{example_id_suffix(example, index)}",
                user_prompt=prompt,
                source_kind=source_kind,
            )
        )
    return converted


def build_conversational_reflection_examples(
    examples: Sequence[ConversationExample],
    *,
    source_kind: str,
    prompt_family: str,
) -> list[ConversationExample]:
    converted: list[ConversationExample] = []
    for index, example in enumerate(examples, start=1):
        if not example.target_text.strip():
            continue
        prompt = build_reflection_prompt(example, prompt_family=prompt_family)
        converted.append(
            make_conversational_example(
                example=example,
                example_id=f"chat-{source_kind}-{example_id_suffix(example, index)}",
                user_prompt=prompt,
                source_kind=source_kind,
            )
        )
    return converted


def build_reflection_prompt(example: ConversationExample, *, prompt_family: str) -> str:
    title = normalize_prompt_text(str(example.metadata.get("title") or ""))
    theme = normalize_prompt_text(str(example.metadata.get("theme") or ""))
    source_kind = normalize_prompt_text(str(example.metadata.get("source_kind") or ""))
    if prompt_family == "interview":
        if theme:
            return f"Can you give me the short version of how you think about this?\n\nTopic: {theme}."
        return "Can you give me the short version of how you think about this?"
    if prompt_family == "longform":
        if title:
            return f"I want to talk through {title}. What are you really getting at here?"
        return "I want to talk through this. What are you really getting at here?"
    if source_kind == "poetry":
        if title:
            return f"Could you write a small poem in your own voice around {title}?"
        return "Could you write a small poem in your own voice?"
    if title:
        return f"What would you jot down privately about {title}?"
    return "What would you jot down privately, just for yourself?"


def make_conversational_example(
    *,
    example: ConversationExample,
    example_id: str,
    user_prompt: str,
    source_kind: str,
) -> ConversationExample:
    metadata = dict(example.metadata)
    if "transform" in metadata and "raw_transform" not in metadata:
        metadata["raw_transform"] = metadata["transform"]
    metadata["source_kind"] = source_kind
    metadata["conversation_format"] = "chat"
    metadata["training_format"] = "chat"
    metadata["transform"] = source_kind
    return ConversationExample(
        example_id=example_id,
        opening_text=user_prompt,
        target_text=example.target_text,
        messages=[
            {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example.target_text},
        ],
        metadata=metadata,
    )


def normalize_prompt_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalized_row_tags(
    row: dict[str, Any],
    *,
    defaults: Sequence[str] | None = None,
) -> list[str]:
    tags: list[str] = []
    for key in ("tags", "labels", "hashtags"):
        tags.extend(parse_tag_filter_values([row.get(key)]))
    source_type = str(row.get("source_type") or row.get("source_kind") or "").strip()
    if source_type:
        tags.append(source_type)
    tags.extend(defaults or [])
    return unique_tags(tags)


def example_id_suffix(example: ConversationExample, index: int) -> str:
    return slugify_name(example.example_id) or f"{index:03d}"


def limit_words(value: str, *, max_words: int) -> str:
    words = value.split()
    if len(words) <= max_words:
        return value
    return " ".join(words[:max_words]).rstrip(",.;:") + "..."


def count_source_kinds(examples: Sequence[ConversationExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        source_kind = str(example.metadata.get("source_kind") or "unknown")
        counts[source_kind] = counts.get(source_kind, 0) + 1
    return counts


def collect_example_tag_counts(examples: Sequence[ConversationExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        for tag in example_tags(example):
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def apply_dataset_variant_tag_filters(
    variants: dict[str, DatasetVariant],
    *,
    include_tags: Sequence[str] | None,
    exclude_tags: Sequence[str] | None,
) -> dict[str, DatasetVariant]:
    include = parse_tag_filter_values(include_tags)
    exclude = parse_tag_filter_values(exclude_tags)
    if not include and not exclude:
        return variants

    filtered_variants: dict[str, DatasetVariant] = {}
    for variant_name, variant in variants.items():
        filtered_train_examples = filter_examples_by_tags(
            variant.train_examples,
            include_tags=include,
            exclude_tags=exclude,
        )
        filtered_variants[variant_name] = DatasetVariant(
            name=variant.name,
            train_examples=filtered_train_examples,
            validation_examples=variant.validation_examples,
            test_examples=variant.test_examples,
            notes=append_filter_note(variant.notes, include_tags=include, exclude_tags=exclude),
            allow_post_train_eval=variant.allow_post_train_eval,
            source_counts=count_source_kinds(filtered_train_examples),
        )
    return filtered_variants


def filter_examples_by_tags(
    examples: Sequence[ConversationExample],
    *,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
) -> list[ConversationExample]:
    include = {normalize_tag(tag) for tag in parse_tag_filter_values(include_tags)}
    exclude = {normalize_tag(tag) for tag in parse_tag_filter_values(exclude_tags)}
    filtered: list[ConversationExample] = []
    for example in examples:
        tags = {normalize_tag(tag) for tag in example_tags(example)}
        if include and not tags.intersection(include):
            continue
        if exclude and tags.intersection(exclude):
            continue
        filtered.append(example)
    return filtered


def example_tags(example: ConversationExample) -> list[str]:
    metadata = example.metadata or {}
    tags: list[str] = []
    for key in ("tags", "labels", "hashtags"):
        tags.extend(parse_tag_filter_values([metadata.get(key)]))
    source_kind = str(metadata.get("source_kind") or "").strip()
    if source_kind:
        tags.append(source_kind)
        tags.extend(part for part in re.split(r"[_\s-]+", source_kind) if part)
    for key in ("conversation_format", "training_format", "transform", "raw_source_kind"):
        value = str(metadata.get(key) or "").strip()
        if value:
            tags.append(value)
            tags.extend(part for part in re.split(r"[_\s-]+", value) if part)
    if metadata.get("synthetic"):
        tags.append("synthetic")
    if metadata.get("is_reply"):
        tags.extend(["bluesky", "reply"])
    elif "created_at" in metadata or "reply_context_text" in metadata:
        tags.extend(["bluesky", "post"])
    if source_kind == "rentry_page":
        tags.extend(["writing", "markdown", "longform"])
    return unique_tags(tags)


def parse_tag_filter_values(values: Sequence[Any] | None) -> list[str]:
    if values is None:
        return []
    tags: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            parsed_values = parse_tag_collection_string(value)
            raw_values = parsed_values if parsed_values is not None else re.split(r"[,;\n]", value)
        elif isinstance(value, dict):
            raw_values = [str(value.get("name") or value.get("label") or "")]
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            raw_values = []
            for item in value:
                if isinstance(item, dict):
                    raw_values.append(str(item.get("name") or item.get("label") or ""))
                else:
                    raw_values.append(str(item))
        else:
            raw_values = [str(value)]
        for raw in raw_values:
            tag = clean_tag_text(raw)
            if tag:
                tags.append(tag)
    return unique_tags(tags)


def parse_tag_collection_string(value: str) -> list[Any] | None:
    stripped = value.strip()
    if not stripped:
        return []
    if not ((stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("(") and stripped.endswith(")"))):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
        return list(parsed)
    return None


def clean_tag_text(value: Any) -> str:
    tag = str(value).strip()
    tag = tag.strip("'\"")
    tag = tag.strip()
    if tag in {"[", "]", "[]", "(", ")", "()"}:
        return ""
    tag = tag.strip("[]()")
    return tag.strip().strip("'\"").strip()


def unique_tags(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip()
        if not tag:
            continue
        normalized = normalize_tag(tag)
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(tag)
    return output


def normalize_tag(value: str) -> str:
    return re.sub(r"\s+", " ", clean_tag_text(value).lower())


def append_filter_note(
    notes: str,
    *,
    include_tags: Sequence[str],
    exclude_tags: Sequence[str],
) -> str:
    filters: list[str] = []
    if include_tags:
        filters.append("include tags: " + ", ".join(include_tags))
    if exclude_tags:
        filters.append("exclude tags: " + ", ".join(exclude_tags))
    if not filters:
        return notes
    return f"{notes} Tag filters applied to training examples only ({'; '.join(filters)})."


def build_longform_examples(
    rows: Sequence[dict[str, Any]],
    *,
    chunk_words: int = 220,
    chunk_overlap_words: int = 60,
    opening_ratio: float = 0.35,
    min_opening_words: int = 24,
    max_opening_words: int = 72,
    min_completion_words: int = 40,
) -> list[ConversationExample]:
    examples: list[ConversationExample] = []
    for row_index, row in enumerate(rows):
        source_text = str(row.get("rendered_text") or row.get("text") or "").strip()
        if not source_text:
            continue
        title = str(row.get("title") or f"essay-{row_index + 1}").strip()
        source_id = str(
            row.get("id")
            or row.get("source_id")
            or row.get("url")
            or row.get("canonical_url")
            or title
        )
        source_path = str(row.get("source_path") or row.get("path") or "")
        row_tags = normalized_row_tags(row, defaults=["rentry_page", "longform"])
        chunks = chunk_text(
            source_text,
            chunk_words=chunk_words,
            chunk_overlap_words=chunk_overlap_words,
            min_chunk_words=max(min_opening_words + min_completion_words, 96),
        )
        for chunk_index, chunk_text_value in enumerate(chunks, start=1):
            opening_text = pick_opening(
                chunk_text_value,
                opening_ratio=opening_ratio,
                min_opening_words=min_opening_words,
                max_opening_words=max_opening_words,
                min_completion_words=min_completion_words,
            )
            example_id = (
                f"{slugify_name(title) or 'essay'}-chunk-{chunk_index:02d}"
            )
            examples.append(
                ConversationExample(
                    example_id=example_id,
                    opening_text=opening_text,
                    target_text=chunk_text_value,
                    messages=[
                        {
                            "role": "user",
                            "content": build_longform_user_prompt(
                                opening_text=opening_text,
                                title=title,
                                row=row,
                            ),
                        },
                        {"role": "assistant", "content": chunk_text_value},
                    ],
                    metadata={
                        "title": title,
                        "url": row.get("url") or row.get("canonical_url"),
                        "word_count": len(chunk_text_value.split()),
                        "source_kind": "rentry_page",
                        "labels": row_tags,
                        "tags": row_tags,
                        "source_path": source_path,
                        "training_format": "completion",
                        "transform": "rentry_page_completion",
                        "raw_source_kind": "longform",
                        "raw_source_id": source_id,
                        "raw_text_sha256": text_sha256(chunk_text_value),
                    },
                )
            )
    return examples


def build_imported_source_examples(rows: Sequence[dict[str, Any]]) -> list[ConversationExample]:
    examples: list[ConversationExample] = []
    for row_index, row in enumerate(rows):
        source_type = str(row.get("source_type") or "notes")
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        title = str(row.get("title") or f"source-{row_index + 1}").strip()
        source_id = str(row.get("id") or row.get("source_id") or row.get("source_path") or title)
        source_path = str(row.get("source_path") or row.get("path") or "")
        row_tags = normalized_row_tags(row, defaults=[source_type])
        if source_type == "longform":
            longform_row = dict(row)
            longform_row["rendered_text"] = text
            longform_row.setdefault("title", title)
            examples.extend(build_longform_examples([longform_row]))
            continue
        chunks = chunk_preserving_lines(
            text,
            max_words=140 if source_type == "poetry" else 120,
            overlap_lines=1 if source_type == "poetry" else 0,
        )
        for chunk_index, chunk in enumerate(chunks or [text], start=1):
            opening_text = pick_opening_preserving_lines(
                chunk,
                min_lines=1 if source_type == "poetry" else 0,
                max_words=36,
            )
            example_id = f"{slugify_name(source_type)}-{slugify_name(title) or row_index + 1}-{chunk_index:02d}"
            examples.append(
                ConversationExample(
                    example_id=example_id,
                    opening_text=opening_text,
                    target_text=chunk,
                    messages=[
                        {
                            "role": "user",
                            "content": build_imported_source_prompt(
                                opening_text=opening_text,
                                row=row,
                                source_type=source_type,
                            ),
                        },
                        {"role": "assistant", "content": chunk},
                    ],
                    metadata={
                        "source_kind": source_type,
                        "title": title,
                        "color": row.get("color"),
                        "labels": row_tags,
                        "tags": row_tags,
                        "word_count": len(chunk.split()),
                        "source_path": source_path,
                        "training_format": "completion",
                        "transform": f"{source_type}_completion",
                        "raw_source_kind": "imported_source",
                        "raw_source_id": source_id,
                        "raw_text_sha256": text_sha256(chunk),
                    },
                )
            )
    return examples


def build_synthetic_source_examples(rows: Sequence[dict[str, Any]]) -> list[ConversationExample]:
    examples: list[ConversationExample] = []
    for row_index, row in enumerate(rows, start=1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        title = str(row.get("title") or f"synthetic-{row_index}").strip()
        row_tags = normalized_row_tags(row, defaults=["synthetic"])
        example_id = str(row.get("id") or f"synthetic-{slugify_name(title) or row_index}")
        source_path = str(row.get("source_path") or row.get("path") or "")
        prompt = build_synthetic_source_prompt(title=title, row=row, tags=row_tags)
        examples.append(
            ConversationExample(
                example_id=example_id,
                opening_text=prompt,
                target_text=text,
                messages=[
                    {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ],
                metadata={
                    "source_kind": "synthetic",
                    "title": title,
                    "labels": row_tags,
                    "tags": row_tags,
                    "source_path": source_path,
                    "word_count": len(text.split()),
                    "conversation_format": "chat",
                    "training_format": "chat",
                    "transform": "synthetic_chat",
                    "synthetic": True,
                    "raw_source_kind": "synthetic",
                    "raw_source_id": example_id,
                    "raw_text_sha256": text_sha256(text),
                },
            )
        )
    return examples


def build_synthetic_source_prompt(*, title: str, row: dict[str, Any], tags: Sequence[str]) -> str:
    tag_text = ", ".join(tags)
    role = str((row.get("metadata") or {}).get("synthetic_role") or "").strip()
    context_lines = [
        "Answer as a compact self-philosophy note in the trained voice.",
        "Keep the meaning specific and internally coherent; do not turn it into a post completion.",
        f"Title: {title}" if title else "",
        f"Synthetic role: {role}" if role else "",
        f"Tags: {tag_text}" if tag_text else "",
    ]
    return "\n".join(line for line in context_lines if line).strip()


def build_imported_source_prompt(*, opening_text: str, row: dict[str, Any], source_type: str) -> str:
    title = str(row.get("title") or "").strip()
    color = str(row.get("color") or "").strip()
    labels = row.get("labels") or []
    labels_text = ", ".join(str(item) for item in labels if str(item).strip())
    if source_type == "poetry":
        instruction = (
            "Write the next lines of a poem in the target author's poetic voice. "
            "Keep the opening exactly as given. Match lineation, image logic, and cadence."
        )
    elif source_type == "google_keep":
        instruction = (
            "Write or continue a note in the target author's private note-taking voice. "
            "Preserve the practical, fragmentary, or reflective texture implied by the note metadata."
        )
    else:
        instruction = (
            "Continue this imported source in the target author's voice. "
            "Match its format, density, and purpose."
        )
    context_lines = [
        instruction,
        f"Title: {title}" if title else "",
        f"Color: {color}" if color else "",
        f"Labels: {labels_text}" if labels_text else "",
        "",
        "Opening:",
        opening_text,
    ]
    return "\n".join(line for line in context_lines if line).strip()


def chunk_preserving_lines(text: str, *, max_words: int, overlap_lines: int = 0) -> list[str]:
    lines = text.strip().splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for line in lines:
        line_words = len(line.split())
        if current and current_words + line_words > max_words:
            chunks.append("\n".join(current).strip())
            current = current[-overlap_lines:] if overlap_lines else []
            current_words = sum(len(item.split()) for item in current)
        current.append(line)
        current_words += line_words
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def pick_opening_preserving_lines(text: str, *, min_lines: int, max_words: int) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if min_lines and len(lines) > 1:
        return "\n".join(lines[:min(max(1, min_lines), len(lines))]).strip()
    words = re.findall(r"\S+", text.strip())
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def build_longform_user_prompt(
    *,
    opening_text: str,
    title: str,
    row: dict[str, Any],
) -> str:
    instructions = [
        "Write the next section of a longform piece in the target author's voice.",
        "Keep the opening exactly as given.",
        "Match the density, argument flow, and cadence of the author's essay writing.",
        "Return only the final continuation text.",
    ]
    url = str(row.get("url") or row.get("canonical_url") or "").strip()
    title_line = f"Title: {title}" if title else ""
    url_line = f"Source URL: {url}" if url else ""
    return "\n".join(
        line
        for line in [
            " ".join(instructions),
            title_line,
            url_line,
            "",
            "Opening:",
            opening_text,
        ]
        if line
    ).strip()


def chunk_text(
    text: str,
    *,
    chunk_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
) -> list[str]:
    words = re.findall(r"\S+", text.strip())
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(1, chunk_words - chunk_overlap_words)
    chunks: list[str] = []
    for start_index in range(0, len(words), step):
        chunk_words_list = words[start_index : start_index + chunk_words]
        if len(chunk_words_list) < min_chunk_words and chunks:
            break
        chunks.append(" ".join(chunk_words_list))
        if start_index + chunk_words >= len(words):
            break
    return chunks


def pick_opening(
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


def dedupe_examples(examples: Sequence[ConversationExample]) -> list[ConversationExample]:
    deduped: list[ConversationExample] = []
    seen_keys: set[tuple[str, str]] = set()
    for example in examples:
        dedupe_key = (example.example_id, example.target_text)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(example)
    return deduped


def format_source_counts(source_counts: dict[str, int]) -> str:
    if not source_counts:
        return ""
    return ", ".join(f"{key}={value}" for key, value in source_counts.items())
