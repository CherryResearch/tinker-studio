from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from tinker_interview_data import (
    build_interview_examples,
    evenly_limit_examples,
    load_processed_interview_rows,
)
from tinker_training_utils import ConversationExample, DatasetBundle, build_post_examples, slugify_name


PROCESSED_POSTS_PATH = Path("processed") / "posts.jsonl"
RENTRY_PAGES_PATH = Path("processed") / "rentry_pages.jsonl"
IMPORTED_SOURCES_PATH = Path("processed") / "imported_sources.jsonl"


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
    }
    return variants


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
                        "labels": row.get("labels") or [],
                        "word_count": len(chunk.split()),
                    },
                )
            )
    return examples


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
