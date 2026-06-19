from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tinker_training_utils import ConversationExample, slugify_name, text_sha256


INTERVIEW_RAW_PATH = Path("raw") / "interview_qa_raw.jsonl"
INTERVIEW_PROCESSED_PATH = Path("processed") / "interview_qa.jsonl"


@dataclass(frozen=True)
class InterviewPromptSet:
    name: str
    label: str
    description: str
    prompts: list[str]


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_interview_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.replace("deferal", "deferral")
    return normalized.strip()


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_interview_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def build_training_answer(row: dict[str, Any]) -> str:
    parts: list[str] = []
    answer_edited = normalize_interview_text(str(row.get("answer_edited") or ""))
    follow_up_question = normalize_interview_text(str(row.get("follow_up_question") or ""))
    follow_up_answer = normalize_interview_text(str(row.get("follow_up_answer_edited") or ""))
    if answer_edited:
        parts.append(answer_edited)
    if follow_up_question and follow_up_answer:
        parts.append(follow_up_answer)
    return "\n\n".join(part for part in parts if part).strip()


def build_interview_units(answer_text: str, *, max_units: int = 3) -> list[str]:
    normalized = normalize_interview_text(answer_text)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    candidate_units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph.split()) <= 80:
            candidate_units.append(paragraph)
            continue

        sentence_buffer: list[str] = []
        sentence_word_count = 0
        for sentence in split_sentences(paragraph):
            sentence_words = sentence.split()
            if sentence_buffer and sentence_word_count + len(sentence_words) > 70:
                candidate_units.append(" ".join(sentence_buffer).strip())
                sentence_buffer = [sentence]
                sentence_word_count = len(sentence_words)
            else:
                sentence_buffer.append(sentence)
                sentence_word_count += len(sentence_words)
        if sentence_buffer:
            candidate_units.append(" ".join(sentence_buffer).strip())

    deduped: list[str] = []
    seen_units: set[str] = set()
    for unit in candidate_units:
        unit = normalize_interview_text(unit)
        if not unit or unit in seen_units:
            continue
        seen_units.add(unit)
        deduped.append(unit)
        if len(deduped) >= max_units:
            break
    return deduped or [normalized]


def derive_opening_candidates(target_text: str, *, max_openings: int = 3) -> list[str]:
    normalized = normalize_interview_text(target_text)
    words = normalized.split()
    if len(words) < 4:
        return []

    candidates: list[str] = []

    if 4 <= len(words) <= 7:
        short_opening = " ".join(words[:-1]).strip()
        if short_opening:
            candidates.append(short_opening)

    fragment_word_count = min(max(6, round(len(words) * 0.35)), 22)
    fragment_word_count = min(fragment_word_count, max(1, len(words) - 3))
    if fragment_word_count < len(words):
        candidates.append(" ".join(words[:fragment_word_count]).strip())

    sentences = split_sentences(normalized)
    if len(sentences) >= 2:
        first_sentence = sentences[0].strip()
        if first_sentence and len(first_sentence.split()) < len(words) - 3:
            candidates.append(first_sentence)

    clause_match = re.match(r"^(.{20,140}?[,;:—-])\s+\S+", normalized)
    if clause_match:
        clause = clause_match.group(1).strip()
        if clause and len(clause.split()) < len(words) - 3:
            candidates.append(clause)

    deduped: list[str] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        candidate = normalize_interview_text(candidate).rstrip(",;:—-").strip()
        if not candidate:
            continue
        if candidate == normalized:
            continue
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        deduped.append(candidate)
        if len(deduped) >= max_openings:
            break
    return deduped


def build_interview_qa_user_prompt(*, question: str, row: dict[str, Any]) -> str:
    instructions = [
        "Answer the question in the target author's voice.",
        "Keep the answer grounded in the author's actual framing and argumentative texture.",
        "Return only the answer text.",
    ]
    theme = normalize_interview_text(str(row.get("theme") or ""))
    if theme:
        instructions.append(f"Theme: {theme}.")
    return textwrap.dedent(
        f"""
        {" ".join(instructions)}

        Question:
        {normalize_interview_text(question)}
        """
    ).strip()


def build_interview_post_user_prompt(
    *,
    opening_text: str,
    row: dict[str, Any],
) -> str:
    instructions = [
        "Write a finished Bluesky post or compact reflective note in the target author's voice.",
        "Keep the opening exactly as given.",
        "Match the tone, argument shape, compression, and internet-native style of the author's posts and reflective writing.",
        "Return only the final text.",
    ]
    theme = normalize_interview_text(str(row.get("theme") or ""))
    if theme:
        instructions.append(f"Theme: {theme}.")
    if row.get("is_reply_style"):
        instructions.append("Make it read naturally as a reply.")
    return textwrap.dedent(
        f"""
        {" ".join(instructions)}

        Opening:
        {normalize_interview_text(opening_text)}
        """
    ).strip()


def build_processed_interview_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    processed = dict(raw_row)
    processed["question"] = normalize_interview_text(str(processed.get("question") or ""))
    processed["answer_raw"] = normalize_interview_text(str(processed.get("answer_raw") or ""))
    processed["answer_edited"] = normalize_interview_text(
        str(processed.get("answer_edited") or processed["answer_raw"])
    )
    processed["follow_up_question"] = normalize_interview_text(
        str(processed.get("follow_up_question") or "")
    )
    processed["follow_up_answer_raw"] = normalize_interview_text(
        str(processed.get("follow_up_answer_raw") or "")
    )
    processed["follow_up_answer_edited"] = normalize_interview_text(
        str(processed.get("follow_up_answer_edited") or processed.get("follow_up_answer_raw") or "")
    )
    tags = processed.get("tags") or []
    if isinstance(tags, str):
        tags = [part.strip() for part in tags.split(",") if part.strip()]
    processed["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
    processed["theme"] = normalize_interview_text(str(processed.get("theme") or ""))
    processed["notes"] = normalize_interview_text(str(processed.get("notes") or ""))
    processed["source_session_id"] = str(processed.get("source_session_id") or "").strip()
    processed["is_reply_style"] = bool(processed.get("is_reply_style"))
    processed["training_answer"] = build_training_answer(processed)
    processed["derived_units"] = build_interview_units(processed["training_answer"])

    derived_targets: list[str] = []
    derived_openings: list[str] = []
    for unit in processed["derived_units"]:
        opening_candidates = derive_opening_candidates(unit, max_openings=1)
        if not opening_candidates:
            continue
        derived_targets.append(unit)
        derived_openings.append(opening_candidates[0])

    processed["derived_post_targets"] = derived_targets
    processed["derived_openings"] = derived_openings
    return processed


def validate_processed_interview_row(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not str(row.get("question") or "").strip():
        issues.append("question is required")
    if not str(row.get("training_answer") or "").strip():
        issues.append("training_answer is required")
    if not isinstance(row.get("tags"), list):
        issues.append("tags must be a list")
    derived_targets = row.get("derived_post_targets")
    derived_openings = row.get("derived_openings")
    if not isinstance(derived_targets, list) or not derived_targets:
        issues.append("derived_post_targets must contain at least one item")
    if not isinstance(derived_openings, list) or not derived_openings:
        issues.append("derived_openings must contain at least one item")
    if isinstance(derived_targets, list) and isinstance(derived_openings, list):
        if len(derived_targets) != len(derived_openings):
            issues.append("derived_post_targets and derived_openings must be the same length")
        for opening, target in zip(derived_openings, derived_targets):
            if not str(opening).strip():
                issues.append("derived opening cannot be blank")
                continue
            if not str(target).strip():
                issues.append("derived target cannot be blank")
                continue
            if len(str(opening).split()) >= len(str(target).split()):
                issues.append("derived opening must be shorter than its target")
            if not str(target).startswith(str(opening)):
                issues.append("derived opening must match the start of its target exactly")
    return issues


def load_processed_interview_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(Path(dataset_root) / INTERVIEW_PROCESSED_PATH)
    processed_rows: list[dict[str, Any]] = []
    for row in rows:
        processed_row = build_processed_interview_row(row)
        issues = validate_processed_interview_row(processed_row)
        if issues:
            raise ValueError(
                f"Invalid interview row {processed_row.get('interview_id') or processed_row.get('question')!r}: "
                + "; ".join(issues)
            )
        processed_rows.append(processed_row)
    return processed_rows


def build_interview_examples(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[ConversationExample], list[ConversationExample]]:
    qa_examples: list[ConversationExample] = []
    post_examples: list[ConversationExample] = []

    for index, row in enumerate(rows, start=1):
        question = normalize_interview_text(str(row.get("question") or ""))
        training_answer = normalize_interview_text(str(row.get("training_answer") or ""))
        if not question or not training_answer:
            continue

        interview_id = str(row.get("interview_id") or f"interview-{index:03d}")
        source_session_id = str(row.get("source_session_id") or "").strip()
        raw_source_id = source_session_id or interview_id
        qa_examples.append(
            ConversationExample(
                example_id=f"{interview_id}-qa",
                opening_text=question,
                target_text=training_answer,
                messages=[
                    {
                        "role": "user",
                        "content": build_interview_qa_user_prompt(question=question, row=row),
                    },
                    {"role": "assistant", "content": training_answer},
                ],
                metadata={
                    "source_kind": "interview_qa",
                    "theme": row.get("theme"),
                    "tags": list(row.get("tags") or []),
                    "source_session_id": source_session_id,
                    "training_format": "completion",
                    "transform": "interview_qa_answer",
                    "raw_source_kind": "interview",
                    "raw_source_id": raw_source_id,
                    "raw_text_sha256": text_sha256(training_answer),
                },
            )
        )

        for prompt_index, (opening, target) in enumerate(
            zip(row.get("derived_openings") or [], row.get("derived_post_targets") or []),
            start=1,
        ):
            post_examples.append(
                ConversationExample(
                    example_id=f"{interview_id}-post-{prompt_index:02d}",
                    opening_text=str(opening),
                    target_text=str(target),
                    messages=[
                        {
                            "role": "user",
                            "content": build_interview_post_user_prompt(
                                opening_text=str(opening),
                                row=row,
                            ),
                        },
                        {"role": "assistant", "content": str(target)},
                    ],
                    metadata={
                        "source_kind": "interview_post",
                        "theme": row.get("theme"),
                        "tags": list(row.get("tags") or []),
                        "source_session_id": source_session_id,
                        "training_format": "completion",
                        "transform": "interview_post_completion",
                        "raw_source_kind": "interview",
                        "raw_source_id": raw_source_id,
                        "raw_text_sha256": text_sha256(str(target)),
                    },
                )
            )

    return qa_examples, post_examples


def evenly_limit_examples(
    examples: Sequence[ConversationExample],
    *,
    limit: int | None,
) -> list[ConversationExample]:
    items = list(examples)
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (limit - 1)
    selected: list[ConversationExample] = []
    used_indices: set[int] = set()
    for item_index in range(limit):
        index = round(item_index * step)
        if index not in used_indices:
            used_indices.add(index)
            selected.append(items[index])
    return selected


def build_interview_prompt_sets(
    rows: Sequence[dict[str, Any]],
    *,
    max_prompts_per_theme: int = 5,
) -> dict[str, InterviewPromptSet]:
    grouped_prompts: dict[str, list[str]] = {}
    theme_labels: dict[str, str] = {}

    for row in rows:
        theme_label = normalize_interview_text(str(row.get("theme") or "Interview themes")) or "Interview themes"
        theme_key = slugify_name(theme_label) or "interview-themes"
        theme_name = f"interview_{theme_key}"
        theme_labels[theme_name] = theme_label
        grouped_prompts.setdefault(theme_name, [])
        for prompt in row.get("derived_openings") or []:
            prompt_text = normalize_interview_text(str(prompt))
            if prompt_text and prompt_text not in grouped_prompts[theme_name]:
                grouped_prompts[theme_name].append(prompt_text)

    prompt_sets: dict[str, InterviewPromptSet] = {}
    for theme_name, prompts in grouped_prompts.items():
        prompt_sets[theme_name] = InterviewPromptSet(
            name=theme_name,
            label=f"Interview: {theme_labels[theme_name]}",
            description="Openings derived from the interview corpus for this theme.",
            prompts=prompts[:max_prompts_per_theme],
        )
    return prompt_sets
