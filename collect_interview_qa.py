from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinker_interview_data import (
    INTERVIEW_PROCESSED_PATH,
    INTERVIEW_RAW_PATH,
    append_jsonl_row,
    build_processed_interview_row,
    load_jsonl_rows,
    normalize_interview_text,
    validate_processed_interview_row,
)
from tinker_training_utils import find_dataset_root, slugify_name


PROMPT_ROUNDS_FILENAME = "interview_prompt_rounds.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect interview-style Q&A items for the Tinker dataset."
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Workspace root containing the dataset and prompt library.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-rounds", help="List the available interview prompt rounds.")

    append_parser = subparsers.add_parser("append", help="Append a new interview item.")
    configure_common_entry_args(append_parser)

    followup_parser = subparsers.add_parser(
        "followup",
        help="Append a new interview item that includes one follow-up question and answer.",
    )
    configure_common_entry_args(followup_parser, include_followup=True)

    review_parser = subparsers.add_parser("review", help="Preview recent processed interview rows.")
    review_parser.add_argument("--last", type=int, default=5, help="How many recent rows to show.")
    review_parser.add_argument(
        "--theme",
        help="Optional theme filter for review output.",
    )

    export_parser = subparsers.add_parser(
        "export-prompts",
        help="Print derived openings grouped by theme from the processed interview corpus.",
    )
    export_parser.add_argument(
        "--theme",
        help="Optional theme filter for derived opening export.",
    )
    return parser.parse_args()


def configure_common_entry_args(
    parser: argparse.ArgumentParser,
    *,
    include_followup: bool = False,
) -> None:
    parser.add_argument("--interview-id", help="Stable ID for this interview row.")
    parser.add_argument("--session-id", help="Conversation/session ID for grouping related entries.")
    parser.add_argument("--theme", help="Theme label for the item.")
    parser.add_argument("--tags", nargs="*", default=[], help="Optional tags to attach.")
    parser.add_argument("--notes", help="Optional notes for review/export.")
    parser.add_argument(
        "--prompt-round",
        help="Prompt round key from interview_prompt_rounds.json to help prefill theme and question.",
    )
    parser.add_argument(
        "--question-index",
        type=int,
        default=1,
        help="1-based question index inside the selected prompt round.",
    )
    parser.add_argument("--question", help="Primary interview question.")
    parser.add_argument("--answer", help="Raw primary answer text.")
    parser.add_argument("--answer-edited", help="Optional lightly edited primary answer.")
    parser.add_argument(
        "--is-reply-style",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mark the derived post examples as reply-style.",
    )
    if include_followup:
        parser.add_argument("--follow-up-question", help="Optional follow-up question.")
        parser.add_argument("--follow-up-answer", help="Optional follow-up answer text.")
        parser.add_argument(
            "--follow-up-answer-edited",
            help="Optional lightly edited follow-up answer.",
        )


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_prompt_rounds(workspace_root: Path) -> dict[str, dict[str, Any]]:
    path = workspace_root / PROMPT_ROUNDS_FILENAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_prompt_round(
    *,
    prompt_rounds: dict[str, dict[str, Any]],
    round_name: str | None,
    question_index: int,
) -> tuple[str | None, str | None]:
    if not round_name:
        return None, None
    if round_name not in prompt_rounds:
        available = ", ".join(sorted(prompt_rounds)) or "none"
        raise KeyError(f"Unknown prompt round: {round_name}. Available rounds: {available}")
    round_payload = prompt_rounds[round_name]
    questions = round_payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Prompt round {round_name} does not contain any questions.")
    resolved_index = max(1, question_index) - 1
    if resolved_index >= len(questions):
        raise ValueError(
            f"Prompt round {round_name} only has {len(questions)} question(s); "
            f"cannot use index {question_index}."
        )
    theme = str(round_payload.get("theme") or round_payload.get("label") or "").strip() or None
    question = str(questions[resolved_index]).strip()
    return theme, question


def prompt_for_value(prompt_text: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt_text}{suffix}: ").strip()
    return value or (default or "")


def build_entry(
    *,
    args: argparse.Namespace,
    prompt_rounds: dict[str, dict[str, Any]],
    include_followup: bool,
) -> dict[str, Any]:
    round_theme, round_question = resolve_prompt_round(
        prompt_rounds=prompt_rounds,
        round_name=getattr(args, "prompt_round", None),
        question_index=int(getattr(args, "question_index", 1)),
    )
    question = normalize_interview_text(str(args.question or round_question or ""))
    theme = normalize_interview_text(str(args.theme or round_theme or ""))
    answer_raw = normalize_interview_text(str(args.answer or ""))
    answer_edited = normalize_interview_text(str(args.answer_edited or answer_raw))
    notes = normalize_interview_text(str(args.notes or ""))

    if not question:
        question = normalize_interview_text(prompt_for_value("Question"))
    if not theme:
        theme = normalize_interview_text(prompt_for_value("Theme"))
    if not answer_raw:
        answer_raw = normalize_interview_text(prompt_for_value("Answer"))
    if not answer_edited:
        answer_edited = normalize_interview_text(prompt_for_value("Edited answer", default=answer_raw))

    follow_up_question = ""
    follow_up_answer_raw = ""
    follow_up_answer_edited = ""
    if include_followup:
        follow_up_question = normalize_interview_text(str(getattr(args, "follow_up_question", "") or ""))
        follow_up_answer_raw = normalize_interview_text(str(getattr(args, "follow_up_answer", "") or ""))
        follow_up_answer_edited = normalize_interview_text(
            str(getattr(args, "follow_up_answer_edited", "") or follow_up_answer_raw or "")
        )
        if not follow_up_question:
            follow_up_question = normalize_interview_text(prompt_for_value("Follow-up question"))
        if not follow_up_answer_raw:
            follow_up_answer_raw = normalize_interview_text(prompt_for_value("Follow-up answer"))
        if not follow_up_answer_edited:
            follow_up_answer_edited = normalize_interview_text(
                prompt_for_value("Edited follow-up answer", default=follow_up_answer_raw)
            )

    session_id = str(args.session_id or slugify_name(theme) or "interview-session").strip()
    interview_id = str(
        args.interview_id
        or f"{slugify_name(theme) or 'interview'}-{utc_now_compact().lower()}"
    ).strip()
    tags = [str(tag).strip() for tag in (args.tags or []) if str(tag).strip()]

    return {
        "interview_id": interview_id,
        "question": question,
        "answer_raw": answer_raw,
        "answer_edited": answer_edited,
        "follow_up_question": follow_up_question,
        "follow_up_answer_raw": follow_up_answer_raw,
        "follow_up_answer_edited": follow_up_answer_edited,
        "theme": theme,
        "tags": tags,
        "notes": notes,
        "is_reply_style": bool(args.is_reply_style),
        "source_session_id": session_id,
        "created_at_utc": utc_now_iso(),
    }


def print_processed_preview(row: dict[str, Any]) -> None:
    print(f"Interview ID: {row.get('interview_id')}")
    print(f"Theme: {row.get('theme') or '-'}")
    print(f"Question: {row.get('question') or '-'}")
    print("Raw answer:")
    print(row.get("answer_raw") or "")
    print()
    print("Edited/training answer:")
    print(row.get("training_answer") or "")
    if row.get("follow_up_question"):
        print()
        print(f"Follow-up: {row.get('follow_up_question')}")
        print(row.get("follow_up_answer_edited") or row.get("follow_up_answer_raw") or "")
    print()
    print("Derived openings:")
    for opening in row.get("derived_openings") or []:
        print(f"- {opening}")


def command_list_rounds(workspace_root: Path) -> int:
    prompt_rounds = load_prompt_rounds(workspace_root)
    if not prompt_rounds:
        print("No interview prompt rounds found.")
        return 0
    print("Interview prompt rounds")
    print("----------------------")
    for key, payload in prompt_rounds.items():
        print(f"{key}: {payload.get('label') or key}")
        if payload.get("description"):
            print(f"  {payload['description']}")
        for index, question in enumerate(payload.get("questions") or [], start=1):
            print(f"  {index}. {question}")
        print()
    return 0


def command_append_or_followup(args: argparse.Namespace, *, include_followup: bool) -> int:
    workspace_root = Path(args.workspace).resolve()
    dataset_root = find_dataset_root(workspace_root)
    prompt_rounds = load_prompt_rounds(workspace_root)
    raw_row = build_entry(args=args, prompt_rounds=prompt_rounds, include_followup=include_followup)
    processed_row = build_processed_interview_row(raw_row)
    issues = validate_processed_interview_row(processed_row)
    if issues:
        raise ValueError("; ".join(issues))

    raw_path = dataset_root / INTERVIEW_RAW_PATH
    processed_path = dataset_root / INTERVIEW_PROCESSED_PATH
    append_jsonl_row(raw_path, raw_row)
    append_jsonl_row(processed_path, processed_row)

    print(f"[SAVED] raw={raw_path}")
    print(f"[SAVED] processed={processed_path}")
    print()
    print_processed_preview(processed_row)
    return 0


def command_review(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).resolve()
    dataset_root = find_dataset_root(workspace_root)
    rows = load_jsonl_rows(dataset_root / INTERVIEW_PROCESSED_PATH)
    if args.theme:
        theme_filter = normalize_interview_text(str(args.theme))
        rows = [row for row in rows if normalize_interview_text(str(row.get("theme") or "")) == theme_filter]
    if not rows:
        print("No processed interview rows found.")
        return 0
    selected_rows = rows[-max(1, int(args.last)) :]
    for row in selected_rows:
        processed_row = build_processed_interview_row(row)
        print_processed_preview(processed_row)
        print()
        print("-" * 60)
        print()
    return 0


def command_export_prompts(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).resolve()
    dataset_root = find_dataset_root(workspace_root)
    rows = load_jsonl_rows(dataset_root / INTERVIEW_PROCESSED_PATH)
    if not rows:
        print("No processed interview rows found.")
        return 0

    theme_filter = normalize_interview_text(str(args.theme or ""))
    grouped: dict[str, list[str]] = {}
    for row in rows:
        processed_row = build_processed_interview_row(row)
        theme = normalize_interview_text(str(processed_row.get("theme") or "Interview themes")) or "Interview themes"
        if theme_filter and theme != theme_filter:
            continue
        grouped.setdefault(theme, [])
        for opening in processed_row.get("derived_openings") or []:
            if opening not in grouped[theme]:
                grouped[theme].append(opening)

    if not grouped:
        print("No derived openings matched that filter.")
        return 0

    for theme, prompts in grouped.items():
        print(theme)
        print("-" * len(theme))
        for prompt in prompts:
            print(f"- {prompt}")
        print()
    return 0


def main() -> int:
    args = parse_args()
    command = args.command
    if command == "list-rounds":
        return command_list_rounds(Path(args.workspace).resolve())
    if command == "append":
        return command_append_or_followup(args, include_followup=False)
    if command == "followup":
        return command_append_or_followup(args, include_followup=True)
    if command == "review":
        return command_review(args)
    if command == "export-prompts":
        return command_export_prompts(args)
    raise RuntimeError(f"Unsupported command: {command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
