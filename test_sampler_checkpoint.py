from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook.renderers import TrainOnWhat

from run_tinker_experiment import find_latest_payload, get_experiment_specs
from tinker_interview_data import build_interview_prompt_sets, load_processed_interview_rows
from tinker_experiment_manager import build_experiment_dataset_variants
from tinker_notebook_env import describe_tinker_api_key, ensure_tinker_api_key
from tinker_training_utils import (
    ConversationExample,
    build_datums,
    build_eval_prompts,
    find_dataset_root,
    load_dataset_bundle,
    sample_generations,
    select_renderer_name,
    slugify_name,
)


RUN_OUTPUTS_DIRNAME = "run_outputs"
SAMPLER_TESTS_DIRNAME = "sampler_tests"
PROMPT_LIBRARY_FILENAME = "sampler_prompt_sets.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a Tinker sampler checkpoint using held-out or custom prompts."
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Workspace root containing the dataset and run_outputs.",
    )
    parser.add_argument(
        "--run-name",
        default="essay_recent_r16",
        help="Experiment run name used to infer the latest sampler checkpoint.",
    )
    parser.add_argument(
        "--sampler-checkpoint",
        help="Explicit sampler checkpoint path. Defaults to the latest local sampler path for --run-name.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Custom opening text to continue. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        help="UTF-8 text file with one custom opening per line.",
    )
    parser.add_argument(
        "--prompt-set",
        action="append",
        default=[],
        help="Named prompt set from sampler_prompt_sets.json. Repeat to combine sets. Use 'all' for every set.",
    )
    parser.add_argument(
        "--list-prompt-sets",
        action="store_true",
        help="Print the available built-in prompt sets and exit.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation", "train"),
        default="test",
        help="Dataset split to draw held-out prompts from when no custom prompts are provided.",
    )
    parser.add_argument(
        "--heldout-limit",
        type=int,
        default=3,
        help="How many held-out prompts to sample when no custom prompts are provided.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Maximum tokens to generate for each sample.",
    )
    parser.add_argument(
        "--is-reply",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat custom prompts as reply-style prompts.",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a JSON record of the sampling run under run_outputs/sampler_tests.",
    )
    return parser.parse_args()


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_custom_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [str(item).strip() for item in args.prompt if str(item).strip()]
    if args.prompt_file:
        prompt_file = Path(args.prompt_file).expanduser().resolve()
        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def load_prompt_library(workspace_root: Path) -> dict[str, dict[str, Any]]:
    library_path = workspace_root / PROMPT_LIBRARY_FILENAME
    if not library_path.exists():
        return {}
    data = json.loads(library_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{library_path} must contain a JSON object.")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        prompts = value.get("prompts")
        if not isinstance(prompts, list):
            continue
        normalized[str(key)] = {
            "label": str(value.get("label") or key),
            "description": str(value.get("description") or ""),
            "prompts": [str(item).strip() for item in prompts if str(item).strip()],
        }
    return normalized


def merge_interview_prompt_sets(
    workspace_root: Path,
    prompt_library: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = dict(prompt_library)
    dataset_root = find_dataset_root(workspace_root)
    interview_rows = load_processed_interview_rows(dataset_root)
    for set_name, prompt_set in build_interview_prompt_sets(interview_rows).items():
        merged[set_name] = {
            "label": prompt_set.label,
            "description": prompt_set.description,
            "prompts": list(prompt_set.prompts),
        }
    return merged


def print_prompt_library(prompt_library: dict[str, dict[str, Any]]) -> None:
    if not prompt_library:
        print("No built-in prompt sets found.")
        return
    print("Built-in prompt sets")
    print("--------------------")
    for key, entry in prompt_library.items():
        print(f"{key}: {entry['label']}")
        description = str(entry.get("description") or "").strip()
        if description:
            print(f"  {description}")
        for prompt in entry.get("prompts", []):
            print(f"  - {prompt}")
        print()


def load_prompt_sets(
    *,
    requested_sets: list[str],
    prompt_library: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if not requested_sets:
        return [], []

    expanded_set_names: list[str] = []
    prompts: list[str] = []
    seen_prompts: set[str] = set()

    for requested_name in requested_sets:
        normalized_name = str(requested_name).strip()
        if not normalized_name:
            continue
        if normalized_name == "all":
            selected_names = list(prompt_library)
        else:
            if normalized_name not in prompt_library:
                available = ", ".join(sorted(prompt_library)) or "none"
                raise KeyError(
                    f"Unknown prompt set: {normalized_name}. Available prompt sets: {available}"
                )
            selected_names = [normalized_name]

        for set_name in selected_names:
            if set_name not in expanded_set_names:
                expanded_set_names.append(set_name)
            for prompt in prompt_library[set_name]["prompts"]:
                if prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    prompts.append(prompt)

    return prompts, expanded_set_names


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


def build_custom_user_prompt(opening_text: str, *, is_reply: bool) -> str:
    instructions = [
        "Write a finished Bluesky post in the target author's voice.",
        "Keep the opening exactly as given.",
        "Match the tone, brevity, formatting, and internet-native style of the author's posts.",
        "Return only the final post text.",
    ]
    if is_reply:
        instructions.append("Make it read naturally as a reply.")
    return (
        " ".join(instructions)
        + "\n\nOpening:\n"
        + opening_text.strip()
    )


def build_custom_examples(openings: list[str], *, is_reply: bool) -> list[ConversationExample]:
    examples: list[ConversationExample] = []
    for index, opening_text in enumerate(openings, start=1):
        prompt = build_custom_user_prompt(opening_text, is_reply=is_reply)
        examples.append(
            ConversationExample(
                example_id=f"custom-{index:02d}",
                opening_text=opening_text,
                target_text="",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ],
                metadata={
                    "source": "custom",
                    "is_reply": bool(is_reply),
                },
            )
        )
    return examples


def get_experiment_specs_by_name() -> dict[str, dict[str, Any]]:
    return {item["run_name"]: item for item in get_experiment_specs(smoke_test=False)}


def choose_dataset_examples(
    variants: dict[str, Any],
    *,
    variant_name: str,
    split_name: str,
    limit: int,
) -> list[ConversationExample]:
    if variant_name not in variants:
        raise KeyError(f"Unknown dataset variant: {variant_name}")
    variant = variants[variant_name]
    split_examples = {
        "train": variant.train_examples,
        "validation": variant.validation_examples,
        "test": variant.test_examples,
    }[split_name]
    return build_eval_prompts(split_examples, limit=limit)


def resolve_sampler_checkpoint(
    *,
    run_dir: Path,
    run_name: str,
    explicit_checkpoint: str | None,
) -> tuple[str, dict[str, Any] | None]:
    if explicit_checkpoint:
        return explicit_checkpoint.strip(), None

    payload = find_latest_payload(run_dir, run_name=run_name)
    checkpoint = extract_sampler_checkpoint(payload)
    if checkpoint:
        return checkpoint, payload

    raise RuntimeError(
        f"No sampler checkpoint recorded for {run_name}. "
        f"Pass --sampler-checkpoint explicitly or run post-train sampling first."
    )


def infer_variant_name(
    *,
    run_name: str,
    payload: dict[str, Any] | None,
    experiment_specs_by_name: dict[str, dict[str, Any]],
) -> str:
    if isinstance(payload, dict):
        dataset_variant = payload.get("dataset_variant")
        if isinstance(dataset_variant, str) and dataset_variant.strip():
            return dataset_variant
    spec = experiment_specs_by_name.get(run_name)
    if spec is None:
        raise KeyError(f"Unknown run name: {run_name}")
    return str(spec["dataset_variant"])


def print_samples(
    rows: list[dict[str, str]],
    *,
    show_target: bool,
    checkpoint_path: str,
    model_name: str,
    source_label: str,
) -> None:
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[MODEL] {model_name}")
    print(f"[PROMPTS] {source_label}")
    print()

    for index, row in enumerate(rows, start=1):
        print(f"=== Sample {index} :: {row.get('example_id') or 'unknown'} ===")
        print("Opening:")
        print(row.get("opening_text") or "")
        if show_target:
            target_text = str(row.get("target_text") or "").strip()
            if target_text:
                print()
                print("Held-out target:")
                print(target_text)
        print()
        print("Generated:")
        print(row.get("generated_text") or "")
        print()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace).resolve()
    run_dir = workspace_root / RUN_OUTPUTS_DIRNAME
    prompt_library = merge_interview_prompt_sets(
        workspace_root,
        load_prompt_library(workspace_root),
    )

    if args.list_prompt_sets:
        print_prompt_library(prompt_library)
        return 0

    key_info = ensure_tinker_api_key(required=True)
    print(describe_tinker_api_key(key_info))

    custom_prompts = load_custom_prompts(args)
    prompt_set_prompts, prompt_set_names = load_prompt_sets(
        requested_sets=args.prompt_set,
        prompt_library=prompt_library,
    )
    combined_custom_prompts: list[str] = []
    seen_custom_prompts: set[str] = set()
    for prompt in [*custom_prompts, *prompt_set_prompts]:
        if prompt not in seen_custom_prompts:
            seen_custom_prompts.add(prompt)
            combined_custom_prompts.append(prompt)
    explicit_custom_mode = bool(combined_custom_prompts)

    sampler_checkpoint, payload = resolve_sampler_checkpoint(
        run_dir=run_dir,
        run_name=args.run_name,
        explicit_checkpoint=args.sampler_checkpoint,
    )

    experiment_specs_by_name = get_experiment_specs_by_name()
    variant_name = infer_variant_name(
        run_name=args.run_name,
        payload=payload,
        experiment_specs_by_name=experiment_specs_by_name,
    )

    dataset_root = find_dataset_root(workspace_root)
    bundle = load_dataset_bundle(dataset_root)
    variants = build_experiment_dataset_variants(bundle)

    if explicit_custom_mode:
        prompt_examples = build_custom_examples(combined_custom_prompts, is_reply=bool(args.is_reply))
        if prompt_set_names:
            joined_names = ", ".join(prompt_set_names)
            source_label = (
                f"{len(prompt_examples)} custom opening(s) "
                f"from prompt set(s): {joined_names}"
            )
        else:
            source_label = f"{len(prompt_examples)} custom opening(s)"
        show_target = False
    else:
        prompt_examples = choose_dataset_examples(
            variants,
            variant_name=variant_name,
            split_name=args.split,
            limit=max(1, int(args.heldout_limit)),
        )
        source_label = f"{len(prompt_examples)} held-out {args.split} prompt(s) from {variant_name}"
        show_target = True

    if not prompt_examples:
        raise RuntimeError("No prompts available to sample from.")

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    checkpoint_info = rest_client.get_weights_info_by_tinker_path(sampler_checkpoint).result()
    model_name = str(getattr(checkpoint_info, "base_model", None) or "unknown")

    sampling_client = service_client.create_sampling_client(sampler_checkpoint)
    tokenizer = sampling_client.get_tokenizer()
    renderer_name = select_renderer_name(model_name)
    renderer, _ = build_datums(
        prompt_examples[:1],
        tokenizer,
        model_name,
        renderer_name=renderer_name,
        max_length=512,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    rows = sample_generations(
        sampling_client,
        renderer,
        prompt_examples,
        max_tokens=max(1, int(args.max_tokens)),
        temperature=float(args.temperature),
    )
    print_samples(
        rows,
        show_target=show_target,
        checkpoint_path=sampler_checkpoint,
        model_name=model_name,
        source_label=source_label,
    )

    if args.save_json:
        timestamp = now_utc_compact()
        output_dir = run_dir / SAMPLER_TESTS_DIRNAME
        output_path = output_dir / f"{timestamp}-{slugify_name(args.run_name) or 'sampler-test'}.json"
        write_json(
            output_path,
            {
                "generated_at_utc": timestamp,
                "workspace_root": str(workspace_root),
                "run_name": args.run_name,
                "dataset_variant": variant_name,
                "split": args.split,
                "sampler_checkpoint": sampler_checkpoint,
                "model_name": model_name,
                "temperature": float(args.temperature),
                "max_tokens": int(args.max_tokens),
                "source_label": source_label,
                "custom_prompt_mode": explicit_custom_mode,
                "prompt_sets": prompt_set_names,
                "rows": rows,
            },
        )
        print(f"[SAVED] {output_path}")

    if explicit_custom_mode:
        print("[NOTE] These were custom openings you supplied, so there is no held-out target to compare against.")
        if prompt_set_names:
            print(f"[TIP] Re-run with --prompt-set {' --prompt-set '.join(prompt_set_names)} to compare variations in the same batch.")
    else:
        print("[NOTE] These were held-out dataset openings, so compare the target and generated text for tone and continuation quality.")
        print("[TIP] Add --prompt \"your opening here\" or --prompt-set targeted to test how it handles your own starts.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
