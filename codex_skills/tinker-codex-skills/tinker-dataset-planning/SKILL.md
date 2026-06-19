---
name: tinker-dataset-planning
description: Inspect a Tinker Studio workspace before training by finding the dataset root, summarizing Bluesky train/eval splits, checking refreshed post and reply-context counts, listing shortform/long-form/interview/imported source mixes, comparing experiment dataset variants, and surfacing evaluation caveats. Use when Codex needs to understand what data a Tinker run will train on, compare predefined run plans, or verify the Streamlit dashboard's dataset view.
---

# Tinker Dataset Planning

Use this inside a Tinker run-manager workspace that contains `tinker_training_utils.py`,
`tinker_experiment_manager.py`, `run_tinker_experiment.py`, and usually
`tinker_env\Scripts\python.exe`.

Prefer read-only inspection first. Use the workspace virtual environment when it exists.

## Setup And Safety

- Resolve data through `find_dataset_root(Path.cwd())`; it prefers `TINKER_STUDIO_DATASET_ROOT`, `TINKER_DATASET_ROOT`, then `data\training_data`.
- Treat `data\training_data` as private storage. It should be ignored by the main Tinker repo and may be its own separate private storage repo.
- Do not print or commit real env values. `.env.example` may contain empty placeholders and relative dataset paths only.

## Quick Start

List the predefined runs first:

```powershell
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --list-runs
```

Then inspect the dataset bundle, variant mix, and experiment plan:

```powershell
@'
from pathlib import Path
from tinker_training_utils import find_dataset_root, load_dataset_bundle, dataset_summary
from tinker_experiment_manager import (
    build_experiment_dataset_variants,
    build_dataset_variant_summary_df,
    build_experiment_plan_df,
    build_training_example_preview_rows,
)
from run_tinker_experiment import get_experiment_specs

root = find_dataset_root(Path.cwd())
bundle = load_dataset_bundle(root)
variants = build_experiment_dataset_variants(bundle)

print("dataset_root:", root)
print("train:", dataset_summary(bundle.train_rows))
print("validation:", dataset_summary(bundle.validation_rows))
print("test:", dataset_summary(bundle.test_rows))
print(build_dataset_variant_summary_df(variants).to_string(index=False))
print(build_experiment_plan_df(get_experiment_specs(smoke_test=False), variants, default_config={}).to_string(index=False))
preview = build_training_example_preview_rows(variants["conversational_voice_mix"], dataset_variant_name="conversational_voice_mix", limit=3)
for row in preview:
    print(row["example_id"], row["training_format"], row["transform"], row["raw_source_id"], row["message_roles"])

synthetic = root / "processed" / "synthetic_sources.jsonl"
print("synthetic_sources:", synthetic, "exists=", synthetic.exists())
'@ | .\tinker_env\Scripts\python.exe -
```

Check the refreshed Bluesky manifest and latest post timestamp:

```powershell
@'
import json
import pandas as pd
from pathlib import Path

root = Path("data") / "training_data"
manifest = json.loads((root / "tinker" / "dataset_manifest.json").read_text(encoding="utf-8"))
posts = pd.read_csv(root / "processed" / "posts.csv")
print(manifest["collected_at_utc"])
print(manifest["counts"])
print("latest_post:", posts["created_at"].max())
'@ | .\tinker_env\Scripts\python.exe -
```

## Dashboard Cross-Check

Use `streamlit_tinker_dashboard.py` or `launch_streamlit_dashboard.bat` when the user wants an
interactive dataset view. The dashboard reads:

- `data\training_data\processed\posts.csv`
- `data\training_data\tinker\dataset_manifest.json`
- `data\training_data\processed\rentry_pages.jsonl` for current long-form seed docs
- `data\training_data\processed\interview_qa.jsonl` for interview-derived Q&A and post-continuation examples
- `data\training_data\processed\imported_sources.jsonl` for Google Keep, poetry, notes, and other local imports
- `data\training_data\processed\synthetic_sources.jsonl` for curated synthetic rows

If the dashboard counts look wrong, validate these files directly before changing dashboard code.

## What To Look For

- Confirm the workspace really points at the intended `tinker\dataset_manifest.json`.
- Compare `initial_posts`, `recent_posts_plus_essays`, `recent_posts_essays_interview`, `personal_sources_mix`, and `conversational_voice_mix` before discussing metrics.
- Treat `personal_sources_mix` as the legacy completion-shaped blend and `conversational_voice_mix` as the chat-shaped blend with a conversational system prompt, direct interview Q&A, reply-context chat rows, capped short posts, and reflective longform/imported answers.
- Raw sources remain in processed/source JSONL/CSV files; trainable examples are materialized as `ConversationExample.messages` with metadata such as `training_format`, `transform`, `raw_transform`, `raw_source_id`, `raw_source_kind`, and `raw_text_sha256`.
- Synthetic rows are private generated data, currently loaded from `processed\synthetic_sources.jsonl`; verify titles and tags locally without quoting private identifiers into public reports.
- Use `--include-tag <tag>` and `--exclude-tag <tag>` on `run_tinker_experiment.py` to filter training examples only. Validation/test splits stay unchanged.
- Use `--export-training-preview` or the dashboard Training tab's example inspector to audit raw/source metadata next to exact rendered chat/completion messages before launching a full run.
- Treat shortform posts, long-form documents, conversations/reply context, interview rows, and imported notes as distinct source categories.
- Check `source_counts`, `train_examples`, `max_length`, `batch_size`, and `num_epochs` together.
- Use `get_experiment_specs(smoke_test=True)` when you want the run manager's reduced smoke-test plan.
- Treat the dataset directory as generated data. It may be ignored by Git even when refreshes rewrote files.

## Guardrails

- Do not describe `recent_posts_plus_essays` as a clean held-out eval setup. Its notes explicitly say later posts cross the original validation and test horizon, so post-train eval is disabled by default.
- If the venv is missing, fall back to reading the local Python modules instead of guessing the dataset shape.
- Prefer summarizing the actual variant tables and manifest over paraphrasing from memory.
