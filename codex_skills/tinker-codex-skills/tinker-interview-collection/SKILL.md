---
name: tinker-interview-collection
description: Collect, review, and save interview-style Q&A source material for Tinker Studio datasets. Use when Codex needs to interview a person, generate interview prompts, turn answers into chunked local training rows, use collect_interview_qa.py, review processed interview rows, or prepare interview-derived examples for a Tinker run.
---

# Tinker Interview Collection

## Overview

Use this in a Tinker Studio workspace to gather consented interview answers and write them into the generated dataset folder. The collector stores both raw answers and processed rows; `tinker_experiment_manager.py` later converts processed rows into direct Q&A examples and compact continuation chunks.

## Interview Guide

Before conducting or scripting an interview, read `references/interview-assistant-guide.md`. It contains template questions, conditional follow-ups, privacy guidance, and save/review commands.

## Setup And Safety

- Write interview rows only through `collect_interview_qa.py`; it resolves the dataset root through the Tinker helpers and should land under private dataset storage such as `data\training_data_cerise`.
- Confirm the user wants answers saved before writing raw or processed rows.
- Do not store third-party secrets, private contact details, or real env keys in interview rows.
- After saving, use `review --last 5` and `tinker-dataset-planning` to verify derived examples before training.

## Quick Commands

List available prompt rounds:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . list-rounds
```

Append an interview item with interactive prompts for missing fields:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . append --theme "taste and judgment" --tags taste judgment
```

Append an item with one follow-up answer and mark it as reply-style:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . followup --theme "reply style" --is-reply-style --tags replies shortform
```

Review processed rows:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . review --last 5
```

## Workflow

1. Read the guide and collect one theme at a time.
2. Preserve raw answers; only lightly edit for privacy, transcription, or user-approved cleanup.
3. Save via `append` or `followup`; the command writes raw and processed JSONL below the generated dataset root.
4. Run `review --last 5` and check the derived openings before using the rows in a run.
5. Use `tinker-dataset-planning` to confirm `recent_posts_essays_interview` or `personal_sources_mix` includes the processed interview corpus.

## Guardrails

- Do not save sensitive or third-party private details without explicit confirmation.
- Do not over-polish answers; the value is the person's real wording and reasoning shape.
- Keep poetry and deliberately fragmented prose line-preserved.
- If validation fails, fix the source answer or theme before forcing a row into the dataset.
