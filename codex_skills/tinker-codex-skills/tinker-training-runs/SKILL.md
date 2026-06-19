---
name: tinker-training-runs
description: Launch, list, describe, smoke-test, and resume Tinker training runs from a run-manager workspace through run_tinker_experiment.py, then verify state through CLI monitors or the Streamlit dashboard. Use when Codex needs to start or continue a predefined experiment run, check the latest local run record, or choose between fresh, resumed, and auto-resumed execution.
---

# Tinker Training Runs

Work from the Tinker run-manager workspace root. Prefer the direct Python CLI over the batch
launchers so Codex can capture output without an extra interactive window.

## Setup And Safety

- Use the workspace venv when present: `.\tinker_env\Scripts\python.exe`.
- Check key availability with `describe_tinker_api_key()` or existing helpers; never echo the real `TINKER_API_KEY`.
- Resolve the dataset through the workspace helpers. The default private dataset path is `data\training_data`.
- Before a full run after dataset or code changes, run a smoke test and check monitors for duplicate active runs.

## Quick Start

Check the environment and available runs:

```powershell
@'
from tinker_notebook_env import describe_tinker_api_key
print(describe_tinker_api_key())
'@ | .\tinker_env\Scripts\python.exe -

.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --list-runs
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name essay_recent_r16 --describe-latest
```

Common execution patterns:

```powershell
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name essay_recent_r16 --smoke-test
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name essay_recent_r16
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name essay_recent_r16 --resume
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name essay_recent_r16 --auto-resume
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name essay_recent_r16 --resume-from-checkpoint "<checkpoint-path>"
```

Preview the exact train/validation/test examples before creating a Tinker client:

```powershell
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name conversational_120b_r8_lr3e5_b6 --export-training-preview --preview-limit 25
```

Tag-filtered runs:

```powershell
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name conversational_120b_r8_lr3e5_b6 --smoke-test --include-tag synthetic
.\tinker_env\Scripts\python.exe .\run_tinker_experiment.py --workspace . --run-name conversational_120b_r8_lr3e5_b6 --exclude-tag synthetic
```

`--include-tag` keeps training examples that match at least one selected tag. `--exclude-tag`
drops any training example matching excluded tags. Both flags may be repeated or comma-separated;
validation/test splits stay unchanged for comparison.

After starting or resuming a run, verify state:

```powershell
.\tinker_env\Scripts\python.exe .\monitor_tinker_runs.py --workspace . --recent 6 --iterations 1
.\launch_streamlit_dashboard.bat
```

Use `launch_tinker_experiment.bat` only when the user explicitly wants the native Windows launcher
behavior with `pause` on exit.

## Workflow

1. Run `--list-runs` or `--describe-latest` when the current state is unclear.
2. Use `--export-training-preview` first after dataset-format or tag-filter changes when the exact rendered examples matter.
3. Use `--smoke-test` first after code, dataset, or config changes.
4. Prefer `--resume` or `--auto-resume` when `run_outputs\latest_active_run.json` already points at an interrupted run.
5. Use `--resume-from-checkpoint` only when you have an explicit checkpoint path and need to bypass the default local record.
6. Use the Streamlit dashboard for browser-visible confirmation after the CLI command starts, not as a substitute for captured command output.

## Guardrails

- Do not start a duplicate full run if the monitor scripts still show an active heartbeat for the same run.
- `run_tinker_experiment.py` clears an existing stop request by default when starting a new run. Mention that when explaining why a pending stop file disappeared.
- Prefer the predefined run specs from `get_experiment_specs()` over ad hoc hyperparameter guessing unless the user asked for code changes.
- For small stylized conversational corpora on `gpt-oss-120b`, prefer `conversational_120b_r8_lr3e5_b6` first, then `conversational_120b_r16_lr4e5_b6` if rank 8 underfits. Keep rank 32 for comparison when stronger adapter imprinting is intentional.
- If the dataset was just refreshed from Bluesky, run `--smoke-test` before a full run.
