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
- Resolve the dataset through the workspace helpers. The default private dataset path is `data\training_data_cerise`.
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

After starting or resuming a run, verify state:

```powershell
.\tinker_env\Scripts\python.exe .\monitor_tinker_runs.py --workspace . --recent 6 --iterations 1
.\launch_streamlit_dashboard.bat
```

Use `launch_tinker_experiment.bat` only when the user explicitly wants the native Windows launcher
behavior with `pause` on exit.

## Workflow

1. Run `--list-runs` or `--describe-latest` when the current state is unclear.
2. Use `--smoke-test` first after code, dataset, or config changes.
3. Prefer `--resume` or `--auto-resume` when `run_outputs\latest_active_run.json` already points at an interrupted run.
4. Use `--resume-from-checkpoint` only when you have an explicit checkpoint path and need to bypass the default local record.
5. Use the Streamlit dashboard for browser-visible confirmation after the CLI command starts, not as a substitute for captured command output.

## Guardrails

- Do not start a duplicate full run if the monitor scripts still show an active heartbeat for the same run.
- `run_tinker_experiment.py` clears an existing stop request by default when starting a new run. Mention that when explaining why a pending stop file disappeared.
- Prefer the predefined run specs from `get_experiment_specs()` over ad hoc hyperparameter guessing unless the user asked for code changes.
- If the dataset was just refreshed from Bluesky, run `--smoke-test` before a full run.
